# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/extract.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Stage 2 holistic extraction (ported from KPaperFlux
                core/ai/stage2.py): one call per logical document with the
                page images (primary) and auxiliary text/tables, type-specific
                questions and evidence schema from the catalog, optional
                forensic audit (Stage 1.5) folded into the same call.
                Large tables switch to page-chunked extraction with carry
                rows and deterministic merging. Schema violations trigger
                correction retries with error feedback; arithmetic checks are
                NOT fed back here (no pressure to invent values) but run in
                the plausibility stage.
------------------------------------------------------------------------------
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from fluxsense.catalog.registry import AUDIT_FULL, AUDIT_NONE, Catalog, TableSpec, TypeSpec
from fluxsense.config import Settings
from fluxsense.extract.base import TextLayer
from fluxsense.grounding.evidence import Obj, Rows, Scalar, Unwrapped, build_plan, unwrap, wire_schema
from fluxsense.grounding.normalize import dehyphenate, split_postcode_town
from fluxsense.grounding.pointer import native_pages, numbered_text, resolve_pointers
from fluxsense.layout import din5008
from fluxsense.llm import budget
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, page_images
from fluxsense.log import get_logger
from fluxsense.models.identity import IdentityProfile
from fluxsense.models.result import FieldEvidence, TableView
from fluxsense.models.semantic import BODY_MODELS, MetaHeader
from fluxsense.prompts import guide
from fluxsense.prompts import stage2 as prompts
from fluxsense.extract.fieldread import form_fields_block
from fluxsense.extract.handread import handwriting_block
from fluxsense.tables.stitch import drop_artefact_rows, merge_chunks, render_markdown, stitch_tables

logger = get_logger("stages.extract")

STAMP_TYPES = ["RECEIVED", "PAID", "COMPANY", "INTERNAL_FORM", "HANDWRITTEN_NOTE"]
LOCATIONS = ["TOP_RIGHT", "TOP_LEFT", "BOTTOM", "CENTER"]
_AMOUNT_LINE_RE = re.compile(r"\d[\d.]*,\d{2}\b|\d[\d,]*\.\d{2}\b")
TEXT_FIELDS_FOR_ARTEFACTS: Tuple[str, ...] = ("description", "counterparty")


def audit_schema(mode: str, type_names: List[str]) -> Dict[str, Any]:
    """JSON schema of the folded Stage 1.5 audit block."""
    props: Dict[str, Any] = {
        "stamps": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
            "type": {"type": "string", "enum": STAMP_TYPES}, "location": {"type": "string", "enum": LOCATIONS},
            "page": {"type": "integer"}, "text": {"type": "string", "maxLength": 120},
            "form_fields": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                "label": {"type": "string", "maxLength": 40}, "value": {"type": ["string", "null"], "maxLength": 60},
                "value_type": {"type": "string", "enum": ["DATE", "TIME", "NUMBER", "TEXT", "SIGNATURE", "EMPTY"]}}}}}}},
        "has_handwriting": {"type": "boolean"},
        "type_matches": {"type": "boolean"},
        "suggested_types": {"type": "array", "items": {"type": "string", "enum": type_names}},
    }
    if mode == AUDIT_FULL:
        props["signature"] = {"type": "object", "additionalProperties": False, "properties": {
            "has_signature": {"type": "boolean"}, "count": {"type": "integer"},
            "type": {"type": "string", "enum": ["HANDWRITTEN", "DIGITAL", "NONE"]}, "page": {"type": ["integer", "null"]}}}
    return {"type": "object", "additionalProperties": False, "properties": props, "required": ["type_matches"]}


def audit_to_kpaperflux(audit: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Maps the folded audit block to KPaperFlux's VisualAuditResult layout."""
    stamps = []
    for stamp in audit.get("stamps") or []:
        stamps.append({"raw_content": stamp.get("text"), "type": stamp.get("type"), "location": stamp.get("location"),
                       "page": stamp.get("page"),
                       "form_fields": [{"label": f.get("label"), "raw_value": f.get("value"), "normalized_value": f.get("value"),
                                        "value_type": f.get("value_type")} for f in stamp.get("form_fields") or []]})
    result: Dict[str, Any] = {
        "audit_summary": {"was_stamp_interference": bool(stamps), "has_handwriting": bool(audit.get("has_handwriting"))},
        "layer_stamps": stamps,
        "integrity": {"is_type_match": audit.get("type_matches", True), "suggested_types": audit.get("suggested_types") or []},
        "arbiter_decision": {"primary_source_recommendation": "AI_VISION"},
        "meta_mode": mode,
    }
    sig = audit.get("signature")
    if isinstance(sig, dict):
        result["signatures"] = {"has_signature": bool(sig.get("has_signature")), "count": int(sig.get("count") or 0),
                                "type": sig.get("type") or "NONE", "details": f"page {sig.get('page')}" if sig.get("page") else ""}
    return result


@dataclass
class ExtractionOutcome:
    """Result of Stage 2 for one logical document."""

    fields: Dict[str, Any]
    evidence: List[FieldEvidence] = field(default_factory=list)
    tables: List[TableView] = field(default_factory=list)
    audit: Optional[Dict[str, Any]] = None
    ai_confidence: float = 1.0
    strategy: str = "single_call"
    errors: List[str] = field(default_factory=list)
    removed_artefacts: int = 0
    image_pages: List[int] = field(default_factory=list)
    coverage_filled: List[int] = field(default_factory=list)
    pointer_stats: Optional[Dict[str, int]] = None


@dataclass
class _Build:
    """Cached schema/prompt build for a set of types."""

    plan: Obj
    schema: Dict[str, Any]
    system: str
    table_specs: List[TableSpec]
    body_keys: List[str]
    audit_mode: str
    single_page: Optional[int] = None


def select_image_pages(pages: List[int], limit: int) -> List[int]:
    """
    Pages to send as images: all if within the limit, else first and last pages.

    Args:
        pages: Pages of the logical document.
        limit: Max images per call.

    Returns:
        Selected pages.
    """
    if len(pages) <= limit:
        return list(pages)
    if limit <= 1:
        return list(pages[:1])
    head = (limit + 1) // 2
    return list(pages[:head]) + list(pages[-(limit - head):])


_DATE_RE = re.compile(r"(?<!\d)\d{1,2}\.\d{1,2}\.(?:\d{2,4})?(?!\d)|\d{4}-\d{2}-\d{2}")


def transaction_pages(layer: TextLayer, pages: List[int], min_lines: int = 2) -> List[int]:
    """Pages whose text shows at least `min_lines` lines with a date and an amount."""
    found: List[int] = []
    for p in pages:
        page = layer.page(p)
        if page is None:
            continue
        lines = page.all_text().splitlines()
        if sum(1 for line in lines if _DATE_RE.search(line) and _AMOUNT_LINE_RE.search(line)) >= min_lines:
            found.append(p)
    return found


def merge_by_page(rows: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merges rows of newly extracted pages into existing rows, ordered by source page (stable)."""
    combined = list(enumerate(rows)) + [(len(rows) + i, r) for i, r in enumerate(new_rows)]
    combined.sort(key=lambda item: (item[1].get("source_page") or 0, item[0]))
    return [r for _, r in combined]


_ROW_ANCHOR_RE = re.compile(r"^\s*\|?\s*(\d{1,4})\s*(?:\||\s{2,}|\t)")
# Header words of an invoice line-item table (an itemised call / usage list or booking details has none of them).
_ITEM_HEADER_RE = re.compile(r"\b(menge|anzahl|stück|stk\.?|einzelpreis|e-preis|stückpreis|artikel|art\.?-?nr|bezeichnung|"
                             r"beschreibung|leistungsbeschreibung|qty|quantity|unit\s+price|description|part\s+(?:no|number)|"
                             r"extended|pos\.?|position)\b", re.IGNORECASE)
# A clock time or duration ('14:32', '00:03:12') marks a usage / call row, never an invoice item row.
_CLOCK_RE = re.compile(r"(?<![\d,.])\d{1,2}:\d{2}(?::\d{2})?(?![\d,.])")
# Cues of the grand total (amount due) - stronger than the totals cues, which also match subtotals and table headers.
_GRAND_TOTAL_RE = re.compile(r"(rechnungsbetrag|gesamtbetrag|endbetrag|zu\s+zahlen|zahlbetrag|bruttobetrag|summe\s+brutto|"
                             r"gesamtsumme|amount\s+due|total\s+amount|grand\s+total|invoice\s+total|total\s+due)", re.IGNORECASE)
_TOTALS_CUE_RE = re.compile(r"(gesamt|total|zwischensumme|summe|rechnungsbetrag|zu zahlen|amount due|endbetrag|nettobetrag|"
                            r"vat on|mwst|umsatzsteuer|ust\.)", re.IGNORECASE)


def _page_lines(layer: TextLayer, page: int) -> List[str]:
    """Lines of a page for layout heuristics: the native text (column layout kept), else the best text."""
    text = layer.page(page)
    if text is None:
        return []
    native = text.sources.get("native", "")
    return (native if native.strip() else text.best_text()).splitlines()


def row_anchors(layer: TextLayer, pages: List[int]) -> Dict[int, List[int]]:
    """
    Item-row anchors per page: lines that start with a position number
    followed by a column gap or cell border and carry an amount (the shape
    of an invoice line-item row). Rows with a clock time (itemised call /
    usage lists) are no item rows, and a page counts only if it prints an
    item-table header (quantity, unit price, description, ...) or directly
    continues an accepted item page. Returns page -> printed position numbers.
    """
    found: Dict[int, List[int]] = {}
    for page in pages:
        lines = _page_lines(layer, page)
        positions = []
        for line in lines:
            match = _ROW_ANCHOR_RE.match(line)
            if match and _AMOUNT_LINE_RE.search(line[match.end():]) and not _CLOCK_RE.search(line):
                positions.append(int(match.group(1)))
        has_header = any(_ITEM_HEADER_RE.search(line) for line in lines)
        if len(positions) >= 2 and (has_header or (page - 1) in found):
            found[page] = positions
    return found


def grand_total_pages(layer: TextLayer, pages: List[int]) -> List[int]:
    """Pages printing the grand total (amount due) with an amount on its line or the next one."""
    out: List[int] = []
    for page in pages:
        lines = _page_lines(layer, page)
        for index, line in enumerate(lines):
            nearby = line + " " + (lines[index + 1] if index + 1 < len(lines) else "")
            if _GRAND_TOTAL_RE.search(line) and _AMOUNT_LINE_RE.search(nearby):
                out.append(page)
                break
    return out


def totals_pages(layer: TextLayer, pages: List[int]) -> List[int]:
    """Pages printing a totals block: a totals cue with an amount on its line or the next one."""
    out: List[int] = []
    for page in pages:
        lines = _page_lines(layer, page)
        for index, line in enumerate(lines):
            nearby = line + " " + (lines[index + 1] if index + 1 < len(lines) else "")
            if _TOTALS_CUE_RE.search(line) and _AMOUNT_LINE_RE.search(nearby):
                out.append(page)
                break
    return out


def position_gaps(rows: List[Dict[str, Any]], anchors: Dict[int, List[int]]) -> List[int]:
    """
    Pages whose printed positions are missing from the extracted rows: a gap
    in the position sequence or positions printed after the last extracted
    one (early stop).
    """
    have = set()
    for row in rows:
        try:
            have.add(int(str(row.get("pos")).strip()))
        except (TypeError, ValueError):
            continue
    if not have:
        return sorted(anchors)
    return sorted(page for page, positions in anchors.items() if any(p not in have for p in positions))


def percent_rates(rows: List[Dict[str, Any]]) -> int:
    """Tax rates returned as fractions (0.19) are normalised to percent (19); returns the number of changed rows."""
    changed = 0
    for row in rows:
        try:
            rate = float(row.get("tax_rate"))
        except (TypeError, ValueError):
            continue
        if 0 < rate < 1:
            row["tax_rate"] = round(rate * 100, 2)
            changed += 1
    return changed


def required_cells(rows_schema: Dict[str, Any], spec: TableSpec) -> Dict[str, Any]:
    """
    Chunk schema: the description and the row amount cells are not nullable
    (with all-nullable tuple cells the constrained decoder can return rows of
    nulls; every printed row has a text and an amount).
    """
    cells = ((rows_schema.get("items") or {}).get("prefixItems")) or []
    must = {"description", spec.amount_fields[-1] if spec.amount_fields else ""}
    for column, cell in zip(spec.columns, cells):
        if column in must and isinstance(cell.get("type"), list):
            cell["type"] = [t for t in cell["type"] if t != "null"]
    return rows_schema


def item_page_range(anchors: Dict[int, List[int]], sums: List[int], pages: List[int]) -> List[int]:
    """Pages of an invoice item table: from the first page with row anchors to the last anchor or totals page."""
    if len(anchors) < 1 or sum(len(v) for v in anchors.values()) < 3:
        return []
    first = min(anchors)
    last = max(list(anchors) + [p for p in sums if p >= first])
    return [p for p in pages if first <= p <= last]


def header_pages(pages: List[int], sums: List[int], limit: int, grand: Optional[List[int]] = None) -> List[int]:
    """
    Pages of the header call of a chunked invoice, by priority: the first
    page, the grand-total page(s), the last page, then other totals pages.
    """
    chosen = list(dict.fromkeys([pages[0]] + list(grand or [])[:2] + [pages[-1]] + [p for p in reversed(sums)]))
    return sorted(chosen[:max(1, limit)])


def missing_header_or_totals(fields: Dict[str, Any]) -> List[str]:
    """Header parts a chunked invoice's header call failed to deliver (number and date, or the totals)."""
    meta = fields.get("meta_header") or {}
    finance = (fields.get("bodies") or {}).get("finance_body") or {}
    sums = finance.get("monetary_summation") or {}
    missing = []
    if not (meta.get("doc_number") or finance.get("invoice_number")) and not (meta.get("doc_date") or finance.get("invoice_date")):
        missing.append("header")
    if sums.get("grand_total_amount") in (None, "") and sums.get("due_payable_amount") in (None, ""):
        missing.append("totals")
    return missing


def dedupe_positions(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drops repeated rows of the same printed position (chunk borders); rows without a position are kept."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("pos") or "").strip(), str(row.get("total_price")))
        if key[0] and key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def estimate_rows(layer: TextLayer, pages: List[int], stitched_rows: int) -> int:
    """Upper estimate of table rows (stitched rows or lines with amounts)."""
    lines = 0
    for p in pages:
        page = layer.page(p)
        if page is not None:
            lines += sum(1 for line in page.best_text().splitlines() if _AMOUNT_LINE_RE.search(line))
    return max(stitched_rows, lines)


TAIL_INSTRUCTION: str = "\nExtract now. Return ONLY the JSON.\n"
POINTER_RULE: str = (
    "\n### LINE POINTERS\nThe auxiliary text lines are numbered like [p1:L37]. For every field that has an \"l\" key, set \"l\" "
    "to the id of the text line where the value is printed (and still give your reading in \"v\").\n")


class Extractor:
    """Stage 2 controller."""

    def __init__(self, backend: LLMBackend, settings: Settings, catalog: Catalog) -> None:
        """
        Args:
            backend: LLM backend.
            settings: Settings.
            catalog: Type catalog.
        """
        self.backend = backend
        self.settings = settings
        self.catalog = catalog
        self._builds: Dict[str, _Build] = {}

    # ---------------------------------------------------------------- schema

    def build(self, specs: List[TypeSpec], with_tables: bool, identity_text: str, audit: bool, single_page: bool = False,
              pointer: bool = False) -> _Build:
        """
        Builds (cached) plan, wire schema and static system prompt for a type set.

        Args:
            specs: Type specs of the logical document.
            with_tables: Include big tables in the schema.
            identity_text: Unused (kept for the cache key API); the identity is NOT shown to the
                extraction model - it caused invented party names.
            audit: Fold the visual audit into the call.
            single_page: Omit page numbers from the wire schema.

        Returns:
            The build.
        """
        audit_mode = self.catalog.audit_mode([s.name for s in specs]) if audit else AUDIT_NONE
        key = f"{'+'.join(s.name for s in specs)}|{with_tables}|{audit_mode}|{single_page}|{pointer}"
        if key in self._builds:
            return self._builds[key]
        body_keys: List[str] = []
        table_specs: List[TableSpec] = []
        for spec in specs:
            for body_key in spec.body_keys():
                if body_key not in body_keys:
                    body_keys.append(body_key)
            table_specs += [t for t in spec.tables if t not in table_specs]
        bodies = Obj()
        for body_key in body_keys:
            tables = {t.field_name: t for t in table_specs if t.body_key == body_key}
            body_plan = build_plan(BODY_MODELS[body_key], tables=tables)
            if not with_tables:
                for name in tables:
                    body_plan.fields.pop(name, None)
            bodies.fields[body_key] = body_plan
        plan = Obj({"meta_header": build_plan(MetaHeader), "bodies": bodies})
        chars = self.settings.grounding.evidence_max_chars
        schema = wire_schema(plan, chars, with_page=not single_page, with_line=pointer)
        schema["properties"]["ai_confidence"] = {"type": "number", "description": "Your confidence 0.0-1.0 (derived)"}
        if audit_mode != AUDIT_NONE:
            schema["properties"]["audit"] = audit_schema(audit_mode, self.catalog.names())
        primary = specs[0]
        questions = []
        for spec in specs:
            questions += [q for q in spec.questions if q not in questions]
            if spec.doc_number_rule and spec.doc_number_rule not in questions:
                questions.append(spec.doc_number_rule)
        audit_text = ""
        if audit_mode == AUDIT_FULL:
            audit_text = prompts.PROMPT_STAGE_2_AUDIT_FULL
        elif audit_mode != AUDIT_NONE:
            audit_text = prompts.PROMPT_STAGE_2_AUDIT_STAMP
        system = prompts.PROMPT_STAGE_2_SYSTEM.format(
            entity_type=primary.name, type_description=primary.description,
            entity_types=", ".join(s.name for s in specs), body_keys=", ".join(body_keys),
            questions="\n".join(f"  {i + 1}. {q}" for i, q in enumerate(questions)), audit_mission=audit_text,
            target_schema_json=json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
            zugferd_guide=guide.guide_for(body_keys))
        if pointer:
            system += POINTER_RULE
        build = _Build(plan, schema, system, table_specs, body_keys, audit_mode)
        self._builds[key] = build
        return build

    def _with_page(self, build: _Build, pages: List[int]) -> _Build:
        """Returns the build annotated with the page of a single-page document (no shared mutation)."""
        if len(pages) != 1:
            return build
        return _Build(build.plan, build.schema, build.system, build.table_specs, build.body_keys, build.audit_mode, pages[0])

    # --------------------------------------------------------------- helpers

    def _images(self, pdf_path: Path, pages: List[int], max_px: Optional[int] = None) -> List[ImagePart]:
        """Medium-resolution page images (escalation level 1)."""
        llm = self.settings.llm
        return page_images(pdf_path, pages, [f"PAGE {p}" for p in pages], max_px or llm.max_image_px,
                           llm.image_format, dpi=llm.render_dpi)

    def _text(self, layer: TextLayer, pages: List[int], limit: int, strip_tables: bool = False) -> str:
        """
        Auxiliary text with page markers: ONE source per page (the best one),
        markdown tables compacted, and removed when they are sent stitched.
        """
        parts = []
        for p in pages:
            page = layer.page(p)
            text = dehyphenate(budget.compact_markdown(page.best_text())) if page else ""
            if strip_tables:
                text = budget.strip_markdown_tables(text)
            block = (form_fields_block(page) + handwriting_block(page)) if page else ""
            parts.append(f"=== PAGE {p} ===\n{block}{text}")
        return "\n\n".join(parts)[:limit]

    def validate(self, clean: Dict[str, Any]) -> List[str]:
        """
        Pydantic validation of the clean structure with KPaperFlux-style hints
        (missing vs. similar keys, value errors).

        Args:
            clean: Clean fields ({"meta_header", "bodies"}).

        Returns:
            Error strings.
        """
        errors: List[str] = []
        checks = [("meta_header", MetaHeader, clean.get("meta_header") or {})]
        checks += [(f"bodies -> {k}", BODY_MODELS[k], v) for k, v in (clean.get("bodies") or {}).items() if k in BODY_MODELS]
        for label, model, data in checks:
            try:
                model.model_validate(data)
            except ValidationError as e:
                for err in e.errors():
                    loc = " -> ".join(str(x) for x in err["loc"])
                    if err["type"] == "missing":
                        parent = data
                        for part in err["loc"][:-1]:
                            parent = parent[part] if isinstance(parent, (dict, list)) else None
                        similar = get_close_matches(str(err["loc"][-1]), list(parent.keys()) if isinstance(parent, dict) else [], n=3, cutoff=0.5)
                        hint = f" Similar keys found: {similar}." if similar else ""
                        errors.append(f"MISSING_FIELD [{label} -> {loc}]: mandatory field is missing.{hint}")
                    else:
                        errors.append(f"VALUE_ERROR [{label} -> {loc}]: {err['msg']}. You provided: {err.get('input')}.")
        return errors

    @staticmethod
    def sanitize(clean: Dict[str, Any]) -> List[str]:
        """
        Deterministic clean-up instead of a costly correction call: drops tax
        rows lacking mandatory values and references without value.

        Returns:
            Notes about removed items.
        """
        notes: List[str] = []
        finance = (clean.get("bodies") or {}).get("finance_body")
        if isinstance(finance, dict) and isinstance(finance.get("tax_breakdown"), list):
            keep = [r for r in finance["tax_breakdown"]
                    if all(r.get(k) is not None for k in ("tax_basis_amount", "tax_amount", "tax_rate"))]
            if len(keep) != len(finance["tax_breakdown"]):
                notes.append(f"dropped {len(finance['tax_breakdown']) - len(keep)} incomplete tax_breakdown rows")
            finance["tax_breakdown"] = keep
        if isinstance(finance, dict):
            fixed = percent_rates(finance.get("line_items") or []) + percent_rates(finance.get("tax_breakdown") or [])
            if fixed:
                notes.append(f"{fixed} tax rates given as fractions normalised to percent")
        meta = clean.get("meta_header")
        if isinstance(meta, dict) and isinstance(meta.get("references"), list):
            meta["references"] = [r for r in meta["references"] if r.get("ref_type") and r.get("ref_value")]
        return notes

    # ------------------------------------------------------------------- run

    def run(self, pdf_path: Path, layer: TextLayer, pages: List[int], type_tags: List[str],
            private_id: Optional[IdentityProfile], business_id: Optional[IdentityProfile],
            zugferd_present: bool = False, audit: bool = True, aux_text: bool = True,
            item_chunking: bool = True) -> Optional[ExtractionOutcome]:
        """
        Extracts one logical document holistically.

        Args:
            pdf_path: PDF file.
            layer: Auxiliary text layer.
            pages: 1-based pages of the logical document (its own pages, never page 1 of the file).
            type_tags: Document types.
            private_id: Private identity.
            business_id: Business identity.
            zugferd_present: The PDF embeds ZUGFeRD XML.
            audit: Fold the visual audit into the call.
            aux_text: Send the auxiliary text (False = images only, used for self-consistency).
            item_chunking: Allow the upfront page-chunked line-item extraction (False = single call).

        Returns:
            ExtractionOutcome, or None if the model failed.
        """
        cfg = self.settings.pipeline
        specs = self.catalog.specs_for(type_tags)
        identity = (f"Private: {private_id.model_dump_json(exclude_none=True) if private_id else '{}'}\n"
                    f"Business: {business_id.model_dump_json(exclude_none=True) if business_id else '{}'}")
        page_tables = [t for p in pages for t in (layer.page(p).tables if layer.page(p) else [])]
        stitched = stitch_tables(page_tables)
        has_big_tables = any(s.tables for s in specs)
        stitched_rows = sum(len(t.rows) for t in stitched)
        expected = 1500 + (estimate_rows(layer, pages, stitched_rows) * cfg.tokens_per_row if has_big_tables else 0)
        # Upfront chunking only for statements (rows are bookings); other types rely on the truncation fallback,
        # because amount-line counts over-estimate (e.g. itemised call lists are not line items).
        is_statement = any(t.body_key == "statement_body" for s in specs for t in s.tables)
        rows_expected = estimate_rows(layer, pages, stitched_rows)
        chunked = is_statement and len(pages) > 1 and (
            expected > cfg.output_budget_tokens or len(pages) > cfg.statement_chunk_pages or rows_expected > cfg.statement_chunk_rows)
        # Invoice line items: chunk when the printed item rows reach beyond the pages one call can see
        is_items = any(t.field_name == "line_items" for s in specs for t in s.tables)
        anchors = row_anchors(layer, pages) if is_items and len(pages) > 1 else {}
        sums = totals_pages(layer, pages) if is_items else []
        grand = grand_total_pages(layer, pages) if is_items else []
        item_pages = item_page_range(anchors, sums, pages)
        default_images = select_image_pages(pages, self.settings.llm.max_images_per_call)
        chunk_items = item_chunking and bool(item_pages) and (any(p not in default_images for p in item_pages)
                                             or sum(len(v) for v in anchors.values()) * cfg.tokens_per_row > cfg.output_budget_tokens)
        chunked = chunked or chunk_items
        pointer = cfg.pointer_extraction and aux_text and native_pages(layer, pages)
        build = self._with_page(self.build(specs, with_tables=not chunked, identity_text=identity, audit=audit,
                                           single_page=len(pages) == 1, pointer=pointer), pages)
        # Chunk / coverage calls use the system prompt WITH the table schema: under the header-only prompt the model
        # returns rows of nulls.
        table_system = build.system if not chunked else self.build(
            specs, with_tables=True, identity_text=identity, audit=False, single_page=False, pointer=False).system
        image_pages = select_image_pages(pages, self.settings.llm.max_images_per_call)
        if chunked:
            image_pages = select_image_pages(pages, min(4, self.settings.llm.max_images_per_call))
        if chunk_items:
            image_pages = header_pages(pages, sums, min(4, self.settings.llm.max_images_per_call), grand)
        hints = din5008.analyze(layer.page(pages[0]), layer.page(pages[-1]))
        line_map: Dict[str, str] = {}

        def compose(text_pages: List[int]) -> str:
            """User prompt whose auxiliary text/tables cover EXACTLY the pages whose images are attached
            (models follow the text over the images: extra text pages make them invent rows)."""
            nonlocal line_map
            page_tables = [t for p in text_pages for t in (layer.page(p).tables if layer.page(p) else [])]
            useful = [t for t in stitch_tables(page_tables) if len(t.rows) >= 3] if aux_text and not chunked else []
            tables_hint = prompts.PROMPT_TABLES_HINT.format(tables_markdown="\n\n".join(render_markdown(t) for t in useful)) \
                if useful else ""
            if pointer:
                document_text, line_map = numbered_text(layer, text_pages, cfg.max_text_chars)
            elif aux_text:
                document_text = self._text(layer, text_pages, cfg.max_text_chars, strip_tables=bool(tables_hint))
            else:
                document_text = "(not provided - read the images)"
            return prompts.PROMPT_STAGE_2_USER.format(
                page_list=", ".join(f"PAGE {p}" for p in text_pages), document_text=document_text, tables_hint=tables_hint,
                layout_hint=(prompts.PROMPT_LAYOUT_HINT.format(page=pages[0], regions=hints.prompt_block())
                             if not hints.empty() and pages[0] in text_pages else ""),
                zugferd_hint=prompts.PROMPT_ZUGFERD_HINT if zugferd_present else "")

        request = LLMRequest(prompt=compose(image_pages), system=build.system, label=f"Stage 2 ({specs[0].name})",
                             model=self.settings.models.primary, images=self._images(pdf_path, image_pages),
                             schema=build.schema, schema_name="extraction",
                             max_tokens=min(self.settings.llm.max_output_tokens, max(4096, expected + 2048)))
        fit_notes = budget.fit(request, self.settings.llm.context_tokens,
                               rerender=lambda px: self._images(pdf_path, image_pages, px), tail=TAIL_INSTRUCTION,
                               protect=([pages[0]] + grand[:2]) if grand else ())
        if fit_notes:
            logger.info(f"Stage 2 request fitted to the slot context: {fit_notes}")
            kept = [int(img.label.split()[-1]) for img in request.images]
            if kept != image_pages:
                image_pages = kept
                request.prompt = compose(image_pages)
                budget.fit(request, self.settings.llm.context_tokens, tail=TAIL_INSTRUCTION)
        wire = self.backend.generate_json(request)
        if request.truncated and has_big_tables and not chunked and len(pages) > 1:
            logger.warning("Stage 2 output truncated: switching to page-chunked table extraction")
            chunked = True
            build = self._with_page(self.build(specs, with_tables=False, identity_text=identity, audit=audit), pages)
            request = replace(request, system=build.system, schema=build.schema, label=f"Stage 2 ({specs[0].name}) header",
                              prompt=compose(image_pages),
                              max_tokens=4096, truncated=False)
            wire = self.backend.generate_json(request)
        if not isinstance(wire, dict):
            return None
        outcome = self._unwrap(build, wire, chunked)
        attempt = 0
        while outcome.errors and attempt < cfg.ai_retries:
            attempt += 1
            logger.info(f"Stage 2 validation failed (attempt {attempt}): {outcome.errors[:5]}")
            retry_request = replace(request, label=f"Stage 2 Correction {attempt}", truncated=False, followups=[
                {"role": "assistant", "content": json.dumps(wire, ensure_ascii=False, separators=(",", ":"))},
                {"role": "user", "content": prompts.PROMPT_STAGE_2_CORRECTION.format(
                    error_msg="\n".join(f"- {e}" for e in outcome.errors))}])
            retry = self.backend.generate_json(retry_request)
            if not isinstance(retry, dict):
                break
            wire = retry
            outcome = self._unwrap(build, wire, chunked)
        outcome.image_pages = image_pages
        if chunk_items and missing_header_or_totals(outcome.fields):
            logger.warning(f"chunked invoice: header call returned no {' / '.join(missing_header_or_totals(outcome.fields))}; "
                           f"falling back to the single-call extraction")
            fallback = self.run(pdf_path, layer, pages, type_tags, private_id, business_id, zugferd_present, audit, aux_text,
                                item_chunking=False)
            if fallback is not None:
                fallback.strategy = "single_call_fallback"
                return fallback
        if pointer:
            outcome.pointer_stats = resolve_pointers(outcome.evidence, outcome.fields, line_map)
        if chunked:
            self._extract_chunks(pdf_path, layer, pages, build, outcome, item_pages if chunk_items else None, table_system)
        self._ensure_coverage(pdf_path, layer, pages, build, outcome, anchors, table_system)
        if is_items:
            self._ensure_totals(pdf_path, layer, build, outcome, sums, image_pages, compose)
        outcome.tables = self._table_views(build, outcome, "chunked" if chunked else "single_call")
        return outcome

    def _unwrap(self, build: _Build, wire: Dict[str, Any], chunked: bool) -> ExtractionOutcome:
        """Unwraps the wire answer and validates it."""
        collector = Unwrapped(None)
        clean = unwrap(build.plan, wire, "", collector, self.settings.grounding.evidence_max_chars) or {}
        if build.single_page:
            for ev in collector.evidence:
                ev.page = build.single_page
            for spec in build.table_specs:
                rows = ((clean.get("bodies") or {}).get(spec.body_key) or {}).get(spec.field_name) or []
                for row in rows:
                    row.setdefault("source_page", build.single_page)
        clean.setdefault("meta_header", {})
        clean.setdefault("bodies", {})
        notes = self.sanitize(clean)
        split = split_postcode_town(clean)
        if split:
            notes.append(f"postcode and town separated: {', '.join(sorted(split))}")
            for ev in collector.evidence:
                if ev.path in split:
                    ev.value = split[ev.path]
        removed = 0
        for spec in build.table_specs:
            body = clean["bodies"].get(spec.body_key)
            if isinstance(body, dict) and isinstance(body.get(spec.field_name), list):
                body[spec.field_name], dropped = drop_artefact_rows(body[spec.field_name], TEXT_FIELDS_FOR_ARTEFACTS)
                removed += dropped
        audit = None
        if build.audit_mode != AUDIT_NONE and isinstance(wire.get("audit"), dict):
            audit = audit_to_kpaperflux(wire["audit"], build.audit_mode)
        try:
            confidence = float(wire.get("ai_confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        outcome = ExtractionOutcome(fields=clean, evidence=collector.evidence, audit=audit,
                                    ai_confidence=max(0.0, min(1.0, confidence)),
                                    strategy="chunked" if chunked else "single_call",
                                    errors=self.validate(clean), removed_artefacts=removed)
        if notes:
            logger.info(f"Sanitized extraction: {notes}")
        return outcome

    def _extract_table_pages(self, pdf_path: Path, layer: TextLayer, pages: List[int], system: str,
                             spec: TableSpec, parallel: bool = False) -> Tuple[List[Dict[str, Any]], int, List[List[int]]]:
        """
        Page-chunked extraction of one table with carry rows and deterministic merge.

        Args:
            pdf_path: PDF file.
            layer: Text layer.
            pages: Pages to extract (ascending).
            system: Static system prompt (KV cache).
            spec: Table spec.
            parallel: Run the chunks concurrently without carry rows (invoice line items: rows are
                independent); duplicates at chunk borders are removed by the merge.

        Returns:
            Tuple (merged rows, removed artefact rows, failed chunks).
        """
        cfg = self.settings.pipeline
        rows_plan: Rows = build_plan(BODY_MODELS[spec.body_key], tables={spec.field_name: spec}).fields[spec.field_name]
        schema = {"type": "object", "additionalProperties": False, "properties": {"rows": required_cells(wire_schema(rows_plan), spec)},
                  "required": ["rows"]}
        chunks: List[List[Dict[str, Any]]] = []
        merged: List[Dict[str, Any]] = []
        removed = 0
        failures: List[List[int]] = []
        size = 1 if parallel else cfg.chunk_pages
        groups = [pages[start:start + size] for start in range(0, len(pages), size)]

        def request_for(chunk_pages: List[int], carry: List[List[Any]]) -> LLMRequest:
            prompt = prompts.PROMPT_TABLE_CHUNK_USER.format(
                table=spec.name, pages=", ".join(str(p) for p in chunk_pages), columns=spec.column_hint(),
                carry_rows=json.dumps(carry, ensure_ascii=False, default=str) if carry else "(none - this is the first chunk)",
                document_text=self._text(layer, chunk_pages, cfg.max_text_chars))
            request = LLMRequest(prompt=prompt, system=system, label=f"Stage 2 chunk {spec.name} p{chunk_pages[0]}-{chunk_pages[-1]}",
                                 model=self.settings.models.primary, images=self._images(pdf_path, chunk_pages),
                                 schema=schema, schema_name="table_chunk",
                                 max_tokens=min(self.settings.llm.max_output_tokens, 800 + 60 * len(chunk_pages) * cfg.tokens_per_row))
            budget.fit(request, self.settings.llm.context_tokens, rerender=lambda px, cp=chunk_pages: self._images(pdf_path, cp, px))
            return request

        def rows_of(chunk_pages: List[int], wire: Any) -> List[Dict[str, Any]]:
            nonlocal removed
            rows: List[Dict[str, Any]] = []
            if isinstance(wire, dict):
                rows = unwrap(rows_plan, wire.get("rows") or [], spec.path, Unwrapped(None)) or []
                rows, dropped = drop_artefact_rows(rows, TEXT_FIELDS_FOR_ARTEFACTS)
                removed += dropped
            else:
                failures.append(chunk_pages)
            for row in rows:
                if row.get("source_page") not in chunk_pages:
                    row["source_page"] = chunk_pages[0] if len(chunk_pages) == 1 else None
            percent_rates(rows)
            return rows

        if parallel:
            requests = [request_for(g, []) for g in groups]
            with ThreadPoolExecutor(max_workers=max(1, self.settings.llm.max_concurrency)) as pool:
                answers = list(pool.map(self.backend.generate_json, requests))
            chunks = [rows_of(g, a) for g, a in zip(groups, answers)]
            merged, _ = merge_chunks(chunks, cfg.carry_rows)
            return merged, removed, failures
        for chunk_pages in groups:
            carry = [[r.get(c) for c in spec.columns] for r in merged[-cfg.carry_rows:]]
            chunks.append(rows_of(chunk_pages, self.backend.generate_json(request_for(chunk_pages, carry))))
            merged, _ = merge_chunks(chunks, cfg.carry_rows)
        return merged, removed, failures

    def _extract_chunks(self, pdf_path: Path, layer: TextLayer, pages: List[int], build: _Build,
                        outcome: ExtractionOutcome, item_pages: Optional[List[int]] = None,
                        table_system: Optional[str] = None) -> None:
        """Page-chunked extraction of all big tables of the build (invoice line items: their pages, in parallel)."""
        for spec in build.table_specs:
            items = spec.field_name == "line_items" and item_pages is not None
            rows, removed, failures = self._extract_table_pages(pdf_path, layer, item_pages if items else pages,  # type: ignore[arg-type]
                                                                table_system or build.system, spec, parallel=items)
            if items:
                rows = dedupe_positions(rows)
            outcome.removed_artefacts += removed
            outcome.errors += [f"chunk {f} of {spec.name} failed" for f in failures]
            outcome.fields["bodies"].setdefault(spec.body_key, {})[spec.field_name] = rows

    def _ensure_coverage(self, pdf_path: Path, layer: TextLayer, pages: List[int], build: _Build,
                         outcome: ExtractionOutcome, anchors: Optional[Dict[int, List[int]]] = None,
                         table_system: Optional[str] = None) -> None:
        """
        Coverage checks that do not depend on totals. Statements: every page
        with transaction-like lines (date + amount) must contribute rows.
        Invoice line items: every page with item-row anchors must contribute
        rows, and the printed position sequence must have no gap or early
        stop. Missing pages are extracted page-chunked (in parallel for line
        items) and merged in page order.
        """
        for spec in build.table_specs:
            if spec.field_name == "line_items" and anchors:
                body = outcome.fields["bodies"].setdefault(spec.body_key, {})
                rows = body.get(spec.field_name) or []
                covered = {r.get("source_page") for r in rows}
                missing = sorted(set(position_gaps(rows, anchors)) | {p for p in anchors if p not in covered})
                if not missing:
                    continue
                logger.warning(f"Coverage gap in {spec.name}: rows of pages {missing} missing (positions / row anchors); "
                               f"extracting them page-chunked")
                new_rows, removed, _failures = self._extract_table_pages(pdf_path, layer, missing, table_system or build.system, spec,
                                                                         parallel=True)
                outcome.removed_artefacts += removed
                replaced = [p for p in missing if sum(1 for r in new_rows if r.get("source_page") == p)
                            > sum(1 for r in rows if r.get("source_page") == p)]
                outcome.coverage_filled += replaced
                kept = [r for r in rows if r.get("source_page") not in replaced]
                fresh = [r for r in new_rows if r.get("source_page") in replaced]
                body[spec.field_name] = dedupe_positions(merge_by_page(kept, fresh))
                continue
            if spec.body_key != "statement_body":
                continue
            body = outcome.fields["bodies"].setdefault(spec.body_key, {})
            rows: List[Dict[str, Any]] = body.get(spec.field_name) or []
            expected_pages = transaction_pages(layer, pages)
            if rows and sum(1 for r in rows if isinstance(r.get("source_page"), int)) < 0.8 * len(rows):
                logger.info("coverage check skipped: too many rows without source page")
                continue
            covered = {r.get("source_page") for r in rows}
            missing = [p for p in expected_pages if p not in covered]
            if not missing:
                continue
            logger.warning(f"Coverage gap in {spec.name}: no rows from pages {missing}; extracting them page-chunked")
            new_rows, removed, failures = self._extract_table_pages(pdf_path, layer, missing, table_system or build.system, spec)
            outcome.removed_artefacts += removed
            outcome.coverage_filled += missing
            body[spec.field_name] = merge_by_page(rows, new_rows)

    def _ensure_totals(self, pdf_path: Path, layer: TextLayer, build: _Build, outcome: ExtractionOutcome,
                       sums: List[int], seen: List[int], compose: Any) -> None:
        """
        A totals block printed on a page the header call did not see: re-read
        the totals, tax breakdown and payment data from those pages and fill
        the empty fields.
        """
        finance = outcome.fields["bodies"].get("finance_body")
        ms = (finance or {}).get("monetary_summation") or {}
        unseen = [p for p in sums if p not in seen]
        if not unseen or (ms.get("grand_total_amount") and ms.get("tax_basis_total_amount")):
            return
        pages = unseen[-2:]
        request = LLMRequest(prompt=compose(pages), system=build.system, label=f"Stage 2 totals p{pages[0]}-{pages[-1]}",
                             model=self.settings.models.primary, images=self._images(pdf_path, pages), schema=build.schema,
                             schema_name="extraction", max_tokens=2048)
        budget.fit(request, self.settings.llm.context_tokens, rerender=lambda px: self._images(pdf_path, pages, px),
                   tail=TAIL_INSTRUCTION)
        wire = self.backend.generate_json(request)
        if not isinstance(wire, dict):
            return
        extra = self._unwrap(build, wire, True)
        found = (extra.fields.get("bodies") or {}).get("finance_body") or {}
        target = outcome.fields["bodies"].setdefault("finance_body", {})
        filled = []
        for key in ("monetary_summation", "tax_breakdown", "payment_accounts", "due_date", "payment_terms"):
            value = found.get(key)
            if not value:
                continue
            if key == "monetary_summation":
                current = target.setdefault(key, {})
                for sub, amount in value.items():
                    if current.get(sub) in (None, "") or sub in ("grand_total_amount", "tax_basis_total_amount", "tax_total_amount"):
                        current[sub] = amount
                        filled.append(f"{key}.{sub}")
            elif not target.get(key) or key == "tax_breakdown":
                target[key] = value
                filled.append(key)
        prefixes = tuple(f"bodies.finance_body.{f}" for f in filled)
        outcome.evidence = [e for e in outcome.evidence if not e.path.startswith(prefixes)] + \
            [e for e in extra.evidence if e.path.startswith(prefixes)]
        if filled:
            logger.info(f"Totals block on unseen pages {pages}: filled {filled}")

    def reextract_table(self, pdf_path: Path, layer: TextLayer, pages: List[int], type_tags: List[str],
                        table_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Re-extracts one table of a document page-chunked (used by the correction
        pass when a balance/sum check fails over many pages).

        Returns:
            The rows, or None if the table is unknown or every chunk failed.
        """
        specs = self.catalog.specs_for(type_tags)
        table = next((t for s in specs for t in s.tables if t.name == table_name), None)
        if table is None:
            return None
        build = self.build(specs, with_tables=True, identity_text="(see document)", audit=False)
        rows, _, failures = self._extract_table_pages(pdf_path, layer, pages, build.system, table)
        if failures and len(failures) * self.settings.pipeline.chunk_pages >= len(pages):
            return None
        return rows

    @staticmethod
    def _table_views(build: _Build, outcome: ExtractionOutcome, strategy: str) -> List[TableView]:
        """Builds the logical table views from the clean fields."""
        views: List[TableView] = []
        for spec in build.table_specs:
            rows = ((outcome.fields.get("bodies") or {}).get(spec.body_key) or {}).get(spec.field_name) or []
            columns = [c for c in spec.columns if c != "source_page"]
            pages = sorted({int(r["source_page"]) for r in rows if isinstance(r.get("source_page"), int)})
            views.append(TableView(name=spec.name, path=spec.path, columns=columns, rows=rows, source_pages=pages,
                                   strategy=strategy))
        return views
