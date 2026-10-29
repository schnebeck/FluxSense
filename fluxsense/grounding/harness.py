# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/harness.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Hallucination guard. Grounds every extracted value against
                the text sources, grounds table rows per page, escalates
                uncertain critical values (ungrounded / weak / involved in a
                failed check) to high-DPI region crops (zoom), optionally to a
                text-only second model, then nulls or replaces values that
                stay unverified - never passing them through silently - and
                builds the per-document trust report.
------------------------------------------------------------------------------
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fluxsense.catalog.contract import contract_for, derived_period_value
from fluxsense.catalog.registry import TableSpec, TypeSpec
from fluxsense.config import Settings
from fluxsense.extract.base import TextLayer
from fluxsense.grounding.matcher import (CHECKED_KINDS, KIND_AMOUNT, KIND_DATE, KIND_IBAN, KIND_IDENTIFIER, KIND_INTEGER, Match,
                                         build_index, ground_value)
from fluxsense.grounding.normalize import amount_candidates, compact, parse_date, to_decimal
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.log import get_logger
from fluxsense.grounding.snap import (SNAP_KINDS, best_iban, good_text, iban_candidates, iban_valid, normalize_iban,
                                     snap_value)
from fluxsense.models.result import (CheckResult, CheckStatus, CorrectionRecord, FieldEvidence, GroundingStatus, Severity,
                                     TableView, TrustReport)
from fluxsense.prompts import verify as prompts
from fluxsense.validate.general import validate_iban
from fluxsense.grounding.blz import known_german_blz
from fluxsense.grounding.snap import choose_iban_repair
from fluxsense.vision.zoom import ZoomTarget, Zoomer, binary_choice

logger = get_logger("grounding")

SECOND_OPINION_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["answers"],
    "properties": {"answers": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "supported"],
        "properties": {"id": {"type": "string"}, "supported": {"type": "boolean"},
                       "printed": {"type": ["string", "null"], "maxLength": 80}}}}},
}


def path_matches(path: str, pattern: str) -> bool:
    """Segment-wise match where '*' matches exactly one segment."""
    p_parts, q_parts = path.split("."), pattern.split(".")
    return len(p_parts) == len(q_parts) and all(q == "*" or q == p for p, q in zip(p_parts, q_parts))


IBAN_ZOOM_DPI: int = 400


def is_critical(path: str, specs: Sequence[TypeSpec]) -> bool:
    """True if a path matches a critical-field pattern of any spec; IBANs are always critical."""
    if path.split(".")[-1] in ("iban", "counterparty_iban", "payout_iban"):
        return True
    return any(path_matches(path, pattern) for spec in specs for pattern in spec.critical_fields)


def set_path(data: Dict[str, Any], path: str, value: Any) -> bool:
    """
    Sets (or removes, if value is None) a dotted path in nested dicts/lists.
    Missing intermediate nodes or out-of-range indices are ignored.

    Returns:
        True if the field was changed.
    """
    parts = path.split(".")
    node: Any = data
    for part in parts[:-1]:
        if isinstance(node, list):
            if not part.isdigit() or int(part) >= len(node):
                return False
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return False
        if node is None:
            return False
    last = parts[-1]
    if isinstance(node, list):
        if value is not None and last.isdigit() and int(last) < len(node):
            node[int(last)] = value
            return True
        return False
    if isinstance(node, dict):
        if value is None:
            return node.pop(last, None) is not None
        node[last] = value
        return True
    return False


def holds(fields: Dict[str, Any], ev: FieldEvidence) -> bool:
    """True if the field at the evidence path still holds the evidence value (paths can go stale after
    sanitizing, row filtering or a ZUGFeRD overlay - such fields are never edited from evidence)."""
    current = _get(fields, ev.path)
    if current is None or ev.value is None:
        return False
    if compact(str(current)) == compact(str(ev.value)) or str(current) == str(ev.value):
        return True
    if ev.kind == KIND_AMOUNT:  # serialized amounts differ in format only ('51.3' vs 51.30)
        a, b = to_decimal(current), to_decimal(ev.value)
        return a is not None and a == b
    return False


def parse_printed(kind: str, printed: Optional[str]) -> Optional[Any]:
    """Converts a printed reading back into a normalized value."""
    if not printed:
        return None
    if kind == KIND_AMOUNT:
        sign = -1 if printed.strip().startswith(("-", "−")) or printed.strip().endswith("-") else 1
        values = amount_candidates(printed)
        return float(sign * min(values, key=lambda v: len(str(v)))) if values else None
    if kind == KIND_DATE:
        d = parse_date(printed)
        return d.isoformat() if d else None
    if kind in (KIND_IBAN, KIND_IDENTIFIER):
        return compact(printed) if kind == KIND_IBAN else printed.strip()
    return printed.strip()


def natively_grounded(ev: FieldEvidence) -> bool:
    """True if the value was found exactly in the native PDF text layer (what is printed, no OCR involved)."""
    return ev.status == GroundingStatus.GROUNDED and ev.matched_source == "native"


def label_for(path: str) -> str:
    """Human label of a field path."""
    parts = [p for p in path.split(".") if not p.isdigit() and p not in ("bodies", "meta_header")]
    return " ".join(parts[-2:]).replace("_", " ")


@dataclass
class GroundingOutcome:
    """Result of the grounding harness for one logical document."""

    evidence: List[FieldEvidence]
    checks: List[CheckResult] = field(default_factory=list)
    trust: TrustReport = field(default_factory=TrustReport)
    snaps: List[CorrectionRecord] = field(default_factory=list)


PAYEE_PATHS = ("bodies.finance_body.payment_accounts.", "meta_header.sender.iban")


def _get(data: Dict[str, Any], path: str) -> Any:
    """Dotted-path lookup (list indices allowed)."""
    node: Any = data
    for part in path.split("."):
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _snap_record(path: str, old: Any, new: Any, how: str) -> CorrectionRecord:
    """Correction record of a deterministic snap."""
    return CorrectionRecord(trigger_checks=["grounding.ungrounded"], scope=f"snap {path}", applied=True,
                            note=f"{old!r} -> {new!r} ({how})")


ISSUER_FOOTER_RE = re.compile(r"(vorstand|aufsichtsrat|sitz der gesellschaft|registergericht|amtsgericht|\bhrb\b|handelsregister|"
                              r"ust-?id|steuernummer|geschäftsführ|unsere bankverbindung|bankverbindungen)", re.IGNORECASE)


def _any_iban(fields: Dict[str, Any]) -> bool:
    """True if any IBAN field of the document is filled."""
    meta = fields.get("meta_header") or {}
    bodies = fields.get("bodies") or {}
    values = [(meta.get(r) or {}).get("iban") for r in ("sender", "recipient") if isinstance(meta.get(r), dict)]
    values += [(bodies.get("statement_body") or {}).get("iban")]
    values += [a.get("iban") for a in (bodies.get("finance_body") or {}).get("payment_accounts") or [] if isinstance(a, dict)]
    return any(values)


def snap_ibans(layer: TextLayer, pages: List[int], fields: Dict[str, Any], evidence: List[FieldEvidence],
               specs: Sequence[TypeSpec]) -> List[CorrectionRecord]:
    """
    IBAN hygiene: strip prefixes, never keep invalid/masked IBANs (snap to the
    closest valid printed IBAN of the same role, else remove), move direct-debit
    accounts (BT-91) out of the payee role (BT-84), fill a missing payee /
    statement IBAN from a unique printed candidate.

    Returns:
        Correction records of all changes.
    """
    records: List[CorrectionRecord] = []
    candidates = iban_candidates(layer, pages)
    by_iban = {c.iban: c for c in candidates}
    for ev in [e for e in evidence if e.kind == KIND_IBAN and e.value and holds(fields, e)]:
        clean = normalize_iban(ev.value)
        if clean != ev.value:
            set_path(fields, ev.path, clean)
            ev.value = clean
        role = "payee" if ev.path.startswith(PAYEE_PATHS) else ("debtor" if ev.path == "meta_header.recipient.iban" else None)
        if not iban_valid(clean):
            cand = best_iban(clean, candidates, role)
            ev.original_value = ev.original_value or clean
            if cand is not None:
                set_path(fields, ev.path, cand.iban)
                ev.value, ev.action, ev.status, ev.matched_source = cand.iban, "snapped", GroundingStatus.GROUNDED, "native"
                records.append(_snap_record(ev.path, clean, cand.iban, f"printed {cand.role} IBAN on page {cand.page}"))
            else:
                ev.status = GroundingStatus.UNGROUNDED  # left for the high-DPI zoom; removed afterwards if still invalid
            continue
        cand = by_iban.get(clean)
        if role == "payee" and cand is not None and cand.role == "debtor":
            meta = fields.setdefault("meta_header", {})
            recipient = meta.get("recipient") if isinstance(meta.get("recipient"), dict) else {}
            recipient["iban"] = clean
            meta["recipient"] = recipient
            payee = next((c for c in candidates if c.role == "payee"), None)
            set_path(fields, ev.path, payee.iban if payee else None)
            records.append(_snap_record(ev.path, clean, payee.iban if payee else None,
                                        "direct-debit account moved to recipient.iban (BT-91)"))
            ev.original_value, ev.value = clean, payee.iban if payee else None
            ev.action = "replaced" if payee else "nulled"
    finance = (fields.get("bodies") or {}).get("finance_body")
    payees = [c for c in candidates if c.role == "payee"]
    if isinstance(finance, dict) and not any(isinstance(a, dict) and a.get("iban") for a in finance.get("payment_accounts") or []) \
            and len(payees) == 1:
        finance["payment_accounts"] = [{"iban": payees[0].iban}]
        evidence.append(FieldEvidence(path="bodies.finance_body.payment_accounts.0.iban", kind=KIND_IBAN, value=payees[0].iban,
                                      page=payees[0].page, critical=True, status=GroundingStatus.GROUNDED,
                                      matched_source="native", action="filled_from_text"))
        records.append(_snap_record("bodies.finance_body.payment_accounts.0.iban", None, payees[0].iban, "unique printed payee IBAN"))
    statement = (fields.get("bodies") or {}).get("statement_body")
    # §7 statements: the account's own IBAN - the unique IBAN of the first page, else the unique IBAN of the document;
    # an account in the issuer's legal footer is the issuer's, never the statement's
    # and an IBAN in a booking row is the counterparty's
    booked = compact(json.dumps((statement or {}).get("transactions") or [], default=str)).upper() if isinstance(statement, dict) else ""
    accounts = [c for c in candidates if not ISSUER_FOOTER_RE.search(c.context) and c.iban not in booked]
    first_page = [c for c in accounts if c.page == pages[0]]
    own = first_page if first_page else accounts
    if isinstance(statement, dict) and not statement.get("iban") and len(own) == 1:
        statement["iban"] = own[0].iban
        evidence.append(FieldEvidence(path="bodies.statement_body.iban", kind=KIND_IBAN, value=own[0].iban, page=own[0].page,
                                      critical=True, status=GroundingStatus.GROUNDED, matched_source="native",
                                      action="filled_from_text"))
        records.append(_snap_record("bodies.statement_body.iban", None, own[0].iban,
                                    "unique IBAN on the first page" if first_page else "unique IBAN of the statement"))
    if specs and contract_for(specs[0].name).party_case == "2.4" and not _any_iban(fields):
        # a form filled in by a person: the unique account entered in the form (not the issuer's legal footer) is the
        # filler's account
        entered = [c for c in candidates if not ISSUER_FOOTER_RE.search(c.context)]
        if len(entered) == 1:
            meta = fields.setdefault("meta_header", {})
            sender = meta.get("sender") if isinstance(meta.get("sender"), dict) else {}
            sender["iban"] = entered[0].iban
            meta["sender"] = sender
            evidence.append(FieldEvidence(path="meta_header.sender.iban", kind=KIND_IBAN, value=entered[0].iban,
                                          page=entered[0].page, critical=True, status=GroundingStatus.GROUNDED,
                                          matched_source="native", action="filled_from_text"))
            records.append(_snap_record("meta_header.sender.iban", None, entered[0].iban, "unique IBAN entered in the form"))
    return records


PARTY_NAME_FIELDS = ("company", "name")


def is_party_field(ev: FieldEvidence) -> bool:
    """meta_header.sender.* / recipient.* (IBANs excluded)."""
    parts = ev.path.split(".")
    return len(parts) == 3 and parts[0] == "meta_header" and parts[1] in ("sender", "recipient") and ev.kind != KIND_IBAN


def null_ungrounded_party_names(evidence: List[FieldEvidence], fields: Dict[str, Any],
                                scan_pages: Optional[set] = None, keep: Sequence[FieldEvidence] = ()) -> List[CheckResult]:
    """
    Sender/recipient values (names, address parts, tax ids, contact data) must
    be printed on the document's own pages; ungrounded ones are removed (after
    snapping) instead of being kept with a flag. IBANs are handled separately.
    On scans (no native text layer) OCR can miss printed text: there only names
    are removed, and only after the zoom did not confirm them (`keep` holds the
    confirmed ones); other party values on scans are kept.

    Returns:
        One INFO check listing the removed values (if any).
    """
    removed = []
    scan_pages = scan_pages or set()
    for ev in evidence:
        on_scan = (ev.page or 0) in scan_pages or (not ev.page and bool(scan_pages))
        if on_scan and ev.path.split(".")[-1] not in PARTY_NAME_FIELDS:
            continue
        if is_party_field(ev) and ev not in keep \
                and ev.status == GroundingStatus.UNGROUNDED and ev.value and holds(fields, ev):
            set_path(fields, ev.path, None)
            ev.original_value, ev.action, ev.status = ev.value, "nulled", GroundingStatus.REMOVED
            removed.append(f"{ev.path}={ev.original_value!r}")
    if not removed:
        return []
    return [CheckResult(check_id="grounding.party_removed", rule="grounding", status=CheckStatus.FAIL, severity=Severity.INFO,
                        message=f"party values not printed on the document removed: {', '.join(removed)}")]


def complete_capped_values(layer: TextLayer, pages: List[int], fields: Dict[str, Any],
                           evidence: List[FieldEvidence]) -> List[CheckResult]:
    """
    A value that reached its schema length cap may have been cut by the
    grammar: complete it from a printed token that starts with it, else flag it.

    Returns:
        WARNING checks for values that stay possibly truncated.
    """
    from fluxsense.grounding.evidence import MAX_LENGTH_BY_KIND
    checks: List[CheckResult] = []
    texts = "\n".join(layer.page(p).all_text() for p in pages if layer.page(p))
    tokens = [t for t in re.split(r"\s+", texts) if t]
    for ev in evidence:
        cap = MAX_LENGTH_BY_KIND.get(ev.kind)
        if cap is None or not isinstance(ev.value, str) or len(ev.value) < cap or not holds(fields, ev):
            continue
        longer = sorted({t.strip(".,;:") for t in tokens if t.startswith(ev.value) and len(t.strip(".,;:")) > len(ev.value)}, key=len)
        if longer:
            set_path(fields, ev.path, longer[0])
            ev.original_value, ev.value, ev.action = ev.value, longer[0], "completed"
            continue
        checks.append(CheckResult(check_id="grounding.possibly_truncated", rule="grounding", status=CheckStatus.FAIL,
                                  severity=Severity.WARNING, field=ev.path, actual=ev.value,
                                  message=f"value reached the {cap}-character cap and may be truncated"))
    return checks


# Values the field contract derives instead of copying (§8: period_start = the day after the previous statement date):
# never snapped onto a nearby printed date.
DERIVED_PATHS: Tuple[str, ...] = ("bodies.statement_body.period_start",)


def snap_ungrounded(layer: TextLayer, pages: List[int], fields: Dict[str, Any], evidence: List[FieldEvidence]) -> List[CorrectionRecord]:
    """
    Snaps ungrounded identifiers/amounts/dates to their unique near-duplicate in
    the document's native text layer (distance <= 2; amounts <= 1 digit).

    Returns:
        Correction records of all snaps.
    """
    texts = good_text(layer, pages)
    if not texts:
        return []
    records: List[CorrectionRecord] = []
    for ev in evidence:
        if ev.status != GroundingStatus.UNGROUNDED or ev.kind not in SNAP_KINDS or ev.value is None or ev.action \
                or not holds(fields, ev) or ev.path in DERIVED_PATHS:
            continue
        order = sorted(texts, key=lambda p: (p != ev.page, p))
        for number in order:
            snapped = snap_value(ev.kind, ev.value, texts[number])
            if snapped is None:
                continue
            if ev.kind == "identifier" and isinstance(_get(fields, ev.path), str) and " " not in str(ev.value):
                snapped = str(snapped)
            set_path(fields, ev.path, snapped)
            records.append(_snap_record(ev.path, ev.value, snapped, f"text layer page {number}"))
            ev.original_value, ev.value = ev.value, snapped
            ev.status, ev.action, ev.matched_source, ev.matched_page = GroundingStatus.GROUNDED, "snapped", "native", number
            break
    return records


class GroundingHarness:
    """Deterministic grounding plus escalation."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: LLM backend.
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings
        self.zoomer = Zoomer(backend, settings)

    def ground(self, layer: TextLayer, pages: List[int], evidence: List[FieldEvidence], specs: Sequence[TypeSpec]) -> None:
        """
        Deterministic grounding of all evidence entries (in place). On pages
        without a native text layer (scans), OCR/Docling-only grounding is not
        independent evidence: critical values are downgraded to weakly_grounded
        so that they are visually re-verified.
        """
        index = build_index(layer, pages)
        for ev in evidence:
            ev.critical = is_critical(ev.path, specs)
            if ev.status == GroundingStatus.REMOVED:
                continue  # already retired (e.g. replaced by the layout rules)
            if ev.kind not in CHECKED_KINDS:
                ev.status = GroundingStatus.NOT_CHECKED
                continue
            match = ground_value(ev.kind, ev.value, index, ev.page)
            if ev.kind == KIND_INTEGER and match.status != GroundingStatus.UNGROUNDED:
                # Small integers occur as tokens almost everywhere: require the evidence snippet.
                snippet = ground_value("text", ev.evidence, index, ev.page) if ev.evidence else None
                if snippet is None or snippet.status == GroundingStatus.UNGROUNDED or str(ev.value) not in (ev.evidence or ""):
                    match = Match(GroundingStatus.UNGROUNDED)
            if match.status == GroundingStatus.UNGROUNDED and ev.evidence and ev.kind != KIND_INTEGER:
                snippet = ground_value("text", ev.evidence, index, ev.page)
                parsed = parse_printed(ev.kind, ev.evidence)
                if snippet.status == GroundingStatus.GROUNDED and parsed is not None and str(parsed) == str(ev.value):
                    match = Match(GroundingStatus.WEAKLY_GROUNDED, snippet.source, snippet.page, ev.evidence)
            ev.status = match.status
            ev.matched_source, ev.matched_page, ev.matched_text = match.source, match.page, match.text
            # OCR-only grounding on scans is kept as 'grounded' (matched_source records the source); IBANs are
            # covered by the checksum, values in failed checks are still escalated to the zoom.
            ev.escalation_level = 1
            ev.resolution = f"page@{self.settings.llm.max_image_px}px"

    def ground_tables(self, layer: TextLayer, pages: List[int], tables: List[TableView],
                      specs: Sequence[TypeSpec]) -> List[CheckResult]:
        """Grounds amount/date cells of big-table rows on their source page."""
        index = build_index(layer, pages)
        table_specs: Dict[str, TableSpec] = {t.name: t for s in specs for t in s.tables}
        checks: List[CheckResult] = []
        for view in tables:
            spec = table_specs.get(view.name)
            if spec is None or not view.rows:
                continue
            bad: List[int] = []
            for i, row in enumerate(view.rows):
                page = row.get("source_page") if isinstance(row.get("source_page"), int) else None
                scope = {page: index[page]} if page in index else index
                for field_name in spec.amount_fields + spec.date_fields:
                    value = row.get(field_name)
                    if value is None:
                        continue
                    kind = KIND_AMOUNT if field_name in spec.amount_fields else KIND_DATE
                    if ground_value(kind, value, scope, page).status == GroundingStatus.UNGROUNDED:
                        bad.append(i)
                        break
            pages_bad = sorted({int(view.rows[i]["source_page"]) for i in bad if isinstance(view.rows[i].get("source_page"), int)})
            status = CheckStatus.FAIL if bad else CheckStatus.PASS
            checks.append(CheckResult(check_id="grounding.table_rows", rule="grounding", status=status,
                                      severity=Severity.WARNING if bad else Severity.INFO,
                                      message=f"{len(view.rows) - len(bad)}/{len(view.rows)} rows of {view.name} grounded in the text layer",
                                      rows=bad, pages=pages_bad, table=view.name))
        return checks

    def _candidates(self, evidence: List[FieldEvidence], failed_fields: Sequence[str]) -> List[FieldEvidence]:
        """Uncertain values in priority order, capped."""
        cfg = self.settings.grounding
        ungrounded = [e for e in evidence if e.critical and e.status == GroundingStatus.UNGROUNDED]
        in_failed = [e for e in evidence if e not in ungrounded and any(e.path.startswith(f) or f.startswith(e.path) for f in failed_fields)
                     and e.kind in CHECKED_KINDS and not natively_grounded(e)]
        weak = [e for e in evidence if cfg.verify_weak_critical and e.critical and e.status == GroundingStatus.WEAKLY_GROUNDED
                and e not in in_failed]
        ordered: List[FieldEvidence] = []
        for ev in ungrounded + in_failed + weak:
            if ev not in ordered:
                ordered.append(ev)
        return ordered[:cfg.max_escalations]

    def _repair_iban(self, pdf_path: Any, layer: TextLayer, pages: List[int], ev: FieldEvidence) -> Optional[str]:
        """
        Checksum repair of an invalid IBAN: the unique mod-97-valid candidate one
        visual confusion away (ranked by IBANs known in the file / identity and a
        valid German bank code), confirmed by a binary 400 dpi zoom question.
        Without a PDF to zoom, only a candidate printed elsewhere is accepted.
        """
        known = [c.iban for c in iban_candidates(layer, [p.page for p in layer.pages])]
        known += [i for prof in (self.settings.identity.private, self.settings.identity.business) for i in prof.iban]
        choice = choose_iban_repair(str(ev.value), known, known_german_blz)
        if choice is None:
            return None
        candidate, score = choice
        if pdf_path is None or not self.settings.grounding.enabled:
            return candidate if score >= 3 else None
        target = ZoomTarget(id="IBAN", page=ev.page if ev.page in pages else pages[0], label="IBAN", value=candidate,
                            kind=KIND_IBAN, evidence=ev.evidence, dpi=IBAN_ZOOM_DPI)
        picked = binary_choice(self.backend, self.settings, pdf_path, layer, target, candidate, normalize_iban(str(ev.value)))
        return candidate if picked == candidate else None

    def _second_opinion(self, layer: TextLayer, pages: List[int], evidence: List[FieldEvidence]) -> Dict[str, Dict[str, Any]]:
        """Escalation level 3: text-only second model."""
        if not evidence:
            return {}
        questions = "\n".join(f"- F{i}: {label_for(e.path)} = {e.value}" for i, e in enumerate(evidence))
        text = "\n\n".join(f"=== PAGE {p} ===\n{layer.page(p).all_text() if layer.page(p) else ''}" for p in pages)[:60000]
        result = self.backend.generate_json(LLMRequest(
            prompt=prompts.PROMPT_TEXT_SECOND_OPINION.format(questions=questions, document_text=text),
            label="Grounding second opinion", model=self.settings.models.review_text, schema=SECOND_OPINION_SCHEMA,
            schema_name="second_opinion", max_tokens=100 + 60 * len(evidence)))
        answers: Dict[str, Dict[str, Any]] = {}
        if isinstance(result, dict):
            for entry in result.get("answers") or []:
                if isinstance(entry, dict):
                    answers[str(entry.get("id"))] = entry
        return answers

    def run(self, pdf_path: Any, layer: TextLayer, pages: List[int], fields: Dict[str, Any], evidence: List[FieldEvidence],
            tables: List[TableView], specs: Sequence[TypeSpec], failed_fields: Sequence[str] = ()) -> GroundingOutcome:
        """
        Full harness: ground, escalate, act, report.

        Args:
            pdf_path: PDF file.
            layer: Text layer.
            pages: Pages of the logical document.
            fields: Clean fields (modified in place: nulled/replaced values).
            evidence: Evidence entries (modified in place).
            tables: Logical tables.
            specs: Type specs.
            failed_fields: Field paths involved in failed plausibility checks.

        Returns:
            GroundingOutcome.
        """
        cfg = self.settings.grounding
        self.ground(layer, pages, evidence, specs)
        snaps = snap_ibans(layer, pages, fields, evidence, specs) + snap_ungrounded(layer, pages, fields, evidence)
        scan_pages = {p for p in pages if layer.page(p) and len(layer.page(p).sources.get("native", "").strip()) < 50}
        checks = self.ground_tables(layer, pages, tables, specs) + complete_capped_values(layer, pages, fields, evidence)
        candidates = self._candidates(evidence, failed_fields) if cfg.enabled else []
        scan_names = [e for e in evidence if is_party_field(e) and e.path.split(".")[-1] in PARTY_NAME_FIELDS
                      and e.status == GroundingStatus.UNGROUNDED and scan_pages and cfg.enabled][:3]
        candidates += [e for e in scan_names if e not in candidates]
        targets = []
        for i, ev in enumerate(candidates):
            page = ev.page if ev.page in pages else (ev.matched_page or pages[0])
            targets.append(ZoomTarget(id=f"F{i + 1}", page=page, label=label_for(ev.path), value=ev.value, kind=ev.kind,
                                      evidence=ev.evidence, bbox=ev.bbox, dpi=IBAN_ZOOM_DPI if ev.kind == KIND_IBAN else None))
        answers = self.zoomer.verify(pdf_path, layer, targets) if targets else {}
        unresolved: List[FieldEvidence] = []
        for target, ev in zip(targets, candidates):
            answer = answers.get(target.id)
            if answer is None:
                unresolved.append(ev)
                continue
            ev.verification = {"visible": answer.visible, "matches": answer.matches, "printed": answer.printed,
                               "located_by": answer.located_by}
            ev.escalation_level, ev.resolution = 2, answer.resolution
            if answer.visible and answer.matches:
                ev.status = GroundingStatus.VERIFIED
                ev.action = "kept"
                continue
            printed_value = parse_printed(ev.kind, answer.printed) if answer.visible else None
            if printed_value is not None and str(printed_value) == str(ev.value):
                ev.status, ev.action = GroundingStatus.VERIFIED, "kept"
                continue
            if printed_value is not None and holds(fields, ev):
                ev.original_value, ev.value, ev.action = ev.value, printed_value, "replaced"
                ev.status = GroundingStatus.VERIFIED
                set_path(fields, ev.path, printed_value)
                continue
            unresolved.append(ev)
        confirmed = [e for e in scan_names if e.status == GroundingStatus.VERIFIED]
        checks += null_ungrounded_party_names(evidence, fields, scan_pages, confirmed)
        unresolved = [e for e in unresolved if e.status != GroundingStatus.REMOVED]
        if cfg.second_model and unresolved:
            opinions = self._second_opinion(layer, pages, unresolved)
            for i, ev in enumerate(unresolved):
                opinion = opinions.get(f"F{i}")
                if opinion and opinion.get("supported") is True:
                    ev.status, ev.escalation_level, ev.resolution = GroundingStatus.VERIFIED, 3, "text@second_model"
                    ev.action = "kept"
        for ev in evidence:
            if ev.kind == KIND_IBAN and ev.value and not iban_valid(str(ev.value)):
                repaired = self._repair_iban(pdf_path, layer, pages, ev)
                if repaired is not None and holds(fields, ev):
                    set_path(fields, ev.path, repaired)
                    snaps.append(_snap_record(ev.path, ev.value, repaired, "checksum repair (visual confusion), confirmed on 400 dpi crop"))
                    ev.original_value, ev.value, ev.action, ev.status = ev.value, repaired, "replaced", GroundingStatus.VERIFIED
                    continue
                # A failing deterministic check overrides 'grounded'/'verified' (OCR and VLM can share an error);
                # an invalid or masked IBAN is never output.
                if holds(fields, ev):
                    set_path(fields, ev.path, None)
                ev.original_value = ev.original_value or ev.value
                ev.status, ev.action = GroundingStatus.UNGROUNDED, "nulled"
                checks.append(CheckResult(check_id="grounding.iban_checksum", rule="grounding", status=CheckStatus.FAIL,
                                          severity=Severity.ERROR, field=ev.path, actual=str(ev.original_value),
                                          pages=[ev.page] if ev.page else [],
                                          message=f"IBAN {ev.original_value} fails mod-97/length even after high-DPI re-reading - removed"))
                continue
            weak_unresolved = ev.status == GroundingStatus.WEAKLY_GROUNDED and ev in unresolved
            if ev.status == GroundingStatus.UNGROUNDED or weak_unresolved:
                derived = derived_period_value(
                    ev.path, ev.value, [line for p in pages if layer.page(p) for line in layer.page(p).best_text().splitlines()])
                if derived:  # §8 normalisation of a printed period / previous statement date: grounded in its source text
                    ev.status, ev.action, ev.resolution = GroundingStatus.GROUNDED, "kept", "derived from printed text (§8)"
                    continue
                if ev.critical and cfg.null_unverified_critical and ev.status == GroundingStatus.UNGROUNDED and holds(fields, ev):
                    ev.original_value, ev.action = ev.value, "nulled"
                    set_path(fields, ev.path, None)
                    severity = Severity.WARNING if ev.kind == KIND_INTEGER else Severity.ERROR
                    message = f"critical value {ev.value!r} not found on the page and not visually verified - removed"
                else:
                    ev.action = ev.action or "flagged"
                    severity = Severity.WARNING if ev.critical else Severity.INFO
                    message = (f"value {ev.value!r} only weakly grounded and not confirmed on the zoomed crop" if weak_unresolved
                               else f"value {ev.value!r} not grounded in the text layer")
                checks.append(CheckResult(check_id="grounding.ungrounded", rule="grounding", status=CheckStatus.FAIL,
                                          severity=severity, message=message, field=ev.path, actual=str(ev.original_value or ev.value),
                                          pages=[ev.page] if ev.page else []))
            elif ev.action is None:
                ev.action = "kept"
        return GroundingOutcome(evidence=evidence, checks=checks, snaps=snaps)


def trust_report(evidence: List[FieldEvidence], tables: List[TableView], grounding_checks: List[CheckResult],
                 plausibility_score: float, failed_checks: int, corrections_applied: int,
                 zugferd_mismatches: int = 0, disagreements: int = 0) -> TrustReport:
    """
    Builds the trust report.

    Score = 0.5 * field grounding + 0.15 * table-row grounding + 0.35 * plausibility.

    Returns:
        TrustReport.
    """
    checked = [e for e in evidence if e.status != GroundingStatus.NOT_CHECKED]
    counts = {s: sum(1 for e in checked if e.status == s) for s in GroundingStatus}
    total = len(checked)
    grounding = 1.0
    if total:
        grounding = (counts[GroundingStatus.GROUNDED] + counts[GroundingStatus.VERIFIED] + 0.6 * counts[GroundingStatus.WEAKLY_GROUNDED]) / total
    rows_total = sum(len(t.rows) for t in tables)
    rows_bad = sum(len(c.rows) for c in grounding_checks if c.check_id == "grounding.table_rows")
    rows_score = 1.0 - rows_bad / rows_total if rows_total else 1.0
    escalations: Dict[str, int] = {}
    for ev in checked:
        escalations[str(ev.escalation_level)] = escalations.get(str(ev.escalation_level), 0) + 1
    score = 0.5 * grounding + 0.15 * rows_score + 0.35 * plausibility_score
    score -= 0.05 * min(4, zugferd_mismatches + disagreements)
    from fluxsense.grounding.acceptance import is_core
    provenance: Dict[str, int] = {}
    for ev in evidence:
        if ev.provenance and ev.auto_accepted is not None:
            provenance[ev.provenance] = provenance.get(ev.provenance, 0) + 1
    return TrustReport(
        score=round(max(0.0, min(1.0, score)), 3), fields_total=total, grounded=counts[GroundingStatus.GROUNDED],
        weakly_grounded=counts[GroundingStatus.WEAKLY_GROUNDED], ungrounded=counts[GroundingStatus.UNGROUNDED],
        verified_visual=counts[GroundingStatus.VERIFIED], nulled=sum(1 for e in evidence if e.action == "nulled"),
        replaced=sum(1 for e in evidence if e.action == "replaced"), table_rows_total=rows_total,
        table_rows_ungrounded=rows_bad, failed_checks=failed_checks, corrections_applied=corrections_applied,
        self_consistency_disagreements=disagreements, zugferd_mismatches=zugferd_mismatches, escalations=escalations,
        auto_accepted=sum(1 for e in evidence if e.auto_accepted is True and is_core(e.path)),
        flagged_for_review=sum(1 for e in evidence if e.auto_accepted is False and is_core(e.path)),
        informational_accepted=sum(1 for e in evidence if e.auto_accepted is True and not is_core(e.path)),
        informational_flagged=sum(1 for e in evidence if e.auto_accepted is False and not is_core(e.path)),
        provenance=provenance,
    )


def evidence_json(evidence: List[FieldEvidence]) -> str:
    """Compact JSON of evidence (debugging)."""
    return json.dumps([e.model_dump(exclude_none=True) for e in evidence], ensure_ascii=False, default=str)

