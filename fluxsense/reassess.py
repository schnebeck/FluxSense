# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/reassess.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Offline re-scoring of saved bench results: re-runs the field
                acceptance (provenance / auto_accepted, field contract §0.5)
                on the stored fields, evidence, checks and audit without any
                LLM call. The text layer comes from the bench cache
                (page texts per source; word boxes and layout items when the
                cache has them) plus the native PDF words when the source
                PDF is available. Writes new result files and a summary.
------------------------------------------------------------------------------
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluxsense.catalog import default_catalog
from fluxsense.catalog.contract import contract_for
from fluxsense.extract.base import PageTable, PageText, TextLayer, Word
from fluxsense.extract.native import read_pages
from fluxsense.grounding.acceptance import assess
from fluxsense.grounding.harness import trust_report
from fluxsense.layout.items import LayoutItem
from fluxsense.log import get_logger
from fluxsense.models.result import AnalysisResult, FieldEvidence

logger = get_logger("reassess")


def layer_to_json(layer: TextLayer) -> Dict[str, Any]:
    """Text-layer cache: page texts per source, word boxes, layout items and table boxes (enough to replay acceptance)."""
    pages = []
    for p in layer.pages:
        pages.append({
            "page": p.page, "width": p.width, "height": p.height, "sources": p.sources,
            "words": [[w.x0, w.y0, w.x1, w.y1, w.text, w.cell] for w in p.words],
            "items": [[i.page, list(i.bbox), i.category, i.content, i.source] for i in p.items if isinstance(i, LayoutItem)],
            "tables": [{"rows": t.rows, "source": t.source, "bbox": list(t.bbox) if t.bbox else None} for t in p.tables],
        })
    return {"engine": layer.engine, "pages": pages}


def layer_from_json(data: Dict[str, Any], pdf: Optional[Path] = None) -> TextLayer:
    """Restores a cached text layer; native word boxes and tables are re-read from the PDF when the cache has none."""
    native: Dict[int, PageText] = {}
    if pdf is not None and pdf.is_file():
        native = {p.page: p for p in read_pages(pdf, "native")}
    pages: List[PageText] = []
    for entry in data.get("pages", []):
        number = entry["page"]
        page = PageText(page=number, width=entry.get("width", 595.0), height=entry.get("height", 842.0),
                        sources=entry.get("sources") or {})
        page.words = [Word(*w) for w in entry.get("words") or []]
        page.items = [LayoutItem(i[0], tuple(i[1]), i[2], i[3], i[4]) for i in entry.get("items") or []]  # type: ignore[arg-type]
        page.tables = [PageTable(page=number, rows=t["rows"], source=t["source"], bbox=tuple(t["bbox"]) if t.get("bbox") else None)
                       for t in entry.get("tables") or []]
        source = native.get(number)
        if source is not None:
            if not page.words and source.words and source.sources.get("native", "").strip():
                page.words, page.words_source = source.words, "native"
                page.width, page.height = source.width, source.height
            if not page.tables:
                page.tables = source.tables
        pages.append(page)
    return TextLayer(pages=pages, engine=data.get("engine", "cache"))


def contract_type(doc: Any) -> str:
    """The catalog type whose contract rows applied (as in the pipeline: first type with a §3/§4 row)."""
    catalog = default_catalog()
    names = [n for n in [doc.subtype] + list(doc.type_tags or []) + [doc.type] if n and catalog.has(n)]
    return next((n for n in names if contract_for(n).date_group or contract_for(n).number_group), names[0] if names else "OTHER")


def replay_layout(doc: Any, layer: TextLayer) -> List[str]:
    """
    Replays the geometry decision on a saved document: party values the
    earlier pipeline replaced by the DIN 5008 geometry ('replaced_by_layout'
    evidence) are restored when the current rules keep the model value
    (geometry implausible, not allowed for the type, agreement or conflict);
    conflicts are added as checks. No LLM call is needed.

    Returns:
        Notes about restored parties.
    """
    from fluxsense.catalog.contract import geometry_policy
    from fluxsense.grounding.matcher import build_index, ground_value
    from fluxsense.layout import din5008
    from fluxsense.models.result import CheckResult, CheckStatus, GroundingStatus, Severity
    from fluxsense.stages.identity import unfilled_form
    fields = doc.fields if isinstance(doc.fields, dict) else {}
    meta = fields.setdefault("meta_header", {})
    catalog = default_catalog()
    if any(catalog.has(n) and contract_for(n).party_case == "2.7" for n in [doc.subtype, doc.type] + list(doc.type_tags or []) if n):
        dropped = [key for key in ("sender", "recipient") if meta.pop(key, None)]
        return [f"{'/'.join(dropped)} removed: private notice/card/memo has no parties (§2.7)"] if dropped else []
    text = "\n".join(layer.page(p).best_text() for p in doc.pages if layer.page(p))
    window_ok, letterhead_ok = geometry_policy(contract_type(doc), unfilled_form(doc.visual_audit))
    notes: List[str] = []
    index = None
    pages = list(doc.pages)
    hints = din5008.analyze(layer.page(pages[0]), layer.page(pages[-1])) if pages and layer.page(pages[0]) else None
    if hints is not None and window_ok and letterhead_ok and din5008.swapped_roles(meta, hints):
        meta["sender"], meta["recipient"] = meta.get("recipient"), meta.get("sender")
        for ev in doc.evidence:
            if ev.path.startswith("meta_header.sender."):
                ev.path = "meta_header.recipient." + ev.path[len("meta_header.sender."):]
            elif ev.path.startswith("meta_header.recipient."):
                ev.path = "meta_header.sender." + ev.path[len("meta_header.recipient."):]
        doc.plausibility.checks = [c for c in doc.plausibility.checks if c.check_id != din5008.GEOMETRY_CONFLICT]
        notes.append("sender/recipient swapped (window and letterhead agree)")
    for role, allowed in (("recipient", window_ok), ("sender", letterhead_ok)):
        records = [e for e in doc.evidence if e.path.startswith(f"meta_header.{role}.") and e.action == "replaced_by_layout"]
        if not records:
            continue
        model = {e.path.split(".")[-1]: (e.original_value if e.original_value is not None else e.value) for e in records}
        geometry = meta.get(role) if isinstance(meta.get(role), dict) else {}
        decision = din5008.decide_party(model, geometry, text) if allowed else "keep"
        if decision in ("fill", "replace"):
            continue
        meta[role] = model
        index = index or build_index(layer, list(doc.pages))
        for ev in records:
            ev.value = ev.original_value if ev.original_value is not None else ev.value
            match = ground_value(ev.kind, ev.value, index, ev.page)
            ev.status, ev.action = match.status if match.status != GroundingStatus.NOT_CHECKED else GroundingStatus.GROUNDED, "kept"
        notes.append(f"{role}: {din5008.party_name(geometry)!r} -> {din5008.party_name(model)!r} restored ({decision})")
        if decision == "conflict":
            sub = "company" if model.get("company") else "name"
            doc.plausibility.checks.append(CheckResult(
                check_id=din5008.GEOMETRY_CONFLICT, rule="layout", status=CheckStatus.FAIL, severity=Severity.INFO,
                field=f"meta_header.{role}.{sub}", actual=din5008.party_name(model), expected=din5008.party_name(geometry),
                message="geometry/model conflict (replayed)"))
    return notes


def replay_deterministic(doc: Any, layer: TextLayer) -> List[str]:
    """
    Replays the deterministic post-checks that need no model call on a saved
    document: the IBAN role scan (unique statement / form IBANs, §7) and the
    §8 statement period fills.

    Returns:
        Notes about changes.
    """
    from fluxsense.grounding.harness import snap_ibans
    catalog = default_catalog()
    names = [n for n in [doc.subtype] + list(doc.type_tags or []) if n and catalog.has(n)]
    specs = [catalog.get(n) for n in dict.fromkeys(names)] or [catalog.get("OTHER")]
    fields = doc.fields if isinstance(doc.fields, dict) else {}
    records = snap_ibans(layer, list(doc.pages), fields, doc.evidence, specs)
    notes = [r.note for r in records]
    statement = (fields.get("bodies") or {}).get("statement_body")
    if isinstance(statement, dict):
        from fluxsense.catalog.contract import derived_period_value
        from fluxsense.grounding.harness import _get
        from fluxsense.models.result import GroundingStatus
        from fluxsense.stages.repair import fill_statement_period
        text = "\n".join(layer.page(p).best_text() for p in doc.pages if layer.page(p))
        notes += fill_statement_period(statement, text)
        lines = text.splitlines()
        # harness rule: §8 values normalised from printed text are grounded in that text
        derived = {c.field for c in doc.plausibility.checks if c.check_id == "grounding.ungrounded" and c.field
                   and derived_period_value(c.field, _get(fields, c.field), lines)}
        if derived:
            doc.plausibility.checks = [c for c in doc.plausibility.checks
                                       if not (c.check_id == "grounding.ungrounded" and c.field in derived)]
            for ev in doc.evidence:
                if ev.path in derived:
                    ev.status, ev.action = GroundingStatus.GROUNDED, "kept"
            notes.append(f"derived period values grounded: {sorted(derived)}")
    return notes


def replay_date_consistency(doc: Any) -> List[str]:
    """§9: re-runs the cross-field date check (and the ambiguous-year decision) on a saved document."""
    from datetime import date
    from fluxsense.validate.core import score_report
    from fluxsense.validate.dates import CHECK_CONSISTENCY, CHECK_YEAR, check_dates_consistency
    fields = doc.fields if isinstance(doc.fields, dict) else {}
    doc.plausibility.checks = [c for c in doc.plausibility.checks if c.check_id not in (CHECK_CONSISTENCY, CHECK_YEAR)]
    checks, notes = check_dates_consistency(fields, doc.evidence, date.today())
    doc.plausibility = score_report(doc.plausibility.checks + checks)
    return notes + [c.message for c in checks if c.check_id == CHECK_CONSISTENCY]


def reassess_result(result: AnalysisResult, layer: TextLayer, layout: bool = True) -> Dict[str, int]:
    """Re-runs the acceptance of every logical document of a result (in place). Returns the summed counts."""
    totals = {"auto_accepted": 0, "flagged": 0}
    xml = result.zugferd if isinstance(result.zugferd, dict) and result.zugferd.get("present") else None
    xml_fields = None
    if xml:
        from fluxsense.stages import zugferd
        xml_fields = zugferd.overlay({"meta_header": {}, "bodies": {}}, xml)
    for doc in result.logical_documents:
        if layout:
            for note in replay_layout(doc, layer) + replay_deterministic(doc, layer):
                logger.info(f"{result.source.path}: {note}")
        fields = doc.fields if isinstance(doc.fields, dict) else {}
        for note in replay_date_consistency(doc):
            logger.info(f"{result.source.path}: {note}")
        hint = {(e.path, str(e.value)) for e in doc.evidence if e.provenance == "geometry"}
        evidence: List[FieldEvidence] = [e for e in doc.evidence if e.action != "synthesized"]
        counts = assess(fields, evidence, layer, list(doc.pages), contract_type(doc), doc.plausibility.checks,
                        doc.visual_audit, xml_fields, hint)
        doc.evidence = evidence
        old = doc.trust
        doc.trust = trust_report(evidence, doc.tables, [], doc.plausibility.score, len(doc.plausibility.findings),
                                 old.corrections_applied, old.zugferd_mismatches, old.self_consistency_disagreements)
        doc.trust.score = old.score
        for key in totals:
            totals[key] += counts[key]
    return totals


def reassess_run(run_dir: Path, out_dir: Path, pdf_root: Optional[Path] = None, layout: bool = True) -> Dict[str, Any]:
    """
    Re-scores all results of a bench run directory (results/ + .cache/) into out_dir/results.

    Args:
        run_dir: Bench output directory.
        out_dir: Target directory.
        pdf_root: Directory of the source PDFs (default: bench_meta.json 'source').
        layout: Also replay the geometry decision on parties replaced by the layout rules.

    Returns:
        The acceptance summary (also written to out_dir/summary.json).
    """
    from fluxsense.bench import acceptance_summary
    meta_file = run_dir / "bench_meta.json"
    if pdf_root is None and meta_file.is_file():
        pdf_root = Path(json.loads(meta_file.read_text(encoding="utf-8")).get("source", ""))
    target = out_dir / "results"
    target.mkdir(parents=True, exist_ok=True)
    results: List[AnalysisResult] = []
    for path in sorted((run_dir / "results").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if "logical_documents" not in data:
            continue
        result = AnalysisResult.model_validate(data)
        cache = run_dir / ".cache" / path.name
        pdf = Path(result.source.path)
        if not pdf.is_file() and pdf_root is not None:
            pdf = pdf_root / pdf.name
        layer = layer_from_json(json.loads(cache.read_text(encoding="utf-8")), pdf) if cache.is_file() else TextLayer(pages=[], engine="none")
        if not layer.pages:
            logger.info(f"{path.name}: no cached text layer - acceptance replay skipped")
            continue
        reassess_result(result, layer, layout)
        (target / path.name).write_text(result.model_dump_json(indent=1), encoding="utf-8")
        results.append(result)
    summary = {"files": len(results), "acceptance": acceptance_summary(results)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary
