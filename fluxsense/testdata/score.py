# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/score.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Scores FluxSense results (a 'fluxsense bench' run) against
                the ground truth of a generated test-document set:
                segmentation, document type, core fields and table row
                counts. Small and dependency-free, meant for the public
                smoke benchmark.
------------------------------------------------------------------------------
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AMOUNT_FIELDS = ("total_gross", "total_net", "tax_amount", "opening_balance", "closing_balance")
FIELDS = ("sender", "recipient", "doc_date", "doc_number", "total_gross", "total_net", "tax_amount", "due_date", "iban",
          "period_start", "period_end", "opening_balance", "closing_balance")


def _party(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    return value.get("company") or value.get("name")


def predicted(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Flat core fields of one result logical document."""
    fields = doc.get("fields") or {}
    meta = fields.get("meta_header") or {}
    bodies = fields.get("bodies") or {}
    fin = bodies.get("finance_body") or {}
    st = bodies.get("statement_body") or {}
    corr = bodies.get("correspondence_body") or {}
    sums = fin.get("monetary_summation") or {}
    accounts = [a.get("iban") for a in fin.get("payment_accounts") or [] if isinstance(a, dict) and a.get("iban")]
    return {
        "sender": _party(meta.get("sender")), "recipient": _party(meta.get("recipient")),
        "doc_date": meta.get("doc_date") or fin.get("invoice_date") or st.get("statement_date"),
        "doc_number": meta.get("doc_number") or fin.get("invoice_number") or st.get("statement_number"),
        "total_gross": sums.get("grand_total_amount"), "total_net": sums.get("tax_basis_total_amount"),
        "tax_amount": sums.get("tax_total_amount"),
        "due_date": fin.get("due_date") or st.get("payment_due_date") or corr.get("deadline"),
        "iban": st.get("iban") or (accounts[0] if accounts else None),
        "period_start": st.get("period_start"), "period_end": st.get("period_end"),
        "opening_balance": st.get("opening_balance"), "closing_balance": st.get("closing_balance"),
        "rows": len(st.get("transactions") or fin.get("line_items") or []),
    }


def _norm(value: Any) -> str:
    return re.sub(r"[^0-9a-zäöüß]", "", str(value).lower())


def same(field: str, truth: Any, value: Any) -> bool:
    """True when a predicted value matches the truth (amounts to the cent, names normalised, numbers without separators)."""
    if field in AMOUNT_FIELDS:
        try:
            return abs(float(truth) - float(value)) <= 0.011
        except (TypeError, ValueError):
            return False
    if field in ("sender", "recipient"):
        a, b = _norm(truth), _norm(value)
        return bool(a and b) and (a == b or a in b or b in a)
    if field in ("doc_date", "due_date", "period_start", "period_end"):
        return str(value)[:10] == str(truth)[:10]
    return _norm(truth) == _norm(value)


def load_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Result JSONs of a bench run by source file name."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        source = (data.get("source") or {}).get("path")
        if source:
            out[Path(source).name] = data
    return out


def score(truth: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Scores results against a generated ground truth.

    Args:
        truth: ground_truth.json entries.
        results: Result JSONs by file name (load_results).

    Returns:
        Summary: files, segmentation_exact, type_correct, per-field correct / wrong / missed / extra and table rows.
    """
    counts: Dict[str, Dict[str, int]] = {f: {"correct": 0, "wrong": 0, "missed": 0, "extra": 0} for f in FIELDS}
    summary: Dict[str, Any] = {"files": len(truth), "missing_results": [], "segmentation_exact": 0, "documents": 0,
                               "type_correct": 0, "table_rows_exact": 0, "tables": 0}
    for entry in truth:
        result = results.get(Path(entry["path"]).name)
        if result is None:
            summary["missing_results"].append(entry["path"])
            continue
        gold = entry["logical_documents"]
        pred = result.get("logical_documents") or []
        starts: Tuple[set, set] = ({min(d["pages"]) for d in gold}, {min(d["pages"]) for d in pred if d.get("pages")})
        summary["segmentation_exact"] += starts[0] == starts[1] and len(gold) == len(pred)
        for doc in gold:
            summary["documents"] += 1
            match = next((p for p in pred if p.get("pages") and min(p["pages"]) == min(doc["pages"])), None)
            values = predicted(match) if match else {}
            if match and ({match.get("type"), match.get("subtype")} | set(match.get("type_tags") or [])) & set(doc["doc_types"]):
                summary["type_correct"] += 1
            flat = dict(doc)
            flat.update(doc.get("statement") or {})
            for field in FIELDS:
                gold_value, value = flat.get(field), values.get(field)
                if gold_value is None:
                    counts[field]["extra"] += value not in (None, "")
                elif value in (None, ""):
                    counts[field]["missed"] += 1
                else:
                    counts[field]["correct" if same(field, gold_value, value) else "wrong"] += 1
            if doc.get("table"):
                summary["tables"] += 1
                summary["table_rows_exact"] += values.get("rows") == doc["table"]["row_count"]
    summary["fields"] = counts
    printed = sum(c["correct"] + c["wrong"] + c["missed"] for c in counts.values())
    summary["field_accuracy"] = round(sum(c["correct"] for c in counts.values()) / printed, 3) if printed else None
    return summary


def format_score(summary: Dict[str, Any]) -> str:
    """Human-readable score table."""
    lines = [f"files {summary['files']}  documents {summary['documents']}  segmentation exact {summary['segmentation_exact']}"
             f"  type correct {summary['type_correct']}  table rows exact {summary['table_rows_exact']}/{summary['tables']}",
             f"field accuracy {summary['field_accuracy']}", f"{'field':16} correct wrong missed extra"]
    for field, c in summary["fields"].items():
        lines.append(f"{field:16} {c['correct']:7} {c['wrong']:5} {c['missed']:6} {c['extra']:5}")
    if summary["missing_results"]:
        lines.append(f"no result for: {', '.join(summary['missing_results'])}")
    return "\n".join(lines)
