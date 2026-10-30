# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/pointer.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Pointer extraction (experiment): numbers the native text
                lines of a document ('[p1:L37] ...'); the model answers
                identifiers, amounts, dates and IBANs with a line id, and the
                value is taken from that line by kind-specific parsing, so the
                model locates and the text layer supplies the characters.
------------------------------------------------------------------------------
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from fluxsense.extract.base import TextLayer
from fluxsense.grounding.normalize import compact, dehyphenate, extract_dates, parse_date, to_decimal
from fluxsense.formats.iban import IBAN_CANDIDATE_RE, iban_valid
from fluxsense.grounding.snap import damerau
from fluxsense.models.result import FieldEvidence

LINE_ID_RE = re.compile(r"p(\d+):L(\d+)", re.IGNORECASE)
POINTER_KINDS = frozenset({"identifier", "amount", "date", "iban"})


def native_pages(layer: TextLayer, pages: List[int], min_chars: int = 50) -> bool:
    """True if every page of the document has a native text layer."""
    return all(layer.page(p) is not None and len(layer.page(p).sources.get("native", "").strip()) >= min_chars for p in pages)


def numbered_text(layer: TextLayer, pages: List[int], limit: int) -> Tuple[str, Dict[str, str]]:
    """
    Numbered native text lines.

    Returns:
        Tuple (prompt text, line id -> line text).
    """
    lines: List[str] = []
    mapping: Dict[str, str] = {}
    for number in pages:
        page = layer.page(number)
        text = dehyphenate(page.sources.get("native", "")) if page else ""
        lines.append(f"=== PAGE {number} ===")
        index = 0
        for raw in text.splitlines():
            if not raw.strip():
                continue
            index += 1
            line_id = f"p{number}:L{index}"
            mapping[line_id] = raw.strip()
            lines.append(f"[{line_id}] {raw.strip()}")
    return "\n".join(lines)[:limit], mapping


def _closest(target: str, candidates: List[Tuple[str, Any]], max_distance: int) -> Optional[Any]:
    """Candidate value whose compact form is closest to the target."""
    if not candidates:
        return None
    scored = sorted(((damerau(compact(target), compact(c), 40), v) for c, v in candidates), key=lambda x: x[0])
    return scored[0][1] if scored[0][0] <= max_distance else None


def value_from_line(kind: str, model_value: Any, line: str) -> Optional[Any]:
    """
    Kind-specific parsing of the referenced line, choosing the occurrence
    closest to the model's reading.

    Returns:
        The printed value (normalized) or None.
    """
    if kind == "amount":
        from fluxsense.stages.repair import line_amounts
        amounts = line_amounts(line)
        model = to_decimal(model_value)
        if not amounts or model is None:
            return None
        best = min(amounts, key=lambda a: abs(abs(a) - abs(model)))
        digits = lambda d: f"{abs(d):.2f}".replace(".", "")  # noqa: E731
        if damerau(digits(best), digits(model), 3) > 2:
            return None
        return float(best if model < 0 or best < 0 else best)
    if kind == "date":
        dates, _ = extract_dates(line)
        parsed = parse_date(model_value)
        model = parsed.isoformat() if parsed else str(model_value or "")
        if not dates or not model:
            return None
        best = min(dates, key=lambda d: damerau(d.isoformat(), model, 10))
        return best.isoformat() if damerau(best.isoformat(), model, 10) <= 2 else None
    if kind == "iban":
        found = [compact(m.group(1)) for m in IBAN_CANDIDATE_RE.finditer(line.upper())]
        found = [f for f in found if iban_valid(f)]
        return _closest(str(model_value), [(f, f) for f in found], 4)
    if kind == "identifier":
        tokens = [t for t in re.split(r"[\s,;:()]+", line) if t]
        windows = [" ".join(tokens[i:i + n]) for n in (1, 2, 3) for i in range(len(tokens) - n + 1)]
        target = compact(str(model_value))
        if len(target) < 3:
            return None
        return _closest(str(model_value), [(w, w) for w in windows if abs(len(compact(w)) - len(target)) <= 2],
                        max(1, len(target) // 4))
    return None


def resolve_pointers(evidence: List[FieldEvidence], fields: Dict[str, Any], mapping: Dict[str, str]) -> Dict[str, int]:
    """
    Replaces model readings by the characters of the referenced text line.

    Returns:
        Stats {"pointed", "resolved", "changed"}.
    """
    from fluxsense.grounding.harness import holds, set_path
    stats = {"pointed": 0, "resolved": 0, "changed": 0}
    for ev in evidence:
        if ev.kind not in POINTER_KINDS or not ev.line:
            continue
        match = LINE_ID_RE.search(ev.line)
        if not match:
            continue
        stats["pointed"] += 1
        line = mapping.get(f"p{int(match.group(1))}:L{int(match.group(2))}")
        if line is None or not holds(fields, ev):
            continue
        value = value_from_line(ev.kind, ev.value, line)
        if value is None:
            continue
        stats["resolved"] += 1
        if str(value) != str(ev.value) and compact(str(value)) != compact(str(ev.value)):
            stats["changed"] += 1
            set_path(fields, ev.path, value)
            ev.original_value, ev.value, ev.action = ev.value, value, "pointer"
        ev.page = int(match.group(1))
        ev.evidence = ev.evidence or line[:40]
    return stats
