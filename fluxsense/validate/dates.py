# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/validate/dates.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Cross-field date consistency (field contract §3.1): the
                document date against the other dates of the same logical
                document (contract start, due / order / service dates,
                statement period, reply deadline) within type-aware
                windows. A violation is a WARNING on every field involved,
                never silently fixed. Ambiguous year readings (two-digit
                years, zoom / vision readings) are resolved only when
                exactly one candidate reading is consistent with the other
                grounded dates - and stay flagged; a zoom reading that
                contradicts them no longer counts as visual support.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from fluxsense.grounding.normalize import parse_date
from fluxsense.models.result import CheckResult, CheckStatus, FieldEvidence, GroundingStatus, Severity

DOC = "meta_header.doc_date"
FIN = "bodies.finance_body"
ST = "bodies.statement_body"
CHECK_CONSISTENCY = "dates.consistency"
CHECK_YEAR = "dates.year_ambiguous"


@dataclass(frozen=True)
class Window:
    """'other - anchor' must lie in [low, high] days (None = unbounded)."""

    anchor: str
    other: str
    low: Optional[int]
    high: Optional[int]
    rule: str


# Field contract §3.1 - keep in sync with docs/field_contract.md.
WINDOWS: Tuple[Window, ...] = (
    Window(DOC, f"{FIN}.due_date", -31, 366, "a due date lies at most a month before and a year after the invoice date"),
    Window(DOC, f"{FIN}.order_date", -731, 92, "an order is placed at most 2 years before the invoice date"),
    Window(DOC, f"{FIN}.service_date", -731, 366, "a service / delivery date lies within 2 years before and 1 year after "
                                                  "the invoice date"),
    Window(DOC, "bodies.subscription_info.service_period_start", -1096, 366, "a service period starts at most 3 years before "
                                                                            "and 1 year after the document date"),
    Window(DOC, "bodies.subscription_info.service_period_end", -1096, 731, "a service period ends at most 3 years before and "
                                                                          "2 years after the document date"),
    # later documents of a contract (amendments, confirmations) may follow its start at any time
    Window(DOC, "bodies.legal_body.effective_date", None, 731, "a contract is signed at most 2 years before its start"),
    Window(DOC, f"{ST}.period_start", -500, 31, "a statement is issued at most about 16 months after its period starts"),
    Window(DOC, f"{ST}.period_end", -366, 31, "a statement is issued within a year after its period ends"),
    Window(f"{ST}.statement_date", f"{ST}.period_end", -366, 31, "a statement date lies within a year after the period end"),
    Window(DOC, f"{ST}.payment_due_date", -31, 366, "a debit date lies within a year after the statement date"),
    Window(DOC, "bodies.correspondence_body.deadline", -31, 731, "a reply deadline lies within 2 years after the letter date"),
    Window(DOC, "bodies.payslip_body.pay_period_end", -366, 62, "a payslip is issued within a year after its pay period"),
)
FALLBACK_ANCHORS: Dict[str, Tuple[str, ...]] = {DOC: (f"{FIN}.invoice_date", f"{ST}.statement_date")}

# Single-digit confusions of handwritten / OCR'd year digits (0 vs 2, 6 vs 8, 1 vs 7, ...).
DIGIT_CONFUSIONS: Dict[str, str] = {"0": "2689", "1": "7", "2": "07", "3": "58", "4": "9", "5": "36", "6": "058", "7": "12",
                                    "8": "0369", "9": "048"}
_TWO_DIGIT_YEAR_RE = re.compile(r"(?<!\d)\d{1,2}\s*[./-]\s*\d{1,2}\s*[./-]\s*\d{2}(?!\d)")


@dataclass
class Conflict:
    """Two dates of one document outside their window."""

    window: Window
    anchor_path: str
    anchor: date
    other_path: str
    other: date

    def message(self, path: str) -> str:
        """Review text for one of the two fields, naming the conflicting field and value."""
        mine, theirs = (self.anchor, (self.other_path, self.other)) if path == self.anchor_path else \
            (self.other, (self.anchor_path, self.anchor))
        return (f"date conflict: {path.split('.')[-1]} {mine} vs {theirs[0].split('.')[-1]} {theirs[1]} "
                f"({self.window.rule})")


def _get(fields: Dict[str, Any], path: str) -> Any:
    node: Any = fields
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _set(fields: Dict[str, Any], path: str, value: Any) -> None:
    node = fields
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _anchor(fields: Dict[str, Any], path: str) -> Tuple[str, Optional[date]]:
    for candidate in (path,) + FALLBACK_ANCHORS.get(path, ()):
        parsed = parse_date(_get(fields, candidate))
        if parsed:
            return candidate, parsed
    return path, None


def inside(window: Window, anchor: date, other: date) -> bool:
    """True if 'other - anchor' lies inside the window."""
    delta = (other - anchor).days
    return (window.low is None or delta >= window.low) and (window.high is None or delta <= window.high)


def conflicts(fields: Dict[str, Any], overrides: Optional[Dict[str, date]] = None) -> List[Conflict]:
    """
    All window violations between the dates of one logical document.

    Args:
        fields: Clean fields.
        overrides: path -> date used instead of the field value (candidate readings).

    Returns:
        The conflicts (empty when consistent or when a date is missing).
    """
    overrides = overrides or {}
    out: List[Conflict] = []
    for window in WINDOWS:
        anchor_path, anchor = _anchor(fields, window.anchor)
        anchor = overrides.get(anchor_path, anchor)
        other = overrides.get(window.other, parse_date(_get(fields, window.other)))
        if anchor and other and anchor_path != window.other and not inside(window, anchor, other):
            out.append(Conflict(window, anchor_path, anchor, window.other, other))
    return out


def _year_variants(value: date) -> Set[date]:
    """Readings of an ambiguous year: its last two digits with at most one confused digit, in either century."""
    two = f"{value.year % 100:02d}"
    variants = {two} | {two[:i] + alt + two[i + 1:] for i, digit in enumerate(two) for alt in DIGIT_CONFUSIONS.get(digit, "")}
    out: Set[date] = set()
    for year in {century + int(v) for v in variants for century in (1900, 2000)}:
        try:
            out.add(value.replace(year=year))
        except ValueError:  # 29 February in a non-leap year
            continue
    return out


def _dates_in(text: Any) -> List[date]:
    from fluxsense.grounding.normalize import extract_dates
    if not text:
        return []
    found, _ = extract_dates(str(text))
    return list(found)


def visual_reading(ev: FieldEvidence) -> bool:
    """The value comes from a zoom / vision reading (or replaced a first reading)."""
    return ev.escalation_level >= 2 or ev.status == GroundingStatus.VERIFIED or ev.action == "replaced" \
        or ev.matched_source == "fieldread"


def ambiguous_year(ev: FieldEvidence) -> bool:
    """A reading whose year is uncertain: a two-digit year that is not grounded in the text, or a zoom / vision reading."""
    snippets = [ev.evidence, ev.matched_text, (ev.verification or {}).get("printed")]
    two_digit = any(s and _TWO_DIGIT_YEAR_RE.search(str(s)) for s in snippets)
    return visual_reading(ev) or (two_digit and ev.status != GroundingStatus.GROUNDED)


def reference_date(ev: Optional[FieldEvidence]) -> bool:
    """A date that may decide an ambiguous reading: grounded in the text layer, not a visual reading."""
    return ev is not None and ev.status == GroundingStatus.GROUNDED and not visual_reading(ev)


def candidates(ev: FieldEvidence) -> Set[date]:
    """Candidate readings of a date: current and first model value, dates in the snippets, and their year variants."""
    base: Set[date] = set()
    for value in (ev.value, ev.original_value):
        parsed = parse_date(value)
        if parsed:
            base.add(parsed)
    for snippet in (ev.evidence, ev.matched_text, (ev.verification or {}).get("printed")):
        base.update(_dates_in(snippet))
    out = set(base)
    for value in base:
        out |= _year_variants(value)
    return out


def check_dates_consistency(fields: Dict[str, Any], evidence: List[FieldEvidence], today: date,
                            future_days: int = 31) -> Tuple[List[CheckResult], List[str]]:
    """
    Field contract §3.1: resolves ambiguous year readings by consistency and
    reports the remaining date conflicts.

    For each conflict, a field whose year reading is ambiguous is re-read
    from its candidates; exactly one candidate that is consistent with the
    other (grounded) dates and not in the future replaces the value, which
    stays flagged ('dates.year_ambiguous'). With none or several, nothing is
    chosen and the field is flagged. Every remaining conflict is a WARNING on
    both fields ('dates.consistency'); a zoom-verified value in a conflict
    loses its visual support (status ungrounded).

    Args:
        fields: Clean fields (an ambiguous year may be replaced in place).
        evidence: Evidence records (status / value updated in place).
        today: Reference date for the not-in-future bound.
        future_days: Tolerance of that bound.

    Returns:
        Tuple (check results, notes about replaced values).
    """
    checks: List[CheckResult] = []
    notes: List[str] = []
    by_path = {e.path: e for e in evidence if e.kind == "date" and e.status != GroundingStatus.REMOVED}
    limit = today + timedelta(days=future_days)
    found = conflicts(fields)
    handled: Set[str] = set()
    for conflict in found:
        for path, reference in ((conflict.anchor_path, conflict.other_path), (conflict.other_path, conflict.anchor_path)):
            ev = by_path.get(path)
            ref = by_path.get(reference)
            if path in handled or ev is None or not ambiguous_year(ev) or not reference_date(ref):
                continue  # only an ambiguous reading is re-read, and only a grounded, text-printed date may decide
            handled.add(path)
            current = parse_date(_get(fields, path))
            options = sorted(c for c in candidates(ev) if c <= limit)
            consistent = [c for c in options if not any(path in (x.anchor_path, x.other_path)
                                                        for x in conflicts(fields, {path: c}))]
            ref_name = reference.split(".")[-1]
            if len(consistent) == 1 and consistent[0] != current:
                chosen = consistent[0]
                ev.original_value = ev.original_value if ev.original_value is not None else ev.value
                ev.value, ev.action, ev.status = chosen.isoformat(), "replaced", GroundingStatus.UNGROUNDED
                ev.resolution = f"year chosen by consistency with {ref_name}"
                _set(fields, path, chosen.isoformat())
                notes.append(f"{path}: {current} -> {chosen} (ambiguous year, the only reading consistent with {ref_name})")
                checks.append(CheckResult(check_id=CHECK_YEAR, rule="dates", status=CheckStatus.FAIL, severity=Severity.WARNING,
                                          field=path, expected=f"consistent with {ref_name}", actual=chosen.isoformat(),
                                          message=f"year ambiguous, chosen by consistency with {ref_name} "
                                                  f"({current} -> {chosen})"))
            elif len(consistent) != 1:
                checks.append(CheckResult(check_id=CHECK_YEAR, rule="dates", status=CheckStatus.FAIL, severity=Severity.WARNING,
                                          field=path, actual=str(current),
                                          message=f"year ambiguous: {'no' if not consistent else 'several'} reading(s) "
                                                  f"consistent with {ref_name}"
                                                  + (f" ({', '.join(str(c) for c in consistent[:4])})" if consistent else "")))
    for conflict in conflicts(fields):
        for path in (conflict.anchor_path, conflict.other_path):
            checks.append(CheckResult(check_id=CHECK_CONSISTENCY, rule="dates", status=CheckStatus.FAIL, severity=Severity.WARNING,
                                      field=path, message=conflict.message(path)))
            ev = by_path.get(path)
            if ev is not None and ev.status == GroundingStatus.VERIFIED:
                # a zoom reading that contradicts the document's other dates is no visual support
                ev.status = GroundingStatus.UNGROUNDED
                ev.resolution = f"{ev.resolution or 'visual'} (not counted: contradicts "\
                                f"{(conflict.other_path if path == conflict.anchor_path else conflict.anchor_path).split('.')[-1]})"
    return checks, notes


def consistency_paths(checks: Iterable[CheckResult]) -> Set[str]:
    """Fields named by date consistency / ambiguity checks."""
    return {c.field for c in checks if c.check_id in (CHECK_CONSISTENCY, CHECK_YEAR) and c.field}


def date_check_messages(checks: Sequence[CheckResult], path: str) -> List[str]:
    """Review reasons of the date consistency / ambiguity checks of one field."""
    return list(dict.fromkeys(c.message for c in checks if c.check_id in (CHECK_CONSISTENCY, CHECK_YEAR) and c.field == path))
