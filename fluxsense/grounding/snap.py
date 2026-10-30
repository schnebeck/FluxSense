# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/snap.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Snap-to-text-layer: when a good text layer exists, an
                ungrounded identifier/amount/date with a near-duplicate in the
                document's own page text (same kind, edit/transposition
                distance <= 2 after normalization) is replaced by the printed
                value. IBANs: candidate extraction from the text with payee
                vs. debtor (direct debit) role detection (format checks and
                repairs live in the format registry, fluxsense.formats).
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fluxsense.extract.base import TextLayer
from fluxsense.formats.iban import normalize_iban, printed_ibans
from fluxsense.grounding.normalize import CENT, compact, extract_amounts, extract_dates, parse_date, to_decimal

DEBIT_CONTEXT_RE = re.compile(r"(buchen wir|abbuchen|abgebucht|lastschrift|einzug|ziehen wir|von dem uns genannten konto|"
                              r"von ihrem konto|ihr konto|ihre bankverbindung|belasten|debit|sepa-mandat|mandatsreferenz)", re.IGNORECASE)
SNAP_KINDS: frozenset = frozenset({"identifier", "amount", "date"})


def damerau(a: str, b: str, limit: int = 3) -> int:
    """Optimal-string-alignment distance (edits + adjacent transpositions), capped at limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2: List[int] = []
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


@dataclass
class IbanCandidate:
    """A valid IBAN printed in the document."""

    iban: str
    page: int
    role: str  # payee | debtor
    context: str


def iban_candidates(layer: TextLayer, pages: Sequence[int]) -> List[IbanCandidate]:
    """
    Valid IBANs printed on the pages with their role: 'debtor' if the line or
    the preceding line speaks about a direct debit from the customer's account
    (BT-91), else 'payee' (BT-84).

    Args:
        layer: Text layer.
        pages: Pages of the logical document.

    Returns:
        Unique candidates in reading order.
    """
    found: Dict[str, IbanCandidate] = {}
    for number in pages:
        page = layer.page(number)
        if page is None:
            continue
        text = page.sources.get("native") or page.all_text()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            for iban in printed_ibans(line):
                if iban in found:
                    continue
                # The previous lines only count when this line continues them (starts with the bank details).
                continuation = re.match(r"\s*(bankverbindung|iban|konto|kto)", line, re.IGNORECASE) is not None
                context = " ".join(lines[max(0, i - 2):i + 1]) if continuation else line
                role = "debtor" if DEBIT_CONTEXT_RE.search(context) else "payee"
                found[iban] = IbanCandidate(iban, number, role, context[-160:])
    return list(found.values())


def best_iban(value: str, candidates: List[IbanCandidate], role: Optional[str]) -> Optional[IbanCandidate]:
    """Closest candidate of the role (or any role if none of the role exists)."""
    pool = [c for c in candidates if role is None or c.role == role] or ([] if role == "debtor" else candidates)
    if not pool:
        return None
    target = normalize_iban(value)
    return min(pool, key=lambda c: damerau(target, c.iban, 30))


# ----------------------------------------------------------------- snapping


def _identifier_candidates(text: str) -> List[str]:
    """Printed tokens and joined 2-3 token windows (e.g. 'DE 344487914')."""
    tokens = [t for t in re.split(r"[\s,;:()]+", text) if t]
    out = []
    for size in (1, 2, 3):
        for i in range(len(tokens) - size + 1):
            window = " ".join(tokens[i:i + size])
            if len(compact(window)) >= 4:
                out.append(window)
    return out


def snap_value(kind: str, value: Any, text: str, max_distance: int = 2) -> Optional[Any]:
    """
    Finds the unique near-duplicate of a value in a page text.

    Args:
        kind: identifier | amount | date.
        value: Extracted (ungrounded) value.
        text: Page text (good text layer).
        max_distance: Max edit/transposition distance after normalization.

    Returns:
        The printed value (normalized for amounts/dates), or None if there is
        no candidate or the best one is ambiguous.
    """
    if kind == "identifier":
        target = compact(str(value))
        if len(target) < 5:
            return None
        scored = {}
        for cand in _identifier_candidates(text):
            c = compact(cand)
            if c == target or abs(len(c) - len(target)) > max_distance:
                continue
            d = damerau(target, c, max_distance)
            if d <= max_distance:
                scored.setdefault(c, (d, cand))
        return _unique(scored)
    if kind == "amount":
        amount = to_decimal(value)
        if amount is None:
            return None
        target = f"{abs(amount).quantize(CENT):.2f}".replace(".", "")
        scored = {}
        for cand in extract_amounts(text):
            c = f"{cand:.2f}".replace(".", "")
            if c == target or len(c) != len(target):
                continue
            d = damerau(target, c, 1)
            if d <= 1:
                scored.setdefault(c, (d, cand if amount >= 0 else -cand))
        hit = _unique(scored)
        return float(hit) if hit is not None else None
    if kind == "date":
        parsed = parse_date(value)
        if parsed is None:
            return None
        target = parsed.strftime("%Y%m%d")
        dates, _ = extract_dates(text)
        scored = {}
        for cand in dates:
            c = cand.strftime("%Y%m%d")
            if c == target:
                continue
            d = damerau(target, c, max_distance)
            if d <= max_distance:
                scored.setdefault(c, (d, cand.isoformat()))
        return _unique(scored)
    return None


def _unique(scored: Dict[str, Tuple[int, Any]]) -> Optional[Any]:
    """Returns the best candidate if it is strictly better than the runner-up."""
    if not scored:
        return None
    ranked = sorted(scored.values(), key=lambda x: x[0])
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def good_text(layer: TextLayer, pages: Sequence[int]) -> Dict[int, str]:
    """Page texts usable for snapping (native text layer present)."""
    out: Dict[int, str] = {}
    for number in pages:
        page = layer.page(number)
        if page is not None and len(page.sources.get("native", "").strip()) >= 50:
            out[number] = page.sources["native"]
    return out
