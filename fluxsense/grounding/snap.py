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
                value. IBANs: normalization (prefix, spaces), mod-97 / length
                / masking checks, candidate extraction from the text with
                payee vs. debtor (direct debit) role detection.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fluxsense.extract.base import TextLayer
from fluxsense.grounding.normalize import CENT, compact, extract_amounts, extract_dates, parse_date, to_decimal

IBAN_CANDIDATE_RE = re.compile(r"\b([A-Z]{2}[ ]?\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?)\b")
DEBIT_CONTEXT_RE = re.compile(r"(buchen wir|abbuchen|abgebucht|lastschrift|einzug|ziehen wir|von dem uns genannten konto|"
                              r"von ihrem konto|ihr konto|ihre bankverbindung|belasten|debit|sepa-mandat|mandatsreferenz)", re.IGNORECASE)
IBAN_LENGTHS: Dict[str, int] = {"DE": 22, "AT": 20, "CH": 21, "NL": 18, "BE": 16, "FR": 27, "IT": 27, "ES": 24, "LU": 20, "GB": 22,
                                "PL": 28, "DK": 18, "SE": 24, "NO": 15, "FI": 18, "IE": 22, "PT": 25, "CZ": 24, "LI": 21}
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


def normalize_iban(value: Any) -> str:
    """Strips an 'IBAN' prefix, spaces and punctuation; uppercase."""
    text = re.sub(r"^\s*IBAN\s*[:.]?\s*", "", str(value or ""), flags=re.IGNORECASE)
    return compact(text)


def iban_valid(value: str) -> bool:
    """mod-97 + country length + not masked."""
    iban = compact(value)
    if re.search(r"[X*]{2,}", str(value).upper()[4:]):
        return False  # masked (e.g. DE44 9901 ... XXXX) - never output
    if not 15 <= len(iban) <= 34 or not iban[:2].isalpha() or not iban[2:4].isdigit():
        return False
    expected = IBAN_LENGTHS.get(iban[:2])
    if expected and len(iban) != expected:
        return False
    try:
        return int("".join(str(int(c, 36)) for c in iban[4:] + iban[:4])) % 97 == 1
    except ValueError:
        return False


@dataclass
class IbanCandidate:
    """A valid IBAN printed in the document."""

    iban: str
    page: int
    role: str  # payee | debtor
    context: str


_IBAN_START_RE = re.compile(r"\b([A-Z]{2})\s?(\d{2})\s?(?=[A-Z0-9])")


def printed_ibans(line: str) -> List[str]:
    """
    Valid IBANs printed on a line, also with irregular grouping
    ('DE44 990 123 45 0532013000'): from each country code + check digits,
    the following characters (spaces ignored) are cut to the country's IBAN
    length and checked with mod-97.
    """
    text = line.upper()
    found: List[str] = []
    for match in IBAN_CANDIDATE_RE.finditer(text):
        iban = compact(match.group(1))
        if iban_valid(iban) and iban not in found:
            found.append(iban)
    for match in _IBAN_START_RE.finditer(text):
        length = IBAN_LENGTHS.get(match.group(1))
        if not length:
            continue
        tail = re.match(r"[A-Z0-9 ]{1,48}", text[match.end():])
        body = compact(tail.group(0)) if tail else ""
        iban = match.group(1) + match.group(2) + body[:length - 4]
        if len(iban) == length and iban_valid(iban) and iban not in found:
            found.append(iban)
    return found


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



# Visual / OCR confusion pairs (symmetric for digits) used to repair IBANs via the checksum.
_DIGIT_CONFUSIONS = [("0", "6"), ("0", "8"), ("6", "5"), ("6", "8"), ("8", "3"), ("1", "7"), ("3", "8"), ("5", "6")]
_LETTER_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5", "B": "8", "Z": "2", "G": "6"}
CONFUSIONS: Dict[str, str] = {}
for _a, _b in _DIGIT_CONFUSIONS:
    CONFUSIONS[_a] = CONFUSIONS.get(_a, "") + _b
    CONFUSIONS[_b] = CONFUSIONS.get(_b, "") + _a
for _a, _b in _LETTER_TO_DIGIT.items():
    CONFUSIONS[_a] = CONFUSIONS.get(_a, "") + _b


def iban_repair_candidates(value: str) -> List[str]:
    """
    Mod-97-valid IBANs one visual confusion (or one adjacent transposition)
    away from an invalid IBAN.

    Args:
        value: Invalid IBAN (normalized).

    Returns:
        Valid candidates (unique, in position order).
    """
    iban = normalize_iban(value)
    out: List[str] = []
    for i in range(2, len(iban)):
        for alt in CONFUSIONS.get(iban[i], ""):
            cand = iban[:i] + alt + iban[i + 1:]
            if cand not in out and iban_valid(cand):
                out.append(cand)
        if i + 1 < len(iban) and iban[i] != iban[i + 1]:
            cand = iban[:i] + iban[i + 1] + iban[i] + iban[i + 2:]
            if cand not in out and iban_valid(cand):
                out.append(cand)
    return out


def rank_iban_candidates(candidates: List[str], known: Sequence[str], blz_check: Any = None) -> List[Tuple[int, str]]:
    """
    Scores candidates: +3 if the IBAN is known (printed elsewhere in the file,
    the user's identity), +1 if its German bank code exists.

    Args:
        candidates: Valid candidates.
        known: Known IBANs (normalized).
        blz_check: Callable iban -> True/False/None (None = unknown).

    Returns:
        (score, iban) sorted best first.
    """
    known_set = {normalize_iban(k) for k in known if k}
    scored = []
    for cand in candidates:
        score = 3 if cand in known_set else 0
        if blz_check is not None and blz_check(cand) is True:
            score += 1
        scored.append((score, cand))
    return sorted(scored, key=lambda x: -x[0])


def choose_iban_repair(value: str, known: Sequence[str], blz_check: Any = None) -> Optional[Tuple[str, int]]:
    """
    The unique best repair candidate (score, uniqueness) or None.

    Returns:
        Tuple (iban, score) if exactly one candidate remains after ranking.
    """
    candidates = iban_repair_candidates(value)
    if not candidates:
        return None
    ranked = rank_iban_candidates(candidates, known, blz_check)
    if len(ranked) == 1 or ranked[0][0] > ranked[1][0]:
        return ranked[0][1], ranked[0][0]
    return None
