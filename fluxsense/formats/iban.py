# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/formats/iban.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    IBAN helpers on top of the registry spec: validity (mod 97,
                country length and structure, never a masked IBAN), IBANs
                printed on a line (also with irregular grouping), repair
                candidates one visual confusion or transposition away and
                their ranking (known IBANs, German bank code).
------------------------------------------------------------------------------
"""

import re
from typing import Any, List, Optional, Sequence, Tuple

from fluxsense.formats.builtin import IBAN, IBAN_LENGTHS, normalize_iban
from fluxsense.formats.spec import DIGIT_DIGIT, LETTER_DIGIT, confusion_map

IBAN_CANDIDATE_RE = re.compile(r"\b([A-Z]{2}[ ]?\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?)\b")
_IBAN_START_RE = re.compile(r"\b([A-Z]{2})\s?(\d{2})\s?(?=[A-Z0-9])")
# Visual / OCR confusions used to repair IBANs via the checksum: digits both ways, letters only towards digits.
IBAN_CONFUSIONS = {**confusion_map(DIGIT_DIGIT)}
for _letter, _digit in LETTER_DIGIT:
    if _letter not in ("T", "A"):
        IBAN_CONFUSIONS[_letter] = IBAN_CONFUSIONS.get(_letter, "") + _digit

__all__ = ["IBAN", "IBAN_CANDIDATE_RE", "IBAN_LENGTHS", "choose_iban_repair", "iban_repair_candidates", "iban_valid",
           "normalize_iban", "printed_ibans", "rank_iban_candidates"]


def iban_valid(value: Any) -> bool:
    """mod-97 + country length and structure + not masked."""
    if re.search(r"[X*]{2,}", str(value).upper()[4:]):
        return False  # masked (e.g. DE44 9901 ... XXXX) - never output
    return IBAN.valid(value)


def printed_ibans(line: str) -> List[str]:
    """
    Valid IBANs printed on a line, also with irregular grouping
    ('DE44 990 123 45 0532013000'): from each country code + check digits,
    the following characters (spaces ignored) are cut to the country's IBAN
    length and checked with mod-97.
    """
    from fluxsense.grounding.normalize import compact
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


def iban_repair_candidates(value: str) -> List[str]:
    """
    Mod-97-valid IBANs one visual confusion (or one adjacent transposition)
    away from an invalid IBAN.

    Args:
        value: Invalid IBAN (normalized).

    Returns:
        Valid candidates (unique, in position order).
    """
    return [c for c in IBAN.free_repairs(normalize_iban(value), IBAN_CONFUSIONS, start=2) if iban_valid(c)]


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
