# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/normalize.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Value normalizers for grounding: amounts (1.234,56 <->
                1234.56, trailing minus, S/H), dates (ISO, German numeric,
                German/English month names, 2-digit years), IBANs, identifiers,
                free text, plus OCR-confusion folding (O/0, l/1, S/5, B/8 ...).
------------------------------------------------------------------------------
"""

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Optional, Set, Tuple

CENT: Decimal = Decimal("0.01")

# Characters OCR engines (and VLMs) commonly confuse, folded to a canonical digit/letter.
_OCR_FOLD = str.maketrans({"O": "0", "Q": "0", "o": "0", "I": "1", "l": "1", "L": "1", "i": "1", "|": "1", "!": "1",
                           "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6"})

MONTHS = {
    "januar": 1, "jänner": 1, "jan": 1, "january": 1, "februar": 2, "feb": 2, "february": 2, "märz": 3, "maerz": 3,
    "mrz": 3, "mär": 3, "march": 3, "mar": 3, "april": 4, "apr": 4, "mai": 5, "may": 5, "juni": 6, "jun": 6,
    "june": 6, "juli": 7, "jul": 7, "july": 7, "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "oktober": 10, "okt": 10, "october": 10, "oct": 10, "november": 11, "nov": 11, "dezember": 12, "dez": 12,
    "december": 12, "dec": 12,
}
_MONTH_RE = "|".join(sorted((re.escape(m) for m in MONTHS), key=len, reverse=True))

_NUM_TOKEN_RE = re.compile(r"(?<![\w])[-−+]?\d[\d.,'’   ]*\d(?:\s?-)?|(?<![\w.,])\d(?![\w])")
_DMY_RE = re.compile(r"(?<!\d)(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4}|\d{2})(?!\d)")
_DM_RE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(?![\d])")
_YMD_RE = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_D_MONTH_Y_RE = re.compile(rf"(?<!\d)(\d{{1,2}})\.?\s*({_MONTH_RE})\.?\s*(\d{{4}}|\d{{2}})(?!\d)", re.IGNORECASE)
_MONTH_D_Y_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})(?!\d)", re.IGNORECASE)


def fold_ocr(text: str) -> str:
    """Folds OCR-confusable characters to a canonical form (for weak matches)."""
    return text.translate(_OCR_FOLD)


def strip_accents(text: str) -> str:
    """Maps umlauts to ASCII digraphs and removes remaining diacritics."""
    text = text.replace("ß", "ss").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = text.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """Lowercase, ASCII-folded, punctuation-free, single-spaced text."""
    text = strip_accents(str(text)).lower()
    text = re.sub(r"[^a-z0-9@]+", " ", text)
    return " ".join(text.split())


def compact(text: str) -> str:
    """Uppercase alphanumerics only (for IBANs and identifiers)."""
    return re.sub(r"[^A-Z0-9]", "", strip_accents(str(text)).upper())


def to_decimal(value: object) -> Optional[Decimal]:
    """Converts a model value (number or numeric string) to Decimal."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        return Decimal(str(value))
    text = str(value).strip().replace("−", "-")
    try:
        return Decimal(text)
    except InvalidOperation:
        candidates = amount_candidates(text)
    if not candidates:
        return None
    negative = text.startswith("-") or text.endswith("-") or bool(re.search(r"\s[sS]$", text))
    best = min(candidates, key=lambda d: len(str(d)))
    return -best if negative else best


def _interpret(body: str) -> Set[Decimal]:
    """Interprets a separator-laden digit string under DE and EN conventions."""
    out: Set[Decimal] = set()
    body = body.replace(" ", "").replace(" ", "").replace("'", "").replace("’", "").replace(" ", "")
    if not body or not body[0].isdigit():
        return out
    variants = []
    if "," in body and "." in body:
        if body.rfind(",") > body.rfind("."):
            variants.append(body.replace(".", "").replace(",", "."))
        else:
            variants.append(body.replace(",", ""))
    elif "," in body:
        tail = body.rpartition(",")[2]
        variants.append(body.replace(",", ".") if body.count(",") == 1 else body.replace(",", ""))
        if len(tail) == 3:
            variants.append(body.replace(",", ""))
    elif "." in body:
        _, _, tail = body.rpartition(".")
        if body.count(".") == 1:
            variants.append(body)
        if len(tail) == 3:
            variants.append(body.replace(".", ""))
    else:
        variants.append(body)
    for variant in variants:
        try:
            out.add(Decimal(variant))
        except InvalidOperation:
            continue
    return out


def amount_candidates(token: str) -> Set[Decimal]:
    """
    All absolute Decimal readings of an amount token, e.g. '1.234,56 €' ->
    {1234.56}; '1.234' -> {1.234, 1234}; '12,50-' -> {12.50}.

    Args:
        token: Text token.

    Returns:
        Set of non-negative Decimals.
    """
    cleaned = re.sub(r"(?i)(eur|euro|usd|chf|gbp|€|\$|£|\bs\b|\bh\b)", " ", token).strip()
    cleaned = cleaned.replace("−", "-").strip()
    cleaned = cleaned.strip("+-").strip()
    return {abs(d) for d in _interpret(cleaned)}


def extract_amounts(text: str) -> Set[Decimal]:
    """
    Extracts all absolute amount readings from a text, quantized to cents.
    Tokens with spaces are also split, so '12 345' yields 12345, 12 and 345.

    Args:
        text: Page text.

    Returns:
        Set of Decimals (quantized to 0.01).
    """
    found: Set[Decimal] = set()
    for match in _NUM_TOKEN_RE.finditer(text):
        token = match.group(0).strip()
        pieces = {token} | set(re.split(r"[\s  ]+", token))
        for piece in pieces:
            for value in amount_candidates(piece):
                try:
                    found.add(value.quantize(CENT))
                except InvalidOperation:
                    continue
    return found


def _safe_date(year: int, month: int, day: int) -> Optional[date]:
    """Builds a date or returns None if invalid."""
    if year < 100:
        year += 2000 if year < 70 else 1900
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_dates(text: str) -> Tuple[Set[date], Set[Tuple[int, int]]]:
    """
    Extracts full dates and year-less day/month pairs ('01.03.') from text.

    Args:
        text: Page text.

    Returns:
        Tuple (set of dates, set of (day, month)).
    """
    dates: Set[date] = set()
    partial: Set[Tuple[int, int]] = set()
    for d, m, y in _DMY_RE.findall(text):
        parsed = _safe_date(int(y), int(m), int(d))
        if parsed:
            dates.add(parsed)
        if len(y) == 4:
            us = _safe_date(int(y), int(d), int(m))
            if us and "/" in text:
                dates.add(us)
    for y, m, d in _YMD_RE.findall(text):
        parsed = _safe_date(int(y), int(m), int(d))
        if parsed:
            dates.add(parsed)
    for d, mon, y in _D_MONTH_Y_RE.findall(text):
        parsed = _safe_date(int(y), MONTHS[mon.lower()], int(d))
        if parsed:
            dates.add(parsed)
    for mon, d, y in _MONTH_D_Y_RE.findall(text):
        parsed = _safe_date(int(y), MONTHS[mon.lower()], int(d))
        if parsed:
            dates.add(parsed)
    for d, m in _DM_RE.findall(text):
        if 1 <= int(d) <= 31 and 1 <= int(m) <= 12:
            partial.add((int(d), int(m)))
    return dates, partial


def parse_date(value: object) -> Optional[date]:
    """
    Parses a model date value (ISO preferred; German formats accepted).

    Args:
        value: Date string.

    Returns:
        The date, or None.
    """
    if not value:
        return None
    text = str(value).strip()
    match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ].*)?", text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    found, _ = extract_dates(text)
    return min(found) if len(found) == 1 else None


_SOFT_HYPHEN_RE = re.compile("\u00ad\\s*")
_LINEBREAK_HYPHEN_RE = re.compile(r"(\w)-[ \t]*\r?\n[ \t]*([a-zäöüß])")


def dehyphenate(text: str) -> str:
    """
    Joins soft hyphens (U+00AD) and line-break hyphenation ('Rech-\\nnung' ->
    'Rechnung'; only when the continuation starts lowercase, so compounds such
    as 'Musterstadt-\\nLinden' are kept).

    Args:
        text: Page text.

    Returns:
        The joined text.
    """
    return _LINEBREAK_HYPHEN_RE.sub(r"\1\2", _SOFT_HYPHEN_RE.sub("", text))
