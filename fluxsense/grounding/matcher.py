# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/matcher.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Deterministic grounding: looks for an extracted value in the
                page text sources (native, OCR, Docling) and classifies it as
                grounded (exact after normalization), weakly_grounded (fuzzy,
                OCR confusion, year-less date, other page only) or ungrounded.
------------------------------------------------------------------------------
"""

import difflib
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple

from fluxsense.extract.base import TextLayer
from fluxsense.grounding.normalize import (CENT, compact, dehyphenate, extract_amounts, extract_dates, fold_ocr, normalize_text,
                                           parse_date, to_decimal)
from fluxsense.models.result import GroundingStatus

KIND_AMOUNT: str = "amount"
KIND_DATE: str = "date"
KIND_IBAN: str = "iban"
KIND_IDENTIFIER: str = "identifier"
KIND_TEXT: str = "text"
KIND_INTEGER: str = "integer"
KIND_CODE: str = "code"
KIND_PROSE: str = "prose"  # model-written free text (summaries): not copied, so never grounded
CHECKED_KINDS: frozenset = frozenset({KIND_AMOUNT, KIND_DATE, KIND_IBAN, KIND_IDENTIFIER, KIND_TEXT, KIND_INTEGER})


@dataclass
class SourceIndex:
    """Precomputed lookup structures for one text source of one page."""

    raw: str
    amounts: Set[Decimal]
    amounts_folded: Set[Decimal]
    dates: Set[date]
    partial_dates: Set[Tuple[int, int]]
    compact: str
    compact_folded: str
    norm: str
    tokens: Set[str]


@dataclass
class PageIndex:
    """Lookup structures for all text sources of one page."""

    page: int
    sources: Dict[str, SourceIndex] = field(default_factory=dict)


@dataclass
class Match:
    """Result of grounding one value."""

    status: GroundingStatus
    source: Optional[str] = None
    page: Optional[int] = None
    text: Optional[str] = None


def build_source_index(text: str) -> SourceIndex:
    """Builds the lookup structures for a text."""
    dates, partial = extract_dates(text)
    norm = normalize_text(text)
    return SourceIndex(
        raw=text, amounts=extract_amounts(text), amounts_folded=extract_amounts(fold_ocr(text)),
        dates=dates, partial_dates=partial, compact=compact(text), compact_folded=fold_ocr(compact(text)),
        norm=norm, tokens=set(norm.split()),
    )


def build_index(layer: TextLayer, pages: Optional[List[int]] = None) -> Dict[int, PageIndex]:
    """
    Builds page indexes for the given 1-based pages (all if None).

    Args:
        layer: Text layer.
        pages: Pages of the logical document.

    Returns:
        page -> PageIndex.
    """
    wanted = pages or [p.page for p in layer.pages]
    index: Dict[int, PageIndex] = {}
    for number in wanted:
        page = layer.page(number)
        if page is None:
            continue
        entry = PageIndex(page=number)
        for source, text in page.sources.items():
            if text and text.strip():
                entry.sources[source] = build_source_index(text)
                joined = dehyphenate(text)
                if joined != text:  # second entry: hyphenation joined (the raw one keeps real compounds)
                    entry.sources[f"{source}+joined"] = build_source_index(joined)
        index[number] = entry
    return index


def _match_amount(value: Any, src: SourceIndex) -> Optional[GroundingStatus]:
    """Amount matching on absolute values (sign is often a column or S/H marker)."""
    amount = to_decimal(value)
    if amount is None:
        return None
    try:
        target = abs(amount).quantize(CENT)
    except InvalidOperation:
        return None
    if target in src.amounts:
        return GroundingStatus.GROUNDED
    if target in src.amounts_folded:
        return GroundingStatus.WEAKLY_GROUNDED
    return None


def _match_date(value: Any, src: SourceIndex) -> Optional[GroundingStatus]:
    """Date matching across formats; year-less day/month is a weak match."""
    parsed = parse_date(value)
    if parsed is None:
        return None
    if parsed in src.dates:
        return GroundingStatus.GROUNDED
    if (parsed.day, parsed.month) in src.partial_dates:
        return GroundingStatus.WEAKLY_GROUNDED
    folded_dates, _ = extract_dates(fold_ocr(src.raw))
    if parsed in folded_dates:
        return GroundingStatus.WEAKLY_GROUNDED
    return None


def _match_compact(value: Any, src: SourceIndex, min_len: int = 4) -> Optional[GroundingStatus]:
    """Identifier/IBAN matching on alphanumerics only."""
    target = compact(str(value))
    if not target:
        return None
    if len(target) < min_len:
        return GroundingStatus.GROUNDED if target.lower() in src.tokens or target in src.raw.upper().split() else None
    if target in src.compact:
        return GroundingStatus.GROUNDED
    if fold_ocr(target) in src.compact_folded:
        return GroundingStatus.WEAKLY_GROUNDED
    return None


def _match_text(value: Any, src: SourceIndex) -> Optional[GroundingStatus]:
    """Free text: exact normalized substring, else token-level fuzzy match."""
    target = normalize_text(str(value))
    if not target:
        return None
    if f" {target} " in f" {src.norm} ":
        return GroundingStatus.GROUNDED
    words = [w for w in target.split() if len(w) >= 2]
    if not words:
        return None
    hits = exact = near = 0
    for word in words:
        if word in src.tokens:
            hits += 1
            exact += 1
        elif difflib.get_close_matches(word, src.tokens, n=1, cutoff=0.8):
            hits += 1
            near += bool(difflib.get_close_matches(word, src.tokens, n=1, cutoff=0.9))
    if exact == len(words) or (exact == len(words) - 1 and near == 1 and len(words) > 1):
        return GroundingStatus.GROUNDED  # every word printed (at most one with an OCR-level difference)
    if hits / len(words) >= 0.6:
        return GroundingStatus.WEAKLY_GROUNDED
    return None


def _match_integer(value: Any, src: SourceIndex) -> Optional[GroundingStatus]:
    """Integer as a standalone token."""
    try:
        token = str(int(value))
    except (TypeError, ValueError):
        return None
    return GroundingStatus.GROUNDED if token in src.tokens else None


_MATCHERS = {KIND_AMOUNT: _match_amount, KIND_DATE: _match_date, KIND_IBAN: _match_compact,
             KIND_IDENTIFIER: _match_compact, KIND_TEXT: _match_text, KIND_INTEGER: _match_integer}


def ground_value(kind: str, value: Any, index: Dict[int, PageIndex], page_hint: Optional[int] = None) -> Match:
    """
    Grounds one value against the page indexes of the logical document. The
    hinted page is searched first; a match on any page of the document counts
    (models often misreport the page of totals), the matched page is recorded.

    Args:
        kind: Value kind (amount, date, iban, identifier, text, integer).
        value: Extracted value.
        index: Page indexes of the logical document.
        page_hint: Page the model claimed.

    Returns:
        The Match.
    """
    if value is None or value == "" or kind not in _MATCHERS:
        return Match(GroundingStatus.NOT_CHECKED)
    matcher = _MATCHERS[kind]
    order = sorted(index.keys(), key=lambda p: (p != page_hint, p))
    best: Optional[Match] = None
    for number in order:
        for source, src in index[number].sources.items():
            status = matcher(value, src)
            if status is None:
                continue
            candidate = Match(status, source, number, str(value))
            if status == GroundingStatus.GROUNDED:
                return candidate
            best = best or candidate
    return best or Match(GroundingStatus.UNGROUNDED)
