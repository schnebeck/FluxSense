# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/text/levels.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Text levels of a page. 'raw' is the text as read (verbatim;
                evidence pointers keep referencing it). 'clean' joins
                line-break hyphenation (lowercase continuation only, so
                compounds such as 'Musterstadt-Linden' stay), removes soft
                hyphens and reflows running text to paragraphs - judged per
                layout item (or per blank-line paragraph without layout):
                address blocks, table rows, lists, short lines and header /
                footer stay line-based. 'corrected' is clean plus the
                corrections (dictionary, format, re-read, human), each with
                its offset in the clean text.
------------------------------------------------------------------------------
"""

import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from fluxsense.extract.base import PageText
from fluxsense.grounding.normalize import dehyphenate
from fluxsense.layout.items import TEXT
from fluxsense.models.result import PageTextOut, TextCorrection

_LIST_RE = re.compile(r"^\s*(?:[-•*·–]|\d{1,2}[.)]|[a-z][.)])\s+")
_ZIP_RE = re.compile(r"(?:^|\s)(?:D-|A-|CH-)?\d{4,5}\s+[A-ZÄÖÜ]")
_TABLE_RE = re.compile(r"\||\S\s{3,}\S|\t")
MIN_FILLED: int = 35  # characters of a well-filled line of running text
KIND_PRIORITY = {"human": 0, "reread": 1, "arithmetic": 1, "format": 2, "dictionary": 3}


def running_text(lines: Sequence[str]) -> bool:
    """
    True for lines of running text that a layout wrapped: at least two
    lines, every line but the last well filled, no list item, table row,
    address line (postcode + town) or line ending in a colon.
    """
    clean = [line.strip() for line in lines if line.strip()]
    if len(clean) < 2:
        return False
    if any(_LIST_RE.match(line) or _TABLE_RE.search(line) or _ZIP_RE.search(line) for line in clean):
        return False
    if any(line.endswith(":") for line in clean[:-1]):
        return False
    return all(len(line) >= MIN_FILLED for line in clean[:-1])


def reflow(block: str) -> str:
    """The lines of a paragraph joined with single spaces."""
    return " ".join(" ".join(line.split()) for line in block.splitlines() if line.strip())


def clean_text(page: PageText, raw: Optional[str] = None) -> str:
    """
    The 'clean' level of a page text.

    Args:
        page: Page (layout items decide what running text is).
        raw: The raw text (default: the page's best text).

    Returns:
        Dehyphenated text with running-text paragraphs on one line each.
    """
    raw = page.best_text() if raw is None else raw
    text = dehyphenate(raw)
    items = [i for i in page.items or [] if getattr(i, "category", "") == TEXT and "\n" in i.content.strip()]
    if page.items:
        for item in items:
            block = dehyphenate(item.content.strip())
            if running_text(block.splitlines()) and block in text:
                text = text.replace(block, reflow(block), 1)
        return text
    paragraphs = re.split(r"(\n[ \t]*\n)", text)
    return "".join(reflow(p) if running_text(p.splitlines()) else p for p in paragraphs)


def locate(text: str, before: str, taken: Iterable[int] = ()) -> Optional[int]:
    """Offset of the first whole-word occurrence of 'before' in a text that is not taken yet."""
    used = set(taken)
    for match in re.finditer(rf"(?<![\w]){re.escape(before)}(?![\w])", text):
        if match.start() not in used:
            return match.start()
    return None


def apply_corrections(clean: str, corrections: List[TextCorrection]) -> str:
    """
    Applies corrections to a clean text (offsets are set where missing;
    overlapping ones keep the strongest evidence: human > re-read /
    arithmetic > format > dictionary; a correction whose text is not found is
    dropped).
    """
    kept: List[TextCorrection] = []
    spans: List[range] = []
    for corr in sorted(corrections, key=lambda c: KIND_PRIORITY.get(c.evidence_kind, 9)):
        if corr.offset is None or clean[corr.offset:corr.offset + len(corr.before)] != corr.before:
            corr.offset = locate(clean, corr.before, [c.offset for c in kept if c.offset is not None])
        if corr.offset is None:
            continue
        span = range(corr.offset, corr.offset + max(1, len(corr.before)))
        if any(span.start < s.stop and s.start < span.stop for s in spans):
            continue
        kept.append(corr)
        spans.append(span)
    corrections[:] = sorted(kept, key=lambda c: c.offset or 0)
    out = clean
    for corr in sorted(kept, key=lambda c: -(c.offset or 0)):
        start = corr.offset or 0
        out = out[:start] + corr.after + out[start + len(corr.before):]
    return out


def word_box(page: PageText, word: str, used: Set[int]) -> Optional[List[float]]:
    """The box of a word on the page (first unused word box with that text, punctuation aside)."""
    for index, candidate in enumerate(page.words):
        if index not in used and candidate.text.strip(".,;:!?()\"'") == word:
            used.add(index)
            return [round(candidate.x0, 2), round(candidate.y0, 2), round(candidate.x1, 2), round(candidate.y1, 2)]
    return None


def page_levels(page: PageText, lookup: Optional[Callable[[Sequence[str]], Dict[str, Optional[List[str]]]]] = None,
                protected: Optional[Set[str]] = None, formats: Sequence[Any] = (),
                corrections: Sequence[TextCorrection] = ()) -> PageTextOut:
    """
    The three text levels of a page.

    Args:
        page: Page.
        lookup: Dictionary lookup for the word correction (None: no word correction).
        protected: Name words that are never spell-corrected.
        formats: Formats supplied by the application (corrected like the built-in ones).
        corrections: Further corrections for this page (e.g. human confirmations); pending ones stored on the page by
            the field checks are added.

    Returns:
        The page output with raw / clean / corrected text and the applied corrections. Printed digital text is never
        corrected by the OCR rules (only human confirmations apply).
    """
    from fluxsense.text.correct import format_corrections, word_corrections
    raw = page.best_text()
    clean = clean_text(page, raw)
    pending = [c for c in list(corrections) + list(page.corrections) if isinstance(c, TextCorrection)]
    if page.scanned():
        pending += format_corrections(page.page, clean, formats)
        if lookup is not None:
            pending += word_corrections(page.page, clean, lookup, protected)
    else:
        pending = [c for c in pending if c.evidence_kind == "human"]
    applied = [c.model_copy() for c in pending]
    corrected = apply_corrections(clean, applied)
    used: Set[int] = set()
    for corr in applied:
        if corr.bbox is None and " " not in corr.before:
            corr.bbox = word_box(page, corr.before, used)
    return PageTextOut(page=page.page, text=raw, source=page.best_source() or "native", clean=clean, corrected=corrected,
                       corrections=applied, native=not page.scanned())
