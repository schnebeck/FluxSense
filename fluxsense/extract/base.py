# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/base.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Auxiliary text evidence per page: text per source (native,
                ocr, docling), word boxes for locating evidence, and table
                grids. The VLM reads the page images; this layer is evidence
                for prompting, table stitching and grounding.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SOURCE_NATIVE: str = "native"
SOURCE_OCR: str = "ocr"
SOURCE_DOCLING: str = "docling"
SOURCE_OVIS: str = "ovis"
SOURCE_STREAM: str = "stream"
SOURCE_PRIORITY: Tuple[str, ...] = (SOURCE_STREAM, SOURCE_DOCLING, SOURCE_OVIS, SOURCE_NATIVE, SOURCE_OCR)

# Page text is line-based: line and page breaks become '\n', tabs a space, every other control character goes.
_BREAK_CONTROLS_RE = re.compile("\r\n|[\r\f\v\x1c\x1d\x1e\x85\u2028\u2029]")
_DROPPED_CONTROLS_RE = re.compile("[\x00-\x08\x0e-\x1b\x1f\x7f-\x9f\ufeff\u200b\u2060]")


def strip_controls(text: str) -> str:
    """
    A page text without control characters.

    Args:
        text: Text as an extractor returned it.

    Returns:
        The text with CR / CRLF, form feed, vertical tab and Unicode line separators as '\n', tabs as a space and
        all other control characters (plus BOM, zero-width space and word joiner) removed.
    """
    return _DROPPED_CONTROLS_RE.sub("", _BREAK_CONTROLS_RE.sub("\n", text)).replace("\t", " ")


@dataclass
class Word:
    """A word with its box in PDF points (origin top-left)."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    cell: bool = False


@dataclass
class PageTable:
    """A table grid detected on one page."""

    page: int
    rows: List[List[str]]
    source: str
    bbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def column_count(self) -> int:
        """Number of columns (max row length)."""
        return max((len(r) for r in self.rows), default=0)


@dataclass
class PageText:
    """All auxiliary text evidence of one page."""

    page: int
    width: float = 595.0
    height: float = 842.0
    sources: Dict[str, str] = field(default_factory=dict)
    words: List[Word] = field(default_factory=list)
    words_source: Optional[str] = None
    tables: List[PageTable] = field(default_factory=list)
    items: List[Any] = field(default_factory=list)  # layout.items.LayoutItem in reading order
    blocks: List[Tuple[float, float, float, float, str]] = field(default_factory=list)
    native_ocr: bool = False  # the 'native' text is a scanner's OCR layer over a page image, not printed digital text
    handwriting: Optional[str] = None  # handwriting level none | signature | entries | full (None: unknown, §12)
    corrections: List[Any] = field(default_factory=list)  # pending TextCorrection records from checks / re-reads

    def scanned(self) -> bool:
        """True if the page has no printed digital text (an image page, or a scanner's OCR layer over one)."""
        return not self.sources.get(SOURCE_NATIVE, "").strip() or self.native_ocr

    def best_source(self) -> Optional[str]:
        """Name of the source best_text() returns (None if the page has no text)."""
        return next((source for source in SOURCE_PRIORITY if self.sources.get(source, "").strip()), None)

    def best_text(self) -> str:
        """Returns the most structured available text (docling > native > ocr)."""
        for source in SOURCE_PRIORITY:
            text = self.sources.get(source, "")
            if text.strip():
                return text
        return ""

    def all_text(self) -> str:
        """Concatenation of all sources (for grounding lookups)."""
        return "\n".join(t for t in self.sources.values() if t)


@dataclass
class TextLayer:
    """Auxiliary text evidence of a whole PDF."""

    pages: List[PageText]
    engine: str
    has_text_layer: bool = False
    ocr_applied: bool = False
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
    stream: Optional[Any] = None  # layout.items.ReadingStream

    @property
    def page_count(self) -> int:
        """Number of pages."""
        return len(self.pages)

    def texts(self) -> List[str]:
        """Best text per page."""
        return [p.best_text() for p in self.pages]

    def sources(self) -> List[str]:
        """Text sources present on at least one page."""
        found = {s for p in self.pages for s, t in p.sources.items() if t.strip()}
        return [s for s in SOURCE_PRIORITY if s in found]

    def tables(self) -> List[PageTable]:
        """All page tables in page order."""
        return [t for p in self.pages for t in p.tables]

    def page(self, number: int) -> Optional[PageText]:
        """Returns the 1-based page or None."""
        if 1 <= number <= len(self.pages):
            return self.pages[number - 1]
        return None


def strip_layer_controls(layer: TextLayer) -> None:
    """Removes control characters from every text of the layer (sources, words, blocks, layout items, tables)."""
    for page in layer.pages:
        page.sources = {name: strip_controls(text) for name, text in page.sources.items()}
        for word in page.words:
            word.text = "".join(strip_controls(word.text).split())
        page.blocks = [(b[0], b[1], b[2], b[3], strip_controls(b[4])) for b in page.blocks]
        for item in page.items:
            if isinstance(getattr(item, "content", None), str):
                item.content = strip_controls(item.content)
        for table in page.tables:
            table.rows = [[strip_controls(cell) for cell in row] for row in table.rows]
