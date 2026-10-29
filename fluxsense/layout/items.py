# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/layout/items.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Engine-agnostic layout items (page, bbox 0-1000, category,
                content) with adapters for Docling labels, OvisOCR3
                'document_structured' categories and a native PyMuPDF block
                fallback, plus the document-level reading stream: page
                furniture dropped, paragraphs joined across page breaks,
                table continuations marked, page/bbox provenance kept.
------------------------------------------------------------------------------
"""

import html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from fluxsense.grounding.normalize import normalize_text

HEADER, FOOTER, PAGE_NUMBER, TITLE, TEXT, LIST, TABLE, IMAGE, FORMULA = (
    "header", "footer", "page_number", "title", "text", "list", "table", "image", "formula")
FURNITURE = frozenset({HEADER, FOOTER, PAGE_NUMBER, IMAGE})

DOCLING_CATEGORIES: Dict[str, str] = {
    "page_header": HEADER, "page_footer": FOOTER, "title": TITLE, "section_header": TITLE, "text": TEXT,
    "paragraph": TEXT, "caption": TEXT, "footnote": TEXT, "list_item": LIST, "table": TABLE, "picture": IMAGE,
    "chart": IMAGE, "formula": FORMULA, "code": TEXT, "key_value_region": TEXT, "form": TEXT, "checkbox_selected": TEXT,
    "checkbox_unselected": TEXT, "document_index": TEXT,
}
OVIS_CATEGORIES: Dict[str, str] = {
    "header": HEADER, "header_image": HEADER, "footer": FOOTER, "footer_image": FOOTER, "number": PAGE_NUMBER,
    "doc_title": TITLE, "paragraph_title": TITLE, "figure_title": TEXT, "table": TABLE, "image": IMAGE, "chart": IMAGE,
    "seal": IMAGE, "display_formula": FORMULA, "inline_formula": FORMULA, "formula_number": FORMULA, "text": TEXT,
    "content": TEXT, "abstract": TEXT, "reference_content": TEXT, "aside_text": TEXT, "vertical_text": TEXT,
    "algorithm": TEXT, "footnote": TEXT, "vision_footnote": TEXT,
}
_PAGE_NO_RE = re.compile(r"^\s*(?:seite|page|blatt)?\s*\d{1,3}\s*(?:(?:/|von|of)\s*\d{1,3})?\s*$", re.IGNORECASE)


@dataclass
class LayoutItem:
    """One layout element of a page (bbox relative, 0-1000)."""

    page: int
    bbox: Tuple[float, float, float, float]
    category: str
    content: str
    source: str = ""


class LayoutSource(Protocol):
    """Anything that yields layout items per page (Docling, OvisOCR3, native fallback, ...)."""

    def layout_items(self) -> Dict[int, List[LayoutItem]]:
        """1-based page -> items in reading order."""
        ...


def html_table_rows(content: str) -> List[List[str]]:
    """Rows of an HTML <table> (cells as plain text)."""
    rows: List[List[str]] = []
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", content, re.IGNORECASE | re.DOTALL):
        cells = re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.IGNORECASE | re.DOTALL)
        rows.append([" ".join(html.unescape(re.sub(r"<[^>]+>", " ", c)).split()) for c in cells])
    return [r for r in rows if any(r)]


def item_text(item: LayoutItem) -> str:
    """Plain text of an item (tables as pipe rows)."""
    if item.category == TABLE and "<t" in item.content.lower():
        return "\n".join("| " + " | ".join(r) + " |" for r in html_table_rows(item.content))
    return item.content.strip()


def native_items(page_no: int, blocks: Sequence[Tuple[float, float, float, float, str]], width: float, height: float,
                 repeated: Optional[set] = None) -> List[LayoutItem]:
    """
    Fallback layout from PyMuPDF text blocks: blocks in the top/bottom band that
    repeat on other pages or are page numbers become header/footer/page_number.

    Args:
        page_no: 1-based page.
        blocks: (x0, y0, x1, y1, text) in points.
        width: Page width.
        height: Page height.
        repeated: Normalized texts that occur in the top/bottom band of several pages.

    Returns:
        Items in block order.
    """
    items: List[LayoutItem] = []
    for x0, y0, x1, y1, text in blocks:
        if not text.strip():
            continue
        rel = (x0 / width * 1000, y0 / height * 1000, x1 / width * 1000, y1 / height * 1000)
        category = TEXT
        band = rel[3] < 70 or rel[1] > 930
        if band and _PAGE_NO_RE.match(text):
            category = PAGE_NUMBER
        elif band and repeated and (normalize_text(text) in repeated or normalize_text(re.sub(r"\d+", "#", text)) in repeated):
            category = HEADER if rel[3] < 70 else FOOTER
        items.append(LayoutItem(page_no, rel, category, text.strip(), "native"))
    return items


@dataclass
class StreamLine:
    """A line of the reading stream with provenance."""

    text: str
    pages: List[int]
    bbox: Tuple[float, float, float, float]
    category: str


@dataclass
class ReadingStream:
    """Document-level reading stream."""

    lines: List[StreamLine] = field(default_factory=list)
    dropped: int = 0
    joined: int = 0
    table_continuations: int = 0

    def page_text(self, page: int) -> str:
        """Text of the lines that start on a page."""
        return "\n".join(l.text for l in self.lines if l.pages[0] == page)

    def text(self) -> str:
        """Whole stream with page markers."""
        out: List[str] = []
        current = None
        for line in self.lines:
            if line.pages[0] != current:
                current = line.pages[0]
                out.append(f"=== PAGE {current} ===")
            out.append(line.text)
        return "\n".join(out)


def _table_width(item: LayoutItem) -> int:
    """Column count of a table item."""
    rows = html_table_rows(item.content) if "<t" in item.content.lower() else \
        [r.split("|") for r in item.content.splitlines() if r.strip().startswith("|")]
    return max((len(r) for r in rows), default=0)


def build_stream(pages: Dict[int, List[LayoutItem]]) -> ReadingStream:
    """
    Builds the reading stream: drops running page headers/footers (repeated on
    several pages), page numbers and images,
    joins a paragraph continuing across a page break (previous body text has
    no sentence end), marks table continuations (same column count).

    Args:
        pages: 1-based page -> items in reading order.

    Returns:
        The ReadingStream.
    """
    stream = ReadingStream()
    last_content: Optional[LayoutItem] = None
    seen: Dict[str, set] = {}
    for page, items in pages.items():
        for item in items:
            if item.category in (HEADER, FOOTER):
                seen.setdefault(normalize_text(re.sub(r"\d+", "#", item.content)), set()).add(page)
    for page in sorted(pages):
        first_on_page = True
        for item in pages[page]:
            # Headers/footers are only furniture when they repeat on several pages: a first-page letterhead or a
            # footer with bank details (IBAN, VAT id) is document content.
            running = item.category in (HEADER, FOOTER) and len(seen.get(normalize_text(re.sub(r"\d+", "#", item.content)), ())) >= 2
            page_number = item.category in (HEADER, FOOTER) and _PAGE_NO_RE.match(item.content) is not None
            if running or page_number or item.category in (PAGE_NUMBER, IMAGE) or not item.content.strip():
                stream.dropped += 1
                continue
            text = item_text(item)
            # continuation is judged against the last body line (a kept single footer may follow it)
            previous = next((l for l in reversed(stream.lines) if l.category not in (HEADER, FOOTER)), None)
            if first_on_page and previous is not None and last_content is not None and last_content.page != page:
                # German nouns are capitalized: join whenever the previous body text has no sentence end.
                if item.category in (TEXT, LIST) and last_content.category in (TEXT, LIST) and len(previous.text) >= 20 \
                        and not re.search(r"[.!?:;)]\s*$", previous.text) and re.match(r"^[\wäöüÄÖÜß(\"„-]", text):
                    previous.text = f"{previous.text} {text}" if not previous.text.endswith("-") else previous.text[:-1] + text
                    previous.pages.append(page)
                    stream.joined += 1
                    last_content, first_on_page = item, False
                    continue
                if item.category == TABLE and last_content.category == TABLE and _table_width(item) == _table_width(last_content):
                    text = f"(table continued from page {last_content.page})\n{text}"
                    stream.table_continuations += 1
            stream.lines.append(StreamLine(text, [page], item.bbox, item.category))
            if item.category not in (HEADER, FOOTER):
                last_content = item
            first_on_page = False
    return stream
