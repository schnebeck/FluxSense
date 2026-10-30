# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/native.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Native text layer: font-based text-layer detection (pikepdf
                /Font, ported from KPaperFlux), PyMuPDF text, word boxes and
                table detection (page.find_tables).
------------------------------------------------------------------------------
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz
import pikepdf

from fluxsense.pdflock import PYMUPDF_LOCK
from fluxsense.extract.base import PageTable, PageText, Word
from fluxsense.log import get_logger

logger = get_logger("extract.native")


def has_font_resources(pdf_path: Path) -> bool:
    """
    Heuristic from KPaperFlux: a PDF has a text layer if any page references fonts.

    Args:
        pdf_path: PDF file.

    Returns:
        True if at least one page has /Font resources.
    """
    try:
        with pikepdf.Pdf.open(pdf_path) as pdf:
            for page in pdf.pages:
                resources = page.obj.get("/Resources")
                if resources is not None and "/Font" in resources:
                    return True
    except pikepdf.PdfError as e:
        logger.warning(f"pikepdf could not inspect {pdf_path.name}: {e}")
    return False


def _clean_cell(value: object) -> str:
    """Normalizes a table cell to a single-line string."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _cell_text(bbox: Optional[Tuple[float, float, float, float]], words: List[Word]) -> str:
    """Joins the words whose center lies inside a cell box (reading order)."""
    if bbox is None:
        return ""
    x0, y0, x1, y1 = bbox
    inside = [w for w in words if x0 <= (w.x0 + w.x1) / 2 <= x1 and y0 <= (w.y0 + w.y1) / 2 <= y1]
    inside.sort(key=lambda w: (round(w.y0, 0), w.x0))
    return " ".join(w.text for w in inside)


def detect_tables(page: fitz.Page, page_no: int, source: str, words: Optional[List[Word]] = None) -> List[PageTable]:
    """
    Detects tables on a page with PyMuPDF. Cell text is rebuilt from the
    word boxes (find_tables' own text extraction scrambles some fonts).

    Args:
        page: The PyMuPDF page.
        page_no: 1-based page number.
        source: Source label.
        words: Word boxes of the page.

    Returns:
        Detected tables (rows of cleaned cells).
    """
    tables: List[PageTable] = []
    try:
        found = page.find_tables()
    except (RuntimeError, ValueError, AttributeError) as e:
        logger.debug(f"find_tables failed on page {page_no}: {e}")
        return tables
    words = words if words is not None else [Word(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
    for tab in found.tables:
        rows = [[_clean_cell(_cell_text(cell, words)) for cell in row.cells] for row in tab.rows]
        rows = [r for r in rows if any(c for c in r)]
        if len(rows) >= 2:
            tables.append(PageTable(page=page_no, rows=rows, source=source, bbox=tuple(tab.bbox)))
    return tables


OCR_PRODUCER_RE = re.compile(r"(abbyy|finereader|omnipage|paper\s*capture|ocrmypdf|tesseract|scansnap|readiris|kofax|"
                             r"nuance|naps2|clearscan|ocr)", re.IGNORECASE)


def scanned_text_layer(page: Any, metadata: Dict[str, Any]) -> bool:
    """
    True if a page's text is a scanner's OCR layer: mostly invisible text
    (render mode 3, a 'sandwich' PDF), or a page-covering image under the
    text in a file produced by an OCR tool.
    """
    total = invisible = 0
    for span in page.get_texttrace():
        count = len(span.get("chars") or ())
        total += count
        invisible += count if span.get("type") == 3 else 0
    if total and invisible / total >= 0.5:
        return True
    area = abs(page.rect) or 1.0
    covered = max((abs(page.rect & fitz.Rect(info["bbox"])) / area for info in page.get_image_info()), default=0.0)
    producer = f"{metadata.get('producer') or ''} {metadata.get('creator') or ''}"
    return covered >= 0.85 and bool(OCR_PRODUCER_RE.search(producer))


def read_pages(pdf_path: Path, source: str, with_tables: bool = True) -> List[PageText]:
    """
    Reads text, words and tables of every page.

    Args:
        pdf_path: PDF file (opened read-only).
        source: Source label stored in PageText.sources (native or ocr).
        with_tables: Run table detection.

    Returns:
        One PageText per page.
    """
    pages: List[PageText] = []
    with PYMUPDF_LOCK, fitz.open(pdf_path) as doc:
        for index in range(doc.page_count):
            page = doc.load_page(index)
            text = page.get_text("text", sort=True)
            words = [Word(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words")]
            entry = PageText(page=index + 1, width=page.rect.width, height=page.rect.height,
                             sources={source: text}, words=words, words_source=source,
                             blocks=[(b[0], b[1], b[2], b[3], b[4]) for b in page.get_text("blocks", sort=True) if b[6] == 0],
                             native_ocr=bool(text.strip()) and scanned_text_layer(page, doc.metadata or {}))
            if with_tables and text.strip():
                entry.tables = detect_tables(page, index + 1, source, words)
            pages.append(entry)
    return pages


def page_count(pdf_path: Path) -> int:
    """Returns the number of pages of a PDF."""
    with PYMUPDF_LOCK, fitz.open(pdf_path) as doc:
        return doc.page_count
