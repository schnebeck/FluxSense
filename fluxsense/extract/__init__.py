# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/__init__.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Uniform auxiliary text layer for every PDF: the native PyMuPDF
                layer is always read (cheap, exact, word boxes); Docling
                (layout + tables + OCR) or ocrmypdf adds evidence. The engine
                never selects a different analysis path: the VLM always reads
                the page images.
------------------------------------------------------------------------------
"""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluxsense.config import Settings
from fluxsense.extract import docling_engine
from fluxsense.extract.base import SOURCE_DOCLING, SOURCE_NATIVE, PageText, TextLayer
from fluxsense.extract.native import has_font_resources, read_pages
from fluxsense.extract.ocr import OcrError, run_ocrmypdf
from fluxsense.log import get_logger

logger = get_logger("extract")

ENGINES: List[str] = ["auto", "native", "ocrmypdf", "docling", "docling-vlm", "docling-serve", "ovisocr3"]
DOCLING_ENGINES: frozenset = frozenset({"docling", "docling-vlm", "docling-serve"})


def resolve_engine(engine: str, settings: Optional[Settings] = None) -> str:
    """
    Resolves 'auto': docling-serve if configured and reachable (about
    1-2.5 s/page on a server CPU), else native text + ocrmypdf for scans.
    Local Docling is not chosen automatically: on a 4-core laptop it costs
    15-50 s/page.

    Args:
        engine: Requested engine.
        settings: Settings (for the docling-serve URL).

    Returns:
        Concrete engine name.
    """
    if engine == "auto":
        if settings is not None and docling_engine.serve_available(settings.extract):
            return "docling-serve"
        return "native"
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Choose from {ENGINES}")
    return engine


def _apply_docling(pages: List[PageText], result: docling_engine.DoclingPages) -> None:
    """Merges Docling markdown, tables, item boxes and layout items into the page list."""
    for page in pages:
        page.items = result.layout.get(page.page, [])
        md = result.markdown.get(page.page, "")
        if md.strip():
            page.sources[SOURCE_DOCLING] = md
        docling_tables = [t for t in result.tables if t.page == page.page]
        if docling_tables:
            page.tables = docling_tables
        if not page.words and result.items.get(page.page):
            page.words = result.items[page.page]
            page.words_source = SOURCE_DOCLING


def _apply_ocr(pages: List[PageText], ocr_pages: List[PageText], only: Optional[set]) -> None:
    """Merges OCR text/words/tables into pages (only into `only` page numbers if given)."""
    for page, ocr_page in zip(pages, ocr_pages):
        if only is not None and page.page not in only:
            continue
        page.sources.update(ocr_page.sources)
        if not page.words:
            page.words = ocr_page.words
            page.words_source = ocr_page.words_source
        if not page.tables:
            page.tables = ocr_page.tables


def build_text_layer(pdf_path: Path, settings: Settings, engine: Optional[str] = None, backend: Optional[Any] = None) -> TextLayer:
    """
    Builds the auxiliary text layer of a PDF (input is never modified).

    Args:
        pdf_path: PDF file.
        settings: Settings.
        engine: Engine override (see ENGINES).
        backend: LLM backend for VLM-based engines (ovisocr3); default: OpenAI-compatible client.

    Returns:
        The TextLayer.
    """
    concrete = resolve_engine(engine or settings.extract.engine, settings)
    timings = {}
    start = time.monotonic()
    pages = read_pages(pdf_path, SOURCE_NATIVE, with_tables=settings.extract.detect_tables)
    timings["native"] = round(time.monotonic() - start, 3)
    min_chars = settings.extract.min_chars_per_page
    weak_pages = {p.page for p in pages if len(p.sources.get(SOURCE_NATIVE, "").strip()) < min_chars}
    has_text_layer = has_font_resources(pdf_path) and len(weak_pages) < len(pages)
    layer = TextLayer(pages=pages, engine=concrete, has_text_layer=has_text_layer)

    run_ocr = concrete == "ocrmypdf" or (concrete == "native" and bool(weak_pages) and settings.extract.scan_fallback == "ocrmypdf")
    if concrete in DOCLING_ENGINES:
        start = time.monotonic()
        try:
            if concrete == "docling":
                result = docling_engine.convert_local(pdf_path, settings.extract)
            elif concrete == "docling-vlm":
                result = docling_engine.convert_vlm(pdf_path, settings.llm, settings.models)
            else:
                result = docling_engine.convert_serve(pdf_path, settings.extract)
            _apply_docling(pages, result)
            layer.ocr_applied = bool(weak_pages)
        except docling_engine.DoclingUnavailable as e:
            layer.warnings.append(f"{concrete} unavailable, falling back to ocrmypdf: {e}")
            logger.warning(layer.warnings[-1])
            run_ocr = bool(weak_pages)
        timings[concrete] = round(time.monotonic() - start, 3)

    if concrete == "ovisocr3":
        from fluxsense.extract.ovis import OvisEngine
        from fluxsense.llm.openai_client import OpenAICompatClient
        start = time.monotonic()
        client = backend or OpenAICompatClient(settings.llm, settings.hooks)
        targets = [p for p in pages if settings.extract.ovis_pages != "weak" or p.page in weak_pages]
        done = OvisEngine(client, settings).apply(pdf_path, targets) if targets else 0
        layer.ocr_applied = bool(weak_pages) and done > 0
        if done < len(targets):
            layer.warnings.append(f"ovisocr3 returned no layout for {len(targets) - done} pages")
        timings["ovisocr3"] = round(time.monotonic() - start, 3)
    if run_ocr:
        start = time.monotonic()
        try:
            ocr_pages = run_ocrmypdf(pdf_path, settings.extract)
            _apply_ocr(pages, ocr_pages, None if concrete == "ocrmypdf" else weak_pages)
            layer.ocr_applied = True
        except OcrError as e:
            layer.warnings.append(str(e))
            logger.warning(f"OCR failed for {pdf_path.name}: {e}")
        timings["ocrmypdf"] = round(time.monotonic() - start, 3)

    attach_stream(layer)
    layer.timings = timings
    return layer


def attach_stream(layer: TextLayer) -> None:
    """
    Builds the document reading stream from layout items (Docling / OvisOCR3,
    else native text blocks) and stores each page's part as the 'stream' source.
    """
    from fluxsense.grounding.normalize import normalize_text
    from fluxsense.layout.items import build_stream, native_items
    if not any(p.items for p in layer.pages):
        bands: Dict[str, int] = {}
        for page in layer.pages:
            h = page.height or 842.0
            for x0, y0, x1, y1, text in page.blocks:
                if y1 / h < 0.07 or y0 / h > 0.93:
                    key = normalize_text(re.sub(r"\d+", "#", text))
                    bands[key] = bands.get(key, 0) + 1
        repeated = {k for k, n in bands.items() if n >= 2 and k}
        for page in layer.pages:
            page.items = native_items(page.page, [(b[0], b[1], b[2], b[3], b[4]) for b in page.blocks],
                                      page.width or 595.0, page.height or 842.0,
                                      {normalize_text(re.sub(r"\d+", "#", t)) for t in repeated} | repeated)
    stream = build_stream({p.page: p.items for p in layer.pages if p.items})
    layer.stream = stream
    for page in layer.pages:
        text = stream.page_text(page.page)
        if text.strip():
            page.sources["stream"] = text
