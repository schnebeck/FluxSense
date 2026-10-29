# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/ovis.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    OCR/layout engine 'ovisocr3' (EXPERIMENTAL, not the default):
                one OpenAI-compatible request per page with the verbatim
                OvisOCR3 'document_structured' prompt (temperature 0,
                thinking off, max_tokens 16384, repetition guard). The JSON
                array of {bbox 0-1000, category, content} is parsed robustly
                (pretty-printed, truncated tail, repeated items) into layout
                items, page text, word boxes and table grids.
------------------------------------------------------------------------------
"""

from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluxsense.config import Settings
from fluxsense.extract.base import SOURCE_OVIS, PageTable, PageText, Word
from fluxsense.layout.items import OVIS_CATEGORIES, TABLE, TEXT, LayoutItem, html_table_rows, item_text
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, encode_data_uri, fit_image, render_page
from fluxsense.llm.json_repair import parse_json_response
from fluxsense.log import get_logger

logger = get_logger("extract.ovis")

MAX_REPEATS: int = 2


def structured_prompt() -> str:
    """The verbatim OvisOCR3 'document_structured' prompt (package data)."""
    return resources.files("fluxsense.data").joinpath("ovisocr3_document_structured.txt").read_text(encoding="utf-8")


def parse_structured(raw: Any, page: int) -> List[LayoutItem]:
    """
    Parses a 'document_structured' answer (string or already parsed JSON).

    Handles pretty-printed JSON, a truncated tail (closed by the JSON repair)
    and repetition loops (an item repeated more than twice in a row is cut).

    Args:
        raw: Model output.
        page: 1-based page.

    Returns:
        Layout items in reading order.
    """
    data = parse_json_response(raw)[0] if isinstance(raw, str) else raw
    if isinstance(data, dict):
        data = data.get("items") or data.get("elements") or [data]
    items: List[LayoutItem] = []
    repeats = 0
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict):
            continue
        bbox, content = entry.get("bbox"), entry.get("content")
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
            continue
        if not isinstance(content, str):
            continue
        category = OVIS_CATEGORIES.get(str(entry.get("category", "text")), TEXT)
        item = LayoutItem(page, tuple(float(v) for v in bbox), category, content, SOURCE_OVIS)  # type: ignore[arg-type]
        if items and items[-1].content == item.content and items[-1].category == item.category:
            repeats += 1
            if repeats >= MAX_REPEATS:
                continue
        else:
            repeats = 0
        items.append(item)
    return items


def apply_items(page: PageText, items: List[LayoutItem]) -> None:
    """Stores Ovis items on a page: layout, text source, word boxes (if none) and tables (if none)."""
    page.items = items
    text = "\n".join(item_text(i) for i in items if i.category not in ("image",) and i.content.strip()
                     and not i.content.lstrip().startswith("<img"))
    if text.strip():
        page.sources[SOURCE_OVIS] = text
    width, height = page.width or 595.0, page.height or 842.0
    words: List[Word] = []
    for i in items:
        if not i.content.strip() or i.content.lstrip().startswith("<img"):
            continue
        x0, y0, x1, y1 = (i.bbox[0] / 1000 * width, i.bbox[1] / 1000 * height, i.bbox[2] / 1000 * width,
                          i.bbox[3] / 1000 * height)
        lines = [l for l in item_text(i).splitlines() if l.strip()] if i.category != TABLE else [" ".join(item_text(i).split())]
        step = (y1 - y0) / max(1, len(lines))
        for n, line in enumerate(lines):  # multi-line elements (e.g. the address block) -> one box per line
            words.append(Word(x0, y0 + n * step, x1, y0 + (n + 1) * step, " ".join(line.split()), cell=i.category == TABLE))
    if not page.words:
        page.words, page.words_source = words, SOURCE_OVIS
    tables = [PageTable(page=page.page, rows=html_table_rows(i.content), source=SOURCE_OVIS,
                        bbox=(i.bbox[0] / 1000 * width, i.bbox[1] / 1000 * height, i.bbox[2] / 1000 * width,
                              i.bbox[3] / 1000 * height))
              for i in items if i.category == TABLE and "<t" in i.content.lower()]
    tables = [t for t in tables if len(t.rows) >= 2]
    if tables and not page.tables:
        page.tables = tables


class OvisEngine:
    """Runs OvisOCR3 page by page."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: OpenAI-compatible backend (llama-swap).
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings
        self.prompt = structured_prompt()

    def request(self, pdf_path: Path, page: int) -> LLMRequest:
        """The request for one page (verbatim prompt, one image)."""
        img = fit_image(render_page(pdf_path, page - 1, dpi=300), self.settings.extract.ovis_max_px)
        image = ImagePart(label=f"PAGE {page}", data_uri=encode_data_uri(img, "png"), width=img.width, height=img.height)
        return LLMRequest(prompt=self.prompt, label=f"OCR ovisocr3 p{page}", model=self.settings.models.ovis_ocr,
                          images=[image], max_tokens=self.settings.extract.ovis_max_tokens, verbatim=True)

    def page_items(self, pdf_path: Path, page: int) -> List[LayoutItem]:
        """Layout items of one page ([] if the call failed)."""
        answer = self.backend.generate_json(self.request(pdf_path, page))
        return parse_structured(answer, page) if answer is not None else []

    def apply(self, pdf_path: Path, pages: List[PageText]) -> int:
        """
        Converts all pages (bounded concurrency) and stores the results.

        Returns:
            Number of pages with items.
        """
        workers = max(1, self.settings.extract.ovis_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda p: (p, self.page_items(pdf_path, p.page)), pages))
        done = 0
        for page, items in results:
            if items:
                apply_items(page, items)
                done += 1
        return done
