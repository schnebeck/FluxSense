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
                (pretty-printed, truncated tail, repeated items, runaway
                repetition inside an item) into layout
                items, page text, word boxes and table grids.
------------------------------------------------------------------------------
"""

import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from fluxsense.config import Settings
from fluxsense.extract.base import SOURCE_OVIS, PageTable, PageText, Word
from fluxsense.layout.items import OVIS_CATEGORIES, TABLE, TEXT, LayoutItem, html_table_rows, item_text
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, encode_data_uri, fit_image, render_page
from fluxsense.llm.json_repair import parse_json_response
from fluxsense.log import get_logger
from fluxsense.pdflock import PYMUPDF_LOCK

logger = get_logger("extract.ovis")

MAX_REPEATS: int = 2
# Runaway repetition inside one item: one alphanumeric character >= 25 times, or a 2-6 character unit without
# whitespace (containing a letter or digit) >= 12 times in a row. Leader dots / underscores of form fields do not count.
_CHAR_RUN_RE = re.compile(r"([0-9A-Za-zÄÖÜäöüß])\1{24,}")
_UNIT_RUN_RE = re.compile(r"(\S{2,6}?)\1{11,}")
EARLY_STOP_Y: float = 600.0   # last item ends above this (0-1000) ...
INK_SHARE: float = 0.002       # ... while this share of dark pixels lies below it: the answer stopped early
TILES_A4: Tuple[Tuple[float, float], ...] = ((0.0, 0.55), (0.45, 1.0))
TILES_TALL: Tuple[Tuple[float, float], ...] = ((0.0, 0.40), (0.30, 0.70), (0.60, 1.0))
NATIVE_MAX_DPI: int = 600


def repetition_start(text: str) -> Optional[int]:
    """Index where a runaway repetition starts (the first repeated unit is kept), None if there is none."""
    starts = []
    for pattern in (_CHAR_RUN_RE, _UNIT_RUN_RE):
        match = pattern.search(text)
        if match and (pattern is _CHAR_RUN_RE or re.search(r"[0-9A-Za-zÄÖÜäöüß]", match.group(1))):
            starts.append(match.start() + len(match.group(1)))
    return min(starts) if starts else None


def cut_degenerate(items: List[LayoutItem]) -> Tuple[List[LayoutItem], bool]:
    """Cuts every item at the start of a runaway repetition; returns (items, degenerate?)."""
    degenerate = False
    out: List[LayoutItem] = []
    for item in items:
        start = repetition_start(item.content)
        if start is not None:
            degenerate = True
            item = replace(item, content=item.content[:start].rstrip())
        out.append(item)
    return out, degenerate


def _median(histogram: List[int]) -> int:
    half, seen = sum(histogram) / 2, 0
    for level, count in enumerate(histogram):
        seen += count
        if seen >= half:
            return level
    return 255


def ink_below(image: Image.Image, y_fraction: float, margin: float = 0.03) -> float:
    """
    Share of ink pixels below a height fraction (the bottom margin excluded)
    on a downscaled grey render; ink is darker than the paper by a clear
    margin (a tinted or grey scan background is no ink).
    """
    small = image.convert("L")
    small.thumbnail((400, 400))
    top = int(small.height * min(1.0, y_fraction + margin))
    bottom = int(small.height * (1.0 - margin))
    if bottom <= top:
        return 0.0
    paper = _median(small.histogram())
    region = small.crop((0, top, small.width, bottom))
    histogram = region.histogram()
    dark = sum(histogram[:max(0, min(160, paper - 70))])
    return dark / max(1, region.width * region.height)


def stopped_early(items: List[LayoutItem], image: Image.Image) -> bool:
    """True if the last item ends high on the page while the render still has ink below it."""
    if not items:
        return False
    lowest = max(item.bbox[3] for item in items)
    return lowest < EARLY_STOP_Y and ink_below(image, lowest / 1000.0) > INK_SHARE


def native_dpi(pdf_path: Path, page_index: int) -> Optional[int]:
    """Resolution of the largest image on a page (a scan's native dpi), None without images."""
    import pymupdf
    with PYMUPDF_LOCK, pymupdf.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        best: Optional[int] = None
        for info in page.get_image_info():
            bbox = pymupdf.Rect(info.get("bbox"))
            if bbox.width <= 0 or not info.get("width"):
                continue
            dpi = int(round(info["width"] / (bbox.width / 72.0)))
            best = max(best or 0, dpi)
        return best


def render_region(pdf_path: Path, page_index: int, top: float, bottom: float, dpi: int) -> Image.Image:
    """Renders a horizontal band (height fractions) of a page at a resolution."""
    import pymupdf
    with PYMUPDF_LOCK, pymupdf.open(pdf_path) as doc:
        page = doc.load_page(page_index)
        rect = page.rect
        clip = pymupdf.Rect(rect.x0, rect.y0 + rect.height * top, rect.x1, rect.y0 + rect.height * bottom)
        png = page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
    return Image.open(io.BytesIO(png)).convert("RGB")


def merge_tiles(tiles: Sequence[Tuple[float, float, List[LayoutItem]]]) -> List[LayoutItem]:
    """
    Maps tile items (bboxes relative to their tile) to page coordinates and
    de-duplicates the overlaps: between two tiles the cut line is the middle
    of their overlap; each tile keeps the items whose centre lies on its side,
    and an item repeated with the same text near the cut is kept once.
    """
    merged: List[LayoutItem] = []
    for index, (top, bottom, items) in enumerate(tiles):
        upper = 0.0 if index == 0 else (tiles[index - 1][1] + top) / 2
        lower = 1.0 if index == len(tiles) - 1 else (bottom + tiles[index + 1][0]) / 2
        for item in items:
            x0, y0, x1, y1 = item.bbox
            span = bottom - top
            page_box = (x0, (top + y0 / 1000 * span) * 1000, x1, (top + y1 / 1000 * span) * 1000)
            centre = (page_box[1] + page_box[3]) / 2000
            if not upper <= centre < lower:
                continue
            near = [m for m in merged if m.content == item.content and abs(m.bbox[1] - page_box[1]) < 60]
            if near:
                continue
            merged.append(replace(item, bbox=page_box))
    return merged


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


def _text_size(items: List[LayoutItem]) -> int:
    return sum(len(i.content.strip()) for i in items)


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
        self.warnings: List[str] = []

    def _request(self, img: Image.Image, page: int, label: str) -> LLMRequest:
        img = fit_image(img, self.settings.extract.ovis_max_px)
        image = ImagePart(label=f"PAGE {page}", data_uri=encode_data_uri(img, "png"), width=img.width, height=img.height)
        return LLMRequest(prompt=self.prompt, label=label, model=self.settings.models.ovis_ocr,
                          images=[image], max_tokens=self.settings.extract.ovis_max_tokens, verbatim=True)

    def request(self, pdf_path: Path, page: int) -> LLMRequest:
        """The request for one page (verbatim prompt, one image)."""
        return self._request(render_page(pdf_path, page - 1, dpi=300), page, f"OCR ovisocr3 p{page}")

    def _ask(self, request: LLMRequest, page: int) -> List[LayoutItem]:
        answer = self.backend.generate_json(request)
        return parse_structured(answer, page) if answer is not None else []

    def tile_items(self, pdf_path: Path, page: int) -> Tuple[List[LayoutItem], bool]:
        """
        Re-OCR of a page as overlapping horizontal tiles (2 for A4, 3 for tall
        pages), rendered from the scan's native resolution when it is above
        300 dpi; returns (merged items in page coordinates, degenerate?).
        """
        dpi = min(NATIVE_MAX_DPI, max(300, native_dpi(pdf_path, page - 1) or 300))
        probe = render_page(pdf_path, page - 1, dpi=36)
        bands = TILES_TALL if probe.height > 1.6 * probe.width else TILES_A4
        tiles: List[Tuple[float, float, List[LayoutItem]]] = []
        degenerate = False
        for number, (top, bottom) in enumerate(bands, 1):
            image = render_region(pdf_path, page - 1, top, bottom, dpi)
            items, bad = cut_degenerate(self._ask(self._request(image, page, f"OCR ovisocr3 p{page} tile{number}"), page))
            degenerate = degenerate or bad or stopped_early(items, image)
            tiles.append((top, bottom, items))
        return merge_tiles(tiles), degenerate

    def page_items(self, pdf_path: Path, page: int) -> List[LayoutItem]:
        """
        Layout items of one page ([] if the call failed). A degenerate answer
        (runaway repetition inside an item, or an early stop with ink below the
        last item) is cut and the page is re-read as overlapping tiles; the
        better result is kept and a warning recorded.
        """
        image = render_page(pdf_path, page - 1, dpi=300)
        items, degenerate = cut_degenerate(self._ask(self._request(image, page, f"OCR ovisocr3 p{page}"), page))
        early = stopped_early(items, image)
        if not (degenerate or early):
            return items
        reason = "repetition" if degenerate else "early stop"
        tiled, tile_degenerate = self.tile_items(pdf_path, page)
        if tile_degenerate:
            self.warnings.append(f"ovis_degenerate_p{page}_retile_degenerate")
            logger.warning(f"OvisOCR3 page {page}: {reason}; the retiled answer is degenerate too - keeping the better one")
            return tiled if _text_size(tiled) > _text_size(items) else items
        self.warnings.append(f"ovis_degenerate_p{page}_retiled")
        logger.warning(f"OvisOCR3 page {page}: {reason} ({len(items)} items) - re-read as tiles ({len(tiled)} items)")
        return tiled

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
