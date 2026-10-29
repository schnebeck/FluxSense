# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/vision/zoom.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Adaptive-resolution 'zoom' capability (escalation level 2):
                locates the region of a value (model bbox, text-layer / OCR /
                Docling word boxes of the evidence snippet or value variants,
                else a 3x3 page tile chosen by a cheap localisation question),
                re-renders only that region at high DPI with PyMuPDF clip=,
                and asks closed questions per crop, several fields of one page
                batched into one call.
------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pymupdf as fitz
from PIL import Image

from fluxsense.pdflock import PYMUPDF_LOCK
from fluxsense.config import Settings
from fluxsense.extract.base import PageText, TextLayer, Word
from fluxsense.grounding.matcher import KIND_AMOUNT, KIND_DATE
from fluxsense.grounding.normalize import compact, parse_date, to_decimal
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, encode_data_uri, fit_image
from fluxsense.log import get_logger
from fluxsense.prompts import verify as prompts

logger = get_logger("vision.zoom")

Rect = Tuple[float, float, float, float]
TILES: Dict[str, Tuple[int, int]] = {"TL": (0, 0), "TC": (1, 0), "TR": (2, 0), "ML": (0, 1), "MC": (1, 1), "MR": (2, 1),
                                     "BL": (0, 2), "BC": (1, 2), "BR": (2, 2)}
ANSWER_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["answers"],
    "properties": {"answers": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "visible", "matches"],
        "properties": {"id": {"type": "string"}, "visible": {"type": "boolean"},
                       "printed": {"type": ["string", "null"], "maxLength": 80}, "matches": {"type": "boolean"}}}}},
}
LOCATE_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["tiles"],
    "properties": {"tiles": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["id", "tile"],
        "properties": {"id": {"type": "string"}, "tile": {"type": "string", "enum": list(TILES) + ["NONE"]}}}}},
}


@dataclass
class ZoomTarget:
    """A value to verify on a page region."""

    id: str
    page: int
    label: str
    value: Any
    kind: str
    evidence: Optional[str] = None
    bbox: Optional[List[float]] = None
    dpi: Optional[int] = None


@dataclass
class ZoomAnswer:
    """Answer of the VLM about one crop."""

    id: str
    visible: bool
    matches: bool
    printed: Optional[str]
    resolution: str
    located_by: str


def value_variants(kind: str, value: Any) -> List[str]:
    """
    Printed variants of a value used to find it among the words.

    Args:
        kind: Value kind.
        value: Normalized value.

    Returns:
        Candidate strings (most specific first).
    """
    variants: List[str] = []
    if kind == KIND_AMOUNT:
        amount = to_decimal(value)
        if amount is not None:
            a = abs(amount)
            en = f"{a:,.2f}"
            de = en.replace(",", "X").replace(".", ",").replace("X", ".")
            variants += [de, de.replace(".", ""), en, en.replace(",", "")]
    elif kind == KIND_DATE:
        d = parse_date(value)
        if d is not None:
            variants += [d.strftime("%d.%m.%Y"), f"{d.day}.{d.month}.{d.year}", d.isoformat(), d.strftime("%d.%m.%y"),
                         d.strftime("%d/%m/%Y"), d.strftime("%d.%m.")]
    if value is not None:
        variants.append(str(value))
    return [v for v in dict.fromkeys(variants) if v]


def find_words(words: Sequence[Word], needle: str, max_window: int = 8) -> Optional[Rect]:
    """
    Finds a needle in a word sequence (compact comparison over sliding windows).

    Args:
        words: Word boxes in reading order.
        needle: Text to find.
        max_window: Max words per window.

    Returns:
        Union rectangle of the matching words, or None.
    """
    target = compact(needle)
    if len(target) < 2:
        return None
    for start in range(len(words)):
        joined = ""
        for end in range(start, min(len(words), start + max_window)):
            if end > start:
                prev, cur = words[end - 1], words[end]
                if abs(cur.y0 - prev.y0) > max(4.0, (prev.y1 - prev.y0) * 1.2):
                    break  # never join words across lines
            joined += compact(words[end].text)
            if target in joined:
                chosen = words[start:end + 1]
                return (min(w.x0 for w in chosen), min(w.y0 for w in chosen), max(w.x1 for w in chosen), max(w.y1 for w in chosen))
            if len(joined) > len(target) + 40:
                break
    return None


def locate(page: PageText, target: ZoomTarget) -> Tuple[Optional[Rect], str]:
    """
    Locates a value on a page.

    Args:
        page: Page evidence (word boxes).
        target: Target.

    Returns:
        Tuple (rect in PDF points or None, method).
    """
    for needle in ([target.evidence] if target.evidence else []) + value_variants(target.kind, target.value):
        rect = find_words(page.words, needle)
        if rect:
            return rect, f"words:{page.words_source}"
    if target.bbox and len(target.bbox) == 4:
        x0, y0, x1, y1 = target.bbox
        if 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000:
            return (x0 / 1000 * page.width, y0 / 1000 * page.height, x1 / 1000 * page.width, y1 / 1000 * page.height), "model_bbox"
    return None, "none"


def expand(rect: Rect, width: float, height: float, margin_x: float = 120.0, margin_y: float = 40.0) -> Rect:
    """Adds context around a rectangle (labels left of values) and clamps it to the page."""
    x0, y0, x1, y1 = rect
    return (max(0.0, x0 - margin_x), max(0.0, y0 - margin_y), min(width, x1 + margin_x), min(height, y1 + margin_y))


def tile_rect(tile: str, width: float, height: float) -> Rect:
    """Rectangle of a 3x3 tile with 10 % overlap."""
    col, row = TILES[tile]
    w, h = width / 3, height / 3
    return (max(0.0, col * w - w * 0.1), max(0.0, row * h - h * 0.1), min(width, (col + 1) * w + w * 0.1), min(height, (row + 1) * h + h * 0.1))


def render_crop(pdf_path: Path, page: int, rect: Rect, dpi: int, max_px: int) -> Tuple[Image.Image, int]:
    """
    Renders a page region at high DPI (only the clip is rasterized).

    Args:
        pdf_path: PDF file.
        page: 1-based page.
        rect: Region in PDF points.
        dpi: Target DPI.
        max_px: Long side limit (effective DPI is lowered if needed).

    Returns:
        Tuple (image, effective dpi).
    """
    with PYMUPDF_LOCK, fitz.open(pdf_path) as doc:
        pdf_page = doc.load_page(page - 1)
        clip = fitz.Rect(*rect)
        long_side_pt = max(clip.width, clip.height) or 1.0
        effective = int(min(dpi, max_px / (long_side_pt / 72.0)))
        pix = pdf_page.get_pixmap(clip=clip, dpi=max(72, effective))
        img = Image.frombytes("RGB" if pix.n < 4 else "RGBA", (pix.width, pix.height), pix.samples).convert("RGB")
    return fit_image(img, max_px), max(72, effective)


class Zoomer:
    """Verifies values on high-resolution region crops."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: LLM backend.
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings

    def _locate_by_tiles(self, pdf_path: Path, page: PageText, targets: List[ZoomTarget]) -> Dict[str, str]:
        """Cheap localisation question on a medium page image (one call per page)."""
        from fluxsense.llm.images import page_images
        images = page_images(pdf_path, [page.page], [f"PAGE {page.page}"], self.settings.llm.classify_image_px,
                             self.settings.llm.image_format, dpi=100)
        items = "\n".join(f"- {t.id}: {t.label} = {t.value}" for t in targets)
        result = self.backend.generate_json(LLMRequest(prompt=prompts.PROMPT_LOCATE.format(items=items),
                                                       label=f"Zoom locate p{page.page}", model=self.settings.models.vision,
                                                       images=images, schema=LOCATE_SCHEMA, schema_name="locate", max_tokens=300))
        tiles: Dict[str, str] = {}
        if isinstance(result, dict):
            for entry in result.get("tiles") or []:
                if isinstance(entry, dict) and entry.get("tile") in TILES:
                    tiles[str(entry.get("id"))] = str(entry["tile"])
        return tiles

    def verify(self, pdf_path: Path, layer: TextLayer, targets: List[ZoomTarget]) -> Dict[str, ZoomAnswer]:
        """
        Verifies targets; one VLM call per page with one crop per target.

        Args:
            pdf_path: PDF file.
            layer: Text layer (word boxes for locating).
            targets: Values to verify.

        Returns:
            id -> ZoomAnswer.
        """
        llm = self.settings.llm
        answers: Dict[str, ZoomAnswer] = {}
        by_page: Dict[int, List[ZoomTarget]] = {}
        for target in targets:
            by_page.setdefault(target.page, []).append(target)
        for page_no, page_targets in by_page.items():
            page = layer.page(page_no)
            if page is None:
                continue
            rects: Dict[str, Tuple[Rect, str]] = {}
            unlocated: List[ZoomTarget] = []
            for target in page_targets:
                rect, method = locate(page, target)
                if rect:
                    rects[target.id] = (expand(rect, page.width, page.height), method)
                else:
                    unlocated.append(target)
            if unlocated:
                for target_id, tile in self._locate_by_tiles(pdf_path, page, unlocated).items():
                    rects[target_id] = (tile_rect(tile, page.width, page.height), f"tile:{tile}")
            images: List[ImagePart] = []
            questions: List[str] = []
            meta: Dict[str, Tuple[str, str]] = {}
            for target in page_targets:
                if target.id not in rects:
                    continue
                rect, method = rects[target.id]
                img, dpi = render_crop(pdf_path, page_no, rect, target.dpi or llm.zoom_dpi, llm.zoom_max_px)
                images.append(ImagePart(label=f"CROP {target.id}", data_uri=encode_data_uri(img, llm.image_format),
                                        width=img.width, height=img.height))
                questions.append(f"- {target.id} (crop {target.id}): Does it show the {target.label} '{target.value}'?")
                meta[target.id] = (f"crop@{dpi}dpi", method)
            if not images:
                continue
            result = self.backend.generate_json(LLMRequest(
                prompt=prompts.PROMPT_ZOOM_VERIFY.format(page=page_no, questions="\n".join(questions)),
                label=f"Zoom verify p{page_no} ({len(images)})", model=self.settings.models.vision, images=images,
                schema=ANSWER_SCHEMA, schema_name="zoom_verify", max_tokens=120 + 80 * len(images)))
            if not isinstance(result, dict):
                continue
            for entry in result.get("answers") or []:
                if not isinstance(entry, dict) or str(entry.get("id")) not in meta:
                    continue
                target_id = str(entry["id"])
                resolution, method = meta[target_id]
                answers[target_id] = ZoomAnswer(id=target_id, visible=bool(entry.get("visible")), matches=bool(entry.get("matches")),
                                                printed=entry.get("printed"), resolution=resolution, located_by=method)
        return answers


CHOICE_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["choice"],
    "properties": {"choice": {"type": "string", "enum": ["A", "B", "NEITHER"]}}}
PROMPT_CHOICE = """
You see a high-resolution crop of page {page}. It shows a {label}.
Read it character by character. Which of these two readings is printed?
A: {a}
B: {b}
Answer A, B, or NEITHER. Return ONLY JSON: {{"choice": "A"}}
"""


def binary_choice(backend: LLMBackend, settings: Settings, pdf_path: Path, layer: TextLayer, target: ZoomTarget,
                  option_a: str, option_b: str) -> Optional[str]:
    """
    Binary visual question on a high-DPI crop ('Does the IBAN read X or Y?').

    Returns:
        The chosen option string, or None (neither / not located / call failed).
    """
    page = layer.page(target.page)
    if page is None or pdf_path is None:
        return None
    rect, method = locate(page, target)
    if rect is None:
        for variant in (option_a, option_b):
            rect, method = locate(page, ZoomTarget(id=target.id, page=target.page, label=target.label, value=variant,
                                                   kind=target.kind))
            if rect:
                break
    if rect is None:
        rect = (0.0, 0.0, page.width, page.height)
    img, dpi = render_crop(pdf_path, target.page, expand(rect, page.width, page.height), target.dpi or 400,
                           settings.llm.zoom_max_px)
    image = ImagePart(label=f"CROP {target.id}", data_uri=encode_data_uri(img, settings.llm.image_format),
                      width=img.width, height=img.height)
    answer = backend.generate_json(LLMRequest(
        prompt=PROMPT_CHOICE.format(page=target.page, label=target.label, a=option_a, b=option_b),
        label=f"Zoom choice p{target.page}", model=settings.models.vision, images=[image], schema=CHOICE_SCHEMA,
        schema_name="zoom_choice", max_tokens=20))
    choice = answer.get("choice") if isinstance(answer, dict) else None
    return {"A": option_a, "B": option_b}.get(choice)
