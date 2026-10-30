# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/handread.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Freely handwritten pages (handwriting level 'full', field
                contract §12): the OCR text is only a hint. The page is cut
                into bands of a few text lines at the empty pixel rows
                between them, every band is cropped at the scan's native
                resolution and transcribed twice with two differently worded
                prompts (no OCR text shown). Lines both readings agree on -
                and that look like text, not a runaway repetition - are
                accepted; the others are kept as '[?] reading A | reading B'.
                The result is the 'handread' text source of the page: the
                extraction sees it as a HANDWRITTEN TEXT block, the
                acceptance accepts a value of such a page only when it stands
                in an agreed line.
------------------------------------------------------------------------------
"""

import difflib
import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from fluxsense.config import Settings
from fluxsense.extract.base import PageText
from fluxsense.extract.fieldread import dark_rows
from fluxsense.grounding.normalize import normalize_text
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, encode_data_uri, fit_image, render_page
from fluxsense.log import get_logger
from fluxsense.pdflock import PYMUPDF_LOCK

logger = get_logger("extract.handread")

SOURCE_HANDREAD = "handread"
DISPUTED = "[?] "
LINES_PER_BAND: int = 3
MAX_BANDS: int = 14
INK_ROW: float = 0.004  # a pixel row with more dark pixels than this share carries text
RULE_ROW: float = 0.5  # ... and one darker than this is a printed rule, not text
PROMPT_A = ('This is a horizontal strip of a handwritten page. Transcribe every line of text in it exactly as written, top '
            'to bottom, one entry per text line. Keep spelling, numbers and punctuation as they stand; do not correct, '
            'complete or add anything; skip lines that are cut off at the strip edge. Answer as JSON: '
            '{"lines": ["...", ...]}')
PROMPT_B = ('Read this cut-out of a page written by hand. Write down each text line you can see, from the top line to the '
            'bottom line, character by character as it is written - no corrections, no additions, no summary. Answer '
            'as JSON: {"lines": ["...", ...]}')
SCHEMA: Dict[str, Any] = {"type": "object", "additionalProperties": False, "required": ["lines"],
                          "properties": {"lines": {"type": "array", "maxItems": 24,
                                                   "items": {"type": "string", "maxLength": 300}}}}


@dataclass
class HandReading:
    """The double reading of one handwritten page."""

    page: int
    agreed: List[str] = field(default_factory=list)
    disputed: List[Tuple[str, str]] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)  # the source text in reading order (agreed lines and '[?] a | b')
    bands: int = 0


def line_bands(overview: Image.Image, lines_per_band: int = LINES_PER_BAND, max_bands: int = MAX_BANDS) -> List[Tuple[float, float]]:
    """
    Horizontal bands of a page (fractions of the page height) of a few text
    lines each, cut in the middle of the empty pixel rows between lines so no
    line is split. Printed rules are no text rows.
    """
    rows = dark_rows(overview)
    inked = [INK_ROW < r <= RULE_ROW for r in rows]
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for y, on in enumerate(inked + [False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if segments and start - segments[-1][1] <= 1:
                segments[-1] = (segments[-1][0], y)  # a one-pixel gap inside a line
            else:
                segments.append((start, y))
            start = None
    segments = [s for s in segments if s[1] - s[0] >= 2]
    if not segments:
        return []
    while len(segments) > lines_per_band * max_bands:
        lines_per_band += 1
    height = len(rows)
    groups = [segments[i:i + lines_per_band] for i in range(0, len(segments), lines_per_band)]
    bands: List[Tuple[float, float]] = []
    for index, group in enumerate(groups):
        top = (groups[index - 1][-1][1] + group[0][0]) / 2 if index else max(0, group[0][0] - 4)
        bottom = (group[-1][1] + groups[index + 1][0][0]) / 2 if index + 1 < len(groups) else min(height, group[-1][1] + 4)
        bands.append((top / height, bottom / height))
    return bands


def _plausible(line: str) -> bool:
    """A transcribed line that looks like text: not empty, not a runaway repetition of one token."""
    tokens = line.split()
    if not tokens or len(line) > 280:
        return False
    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return max(counts.values()) <= max(4, len(tokens) // 2)


def _key(line: str) -> str:
    """Comparison key of a transcribed line (case, spacing and edge punctuation do not count)."""
    return re.sub(r"\s+", " ", normalize_text(line)).strip(" .,;:-")


def compare(first: Sequence[str], second: Sequence[str]) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """
    Aligns two transcriptions of a band line by line.

    Returns:
        (source lines in reading order, agreed lines, disputed (a, b) pairs).
    """
    a = [line.strip() for line in first if line.strip()]
    b = [line.strip() for line in second if line.strip()]
    matcher = difflib.SequenceMatcher(a=[_key(x) for x in a], b=[_key(y) for y in b], autojunk=False)
    lines: List[str] = []
    agreed: List[str] = []
    disputed: List[Tuple[str, str]] = []
    for tag, i0, i1, j0, j1 in matcher.get_opcodes():
        if tag == "equal":
            for line in a[i0:i1]:
                if _plausible(line):
                    lines.append(line)
                    agreed.append(line)
                else:
                    lines.append(f"{DISPUTED}{line} | {line}")
                    disputed.append((line, line))
            continue
        left, right = a[i0:i1], b[j0:j1]
        for k in range(max(len(left), len(right))):
            pair = (left[k] if k < len(left) else "", right[k] if k < len(right) else "")
            lines.append(f"{DISPUTED}{pair[0]} | {pair[1]}")
            disputed.append(pair)
    return lines, agreed, disputed


class HandReader:
    """Reads freely handwritten pages twice from native-resolution line bands."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: Vision-language backend.
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings

    def _crop(self, pdf_path: Path, page: int, band: Tuple[float, float], dpi: int) -> Image.Image:
        import pymupdf
        with PYMUPDF_LOCK, pymupdf.open(pdf_path) as doc:
            pdf_page = doc.load_page(page - 1)
            rect = pdf_page.rect
            clip = pymupdf.Rect(rect.x0, rect.y0 + band[0] * rect.height, rect.x1, rect.y0 + band[1] * rect.height)
            png = pdf_page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
        return fit_image(Image.open(io.BytesIO(png)).convert("RGB"), 1600)

    def _transcribe(self, image: Image.Image, page: int, band: int, second: bool) -> List[str]:
        part = ImagePart(label=f"BAND {band}", data_uri=encode_data_uri(image, "png"), width=image.width, height=image.height)
        model = self.settings.models.primary if second else self.settings.models.vision
        answer = self.backend.generate_json(LLMRequest(
            prompt=PROMPT_B if second else PROMPT_A, label=f"Handread p{page} band {band} {'B' if second else 'A'}",
            model=model, images=[part], schema=SCHEMA, schema_name="handwritten_lines", max_tokens=600))
        lines = answer.get("lines") if isinstance(answer, dict) else None
        return [str(line) for line in lines if isinstance(line, (str, int, float))] if isinstance(lines, list) else []

    def read_page(self, pdf_path: Path, page: PageText) -> Optional[HandReading]:
        """
        Double reading of one page (None if the page shows no text rows or no
        band could be read).
        """
        from fluxsense.extract.ovis import NATIVE_MAX_DPI, native_dpi
        bands = line_bands(render_page(pdf_path, page.page - 1, dpi=100))
        if not bands:
            return None
        dpi = min(NATIVE_MAX_DPI, max(300, native_dpi(pdf_path, page.page - 1) or 300))
        crops = [self._crop(pdf_path, page.page, band, dpi) for band in bands]
        jobs = [(index, second) for index in range(len(crops)) for second in (False, True)]
        workers = max(1, self.settings.llm.max_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(lambda job: self._transcribe(crops[job[0]], page.page, job[0] + 1, job[1]), jobs))
        reading = HandReading(page=page.page, bands=len(bands))
        for index in range(len(crops)):
            first, second = answers[2 * index], answers[2 * index + 1]
            if not first and not second:
                continue
            lines, agreed, disputed = compare(first, second)
            reading.lines += lines
            reading.agreed += agreed
            reading.disputed += disputed
        return reading if reading.lines else None

    def run(self, pdf_path: Path, pages: Sequence[PageText], levels: Optional[Mapping[int, str]] = None) -> List[HandReading]:
        """
        Reads the pages whose handwriting level is 'full' and stores each
        reading as the page's 'handread' text source.

        Args:
            pdf_path: PDF file.
            pages: Pages of the logical document.
            levels: Page -> handwriting level (default: the pages' own level).

        Returns:
            The readings.
        """
        from fluxsense.stages.signatures import FULL
        readings: List[HandReading] = []
        for page in pages:
            if ((levels or {}).get(page.page) or page.handwriting) != FULL or not page.scanned():
                continue
            reading = self.read_page(pdf_path, page)
            if reading is None:
                continue
            page.sources[SOURCE_HANDREAD] = "\n".join(reading.lines)
            readings.append(reading)
            logger.info(f"handread p{page.page}: {len(reading.agreed)} lines agreed, {len(reading.disputed)} disputed "
                        f"({reading.bands} bands)")
        return readings


def agreed_text(page: Optional[PageText]) -> str:
    """The agreed lines of a page's double reading ('' without one)."""
    text = page.sources.get(SOURCE_HANDREAD, "") if page is not None else ""
    return "\n".join(line for line in text.splitlines() if line.strip() and not line.startswith(DISPUTED))


def handwriting_block(page: PageText) -> str:
    """The HANDWRITTEN TEXT block of a page for the extraction prompt ('' without a double reading)."""
    text = page.sources.get(SOURCE_HANDREAD, "")
    if not text.strip():
        return ""
    return ("HANDWRITTEN TEXT (the page is written by hand; read twice from native-resolution line crops - lines marked "
            "[?] differ between the two readings 'A | B' and are uncertain; the OCR text that follows is only a hint):\n"
            + text + "\n")
