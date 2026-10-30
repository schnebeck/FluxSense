# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_ovis_degenerate.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Degenerate OvisOCR3 answers on a generated 600 dpi scan of a
                hand-filled form: a runaway digit repetition inside an item
                and an early stop trigger a re-read as overlapping tiles
                rendered from the native resolution; the merged layout is
                complete and the warning recorded. Negatives: long but
                legitimate numbers, leader dots and a short page.
------------------------------------------------------------------------------
"""

import io
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from fluxsense.extract import build_text_layer
from fluxsense.extract.ovis import cut_degenerate, merge_tiles, parse_structured, repetition_start
from fluxsense.layout.items import LayoutItem
from fluxsense.llm.base import LLMRequest
from fluxsense.testdata import build_landlord_confirmation, build_letter
from fluxsense.testdata.layout import Page, render
from tests.conftest import FakeBackend


def _scan(page: Page, target: Path, dpi: int) -> Path:
    """An image-only copy of a page model at a resolution (a scanned form)."""
    digital = render([page], target.with_suffix(".digital.pdf"))
    with pymupdf.open(digital) as src, pymupdf.open() as out:
        pix = src[0].get_pixmap(dpi=dpi)
        new = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
        out.save(target)
    return target


def _items(page: Page, top: float = 0.0, bottom: float = 1.0) -> List[Dict[str, Any]]:
    """Ovis-style items of the page's text runs inside a band, bboxes relative to the band (0-1000)."""
    out = []
    for run in sorted(page.runs, key=lambda r: (r.y, r.left())):
        y0, y1 = (run.y - run.size) / page.height, (run.y + 2) / page.height
        if not top <= (y0 + y1) / 2 < bottom:
            continue
        span = bottom - top
        out.append({"bbox": [round(run.left() / page.width * 1000), round((y0 - top) / span * 1000),
                             round((run.left() + run.width()) / page.width * 1000), round((y1 - top) / span * 1000)],
                    "category": "text", "content": run.text})
    return out


def _answers(page: Page, full: List[Dict[str, Any]]) -> Tuple[Any, List[LLMRequest]]:
    seen: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Optional[Any]:
        seen.append(request)
        if request.label.endswith("tile1"):
            return _items(page, 0.0, 0.55)
        if request.label.endswith("tile2"):
            return _items(page, 0.45, 1.0)
        return full
    return handler, seen


def test_repetition_detection_and_cut() -> None:
    """A digit run or a repeated short unit is cut where it starts; legitimate long numbers and leader dots are not."""
    assert repetition_start("26629 1" + "0" * 100) == len("26629 10")
    assert repetition_start("ab" + "xy" * 20) == len("abxy")
    for text in ("IBAN DE44 9901 2345 0532 0130 00", "Referenz 10000000000000000000", "Name: " + "." * 60,
                 "_" * 40, "| 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |"):
        assert repetition_start(text) is None, text
    items, degenerate = cut_degenerate(parse_structured([{"bbox": [1, 2, 3, 4], "category": "text",
                                                          "content": "26629 1" + "0" * 100}], 1))
    assert degenerate and items[0].content == "26629 10"


def test_merge_tiles_maps_and_deduplicates() -> None:
    """Tile boxes go back to page coordinates; an item seen by both tiles near the cut is kept once."""
    top = [LayoutItem(1, (100, 100, 900, 150), "text", "Kopf"), LayoutItem(1, (100, 880, 900, 920), "text", "Mitte")]
    bottom = [LayoutItem(1, (100, 80, 900, 130), "text", "Mitte"), LayoutItem(1, (100, 900, 900, 950), "text", "Fuss")]
    merged = merge_tiles([(0.0, 0.55, top), (0.45, 1.0, bottom)])
    assert [i.content for i in merged] == ["Kopf", "Mitte", "Fuss"]
    assert round(merged[0].bbox[1]) == 55 and round(merged[2].bbox[1]) == round((0.45 + 0.9 * 0.55) * 1000)


def test_degenerate_answer_is_retiled_from_native_resolution(settings: Any, tmp_path: Path) -> None:
    """A 600 dpi scan of a hand-filled form: the 11th item runs into zeros and the answer stops; the page is re-read
    as two tiles rendered from 600 dpi (fitted to 2880 px, not upsampled from 300 dpi); every printed label and the
    handwritten persons are in the merged layout; the warning is recorded."""
    doc = build_landlord_confirmation(random.Random("wgb-ovis"))
    page = doc.pages[0]
    pdf = _scan(page, tmp_path / "form.pdf", 600)
    full = _items(page)[:10] + [{"bbox": [80, 420, 600, 440], "category": "text", "content": "26629 1" + "0" * 100}]
    handler, seen = _answers(page, full)
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(handler))
    assert [r.label for r in seen] == ["OCR ovisocr3 p1", "OCR ovisocr3 p1 tile1", "OCR ovisocr3 p1 tile2"]
    assert all(r.images[0].width == 2880 for r in seen[1:])  # 600 dpi tiles fitted down; a 300 dpi tile is 2480 px wide
    text = layer.pages[0].sources["ovis"]
    for needle in ["Ort, Datum:", "Unterschrift des Wohnungsgebers", "Anschrift des Wohnungsgebers:"] + doc.facts["persons"]:
        assert needle in text, needle
    assert "0" * 25 not in text and len(set(text.splitlines())) == len(text.splitlines())
    assert "ovis_degenerate_p1_retiled" in layer.warnings


def test_early_stop_is_retiled(settings: Any, tmp_path: Path) -> None:
    """No repetition, but the answer ends at the top third while the page has ink below: re-read as tiles."""
    doc = build_landlord_confirmation(random.Random("wgb-early"))
    page = doc.pages[0]
    pdf = _scan(page, tmp_path / "form.pdf", 300)
    handler, seen = _answers(page, [i for i in _items(page) if i["bbox"][3] < 300])
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(handler))
    assert len(seen) == 3 and "ovis_degenerate_p1_retiled" in layer.warnings
    assert doc.facts["persons"][0] in layer.pages[0].sources["ovis"]


def test_normal_and_short_pages_are_not_retiled(settings: Any, tmp_path: Path) -> None:
    """A complete answer with a long IBAN-like reference, and a short page with nothing below its text: one call each."""
    doc = build_letter(random.Random("ovis-normal"))
    page = doc.pages[0]
    page.text(62, 700, "Referenz 10000000000000000000 · Name: " + "." * 40, 9)
    pdf = _scan(page, tmp_path / "letter.pdf", 300)
    handler, seen = _answers(page, _items(page))
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(handler))
    assert len(seen) == 1 and not any(w.startswith("ovis_degenerate") for w in layer.warnings)
    short = Page()
    short.text(62, 80, "Kurze Notiz", 14, bold=True)
    short.text(62, 110, "Bitte zurückrufen.", 10)
    pdf = _scan(short, tmp_path / "short.pdf", 300)
    handler, seen = _answers(short, _items(short))
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(handler))
    assert len(seen) == 1 and not any(w.startswith("ovis_degenerate") for w in layer.warnings)
