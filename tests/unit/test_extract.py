# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_extract.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for the auxiliary text layer: native detection, word
                boxes, OCR fallback on a temporary copy (input unchanged),
                docling-serve form fields.
------------------------------------------------------------------------------
"""

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict

import pytest

from fluxsense.extract import build_text_layer, resolve_engine
from fluxsense.extract.docling_engine import _form_dict, serve_form_data
from fluxsense.extract.native import has_font_resources
from fluxsense.extract.ocr import build_command


def test_native_layer(settings: Any, samples: Dict[str, Any]) -> None:
    """A reportlab PDF has a text layer, words with boxes and detected tables."""
    layer = build_text_layer(samples["invoice_wrong_vat"].path, settings, "native")
    assert layer.has_text_layer and not layer.ocr_applied and layer.sources() == ["stream", "native"]
    assert "RE-2026-0815" in layer.pages[0].best_text()
    assert layer.pages[0].words and layer.pages[0].words_source == "native"
    assert layer.tables()


def test_scan_has_no_text_layer(samples: Dict[str, Any]) -> None:
    """The image-only copy has no font resources."""
    assert not has_font_resources(samples["invoice_wrong_vat_scanned"].path)


@pytest.mark.skipif(shutil.which("ocrmypdf") is None, reason="ocrmypdf not installed")
def test_ocr_fallback_on_temp_copy(settings: Any, samples: Dict[str, Any]) -> None:
    """Scans are OCR'd via ocrmypdf on a temporary copy; the input file is never modified."""
    path = samples["invoice_wrong_vat_scanned"].path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    settings.extract.scan_fallback = "ocrmypdf"
    layer = build_text_layer(path, settings, "native")
    assert layer.ocr_applied and "ocr" in layer.sources()
    assert "0815" in layer.pages[0].sources["ocr"]
    assert layer.pages[0].words_source == "ocr"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_engine_resolution_and_commands(settings: Any) -> None:
    """Explicit engines resolve to themselves; unknown engines are rejected; commands are well-formed."""
    assert resolve_engine("ocrmypdf") == "ocrmypdf"
    with pytest.raises(ValueError):
        resolve_engine("bogus")
    cmd = build_command(settings.extract, Path("/tmp/a.pdf"), Path("/tmp/b.pdf"))
    assert cmd[0] == "ocrmypdf" and "-l" in cmd and "deu+eng" in cmd
    form = _form_dict(serve_form_data(settings.extract))
    assert form["to_formats"] == ["json", "md"] and form["pipeline"] == "standard" and form["ocr_lang"] == ["de", "en"]


def test_docling_table_cells_become_word_boxes() -> None:
    """Docling table cells (e.g. totals) are usable for zoom locating on scans."""
    from types import SimpleNamespace
    from fluxsense.extract.docling_engine import pages_from_document

    class Box:
        def __init__(self, l: float, t: float, r: float, b: float) -> None:
            self.l, self.t, self.r, self.b = l, t, r, b

        def to_top_left_origin(self, page_height: float) -> "Box":
            return Box(self.l, page_height - self.t, self.r, page_height - self.b)

    cell = SimpleNamespace(text="357,00 EUR", bbox=Box(400, 450, 480, 440))
    table = SimpleNamespace(prov=[SimpleNamespace(page_no=1)],
                            data=SimpleNamespace(grid=[[SimpleNamespace(text="Summe")], [cell]], table_cells=[cell]))
    doc = SimpleNamespace(pages={1: SimpleNamespace(size=SimpleNamespace(height=842.0))}, tables=[table],
                          iterate_items=lambda: [], export_to_markdown=lambda **k: "")
    result = pages_from_document(doc, "test")
    word = result.items[1][0]
    assert word.text == "357,00 EUR" and (word.x0, word.y0) == (400, 392.0)
