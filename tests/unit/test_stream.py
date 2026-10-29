# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_stream.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Engine-agnostic layout items and the document reading
                stream (furniture dropped, paragraphs joined across pages,
                table continuation marked, provenance kept).
------------------------------------------------------------------------------
"""

from fluxsense.extract.base import PageText, TextLayer
from fluxsense.extract import attach_stream
from fluxsense.layout.items import (FOOTER, HEADER, PAGE_NUMBER, TABLE, TEXT, LayoutItem, build_stream, html_table_rows,
                                    native_items)

TABLE_1 = "<table><tr><th>Datum</th><th>Text</th><th>Betrag</th></tr><tr><td>01.09.</td><td>REWE</td><td>-7,35</td></tr></table>"
TABLE_2 = "<table><tr><td>02.09.</td><td>Miete</td><td>-800,00</td></tr></table>"


def item(page: int, category: str, content: str, y: float = 500) -> LayoutItem:
    """A layout item."""
    return LayoutItem(page, (100, y, 900, y + 20), category, content, "test")


def test_stream_drops_furniture_joins_paragraphs_and_marks_tables() -> None:
    """Headers/footers/page numbers dropped; a sentence continuing on the next page joined; table continuation marked."""
    pages = {
        1: [item(1, HEADER, "Musterbank eG", 20), item(1, TEXT, "Wir bestätigen den Eingang Ihrer", 800),
            item(1, PAGE_NUMBER, "Seite 1 von 2", 960)],
        2: [item(2, HEADER, "Musterbank eG", 20), item(2, TEXT, "Beschwerde vom 3. Juli.", 100), item(2, TABLE, TABLE_1, 300),
            item(2, PAGE_NUMBER, "Seite 2 von 2", 960)],
        3: [item(3, TABLE, TABLE_2, 100)],
    }
    pages[1].append(item(1, FOOTER, "Musterbank eG · IBAN DE44 9901 2345 0532 0130 00", 940))
    stream = build_stream(pages)
    assert stream.dropped == 4 and stream.joined == 1 and stream.table_continuations == 1
    assert "IBAN DE44" in stream.page_text(1)  # single footer with bank details is content
    assert stream.lines[0].text == "Wir bestätigen den Eingang Ihrer Beschwerde vom 3. Juli." and stream.lines[0].pages == [1, 2]
    assert "(table continued from page 2)" in stream.page_text(3) and "| 02.09. | Miete | -800,00 |" in stream.page_text(3)
    assert all(l.text != "Musterbank eG" for l in stream.lines) and "=== PAGE 2 ===" in stream.text()
    assert html_table_rows(TABLE_1) == [["Datum", "Text", "Betrag"], ["01.09.", "REWE", "-7,35"]]


def test_native_fallback_detects_repeated_headers_and_page_numbers() -> None:
    """Native blocks: repeated top-band text and page numbers become furniture."""
    blocks = [(50, 20, 300, 40, "Musterbank eG Kontoauszug"), (50, 200, 500, 220, "Text"), (280, 800, 320, 815, "Seite 2")]
    items = native_items(2, blocks, 595, 842, {"musterbank eg kontoauszug"})
    assert [i.category for i in items] == [HEADER, TEXT, PAGE_NUMBER]


def test_attach_stream_sets_stream_source() -> None:
    """The stream becomes the preferred auxiliary text of each page."""
    pages = [PageText(page=1, sources={"native": "x"}, items=[item(1, TEXT, "Hallo Welt")]),
             PageText(page=2, sources={"native": "y"}, items=[item(2, FOOTER, "Seite 2", 960)])]
    layer = TextLayer(pages=pages, engine="t")
    attach_stream(layer)
    assert layer.pages[0].best_text() == "Hallo Welt" and "stream" not in layer.pages[1].sources
