# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_stitch.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for page-table stitching and chunk merging on synthetic grids and on reportlab multi-page fixtures (3-page bank statement with carry-over rows, 2-page invoice).
------------------------------------------------------------------------------
"""

from decimal import Decimal
from typing import Any, Dict

from fluxsense.extract.base import PageTable
from fluxsense.extract.native import read_pages
from fluxsense.grounding.normalize import to_decimal
from fluxsense.tables.stitch import (classify_row, drop_artefact_rows, is_header_row, merge_chunks, render_markdown,
                                     stitch_tables)

HEADER = ["Datum", "Text", "Betrag"]


def test_header_and_artefact_classification() -> None:
    """Header rows, carry-over, balance, total and page-number rows."""
    assert is_header_row(HEADER) and not is_header_row(["01.09.", "REWE", "-7,35"])
    assert classify_row(["", "Übertrag", "1.234,56"]) == "carry_over"
    assert classify_row(["", "Zwischensumme", "100,00"]) == "carry_over"
    assert classify_row(["Alter Kontostand", "", "1.523,40"]) == "balance"
    assert classify_row(["Summe", "", "300,00"]) == "total"
    assert classify_row(["Seite 2 von 3", "", ""]) == "page_number"
    assert classify_row(["", "", ""]) == "empty"
    assert classify_row(["01.09.", "REWE Markt", "-7,35"]) is None


def test_stitch_continuation_with_repeated_header_and_duplicates() -> None:
    """Same width + same header on the next page = one table; boundary duplicates and carry-overs dropped."""
    page1 = PageTable(page=1, source="t", rows=[HEADER, ["01.09.", "A", "-1,00"], ["02.09.", "B", "-2,00"], ["", "Übertrag", "-3,00"]])
    page2 = PageTable(page=2, source="t", rows=[HEADER, ["", "Übertrag", "-3,00"], ["02.09.", "B", "-2,00"], ["03.09.", "C", "5,00"]])
    page3 = PageTable(page=3, source="t", rows=[["04.09.", "D", "1,00"], ["Seite 3 von 3", "", ""]])
    tables = stitch_tables([page1, page2, page3])
    assert len(tables) == 1
    table = tables[0]
    assert [r.cells[1] for r in table.rows] == ["A", "B", "C", "D"]
    assert [r.page for r in table.rows] == [1, 1, 2, 3]
    assert table.pages == [1, 2, 3]
    assert table.removed["carry_over"] == 2 and table.removed["duplicate_boundary_row"] == 1
    assert "| page |" in render_markdown(table)


def test_different_tables_are_not_merged() -> None:
    """A different column count or header starts a new table."""
    a = PageTable(page=1, source="t", rows=[HEADER, ["01.09.", "A", "1,00"]])
    b = PageTable(page=2, source="t", rows=[["Pos", "Artikel"], ["1", "Schraube"]])
    c = PageTable(page=3, source="t", rows=[["Kto", "Name", "Saldo"], ["1", "X", "2,00"]])
    assert len(stitch_tables([a, b, c])) == 3


def test_wrapped_text_lines_are_folded() -> None:
    """A text-only line after a data row belongs to that row's description."""
    table = stitch_tables([PageTable(page=1, source="t", rows=[HEADER, ["01.09.", "SEPA Lastschrift", "-9,99"],
                                                               ["", "Mandat 4711 Stadtwerke", ""], ["02.09.", "REWE", "-1,00"]])])[0]
    assert len(table.rows) == 2 and "Mandat 4711" in table.rows[0].cells[1]


def test_merge_chunks_drops_carried_rows() -> None:
    """Rows repeated at the start of a chunk (context rows) are removed, order kept."""
    a = [{"d": "1", "a": 1}, {"d": "2", "a": 2}]
    b = [{"d": "2", "a": 2}, {"d": "3", "a": 3}]
    c = [{"d": "4", "a": 4}, {"d": "2", "a": 2}]
    merged, dropped = merge_chunks([a, b, c], carry=2)
    assert [r["d"] for r in merged] == ["1", "2", "3", "4", "2"] and dropped == 1


def test_drop_llm_artefact_rows() -> None:
    """Carry-over rows copied by a model are removed from extracted tables."""
    rows = [{"description": "Übertrag"}, {"description": "REWE"}, {"description": "Neuer Kontostand"}]
    clean, removed = drop_artefact_rows(rows, ("description",))
    assert clean == [{"description": "REWE"}] and removed == 2


def test_real_bank_statement_three_pages(samples: Dict[str, Any]) -> None:
    """The reportlab statement (3 pages, carry-over rows, repeated headers) becomes ONE table of all bookings."""
    truth = samples["bank_statement_3p"].data
    pages = read_pages(samples["bank_statement_3p"].path, "native")
    tables = stitch_tables([t for p in pages for t in p.tables])
    assert len(tables) == 1
    table = tables[0]
    assert len(table.rows) == len(truth["entries"]) and table.pages == [1, 2, 3]
    assert table.removed.get("carry_over") == 4 and table.removed.get("repeated_header") == 2
    amounts = [to_decimal(r.cells[3]) for r in table.rows]
    assert sum(amounts, Decimal(0)) == truth["closing"] - truth["opening"]
    assert [to_decimal(r.cells[4]) for r in table.rows] == [e[4] for e in truth["entries"]]


def test_real_invoice_two_pages(samples: Dict[str, Any]) -> None:
    """Line items over two pages become one table whose totals match the net."""
    truth = samples["invoice_multipage"].data
    pages = read_pages(samples["invoice_multipage"].path, "native")
    tables = [t for t in stitch_tables([t for p in pages for t in p.tables]) if t.header[0] == "Pos"]
    assert len(tables) == 1
    rows = tables[0].rows
    assert [r.cells[0] for r in rows] == [str(i) for i in range(1, len(truth["lines"]) + 1)]
    assert sum((to_decimal(r.cells[4]) for r in rows), Decimal(0)) == truth["net"]
    assert {r.page for r in rows} == {1, 2}
