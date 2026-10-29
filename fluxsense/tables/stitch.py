# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/tables/stitch.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Page-independent tables. Stitches per-page table grids
                (PyMuPDF / Docling TableFormer) into logical tables: same
                column count + same or missing header -> continuation;
                removes repeated headers, carry-over/subtotal rows, page
                numbers, duplicate boundary rows; folds multi-line cells;
                keeps order and the source page of each row. Also merges
                page-chunked LLM row lists deterministically.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fluxsense.extract.base import PageTable
from fluxsense.grounding.normalize import extract_amounts, extract_dates, normalize_text

CARRY_OVER_RE = re.compile(
    r"\b(übertrag|uebertrag|ubertrag|vortrag|zwischensumme|seitensumme|summe seite|carried forward|brought forward|"
    r"carry[- ]?over|carryover|sub-?total|page total|transport|à reporter|a reporter)\b", re.IGNORECASE)
BALANCE_ROW_RE = re.compile(
    r"\b(alter kontostand|neuer kontostand|kontostand|anfangssaldo|endsaldo|saldo|opening balance|closing balance|"
    r"previous balance|new balance|balance brought|balance carried)\b", re.IGNORECASE)
TOTAL_ROW_RE = re.compile(r"^\s*(summe|gesamt|gesamtbetrag|total|netto|nettobetrag|mwst|ust|umsatzsteuer|brutto|"
                          r"rechnungsbetrag|endbetrag|zu zahlen)\b", re.IGNORECASE)
PAGE_NO_RE = re.compile(r"^\s*(seite|page|blatt)\s*\d+\s*(von|of|/)\s*\d+\s*$|^\s*-?\s*\d+\s*-?\s*$", re.IGNORECASE)


@dataclass
class StitchedRow:
    """A data row with its source page."""

    cells: List[str]
    page: int


@dataclass
class StitchedTable:
    """A logical table spanning one or more pages."""

    header: List[str]
    rows: List[StitchedRow] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    summary_rows: List[StitchedRow] = field(default_factory=list)
    removed: Dict[str, int] = field(default_factory=dict)

    @property
    def column_count(self) -> int:
        """Number of columns."""
        return len(self.header) if self.header else max((len(r.cells) for r in self.rows), default=0)

    def count(self, reason: str) -> None:
        """Increments a removal counter."""
        self.removed[reason] = self.removed.get(reason, 0) + 1


def _norm_row(cells: Sequence[str]) -> str:
    """Normalized row key."""
    return "|".join(normalize_text(c) for c in cells)


def _has_number_or_date(cell: str) -> bool:
    """True if a cell contains an amount-like number or a date."""
    if not cell:
        return False
    dates, partial = extract_dates(cell)
    return bool(dates or partial or (re.search(r"\d", cell) and extract_amounts(cell)))


_AMOUNT_CELL_RE = re.compile(r"^[-+−]?\s?\d{1,3}(?:[.,'’ ]?\d{3})*(?:[.,]\d{1,2})?\s?-?\s?(?:EUR|€|S|H)?$", re.IGNORECASE)


def is_value_cell(cell: str) -> bool:
    """True if a cell is an amount or a date (not text that merely contains digits)."""
    text = cell.strip()
    if not text:
        return False
    dates, partial = extract_dates(text)
    return bool(dates or partial or _AMOUNT_CELL_RE.match(text))


def is_header_row(cells: Sequence[str]) -> bool:
    """
    Heuristic header detection: mostly filled, no amounts or dates.

    Args:
        cells: Row cells.

    Returns:
        True if the row looks like column titles.
    """
    filled = [c for c in cells if c.strip()]
    if len(filled) < max(2, len(cells) // 2):
        return False
    return not any(_has_number_or_date(c) for c in filled)


def row_text(cells: Sequence[str]) -> str:
    """Joined row text."""
    return " ".join(c for c in cells if c)


def classify_row(cells: Sequence[str]) -> Optional[str]:
    """
    Classifies page artefacts.

    Args:
        cells: Row cells.

    Returns:
        'empty', 'carry_over', 'balance', 'total', 'page_number' or None for data rows.
    """
    text = row_text(cells).strip()
    if not text:
        return "empty"
    if PAGE_NO_RE.match(text):
        return "page_number"
    if CARRY_OVER_RE.search(text):
        return "carry_over"
    first = next((c for c in cells if c.strip()), "")
    if BALANCE_ROW_RE.search(text) and sum(1 for c in cells if _has_number_or_date(c)) <= 2:
        return "balance"
    if TOTAL_ROW_RE.match(first) and sum(1 for c in cells if c.strip()) <= 3:
        return "total"
    return None


def _continues(group: StitchedTable, table: PageTable, header: Optional[List[str]]) -> bool:
    """Decides whether a page table continues the current group."""
    if not group.pages or table.page != group.pages[-1] + 1:
        return False
    if table.column_count != group.column_count:
        return False
    if header is None:
        return True
    return _norm_row(header) == _norm_row(group.header)


def _is_continuation_line(cells: Sequence[str], previous: Optional[StitchedRow]) -> bool:
    """A row with text only (no amount/date) following a data row is a wrapped cell line."""
    if previous is None:
        return False
    filled = [c for c in cells if c.strip()]
    return bool(filled) and len(filled) <= 2 and not any(is_value_cell(c) for c in filled)


def stitch_tables(tables: Sequence[PageTable], fold_lines: bool = True) -> List[StitchedTable]:
    """
    Stitches page tables into logical tables.

    Args:
        tables: Page tables in page order.
        fold_lines: Merge wrapped text lines into the previous row.

    Returns:
        Logical tables.
    """
    result: List[StitchedTable] = []
    group: Optional[StitchedTable] = None
    for table in sorted(tables, key=lambda t: t.page):
        rows = [list(r) for r in table.rows]
        width = table.column_count
        rows = [r + [""] * (width - len(r)) for r in rows]
        header = rows[0] if rows and is_header_row(rows[0]) else None
        if group is None or not _continues(group, table, header):
            group = StitchedTable(header=header or [f"col{i + 1}" for i in range(width)])
            result.append(group)
        if table.page not in group.pages:
            group.pages.append(table.page)
        body = rows[1:] if header is not None else rows
        if header is not None and group.rows:
            group.count("repeated_header")
        first_on_page = True
        for cells in body:
            if group.header and _norm_row(cells) == _norm_row(group.header):
                group.count("repeated_header")
                continue
            kind = classify_row(cells)
            if kind is not None:
                group.count(kind)
                if kind in ("balance", "total", "carry_over"):
                    group.summary_rows.append(StitchedRow(cells, table.page))
                continue
            previous = group.rows[-1] if group.rows else None
            if first_on_page and previous is not None and previous.page != table.page \
                    and _norm_row(cells) == _norm_row(previous.cells):
                group.count("duplicate_boundary_row")
                first_on_page = False
                continue
            if fold_lines and _is_continuation_line(cells, previous) and not first_on_page:
                previous.cells = [f"{p} {c}".strip() if c.strip() else p for p, c in zip(previous.cells, cells)]
                group.count("folded_line")
                continue
            group.rows.append(StitchedRow(cells, table.page))
            first_on_page = False
    return [g for g in result if g.rows]


def render_markdown(table: StitchedTable, max_rows: int = 2000) -> str:
    """
    Renders a stitched table as markdown with a page column.

    Args:
        table: The table.
        max_rows: Row limit.

    Returns:
        Markdown text.
    """
    header = list(table.header) + ["page"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in table.rows[:max_rows]:
        cells = [c.replace("|", "/") for c in row.cells] + [str(row.page)]
        lines.append("| " + " | ".join(cells) + " |")
    if len(table.rows) > max_rows:
        lines.append(f"(+{len(table.rows) - max_rows} more rows)")
    return "\n".join(lines)


RowKey = Callable[[Dict[str, Any]], Tuple[Any, ...]]


def default_row_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Identity key of an extracted row (dates, amounts, description prefix)."""
    parts: List[Any] = []
    for key in sorted(row.keys()):
        if key == "source_page":
            continue
        value = row[key]
        parts.append((key, normalize_text(str(value))[:40] if isinstance(value, str) else str(value)))
    return tuple(parts)


def merge_chunks(chunks: Sequence[List[Dict[str, Any]]], carry: int = 3, key: RowKey = default_row_key) -> Tuple[List[Dict[str, Any]], int]:
    """
    Concatenates row lists of consecutive page chunks. Rows at the start of a
    chunk that repeat one of the last `carry` rows of the merged list (the
    context rows given to the model) are dropped.

    Args:
        chunks: Row lists in chunk order.
        carry: Number of context rows given per chunk.
        key: Row identity function.

    Returns:
        Tuple (merged rows, number of dropped duplicates).
    """
    merged: List[Dict[str, Any]] = []
    dropped = 0
    for chunk in chunks:
        tail = {key(r) for r in merged[-carry:]} if carry > 0 else set()
        leading = True
        for row in chunk:
            if leading and key(row) in tail:
                dropped += 1
                continue
            leading = False
            merged.append(row)
    return merged, dropped


def drop_artefact_rows(rows: List[Dict[str, Any]], text_fields: Sequence[str]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Removes carry-over / subtotal / balance rows an LLM copied into a table.

    Args:
        rows: Extracted rows.
        text_fields: Fields holding the row text.

    Returns:
        Tuple (clean rows, removed count).
    """
    clean: List[Dict[str, Any]] = []
    removed = 0
    for row in rows:
        text = " ".join(str(row.get(f) or "") for f in text_fields)
        if text.strip() and (CARRY_OVER_RE.search(text) or (BALANCE_ROW_RE.search(text) and len(text) < 40)):
            removed += 1
            continue
        clean.append(row)
    return clean, removed
