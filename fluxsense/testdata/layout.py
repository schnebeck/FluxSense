# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/layout.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Page model of the test-document generator: positioned text
                runs, rules and signature strokes in PDF points (A4, origin
                top-left). The same model renders the PDF (PyMuPDF base-14
                fonts, no extra dependency) and yields text lines and word
                boxes for tests that need no PDF at all.
------------------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pymupdf

from fluxsense.extract.base import PageText, Word
from fluxsense.pdflock import PYMUPDF_LOCK

A4: Tuple[float, float] = (595.0, 842.0)
_FONTS = {(False, False): "helv", (True, False): "hebo", (False, True): "cour", (True, True): "cobo"}


@dataclass
class Run:
    """One text run; x is the left edge (align 'left') or the right edge (align 'right')."""

    x: float
    y: float
    text: str
    size: float = 9.0
    bold: bool = False
    mono: bool = False
    align: str = "left"
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand: bool = False  # handwriting-like entry (Times italic, blue)

    @property
    def font(self) -> str:
        """PyMuPDF base-14 font name."""
        return "tiit" if self.hand else _FONTS[(self.bold, self.mono)]

    def width(self) -> float:
        """Rendered width in points."""
        return pymupdf.get_text_length(self.text, fontname=self.font, fontsize=self.size)

    def left(self) -> float:
        """Left edge in points."""
        return self.x - self.width() if self.align == "right" else self.x


@dataclass
class Page:
    """One page of a generated document."""

    width: float = A4[0]
    height: float = A4[1]
    runs: List[Run] = field(default_factory=list)
    rules: List[Tuple[float, float, float, float]] = field(default_factory=list)
    strokes: List[List[Tuple[float, float]]] = field(default_factory=list)

    def text(self, x: float, y: float, text: str, size: float = 9.0, bold: bool = False, mono: bool = False,
             align: str = "left", hand: bool = False) -> None:
        """Adds a text run (y = baseline); hand = a handwriting-like entry."""
        if text:
            self.runs.append(Run(x, y, text, size, bold, mono, align, (0.05, 0.12, 0.5) if hand else (0.0, 0.0, 0.0), hand))

    def rule(self, x0: float, y: float, x1: float) -> None:
        """Adds a horizontal rule."""
        self.rules.append((x0, y, x1, y))

    def lines(self) -> List[str]:
        """Text lines in reading order: runs on one baseline joined left to right."""
        rows: List[Tuple[float, List[Run]]] = []
        for run in sorted(self.runs, key=lambda r: (round(r.y, 1), r.left())):
            if rows and abs(rows[-1][0] - run.y) < 1.0:
                rows[-1][1].append(run)
            else:
                rows.append((run.y, [run]))
        return ["  ".join(r.text for r in sorted(group, key=lambda r: r.left())) for _, group in rows]

    def words(self) -> List[Word]:
        """Word boxes (top-left origin) as the native text layer reports them."""
        words: List[Word] = []
        for run in self.runs:
            cursor = run.left()
            space = pymupdf.get_text_length(" ", fontname=run.font, fontsize=run.size)
            for token in run.text.split(" "):
                if not token:
                    cursor += space
                    continue
                width = pymupdf.get_text_length(token, fontname=run.font, fontsize=run.size)
                words.append(Word(cursor, run.y - run.size * 0.8, cursor + width, run.y + run.size * 0.2, token))
                cursor += width + space
        return words

    def page_text(self, number: int) -> PageText:
        """This page as a native-text PageText (words and text), without rendering."""
        return PageText(page=number, width=self.width, height=self.height, sources={"native": "\n".join(self.lines())},
                        words=self.words(), words_source="native")


def signature(page: Page, x: float, y: float, seed: int) -> None:
    """Adds a scribbled signature stroke (synthetic polyline) at x/y."""
    points: List[Tuple[float, float]] = []
    for i in range(24):
        dx = i * 4.5
        dy = ((seed * 7 + i * 13) % 11 - 5) * 1.6
        points.append((x + dx, y + dy))
    page.strokes.append(points)


def render(pages: Sequence[Page], target: Path, title: str = "", subject: str = "") -> Path:
    """
    Renders pages into a PDF with a text layer.

    Args:
        pages: Page models.
        target: Output path.
        title: PDF metadata title.
        subject: PDF metadata subject.

    Returns:
        The target path.
    """
    with PYMUPDF_LOCK, pymupdf.open() as doc:
        for model in pages:
            page = doc.new_page(width=model.width, height=model.height)
            for x0, y0, x1, y1 in model.rules:
                page.draw_line((x0, y0), (x1, y1), color=(0.45, 0.45, 0.45), width=0.5)
            for points in model.strokes:
                page.draw_polyline(points, color=(0.05, 0.1, 0.45), width=1.1)
            for run in model.runs:
                page.insert_text((run.left(), run.y), run.text, fontname=run.font, fontsize=run.size, color=run.color)
        doc.set_metadata({"title": title, "subject": subject, "producer": "FluxSense test-document generator",
                          "creator": "fluxsense gen-testdocs"})
        doc.save(target, garbage=3, deflate=True)
    return target


def wrap(text: str, width: float, size: float = 9.0, bold: bool = False) -> List[str]:
    """Greedy word wrap to a width in points."""
    font = _FONTS[(bold, False)]
    lines: List[str] = []
    current: Optional[str] = None
    for word in text.split():
        candidate = word if current is None else f"{current} {word}"
        if current is not None and pymupdf.get_text_length(candidate, fontname=font, fontsize=size) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines
