# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/text/render.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Searchable PDF from FluxSense's text and geometry: an
                invisible text layer (render mode 3, Helvetica, scaled to
                each word box) is added on top of the untouched page images
                of scanned pages. Word boxes are used where the OCR has
                them, else the line boxes of the layout items are split in
                proportion to the characters. Corrected words stand at their
                image positions; a hyphenated pair keeps its visible split
                with the hyphen as U+00AD inside one /ActualText span
                carrying the joined word; form-entry readings stand at their
                entry regions; no text is placed inside signature regions.
                ocrmypdf's hOCR renderer is not used: the installed ocrmypdf
                is a command (its Python module is not importable from the
                engine's environment) and its renderer needs reportlab, so
                the layer is written with PyMuPDF.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fluxsense.extract.base import PageText, TextLayer
from fluxsense.log import get_logger
from fluxsense.pdflock import PYMUPDF_LOCK

logger = get_logger("text.render")

Box = Tuple[float, float, float, float]
FONT = "helv"
SOFT_HYPHEN = "­"


@dataclass
class Run:
    """One word of the text layer: its text, its box in page points and the joined word of a hyphenated pair."""

    text: str
    box: Box
    actual: Optional[str] = None  # /ActualText of a hyphenated pair (set on its first part)
    closes: bool = False  # the run ends an /ActualText span


def split_line(text: str, box: Box) -> List[Tuple[str, Box]]:
    """The words of a line box, each with a share of the width proportional to its characters (spaces count once)."""
    words = text.split()
    if not words:
        return []
    x0, y0, x1, y1 = box
    total = sum(len(w) for w in words) + len(words) - 1
    step = (x1 - x0) / max(1, total)
    out: List[Tuple[str, Box]] = []
    cursor = x0
    for word in words:
        out.append((word, (cursor, y0, cursor + len(word) * step, y1)))
        cursor += (len(word) + 1) * step
    return out


def _inside(box: Box, regions: Sequence[Box]) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in regions)


def _norm_box(bbox: Sequence[float], width: float, height: float) -> Box:
    return (bbox[0] / 1000 * width, bbox[1] / 1000 * height, bbox[2] / 1000 * width, bbox[3] / 1000 * height)


def page_words(page: PageText) -> List[Tuple[str, Box]]:
    """Word boxes of an OCR page: real word boxes, else line boxes (layout items / OCR lines) split proportionally."""
    words: List[Tuple[str, Box]] = []
    for word in page.words:
        if " " in word.text.strip():
            words += split_line(word.text, (word.x0, word.y0, word.x1, word.y1))
        elif word.text.strip():
            words.append((word.text.strip(), (word.x0, word.y0, word.x1, word.y1)))
    if words:
        return words
    from fluxsense.extract.fieldread import page_lines
    for text, box in page_lines(page):
        words += split_line(text, box)
    return words


def _lines(words: List[Tuple[str, Box]]) -> List[List[Tuple[str, Box]]]:
    """Words grouped into lines in reading order (same vertical centre within half a line height, left to right)."""
    rows: List[List[Tuple[str, Box]]] = []
    for text, box in sorted(words, key=lambda tb: ((tb[1][1] + tb[1][3]) / 2, tb[1][0])):
        centre, height = (box[1] + box[3]) / 2, box[3] - box[1]
        for row in rows[-3:]:
            top, bottom = min(b[1] for _, b in row), max(b[3] for _, b in row)
            if abs((top + bottom) / 2 - centre) <= 0.5 * max(bottom - top, height):
                row.append((text, box))
                break
        else:
            rows.append([(text, box)])
    return [sorted(row, key=lambda tb: tb[1][0]) for row in rows]


def _apply_corrections(words: List[Tuple[str, Box]], corrections: Sequence[Any]) -> List[Tuple[str, Box]]:
    """Corrected words at their positions: by word box, else by the token sequence of the correction."""
    words = list(words)
    for corr in corrections:
        before, after = str(corr.before), str(corr.after)
        if corr.bbox and " " not in before:
            for index, (text, box) in enumerate(words):
                if text.strip(".,;:!?()") == before and all(abs(a - b) < 1.5 for a, b in zip(box, corr.bbox)):
                    words[index] = (text.replace(before, after), box)
                    break
            continue
        old, new = before.split(), after.split()
        if len(old) != len(new) or not old:
            continue
        for start in range(len(words) - len(old) + 1):
            if [w for w, _ in words[start:start + len(old)]] == old:
                for k, token in enumerate(new):
                    words[start + k] = (token, words[start + k][1])
                break
    return words


def page_runs(page: PageText, corrections: Sequence[Any] = (), signatures: Sequence[Box] = (),
              entries: Sequence[Tuple[str, Box]] = ()) -> List[Run]:
    """
    The runs of one page's text layer in reading order.

    Args:
        page: OCR page (word boxes or layout items).
        corrections: TextCorrection records of the page.
        signatures: Signature regions in page points (no text there).
        entries: Form-entry readings (text, entry region in page points) - OCR words inside an entry region give way.

    Returns:
        Runs; a hyphenated pair at a line end is marked for one /ActualText span.
    """
    regions = list(signatures) + [box for _, box in entries]
    words = [(t, b) for t, b in _apply_corrections(page_words(page), corrections) if not _inside(b, regions)]
    words += [(text, box) for text, box in entries if text.strip()]
    lines = _lines(words)
    runs: List[Run] = []
    for index, line in enumerate(lines):
        for position, (text, box) in enumerate(line):
            last = position == len(line) - 1
            following = lines[index + 1][0][0] if last and index + 1 < len(lines) and lines[index + 1] else ""
            if last and re.search(r"\w-$", text) and re.match(r"[a-zäöüß]", following):
                runs.append(Run(text[:-1] + SOFT_HYPHEN, box, actual=text[:-1] + following))
                continue
            if runs and runs[-1].actual and not runs[-1].closes and position == 0:
                runs.append(Run(text, box, closes=True))
                continue
            runs.append(Run(text, box))
    return runs


def _pdf_string(text: str) -> bytes:
    raw = text.encode("cp1252", "replace")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def _utf16(text: str) -> bytes:
    return ("<FEFF" + text.encode("utf-16-be").hex().upper() + ">").encode("ascii")


def content_stream(runs: Sequence[Run], to_pdf: Any) -> bytes:
    """
    PDF operators of the invisible text: each run with render mode 3, its
    font size from the box height and a horizontal scale that fits the box
    width.
    """
    import pymupdf
    out: List[bytes] = []
    for run in runs:
        x0, y0, x1, y1 = run.box
        height = max(1.0, y1 - y0)
        size = max(2.0, min(72.0, 0.9 * height))
        width = pymupdf.get_text_length(run.text, fontname=FONT, fontsize=size) or 1.0
        scale = max(10.0, min(500.0, 100.0 * max(1.0, x1 - x0) / width))
        origin = pymupdf.Point(x0, y1 - 0.2 * height) * to_pdf
        if run.actual:
            out.append(b"/Span <</ActualText " + _utf16(run.actual) + b">> BDC")
        # a trailing space after every word: text extraction sees the word boundary however tight the boxes are
        out.append(b"BT 3 Tr /%s %.2f Tf %.2f Tz 1 0 0 1 %.2f %.2f Tm %s Tj ET" % (
            FONT.encode(), size, scale, origin.x, origin.y, _pdf_string(run.text + " ")))
        if run.closes:
            out.append(b"EMC")
    if runs and runs[-1].actual and not runs[-1].closes:
        out.append(b"EMC")
    return b"\n".join(out) + b"\n"


def render_text_layer(pdf_path: Path, layer: TextLayer, out_path: Path, result: Optional[Any] = None) -> Dict[str, int]:
    """
    Writes a searchable copy of a PDF: an invisible text layer from
    FluxSense's text and geometry on the scanned pages (pages with a text
    layer of their own - digital text or a scanner's OCR layer - stay as they
    are); the page images are not touched and the source file is never
    modified.

    Args:
        pdf_path: Source PDF.
        layer: Its text layer (after the analysis: corrections pending on the pages are applied).
        out_path: Target PDF.
        result: The AnalysisResult (text-level corrections, form-entry readings, signature regions), optional.

    Returns:
        Counts: pages with a text layer added, runs written.
    """
    import pymupdf
    by_page: Dict[int, List[Any]] = {p.page: list(p.corrections) for p in layer.pages}
    signatures: Dict[int, List[Box]] = {}
    entries: Dict[int, List[Tuple[str, Box]]] = {}
    if result is not None:
        for out in result.text_layer.pages:
            if out.corrections:
                by_page[out.page] = list(out.corrections)
        for doc in result.logical_documents:
            for sig in doc.signatures:
                page = layer.page(sig.page)
                if sig.bbox and page is not None:
                    signatures.setdefault(sig.page, []).append(_norm_box(sig.bbox, page.width, page.height))
            for fld in doc.form_fields:
                page = layer.page(fld.page)
                text = fld.entry if fld.entry and fld.entry.upper() != "EMPTY" and fld.found != "acroform" else None
                if text and fld.entry_bbox and page is not None and fld.script != "typed":
                    entries.setdefault(fld.page, []).append((" ".join(text.split()), _norm_box(fld.entry_bbox, page.width,
                                                                                              page.height)))
    stats = {"pages": 0, "runs": 0}
    with PYMUPDF_LOCK, pymupdf.open(pdf_path) as doc:
        for page in layer.pages:
            if page.sources.get("native", "").strip() or page.page > doc.page_count:
                continue  # digital text or a scanner's OCR layer: already searchable
            runs = page_runs(page, by_page.get(page.page, []), signatures.get(page.page, []), entries.get(page.page, []))
            if not runs:
                continue
            pdf_page = doc.load_page(page.page - 1)
            pdf_page.wrap_contents()
            pdf_page.insert_font(fontname=FONT)
            xref = doc.get_new_xref()
            doc.update_object(xref, "<<>>")
            doc.update_stream(xref, content_stream(runs, ~pdf_page.transformation_matrix))
            contents = pdf_page.get_contents()
            doc.xref_set_key(pdf_page.xref, "Contents", "[" + " ".join(f"{x} 0 R" for x in contents + [xref]) + "]")
            stats["pages"] += 1
            stats["runs"] += len(runs)
        doc.save(out_path)  # no re-compression: the page images keep their encoded streams
    logger.info(f"text layer: {stats['runs']} words on {stats['pages']} page(s) -> {out_path}")
    return stats
