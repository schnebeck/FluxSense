# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/fieldread.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Form fields read by their printed label (field contract
                §12), routed by the page's handwriting level: a page with
                only a signature reads its date / signature lines alone; a
                printed form with handwritten entries is read form-first
                (the vision model locates the label / entry pairs on the
                page, the lexicon only gives the kind); other scanned form
                pages find the printed labels in the layout ('Ort, Datum',
                'Einzug am', 'Name des ...', '..., den') with a lexicon that
                also gives the field kind. Each entry region is cropped from
                the scan's native resolution and read by the vision model
                with an open question plus a type hint - never with the
                full-page value, so the reading is not anchored - and the
                reader says whether the entry is typed, handwritten or
                empty. Native AcroForm fields of digital PDFs are read from
                their widgets without vision. Dates are normalised to ISO
                with their candidate readings ('1|10|26': bars may be
                separators or ones; typed dates have one reading). The
                readings become a FORM FIELDS block of the page text and the
                'fieldread' text source used by grounding.
------------------------------------------------------------------------------
"""

import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

from fluxsense.config import Settings
from fluxsense.extract.base import PageText
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, encode_data_uri, fit_image
from fluxsense.log import get_logger
from fluxsense.pdflock import PYMUPDF_LOCK

logger = get_logger("extract.fieldread")

SOURCE_FIELDREAD = "fieldread"
DATE, PLACE_DATE, NAME, PERSONS, ADDRESS, LOCATION, PHONE, TEXT, CHECKBOX = (
    "date", "place_date", "name", "persons", "address", "location", "phone", "text", "checkbox")
KINDS: Tuple[str, ...] = (DATE, PLACE_DATE, NAME, PERSONS, ADDRESS, LOCATION, PHONE, TEXT, CHECKBOX)
EMPTY = "EMPTY"
TYPED, HANDWRITTEN, BLANK = "typed", "handwritten", "empty"
SCRIPTS: Tuple[str, ...] = (TYPED, HANDWRITTEN, BLANK)
# How a field was found: a lexicon label in the layout, the vision model's form locator, or a digital AcroForm widget.
FOUND_LABEL, FOUND_LOCATOR, FOUND_ACROFORM = "label", "locator", "acroform"
MARKERS: Dict[str, str] = {TYPED: "[typed]", HANDWRITTEN: "[handwritten]", FOUND_ACROFORM: "[form field]"}
MAX_FIELDS: int = 24
MIN_LABELS: int = 3


@dataclass(frozen=True)
class LabelRule:
    """A lexicon entry: label pattern, field kind and (optional) schema target."""

    pattern: "re.Pattern[str]"
    kind: str
    target: Optional[str] = None
    below: bool = False


def _rule(pattern: str, kind: str, target: Optional[str] = None, below: bool = False) -> LabelRule:
    return LabelRule(re.compile(pattern, re.IGNORECASE), kind, target, below)


# Label lexicon (DE + EN), most specific first. Targets are applied after the extraction (see map_fields).
LEXICON: Tuple[LabelRule, ...] = (
    _rule(r"^\W*name\s+und\s+anschrift\s+des\s+wohnungsgebers", ADDRESS, "sender", below=True),
    _rule(r"^\W*(?:name|vor-?\s*und\s+zuname|name,\s*vorname)\s+des\s+wohnungsgebers", NAME, "sender.name"),
    _rule(r"^\W*anschrift\s+des\s+wohnungsgebers", ADDRESS, "sender.address"),
    _rule(r"^\W*(?:name\s+des\s+|vor-?\s*und\s+zuname\s+des\s+)?(?:antragsteller(?:in|s)?|aussteller(?:in|s)?)\b", NAME,
          "sender.name"),
    _rule(r"^\W*(?:\[.?\]\s*)?einzug\s+am", DATE, "event.MOVE_IN"),
    _rule(r"^\W*(?:\[.?\]\s*)?auszug\s+am", DATE, "event.MOVE_OUT"),
    _rule(r"^\W*anschrift\s+der\s+wohnung|^\W*(?:postleitzahl|plz),?\s*ort,?\s*stra(?:ß|ss)e", ADDRESS, "subject_address"),
    _rule(r"^\W*(?:geschoss|stockwerk|etage|lage\b|geschoss\s*/\s*lage)", LOCATION, "subject_address.location_detail"),
    _rule(r"(?:folgende|nachstehende)\s+person|für\s+folgende\s+person|person\(en\)", PERSONS, "certified_persons", below=True),
    _rule(r"^\W*ort\s*,?\s*(?:und\s+)?datum|^\W*place\s*,?\s*date", PLACE_DATE, "doc_date"),
    _rule(r"^[\wÄÖÜäöüß .-]{2,30},\s*den\b", PLACE_DATE, "doc_date"),
    _rule(r"^\W*datum\s*/\s*unterschrift|^\W*(?:geburts)?datum\b|^\W*date\b", DATE),
    _rule(r"^\W*(?:telefon|tel\.|mobil|handy|phone)", PHONE),
    _rule(r"^\W*(?:name|vorname|nachname|familienname|vor-?\s*und\s+zuname|name,\s*vorname)\b", NAME),
    _rule(r"^\W*(?:anschrift|adresse|wohnort|stra(?:ß|ss)e|hausnummer|address)\b", ADDRESS),
)
_BELOW_HINT_RE = re.compile(r"(lauten|lautet|folgende|:\s*$)", re.IGNORECASE)

TYPE_HINTS: Dict[str, str] = {
    DATE: "The entry is a handwritten DATE in German notation day/month/year (separators may be '.', '/' or '|'; the year "
          "may have 2 or 4 digits). Transcribe the separators exactly as they appear ('.', '/', '|' or '-'). Return it as "
          "written AND as ISO date, e.g. '7/8/26 -> 2026-08-07'.",
    PLACE_DATE: "The entry is a PLACE and a handwritten DATE (day/month/year; separators may be '.', '/' or '|'; 2- or "
                "4-digit year). Transcribe the separators exactly as they appear. Return it as written AND the date as "
                "ISO, e.g. 'Musterstadt, 7/8/26 -> 2026-08-07'.",
    NAME: "The entry is the NAME of a person or organisation.",
    PERSONS: "The entry lists one or more PERSONS (first and last name each), often on numbered lines; return one person "
             "per line without the numbers, skip empty lines.",
    ADDRESS: "The entry is a postal ADDRESS (street, house number, postcode, town).",
    LOCATION: "The entry is a LOCATION inside a building (floor and position, e.g. 'EG links', '2. OG rechts').",
    PHONE: "The entry is a PHONE number.",
    TEXT: "",
}
PROMPT = ('This is a crop of a German form. The printed label of the field is: "{label}". Read only the handwritten (or '
          'typed) ENTRY that belongs to this label - not the printed labels. If the field is empty answer EMPTY. {hint} '
          'If the handwriting allows several readings, list every plausible ISO date in "candidates". Say how the entry '
          'is written: "typed" (printed, typewriter or computer), "handwritten" or "empty". '
          'Answer as JSON: {{"entry": "<the entry as written, or EMPTY>", "iso": "<ISO date or null>", '
          '"candidates": ["<ISO date>", ...], "script": "typed" | "handwritten" | "empty"}}')
CHECKBOX_SCHEMA: Dict[str, Any] = {"type": "object", "additionalProperties": False,
                                   "properties": {"state": {"type": "string", "enum": ["yes", "no", "no checkbox"]}},
                                   "required": ["state"]}
SCHEMA: Dict[str, Any] = {"type": "object", "additionalProperties": False,
                          "properties": {"entry": {"type": "string", "maxLength": 200},
                                         "iso": {"type": ["string", "null"], "maxLength": 10},
                                         "candidates": {"type": "array", "items": {"type": "string", "maxLength": 10},
                                                        "maxItems": 4},
                                         "script": {"type": "string", "enum": list(SCRIPTS)}},
                          "required": ["entry"]}
LOCATE_PROMPT = ('This page is a printed form that has been filled in. List its fields in reading order, at most {limit}: '
                 'the printed LABEL of each field (verbatim) and the box of its ENTRY (where the value is written or '
                 'typed), both as [x0, y0, x1, y1] relative to the page in 0-1000. Say how each entry is written: "typed", '
                 '"handwritten" or "empty"; for a checkbox say whether it is ticked. Do not transcribe the entries. A '
                 'signature is no field. Answer as JSON: {{"fields": [{{"label": "...", "label_box": [x0, y0, x1, y1], '
                 '"entry_box": [x0, y0, x1, y1], "kind": "date | place_date | name | persons | address | location | phone '
                 '| text | checkbox", "script": "typed | handwritten | empty", "ticked": true | false | null}}]}}')
_BOX: Dict[str, Any] = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
LOCATE_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["fields"],
    "properties": {"fields": {"type": "array", "maxItems": MAX_FIELDS, "items": {
        "type": "object", "additionalProperties": False, "required": ["label", "entry_box"],
        "properties": {"label": {"type": "string", "maxLength": 90}, "label_box": _BOX, "entry_box": _BOX,
                       "kind": {"type": "string", "enum": list(KINDS)}, "script": {"type": "string", "enum": list(SCRIPTS)},
                       "ticked": {"type": ["boolean", "null"]}}}}}}


@dataclass
class FormField:
    """A printed label, its entry region and the reading."""

    page: int
    label: str
    kind: str
    target: Optional[str]
    bbox: Tuple[float, float, float, float]  # label box in page points
    region: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    entry: Optional[str] = None
    value: Optional[str] = None  # normalised value (ISO date for date kinds)
    candidates: List[str] = field(default_factory=list)
    ticked: Optional[bool] = None  # checkbox before the label: True ticked, False unticked, None no box / unknown
    inside: bool = False  # the layout item already contains the entry after the label ('Musterstadt, den 71806')
    script: Optional[str] = None  # typed | handwritten | empty as the reader saw it (None: not reported)
    found: str = FOUND_LABEL  # label (lexicon label in the layout) | locator (vision form locator) | acroform (widget)

    @property
    def empty(self) -> bool:
        """True if the field was read as empty."""
        return self.entry is not None and self.entry.strip().upper() in ("", EMPTY)

    @property
    def handwritten(self) -> bool:
        """True unless the entry is known to be typed or a digital form value (an unreported script counts as handwriting)."""
        return self.found != FOUND_ACROFORM and self.script != TYPED

    def marker(self) -> str:
        """'[typed] ', '[handwritten] ', '[form field] ' or '' (script not reported) before the label of a filled entry."""
        key = FOUND_ACROFORM if self.found == FOUND_ACROFORM else self.script or ""
        return f"{MARKERS[key]} " if key in MARKERS else ""

    def line(self) -> str:
        """The FORM FIELDS line of this field."""
        if self.entry is None:
            return f"{self.label}: (not read)"
        if self.empty:
            return f"{self.label}: (empty)"
        head = f"{self.marker()}{self.label}"
        if self.value and self.value != self.entry:
            alt = f" (readings: {', '.join(self.candidates)})" if len(self.candidates) > 1 else ""
            return f"{head}: {self.entry} -> {self.value}{alt}"
        if self.kind == NAME and not self.target:
            return f"{head}: {self.entry} [a name field of the form - not by itself the sender or recipient]"
        return f"{head}: {self.entry}"


def classify_label(text: str) -> Optional[LabelRule]:
    """The lexicon rule of a printed label, None if it is no known label."""
    clean = " ".join(text.split())
    if not clean or len(clean) > 90:
        return None
    return next((rule for rule in LEXICON if rule.pattern.search(clean)), None)


Box = Tuple[float, float, float, float]


def page_lines(page: PageText) -> List[Tuple[str, Box]]:
    """
    Text lines of a page with their boxes in page points: the lines of the
    layout items (OvisOCR3 / Docling, bbox 0-1000; a multi-line item is split
    into equal bands), else the native / OCR word boxes grouped into lines.
    """
    width, height = page.width or 595.0, page.height or 842.0
    candidates: List[Tuple[str, Box]] = []
    for item in page.items or []:
        if item.category in ("image", "table") or not item.content.strip():
            continue
        x0, y0, x1, y1 = item.bbox
        for n, line in enumerate(item.content.splitlines() or [item.content]):
            lines = max(1, len(item.content.splitlines()))
            step = (y1 - y0) / lines
            candidates.append((line, (x0 / 1000 * width, (y0 + n * step) / 1000 * height, x1 / 1000 * width,
                                      (y0 + (n + 1) * step) / 1000 * height)))
    if not candidates and page.words:
        from fluxsense.layout.din5008 import group_lines
        candidates = [(line.text, (line.x0, line.y0, line.x1, line.y1)) for line in group_lines(page.words)]
    return candidates


def find_labels(page: PageText) -> List[FormField]:
    """
    Printed field labels of a page: layout items (OvisOCR3 / Docling, bbox
    0-1000) or native text lines, matched against the lexicon. A label item
    that already carries text after the label keeps only the label part.
    """
    candidates = page_lines(page)
    fields: List[FormField] = []
    boxes = [(normalize_checkbox(text).strip(), box) for text, box in candidates if _BOX_ONLY_RE.match(normalize_checkbox(text))]
    for text, box in candidates:
        text = normalize_checkbox(text)
        rule = classify_label(text)
        if rule is None:
            continue
        label = text.strip()
        box_match = _CHECKBOX_RE.match(label)
        ticked = None
        if box_match:
            ticked = box_match.group(1) in TICKED
            label = label[box_match.end():].strip()
        else:  # a box glyph as an item of its own, just left of the label on the same line
            near = [b for b, bb in boxes if bb[2] <= box[0] + 2 and box[0] - bb[2] < 40
                    and abs((bb[1] + bb[3]) / 2 - (box[1] + box[3]) / 2) < max(6.0, box[3] - box[1])]
            if near:
                ticked = near[-1] in TICKED
        inside = False
        match = re.match(r"^(.*?(?::|\bam|\bden|\bvom|\bbis))\s+\S", label)
        if match and rule.kind in (DATE, PLACE_DATE, NAME, ADDRESS, LOCATION, PHONE) and len(match.group(1)) < len(label):
            label, inside = match.group(1).strip(), True  # the entry is inside the item, after the label
        if any(f.label == label and abs((f.bbox[1] + f.bbox[3]) / 2 - (box[1] + box[3]) / 2) < max(6.0, box[3] - box[1])
               for f in fields):
            continue  # the same label repeated on one line (e.g. several 'Tel.:' cells): read once
        fields.append(FormField(page.page, label, rule.kind, rule.target, box, ticked=ticked, inside=inside))
    return fields


_BOX_GLYPHS_UNTICKED = ("□", "☐", "❑", "❒", "◻", "▢")
_BOX_GLYPHS_TICKED = ("☒", "☑", "✓", "✔", "✗", "✘", "■", "▣")


def normalize_checkbox(text: str) -> str:
    """Checkbox glyphs at the start of a label become '[ ]' / '[x]' ('□ Auszug am' -> '[ ] Auszug am')."""
    stripped = text.lstrip()
    for glyph in _BOX_GLYPHS_UNTICKED:
        if stripped.startswith(glyph):
            return "[ ] " + stripped[len(glyph):].lstrip()
    for glyph in _BOX_GLYPHS_TICKED:
        if stripped.startswith(glyph):
            return "[x] " + stripped[len(glyph):].lstrip()
    return text


_BOX_ONLY_RE = re.compile(r"^\s*\[[ xX]\]\s*$")


TICKED = ("[x]", "[X]", "(x)", "(X)", "x", "X")
_CHECKBOX_RE = re.compile(r"^\s*(\[x\]|\[X\]|\[ \]|\(x\)|\(X\)|\( \)|[xX](?=\s+[A-ZÄÖÜ]))\s*")
INLINE_END_RE = re.compile(r"(\bam|\bden|\bvom|\bbis|:)\s*$", re.IGNORECASE)


def entry_region(fld: FormField, width: float, height: float, below: bool, inline: bool = False,
                 others: Sequence[FormField] = ()) -> Tuple[float, float, float, float]:
    """
    The entry region of a label in page points, clipped at the neighbouring
    labels:
    - inline (label ending in 'am', 'den', 'vom', 'bis' or ':' with the entry
      to its right): the label's own text band (y0 - 0.6h .. y1 + 0.6h), from
      the label's end to the next label on that line (else half a page
      width) - never the row above;
    - below (header label '... lauten:'): the lines under the label down to
      the next label below;
    - otherwise (label printed under its entry line): around and above the
      label, up to the next label above.
    """
    x0, y0, x1, y1 = fld.bbox
    h = max(6.0, y1 - y0)

    def overlaps_x(o: FormField, left: float, right: float) -> bool:
        return o.bbox[0] < right and o.bbox[2] > left

    if inline:
        right = min(width, x1 + 0.5 * width)
        same_line = [o.bbox[0] for o in others if o is not fld and o.bbox[0] > x1 and abs((o.bbox[1] + o.bbox[3]) / 2 - (y0 + y1) / 2) < h]
        if same_line:
            right = min(right, min(same_line) - 2)
        return (x1, max(0.0, y0 - 0.6 * h), max(x1 + 10, right), min(height, y1 + 0.6 * h))
    if below:
        left, right = max(0.0, x0 - 20), min(width, max(x1 + 0.45 * width, 0.9 * width))
        bottom = min(height, y1 + 6.5 * h)
        under = [o.bbox[1] for o in others if o is not fld and o.bbox[1] > y1 and overlaps_x(o, left, right)]
        if under:
            bottom = min(bottom, min(under) - 1)
        return (left, y0, right, max(y1 + h, bottom))
    left, right = max(0.0, x0 - 20), min(width, max(x1 + 0.45 * width, x0 + 0.55 * width))
    top = max(0.0, y0 - 3.2 * h)
    above = [o.bbox[3] for o in others if o is not fld and o.bbox[3] <= y0 and overlaps_x(o, left, right)]
    if above:
        top = max(top, max(above) + 1)
    return (left, top, right, min(height, y1 + 1.2 * h))


def ink_share(image: Image.Image, page_width: float, page_height: float, box: Tuple[float, float, float, float]) -> float:
    """Share of dark pixels of a page-point box on a low-resolution page render."""
    sx, sy = image.width / page_width, image.height / page_height
    crop = image.crop((int(box[0] * sx), int(box[1] * sy), max(int(box[0] * sx) + 1, int(box[2] * sx)),
                       max(int(box[1] * sy) + 1, int(box[3] * sy)))).convert("L")
    histogram = crop.histogram()
    return sum(histogram[:160]) / max(1, crop.width * crop.height)


def dark_rows(image: Image.Image) -> List[float]:
    """Share of dark pixels (< 160) of every pixel row of an image."""
    binary = image.convert("L").point(lambda v: 255 if v < 160 else 0)
    column = binary.resize((1, binary.height), Image.Resampling.BOX)
    return [v / 255 for v in column.tobytes()]


def stroke_share(image: Image.Image, page_width: float, page_height: float, box: Tuple[float, float, float, float],
                 rule_share: float = 0.35) -> float:
    """
    Share of dark pixels of a page-point box without printed rules: pixel
    rows that are dark over more than rule_share of the width (an entry or
    signature line, a box border) do not count - pen strokes and letters do.
    """
    sx, sy = image.width / page_width, image.height / page_height
    crop = image.crop((int(box[0] * sx), int(box[1] * sy), max(int(box[0] * sx) + 1, int(box[2] * sx)),
                       max(int(box[1] * sy) + 1, int(box[3] * sy))))
    rows = dark_rows(crop)
    return sum(r for r in rows if r <= rule_share) / max(1, len(rows))


def form_like(page: PageText, labels: Sequence[FormField], handwriting: bool) -> bool:
    """
    Trigger: a scanned page (no native text layer: a digital page prints its
    entries as text) on which the audit saw handwriting next to labels, at
    least MIN_LABELS lexicon labels sit, or a date / signature line sits
    ('..., den', 'Ort, Datum', 'Datum/Unterschrift', 'Datum:' - then only
    those fields are read).
    """
    if page.sources.get("native", "").strip() and not page.native_ocr:
        return False
    return bool(labels) and (handwriting or len(labels) >= MIN_LABELS or any(l.kind in (DATE, PLACE_DATE) for l in labels))


_DATE_PARTS_RE = re.compile(r"(?<!\d)(\d{1,2})\s*([./|])\s*(\d{1,2})\s*([./|])\s*(\d{2}|\d{4})(?!\d)")


def _expand_year(year: str) -> int:
    value = int(year)
    return value + (2000 if value < 70 else 1900) if len(year) == 2 else value


def date_candidates(entry: str, model_iso: Optional[str] = None, today: Optional[date] = None,
                    handwritten: bool = True) -> List[str]:
    """
    ISO readings of a handwritten date: the model's ISO value, the entry with
    '.', '/' or '|' as separators, every single '1' / '|' / '/' swap
    ('11/10/26' may be '1|10|26'), and the readings where a bar next to a '1'
    is part of the digit ('1|10|26' -> 01.10.2026 or 11.10.2026); invalid and
    future dates are dropped. A typed date (handwritten False) has only its
    literal reading.
    """
    today = today or date.today()
    found: List[str] = []

    def add(day: int, month: int, year: int) -> None:
        try:
            value = date(year, month, day)
        except ValueError:
            return
        if value <= today + timedelta(days=31) and value.isoformat() not in found:
            found.append(value.isoformat())

    text = entry.replace(" ", "")
    iso = re.match(r"(\d{4})-(\d{2})-(\d{2})$", (model_iso or "")[:10])
    if not handwritten:
        literal = _DATE_PARTS_RE.search(text)
        if literal:
            add(int(literal.group(1)), int(literal.group(3)), _expand_year(literal.group(5)))
        elif iso:  # a typed date in words ('7. August 2026'): the reader's ISO value
            add(int(iso.group(3)), int(iso.group(2)), int(iso.group(1)))
        return found
    if iso:
        add(int(iso.group(3)), int(iso.group(2)), int(iso.group(1)))
    # '1', '|' and '/' are confused in handwriting: every single swap is a reading (repeated separators collapse)
    def swappable(i: int, ch: str) -> bool:
        if ch == "|":
            return True
        near = text[max(0, i - 1):i] + text[i + 1:i + 2]
        return ch == "1" and any(c in "1|/." for c in near)

    handwritten_seps = "|" in text or "/" in text
    variants = [text] + ([text[:i] + alt + text[i + 1:] for i, ch in enumerate(text) if swappable(i, ch)
                          for alt in "1|/" if alt != ch] if handwritten_seps else [])
    for variant in variants[1:]:
        swapped = _DATE_PARTS_RE.search(re.sub(r"[./|]{2,}", ".", variant))
        if swapped:
            add(int(swapped.group(1)), int(swapped.group(3)), _expand_year(swapped.group(5)))
    match = _DATE_PARTS_RE.search(text)
    if match:
        day, sep1, month, sep2, year = match.groups()
        add(int(day), int(month), _expand_year(year))
        if sep1 == sep2 == "." and not handwritten_seps:
            # a handwritten '1|' is often transcribed as '11': a day / month '11' or ending in '1' may be the digit
            # before a bar separator
            if len(day) == 2 and (day == "11" or day.endswith("1")):
                add(int(day[0]), int(month), _expand_year(year))
            if len(month) == 2 and (month == "11" or month.endswith("1")):
                add(int(day), int(month[0]), _expand_year(year))
        if sep1 == "|" and len(day) == 1:  # '1|10' may be '11.10' or '1.10': the bar is a stroke of a digit 1
            add(int(day + "1"), int(month), _expand_year(year))
        if sep2 == "|" and len(month) == 1:
            add(int(day), int(month + "1"), _expand_year(year))
    if "|" in text and not match:
        bars = text.replace("|", ".")
        again = _DATE_PARTS_RE.search(bars)
        if again:
            add(int(again.group(1)), int(again.group(3)), _expand_year(again.group(5)))
    return found


class FieldReader:
    """Reads the form fields of the given pages by their printed labels (one crop per field, in parallel)."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: Vision-language backend.
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings

    def _crop(self, pdf_path: Path, fld: FormField, dpi: int) -> Image.Image:
        import pymupdf
        with PYMUPDF_LOCK, pymupdf.open(pdf_path) as doc:
            page = doc.load_page(fld.page - 1)
            x0, y0, x1, y1 = fld.region
            clip = pymupdf.Rect(page.rect.x0 + x0, page.rect.y0 + y0, page.rect.x0 + x1, page.rect.y0 + y1)
            png = page.get_pixmap(dpi=dpi, clip=clip).tobytes("png")
        return fit_image(Image.open(io.BytesIO(png)).convert("RGB"), 1280)

    @staticmethod
    def _ink_at_entry(fld: FormField, overview: Image.Image, width: float, height: float, labels: Sequence[FormField]) -> bool:
        """Ink to the right of a label (inline entry) or above it (entry line over the label), the label itself excluded."""
        right = entry_region(fld, width, height, False, True, labels)
        up = entry_region(fld, width, height, False, False, labels)
        above = (up[0], up[1], up[2], max(up[1] + 1, fld.bbox[1] - 1))
        return ink_share(overview, width, height, (right[0] + 2, right[1], right[2], right[3])) > 0.003 or \
            ink_share(overview, width, height, above) > 0.003

    def checkbox_state(self, pdf_path: Path, fld: FormField, dpi: int) -> Optional[bool]:
        """Asks the vision model whether the checkbox left of a label is ticked (True / False / None = no checkbox)."""
        x0, y0, x1, y1 = fld.bbox
        h = max(6.0, y1 - y0)
        probe = FormField(fld.page, fld.label, fld.kind, None, fld.bbox, region=(max(0.0, x0 - 1.8 * h), max(0.0, y0 - 0.4 * h),
                                                                                    x0 + 0.3 * h, y1 + 0.4 * h))
        image = self._crop(pdf_path, probe, dpi)
        part = ImagePart(label="CHECKBOX", data_uri=encode_data_uri(image, "png"), width=image.width, height=image.height)
        answer = self.backend.generate_json(LLMRequest(
            prompt=f'This is a crop of a German form left of the printed label "{fld.label}". Is the checkbox ticked? '
                   'Answer as JSON: {"state": "yes" | "no" | "no checkbox"}',
            label=f"Fieldread checkbox p{fld.page} {fld.label[:30]}", model=self.settings.models.vision, images=[part],
            schema=CHECKBOX_SCHEMA, schema_name="checkbox", max_tokens=30))
        state = str((answer or {}).get("state") or "").lower() if isinstance(answer, dict) else ""
        return True if state == "yes" else False if state == "no" else None

    def read(self, pdf_path: Path, fld: FormField, dpi: int) -> FormField:
        """Reads one field (typed prompt, crop only) and normalises the entry."""
        image = self._crop(pdf_path, fld, dpi)
        part = ImagePart(label=f"FIELD {fld.label[:40]}", data_uri=encode_data_uri(image, "png"), width=image.width,
                         height=image.height)
        request = LLMRequest(prompt=PROMPT.format(label=fld.label, hint=TYPE_HINTS.get(fld.kind, "")),
                             label=f"Fieldread p{fld.page} {fld.label[:30]}", model=self.settings.models.vision,
                             images=[part], schema=SCHEMA, schema_name="form_field", max_tokens=170)
        answer = self.backend.generate_json(request)
        entry = str((answer or {}).get("entry") or "").strip() if isinstance(answer, dict) else ""
        iso = (answer or {}).get("iso") if isinstance(answer, dict) else None
        script = str((answer or {}).get("script") or "").lower() if isinstance(answer, dict) else ""
        fld.script = script if script in SCRIPTS else fld.script if fld.found == FOUND_LOCATOR else None
        fld.entry = entry or EMPTY
        if fld.empty and fld.inside:
            fld.entry, fld.script = None, None  # the layout shows something after the label: an EMPTY answer is no reading
            return fld
        if fld.empty:
            fld.script = BLANK
            return fld
        if fld.script == BLANK:
            fld.script = None  # an entry was read: 'empty' contradicts it
        normalise(fld, entry, iso if isinstance(iso, str) else None,
                  [c for c in (answer.get("candidates") or []) if isinstance(c, str)] if isinstance(answer, dict) else [])
        return fld

    def locate(self, pdf_path: Path, page: PageText) -> Optional[List[FormField]]:
        """
        Form-first reading (handwriting level 'entries'): the vision model
        lists the label / entry pairs of the page with their boxes and how
        each entry is written. The lexicon only refines the kind and gives the
        schema target of a known label. None if the call failed or found
        nothing (the caller falls back to the lexicon labels).
        """
        from fluxsense.llm.images import page_images
        llm = self.settings.llm
        limit = min(MAX_FIELDS, self.settings.pipeline.fieldread_max_fields)
        images = page_images(pdf_path, [page.page], [f"PAGE {page.page}"], llm.max_image_px, llm.image_format,
                             dpi=llm.render_dpi)
        answer = self.backend.generate_json(LLMRequest(
            prompt=LOCATE_PROMPT.format(limit=limit), label=f"Fieldread locate p{page.page}", model=self.settings.models.vision,
            images=images, schema=LOCATE_SCHEMA, schema_name="form_locate", max_tokens=90 * limit + 60))
        entries = answer.get("fields") if isinstance(answer, dict) else None
        if not isinstance(entries, list):
            return None
        width, height = page.width or 595.0, page.height or 842.0
        fields: List[FormField] = []
        for entry in entries[:limit]:
            if not isinstance(entry, dict) or not str(entry.get("label") or "").strip():
                continue
            region = _points(entry.get("entry_box"), width, height)
            if region is None:
                continue
            label = normalize_checkbox(" ".join(str(entry["label"]).split()))
            ticked = entry.get("ticked") if isinstance(entry.get("ticked"), bool) else None
            box_match = _CHECKBOX_RE.match(label)
            if box_match:
                ticked = box_match.group(1) in TICKED if ticked is None else ticked
                label = label[box_match.end():].strip()
            rule = classify_label(label)
            kind = rule.kind if rule else (str(entry.get("kind")) if entry.get("kind") in KINDS else TEXT)
            script = str(entry.get("script") or "").lower()
            x0, y0, x1, y1 = region
            pad = max(3.0, 0.25 * (y1 - y0))
            fld = FormField(page.page, label, kind, rule.target if rule else None,
                            _points(entry.get("label_box"), width, height) or region,
                            region=(max(0.0, x0 - 6), max(0.0, y0 - pad), min(width, x1 + 6), min(height, y1 + pad)),
                            ticked=ticked, script=script if script in SCRIPTS else None, found=FOUND_LOCATOR)
            fields.append(fld)
        return fields or None

    def _located_jobs(self, pdf_path: Path, located: List[FormField], dpi: int, overview: Image.Image, width: float,
                      height: float) -> Tuple[List[FormField], List[Tuple[FormField, int]]]:
        """Settled fields (checkboxes, entries the locator and the page show empty) and crop jobs of located fields."""
        settled: List[FormField] = []
        jobs: List[Tuple[FormField, int]] = []
        for fld in located:
            if fld.kind == CHECKBOX and not (fld.target and fld.target.startswith("event.")):
                if fld.ticked is not None:
                    fld.entry = fld.value = "[x]" if fld.ticked else "[ ]"
                    settled.append(fld)
                continue
            if fld.ticked is None and fld.target and fld.target.startswith("event."):
                fld.ticked = self.checkbox_state(pdf_path, fld, dpi)
            if fld.ticked is False:
                continue  # an unticked option: its field is not filled
            if fld.script == BLANK and stroke_share(overview, width, height, fld.region) < 0.003:
                fld.entry = EMPTY
                settled.append(fld)
                continue
            jobs.append((fld, dpi))
        return settled, jobs

    def run(self, pdf_path: Path, pages: Sequence[PageText], levels: Optional[Mapping[int, str]] = None) -> List[FormField]:
        """
        Reads the form fields of the pages, routed by their handwriting level
        (field contract §12): AcroForm widgets are read directly; 'signature'
        reads only date / signature lines; 'entries' locates the label / entry
        pairs with the vision model (lexicon labels as fallback); 'full' is
        left to the line-tiled reading (extract/handread); 'none' or an
        unknown level uses the lexicon trigger (at least MIN_LABELS labels, or
        date / signature lines with ink).

        Args:
            pdf_path: PDF file.
            pages: Pages of the logical document (with layout items or word boxes).
            levels: Page -> handwriting level (none | signature | entries | full); default: the pages' own level.

        Returns:
            The read fields in page order (empty when no page has form fields).
        """
        from fluxsense.extract.ovis import NATIVE_MAX_DPI, native_dpi
        from fluxsense.llm.images import render_page
        from fluxsense.stages.signatures import ENTRIES, FULL, SIGNATURE
        done: List[FormField] = acroform_fields(pdf_path, pages)
        widget_pages = {f.page for f in done}
        jobs: List[Tuple[FormField, int]] = []
        for page in pages:
            level = (levels or {}).get(page.page) or page.handwriting
            if page.page in widget_pages or level == FULL:
                continue  # digital form values are read from their widgets; a handwritten page is read as line tiles
            labels = find_labels(page)
            hand = level == ENTRIES
            if level != ENTRIES and not form_like(page, labels, False):
                continue
            if level == ENTRIES and not page.scanned():
                continue  # a digital page prints its entries as text
            dpi = min(NATIVE_MAX_DPI, max(300, native_dpi(pdf_path, page.page - 1) or 300))
            overview = render_page(pdf_path, page.page - 1, dpi=100)
            width, height = page.width or 595.0, page.height or 842.0
            if level == ENTRIES:
                located = self.locate(pdf_path, page)
                if located:
                    settled, located_jobs = self._located_jobs(pdf_path, located, dpi, overview, width, height)
                    done += settled
                    jobs += located_jobs
                    continue
                if not labels:
                    continue
            # a page with only a signature reads its date / signature lines alone (never the printed values)
            date_only = level == SIGNATURE or (len(labels) < MIN_LABELS and not hand)
            if date_only:
                labels = [label for label in labels if label.kind in (DATE, PLACE_DATE)]  # date / signature lines only
            for fld in labels[:self.settings.pipeline.fieldread_max_fields]:
                if fld.ticked is None and fld.target and fld.target.startswith("event."):
                    fld.ticked = self.checkbox_state(pdf_path, fld, dpi)
                if fld.ticked is False:
                    continue  # an unticked option: its field is not filled
                if date_only and not fld.inside and not self._ink_at_entry(fld, overview, width, height, labels):
                    continue  # a date line alone triggers only with something written to its right or above it
                if fld.inside:  # the item holds label and entry: its own box plus a right margin
                    x0, y0, x1, y1 = fld.bbox
                    h = max(6.0, y1 - y0)
                    fld.region = (max(0.0, x0 - 5), max(0.0, y0 - 0.6 * h), min(width, x1 + 0.1 * width), min(height, y1 + 0.6 * h))
                    jobs.append((fld, dpi))
                    continue
                rule = classify_label(fld.label)
                below = bool(rule and (rule.below or _BELOW_HINT_RE.search(fld.label) and rule.kind in (ADDRESS, PERSONS)))
                inline = False
                if not below and INLINE_END_RE.search(fld.label):
                    band = entry_region(fld, width, height, False, True, labels)
                    inline = ink_share(overview, width, height, (band[0] + 2, band[1], band[2], band[3])) > 0.003
                fld.region = entry_region(fld, width, height, below, inline, labels)
                jobs.append((fld, dpi))
        if jobs:
            workers = max(1, self.settings.llm.max_concurrency)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                done += list(pool.map(lambda job: self.read(pdf_path, job[0], job[1]), jobs))
        if done:
            logger.info(f"fieldread: {len(done)} fields read on pages {sorted({f.page for f in done})}")
        return sorted(done, key=lambda f: (f.page, round(f.bbox[1]), f.bbox[0]))


def normalise(fld: FormField, entry: str, iso: Optional[str], extra: Sequence[str] = ()) -> None:
    """
    Sets value / candidates of a read entry: dates as ISO with their
    candidate readings (a typed date has its literal reading only), other
    kinds verbatim.
    """
    if fld.kind not in (DATE, PLACE_DATE):
        fld.value = entry
        return
    handwritten = fld.handwritten
    fld.candidates = date_candidates(entry, iso, handwritten=handwritten)
    for value in extra if handwritten else ():  # the reader's own alternative readings (ISO), validated like the others
        iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})$", value)
        for reading in date_candidates(f"{iso_match.group(3)}.{iso_match.group(2)}.{iso_match.group(1)}") if iso_match else []:
            if reading not in fld.candidates:
                fld.candidates.append(reading)
    fld.value = fld.candidates[0] if len(fld.candidates) == 1 else iso


def _points(box: Any, width: float, height: float) -> Optional[Box]:
    """A 0-1000 box [x0, y0, x1, y1] in page points (None if it is no valid box)."""
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
        return None
    x0, y0, x1, y1 = (max(0.0, min(1000.0, float(v))) for v in box)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height)


def _widget_label(widget_box: Box, lines: Sequence[Tuple[str, Box]], checkbox: bool) -> Optional[Tuple[str, Box]]:
    """The printed label of a widget: the nearest line left of it on its row (right of it for a checkbox), else above it."""
    x0, y0, x1, y1 = widget_box
    mid, h = (y0 + y1) / 2, max(6.0, y1 - y0)
    same_row = [(text, box) for text, box in lines if abs((box[1] + box[3]) / 2 - mid) < max(6.0, 0.6 * h)]
    if checkbox:
        right = [(text, box) for text, box in same_row if box[0] >= x1 - 2 and box[0] - x1 < 60]
        if right:
            return min(right, key=lambda tb: tb[1][0])
    left = [(text, box) for text, box in same_row if box[2] <= x0 + 2 and x0 - box[2] < 200]
    if left:
        return max(left, key=lambda tb: tb[1][2])
    above = [(text, box) for text, box in lines if box[3] <= y0 + 1 and y0 - box[3] < 16 and box[0] < x1 and box[2] > x0]
    return max(above, key=lambda tb: tb[1][3]) if above else None


def _label_lines(page: PageText, widgets: Sequence[Box]) -> List[Tuple[str, Box]]:
    """The printed text lines of a form page without the widget appearances (their values and check glyphs)."""
    if not page.words:
        return page_lines(page)
    from fluxsense.layout.din5008 import group_lines

    def in_widget(word: Any) -> bool:
        cx, cy = (word.x0 + word.x1) / 2, (word.y0 + word.y1) / 2
        return any(b[0] - 1 <= cx <= b[2] + 1 and b[1] - 1 <= cy <= b[3] + 1 for b in widgets)
    words = [w for w in page.words if not in_widget(w)]
    return [(line.text, (line.x0, line.y0, line.x1, line.y1)) for line in group_lines(words)] if words else []


def acroform_fields(pdf_path: Path, pages: Sequence[PageText]) -> List[FormField]:
    """
    Native AcroForm fields of a digital PDF: the widget values are the
    entries (no vision); the label is the printed text next to the widget,
    else the widget's own label or name. Checkboxes give ticked / unticked,
    signature widgets are skipped.
    """
    import pymupdf
    out: List[FormField] = []
    try:
        with PYMUPDF_LOCK, pymupdf.open(pdf_path) as doc:
            if not doc.is_form_pdf:
                return []
            for page in pages:
                pdf_page = doc.load_page(page.page - 1)
                origin = (pdf_page.rect.x0, pdf_page.rect.y0)
                widgets = list(pdf_page.widgets() or [])
                lines = _label_lines(page, [(w.rect.x0 - origin[0], w.rect.y0 - origin[1], w.rect.x1 - origin[0],
                                             w.rect.y1 - origin[1]) for w in widgets]) if widgets else []
                for widget in widgets:
                    kind_code = widget.field_type
                    if kind_code in (pymupdf.PDF_WIDGET_TYPE_SIGNATURE, pymupdf.PDF_WIDGET_TYPE_BUTTON):
                        continue
                    rect = widget.rect
                    box = (rect.x0 - origin[0], rect.y0 - origin[1], rect.x1 - origin[0], rect.y1 - origin[1])
                    checkbox = kind_code in (pymupdf.PDF_WIDGET_TYPE_CHECKBOX, pymupdf.PDF_WIDGET_TYPE_RADIOBUTTON)
                    printed = _widget_label(box, lines, checkbox)
                    label = " ".join((printed[0] if printed else widget.field_label or widget.field_name or "").split())
                    if not label:
                        continue
                    rule = classify_label(label)
                    raw = widget.field_value
                    fld = FormField(page.page, label, CHECKBOX if checkbox else rule.kind if rule else TEXT,
                                    rule.target if rule else None, printed[1] if printed else box, region=box,
                                    script=TYPED, found=FOUND_ACROFORM)
                    if checkbox:
                        fld.ticked = raw not in (None, "", False, "Off", "off", "No")
                        fld.entry = fld.value = "[x]" if fld.ticked else "[ ]"
                    else:
                        entry = " ".join(str(raw or "").split())
                        fld.entry = entry or EMPTY
                        if entry:
                            normalise(fld, entry, None)
                        else:
                            fld.script = BLANK
                    out.append(fld)
    except (RuntimeError, ValueError) as e:  # an unreadable form structure: no widget values
        logger.warning(f"AcroForm fields not readable: {e}")
        return []
    if out:
        logger.info(f"fieldread: {len(out)} AcroForm fields read on pages {sorted({f.page for f in out})}")
    return out


def field_out(fld: FormField, width: float, height: float) -> Any:
    """The output record of a form field (boxes normalised to 0-1000)."""
    from fluxsense.models.result import FormFieldOut

    def norm(box: Box) -> Optional[List[float]]:
        if not any(box):
            return None
        return [round(box[0] / width * 1000, 1), round(box[1] / height * 1000, 1), round(box[2] / width * 1000, 1),
                round(box[3] / height * 1000, 1)]
    return FormFieldOut(page=fld.page, label=fld.label, kind=fld.kind, target=fld.target, entry=fld.entry, value=fld.value,
                        candidates=list(fld.candidates), script=fld.script, found=fld.found, ticked=fld.ticked,
                        bbox=norm(fld.bbox), entry_bbox=norm(fld.region))


def apply_fields(pages: Sequence[PageText], fields: Sequence[FormField]) -> None:
    """Stores the readings as the 'fieldread' text source of their pages (the FORM FIELDS block)."""
    for page in pages:
        mine = [f for f in fields if f.page == page.page]
        if mine:
            page.sources[SOURCE_FIELDREAD] = "\n".join(f.line() for f in mine)


def form_fields_block(page: PageText) -> str:
    """The FORM FIELDS block of a page for the extraction prompt ('' without readings)."""
    text = page.sources.get(SOURCE_FIELDREAD, "")
    if not text.strip():
        return ""
    return ("FORM FIELDS (each entry was read from a crop of its printed label; authoritative for these fields - map "
            "these label/value pairs first; [typed] / [handwritten] say how the entry is written, [form field] is the "
            "value of a digital form field; '(empty)' means nothing is written there):\n" + text + "\n")


_ADDRESS_RE = re.compile(r"^\s*(?P<street>.+?)\s+(?P<number>\d+\s?[a-zA-Z]?)\s*,\s*(?P<zip>\d{5})\s+(?P<city>.+?)\s*$")
CHECK_DISAGREE = "fieldread.disagreement"
CHECK_CHECKBOX = "fieldread.checkbox_unknown"


def parse_persons(entry: str) -> List[str]:
    """One person per line or ';' part; line numbers ('1.', '2)') and empty numbered lines are dropped."""
    persons = []
    for part in re.split(r"[;\n]", entry):
        name = re.sub(r"^\s*\d{1,2}\s*[.)]\s*", "", part).strip(" ,.-_")
        if name and not re.fullmatch(r"[\d\s.)_-]*", name):
            persons.append(name)
    return persons


def split_name_address(entry: str) -> Tuple[Optional[str], Dict[str, str]]:
    """
    A 'name and address' entry: the first line (or the first comma part)
    without digits is the name; the rest must parse as 'Street No, Postcode
    Town' to be used. A name never becomes a street.
    """
    parts = [p.strip() for p in (entry.splitlines() if "\n" in entry else entry.split(",")) if p.strip()]
    if not parts:
        return None, {}
    name = parts[0] if not re.search(r"\d", parts[0]) else None
    rest = parts[1:] if name else parts
    return name, parse_address(", ".join(rest)) if rest else {}
CHECK_AMBIGUOUS = "fieldread.ambiguous"


def parse_address(text: str) -> Dict[str, str]:
    """'Street 12, 00123 Town' -> street / house_number / zip_code / city (empty dict if it does not parse)."""
    match = _ADDRESS_RE.match(text.replace("\n", ", "))
    if not match:
        return {}
    return {"street": match.group("street"), "house_number": match.group("number"), "zip_code": match.group("zip"),
            "city": match.group("city")}


def map_fields(fields: Dict[str, Any], evidence: List[Any], form: Sequence[FormField]) -> Tuple[List[str], List[Any]]:
    """
    §12: unambiguous label targets take the field readings ('Ort, Datum' /
    '..., den' -> doc_date, 'Einzug/Auszug am' -> certified_event + event_date,
    persons, issuer name and address -> sender, flat address / floor ->
    subject_address). A full-page value that differs is replaced by the field
    reading and flagged with both values; a field with several date readings
    is flagged with the candidates; an empty field never yields a value.

    Returns:
        Tuple (notes, check results).
    """
    from fluxsense.models.result import CheckResult, CheckStatus, FieldEvidence, GroundingStatus, Severity
    meta = fields.setdefault("meta_header", {})
    bodies = fields.setdefault("bodies", {})
    legal = bodies.get("legal_body") if isinstance(bodies.get("legal_body"), dict) else None
    notes: List[str] = []
    checks: List[Any] = []
    by_path = {e.path: e for e in evidence}

    def put(path: str, value: Any, fld: FormField) -> None:
        node = fields
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        old = node.get(parts[-1])
        if old not in (None, "", []) and str(old) != str(value):
            checks.append(CheckResult(check_id=CHECK_DISAGREE, rule="fieldread", status=CheckStatus.FAIL, severity=Severity.WARNING,
                                      field=path, expected=str(value), actual=str(old),
                                      message=f"full-page reading {old} vs field reading {value} ('{fld.label}')"))
        node[parts[-1]] = value
        ev = by_path.get(path)
        if ev is None:
            ev = FieldEvidence(path=path, kind="date" if "date" in path else "text", value=value, page=fld.page)
            evidence.append(ev)
            by_path[path] = ev
        ev.original_value = old if old not in (None, "", []) and str(old) != str(value) else ev.original_value
        ev.value, ev.status, ev.matched_source, ev.resolution = value, GroundingStatus.GROUNDED, SOURCE_FIELDREAD, SOURCE_FIELDREAD
        ev.action = "replaced" if ev.original_value is not None else "filled_from_fieldread"
        ev.evidence = fld.line()
        if len(fld.candidates) > 1:
            checks.append(CheckResult(check_id=CHECK_AMBIGUOUS, rule="fieldread", status=CheckStatus.FAIL, severity=Severity.WARNING,
                                      field=path, actual=str(value),
                                      message=f"handwritten date ambiguous: '{fld.entry}' reads as {' or '.join(fld.candidates)} "
                                              f"('{fld.label}')"))
        notes.append(f"{path} = {value} from the field '{fld.label}'")

    movement = any(f.target and f.target.startswith("event.") and f.ticked is not False and f.entry and not f.empty
                   for f in form)
    for fld in form:
        if not fld.target or fld.entry is None:
            continue
        if fld.empty:
            current = meta.get("doc_date") if fld.target == "doc_date" else None
            if current:
                checks.append(CheckResult(check_id=CHECK_DISAGREE, rule="fieldread", status=CheckStatus.FAIL,
                                          severity=Severity.WARNING, field="meta_header.doc_date", actual=str(current),
                                          message=f"full-page reading {current}, but the field '{fld.label}' is empty"))
            continue
        value = fld.value or fld.entry
        if fld.target == "doc_date" and fld.value and re.match(r"\d{4}-\d{2}-\d{2}$", fld.value):
            put("meta_header.doc_date", fld.value, fld)
        elif fld.target.startswith("event.") and legal is not None and fld.value and re.match(r"\d{4}-\d{2}-\d{2}$", fld.value):
            legal["certified_event"] = fld.target.split(".", 1)[1]  # an unticked option was never read
            put("bodies.legal_body.event_date", fld.value, fld)
            if fld.ticked is None:
                checks.append(CheckResult(check_id=CHECK_CHECKBOX, rule="fieldread", status=CheckStatus.FAIL,
                                          severity=Severity.WARNING, field="bodies.legal_body.event_date",
                                          message=f"checkbox state of '{fld.label}' unknown - is this option ticked?"))
        elif fld.target == "certified_persons":
            persons = parse_persons(fld.entry)
            if persons:
                legal = bodies.setdefault("legal_body", {}) if legal is None else legal
                if persons != legal.get("certified_persons"):
                    legal["certified_persons"] = persons
                    notes.append(f"certified_persons = {persons} from the field '{fld.label}'")
                if movement and not (meta.get("recipient") or {}).get("name"):
                    put("meta_header.recipient.name", persons[0], fld)  # a move confirmation is about the moving person
        elif fld.target == "sender.name":
            put("meta_header.sender.name", value, fld)
        elif fld.target == "sender.address":
            for key, part in parse_address(fld.entry).items():
                put(f"meta_header.sender.{key}", part, fld)
        elif fld.target == "sender":  # a name + address block: split, or only FORM FIELDS context
            name, address = split_name_address(fld.entry)
            if name:
                put("meta_header.sender.name", name, fld)
            for key, part in address.items():
                put(f"meta_header.sender.{key}", part, fld)
        elif fld.target == "subject_address" and legal is not None:
            for key, part in parse_address(fld.entry).items():
                put(f"bodies.legal_body.subject_address.{key}", part, fld)
        elif fld.target == "subject_address.location_detail" and legal is not None:
            put("bodies.legal_body.subject_address.location_detail", value, fld)
    return notes, checks
