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
                §12). On form-like pages the printed labels are found in the
                layout ('Ort, Datum', 'Einzug am', 'Name des ...', '..., den')
                with a lexicon that also gives the field kind; the entry
                region next to, above or below each label is cropped from
                the scan's native resolution and read by the vision model
                with an open question plus a type hint - never with the
                full-page value, so the reading is not anchored. Dates are
                normalised to ISO with their candidate readings
                ('1|10|26': bars may be separators or ones). The readings
                become a FORM FIELDS block of the page text and the
                'fieldread' text source used by grounding.
------------------------------------------------------------------------------
"""

import io
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from fluxsense.config import Settings
from fluxsense.extract.base import PageText
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, encode_data_uri, fit_image
from fluxsense.log import get_logger
from fluxsense.pdflock import PYMUPDF_LOCK

logger = get_logger("extract.fieldread")

SOURCE_FIELDREAD = "fieldread"
DATE, PLACE_DATE, NAME, PERSONS, ADDRESS, LOCATION, PHONE, TEXT = (
    "date", "place_date", "name", "persons", "address", "location", "phone", "text")
EMPTY = "EMPTY"
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
          "may have 2 or 4 digits). Return it as written AND as ISO date, e.g. '7/8/26 -> 2026-08-07'.",
    PLACE_DATE: "The entry is a PLACE and a handwritten DATE (day/month/year; separators may be '.', '/' or '|'; 2- or "
                "4-digit year). Return it as written AND the date as ISO, e.g. 'Musterstadt, 7/8/26 -> 2026-08-07'.",
    NAME: "The entry is the NAME of a person or organisation.",
    PERSONS: "The entry lists one or more PERSONS (first and last name each); separate them with '; '.",
    ADDRESS: "The entry is a postal ADDRESS (street, house number, postcode, town).",
    LOCATION: "The entry is a LOCATION inside a building (floor and position, e.g. 'EG links', '2. OG rechts').",
    PHONE: "The entry is a PHONE number.",
    TEXT: "",
}
PROMPT = ('This is a crop of a German form. The printed label of the field is: "{label}". Read only the handwritten (or '
          'typed) ENTRY that belongs to this label - not the printed labels. If the field is empty answer EMPTY. {hint} '
          'Answer as JSON: {{"entry": "<the entry as written, or EMPTY>", "iso": "<ISO date or null>"}}')
SCHEMA: Dict[str, Any] = {"type": "object", "additionalProperties": False,
                          "properties": {"entry": {"type": "string", "maxLength": 200},
                                         "iso": {"type": ["string", "null"], "maxLength": 10}},
                          "required": ["entry"]}


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

    @property
    def empty(self) -> bool:
        """True if the field was read as empty."""
        return self.entry is not None and self.entry.strip().upper() in ("", EMPTY)

    def line(self) -> str:
        """The FORM FIELDS line of this field."""
        if self.entry is None:
            return f"{self.label}: (not read)"
        if self.empty:
            return f"{self.label}: (empty)"
        if self.value and self.value != self.entry:
            alt = f" (readings: {', '.join(self.candidates)})" if len(self.candidates) > 1 else ""
            return f"{self.label}: {self.entry} -> {self.value}{alt}"
        return f"{self.label}: {self.entry}"


def classify_label(text: str) -> Optional[LabelRule]:
    """The lexicon rule of a printed label, None if it is no known label."""
    clean = " ".join(text.split())
    if not clean or len(clean) > 90:
        return None
    return next((rule for rule in LEXICON if rule.pattern.search(clean)), None)


def find_labels(page: PageText) -> List[FormField]:
    """
    Printed field labels of a page: layout items (OvisOCR3 / Docling, bbox
    0-1000) or native text lines, matched against the lexicon. A label item
    that already carries text after the label keeps only the label part.
    """
    width, height = page.width or 595.0, page.height or 842.0
    candidates: List[Tuple[str, Tuple[float, float, float, float]]] = []
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
    fields: List[FormField] = []
    for text, box in candidates:
        rule = classify_label(text)
        if rule is None:
            continue
        label = text.strip()
        match = re.match(r"^(.*?(?::|\bam|\bden))\s+\S", label)
        if match and rule.kind in (DATE, PLACE_DATE, NAME, ADDRESS, LOCATION, PHONE) and len(match.group(1)) < len(label):
            label = match.group(1).strip()
        fields.append(FormField(page.page, label, rule.kind, rule.target, box))
    return fields


def entry_region(fld: FormField, width: float, height: float, below: bool) -> Tuple[float, float, float, float]:
    """
    The entry region of a label in page points: beside / above the label
    (German forms put the label under the entry line, or the entry inline to
    the right), or below it for a header label ('... lauten:').
    """
    x0, y0, x1, y1 = fld.bbox
    h = max(6.0, y1 - y0)
    if below:
        return (max(0.0, x0 - 20), y0, min(width, max(x1 + 0.45 * width, x0 + 0.85 * width)), min(height, y1 + 6.5 * h))
    return (max(0.0, x0 - 20), max(0.0, y0 - 3.2 * h), min(width, max(x1 + 0.45 * width, x0 + 0.55 * width)),
            min(height, y1 + 1.2 * h))


def form_like(page: PageText, labels: Sequence[FormField], handwriting: bool) -> bool:
    """
    Trigger: a scanned page (no native text layer: a digital page prints its
    entries as text) on which the audit saw handwriting next to labels, or
    at least MIN_LABELS lexicon labels sit.
    """
    if page.sources.get("native", "").strip():
        return False
    return bool(labels) and (handwriting or len(labels) >= MIN_LABELS)


_DATE_PARTS_RE = re.compile(r"(?<!\d)(\d{1,2})\s*([./|])\s*(\d{1,2})\s*([./|])\s*(\d{2}|\d{4})(?!\d)")


def _expand_year(year: str) -> int:
    value = int(year)
    return value + (2000 if value < 70 else 1900) if len(year) == 2 else value


def date_candidates(entry: str, model_iso: Optional[str] = None, today: Optional[date] = None) -> List[str]:
    """
    ISO readings of a handwritten date: the model's ISO value, the entry with
    '.', '/' or '|' as separators, and the readings where a bar next to a
    '1' is part of the digit ('1|10|26' -> 01.10.2026 or 11.10.2026); invalid
    and future dates are dropped.
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

    iso = re.match(r"(\d{4})-(\d{2})-(\d{2})$", (model_iso or "")[:10])
    if iso:
        add(int(iso.group(3)), int(iso.group(2)), int(iso.group(1)))
    text = entry.replace(" ", "")
    match = _DATE_PARTS_RE.search(text)
    if match:
        day, sep1, month, sep2, year = match.groups()
        add(int(day), int(month), _expand_year(year))
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

    def read(self, pdf_path: Path, fld: FormField, dpi: int) -> FormField:
        """Reads one field (typed prompt, crop only) and normalises the entry."""
        image = self._crop(pdf_path, fld, dpi)
        part = ImagePart(label=f"FIELD {fld.label[:40]}", data_uri=encode_data_uri(image, "png"), width=image.width,
                         height=image.height)
        request = LLMRequest(prompt=PROMPT.format(label=fld.label, hint=TYPE_HINTS.get(fld.kind, "")),
                             label=f"Fieldread p{fld.page} {fld.label[:30]}", model=self.settings.models.vision,
                             images=[part], schema=SCHEMA, schema_name="form_field", max_tokens=160)
        answer = self.backend.generate_json(request)
        entry = str((answer or {}).get("entry") or "").strip() if isinstance(answer, dict) else ""
        iso = (answer or {}).get("iso") if isinstance(answer, dict) else None
        fld.entry = entry or EMPTY
        if fld.empty:
            return fld
        if fld.kind in (DATE, PLACE_DATE):
            fld.candidates = date_candidates(entry, iso if isinstance(iso, str) else None)
            fld.value = fld.candidates[0] if len(fld.candidates) == 1 else (iso if isinstance(iso, str) else None)
        else:
            fld.value = entry
        return fld

    def run(self, pdf_path: Path, pages: Sequence[PageText], handwriting: bool = False) -> List[FormField]:
        """
        Reads the labelled fields of the form-like pages.

        Args:
            pdf_path: PDF file.
            pages: Pages of the logical document (with layout items or word boxes).
            handwriting: The visual audit reported handwriting.

        Returns:
            The read fields (empty when no page is form-like).
        """
        from fluxsense.extract.ovis import NATIVE_MAX_DPI, native_dpi
        jobs: List[Tuple[FormField, int]] = []
        for page in pages:
            labels = find_labels(page)
            if not form_like(page, labels, handwriting):
                continue
            dpi = min(NATIVE_MAX_DPI, max(300, native_dpi(pdf_path, page.page - 1) or 300))
            for fld in labels[:self.settings.pipeline.fieldread_max_fields]:
                rule = classify_label(fld.label)
                below = bool(rule and (rule.below or _BELOW_HINT_RE.search(fld.label) and rule.kind in (ADDRESS, PERSONS)))
                fld.region = entry_region(fld, page.width or 595.0, page.height or 842.0, below)
                jobs.append((fld, dpi))
        if not jobs:
            return []
        workers = max(1, self.settings.llm.max_concurrency)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(lambda job: self.read(pdf_path, job[0], job[1]), jobs))
        logger.info(f"fieldread: {len(done)} fields read on pages {sorted({f.page for f in done})}")
        return done


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
    return ("FORM FIELDS (each entry was read from a crop of its printed label; authoritative for these fields, "
            "'(empty)' means nothing is written there):\n" + text + "\n")


_ADDRESS_RE = re.compile(r"^\s*(?P<street>.+?)\s+(?P<number>\d+\s?[a-zA-Z]?)\s*,\s*(?P<zip>\d{5})\s+(?P<city>.+?)\s*$")
CHECK_DISAGREE = "fieldread.disagreement"
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
            legal["certified_event"] = fld.target.split(".", 1)[1]
            put("bodies.legal_body.event_date", fld.value, fld)
        elif fld.target == "certified_persons" and legal is not None:
            persons = [p.strip() for p in re.split(r"[;\n]", fld.entry) if p.strip()]
            if persons and persons != legal.get("certified_persons"):
                legal["certified_persons"] = persons
                notes.append(f"certified_persons = {persons} from the field '{fld.label}'")
        elif fld.target == "sender.name":
            put("meta_header.sender.name", value, fld)
        elif fld.target in ("sender.address", "sender"):
            parts = parse_address(fld.entry.splitlines()[-1] if fld.target == "sender" else fld.entry)
            for key, part in parts.items():
                put(f"meta_header.sender.{key}", part, fld)
            if fld.target == "sender" and len(fld.entry.splitlines()) > 1:
                put("meta_header.sender.name", fld.entry.splitlines()[0].strip(), fld)
        elif fld.target == "subject_address" and legal is not None:
            for key, part in parse_address(fld.entry).items():
                put(f"bodies.legal_body.subject_address.{key}", part, fld)
        elif fld.target == "subject_address.location_detail" and legal is not None:
            put("bodies.legal_body.subject_address.location_detail", value, fld)
    return notes, checks
