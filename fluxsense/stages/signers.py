# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/signers.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Signatures as structured output (field contract §12.3).
                Signature places are found in the page geometry - a printed
                signature label ('Unterschrift ...', 'Datum/Unterschrift',
                a role word under a line: 'Mieter') with the space above it,
                or the gap of a closing block ('Mit freundlichen Grüßen' /
                organisation / signature / printed name) - and their status
                (signed | empty | unclear) is measured as pen ink in that
                space (printed rules do not count). OCR text of the scribble
                is removed from the scanned page's text before the
                extraction, so it cannot become a party or name. The signer
                is derived from printed content only, never by reading the
                scribble: an 'i. A.' / 'i. V.' name, the printed name under
                the signature, the party the line label names ('Unterschrift
                Mieter' -> the tenant party), or the filler of a §2.4 form;
                without such evidence it stays empty (no error). Place and
                date come from the date-only field readings next to the
                signature, else from a printed '<Ort>, den <Datum>' line.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from fluxsense.catalog.contract import DATE_TOKEN_RE, not_a_party
from fluxsense.extract.base import SOURCE_NATIVE, PageText
from fluxsense.extract.fieldread import DATE, PLACE_DATE, FormField, classify_label, page_lines, stroke_share
from fluxsense.grounding.normalize import parse_date
from fluxsense.log import get_logger
from fluxsense.models.result import SignatureOut
from fluxsense.stages.repair import SIGNATURE_LABEL_RE

logger = get_logger("stages.signers")

Box = Tuple[float, float, float, float]
SIGNED, EMPTY, UNCLEAR = "signed", "empty", "unclear"
SIGNED_INK: float = 0.006  # share of pen-ink pixels in the signature space (rules excluded) above which it is signed
EMPTY_INK: float = 0.0015  # ... and below which it is empty; in between: unclear
LINE, CLOSING, TYPED_SIGNATURE = "line", "closing", "gez"
SOURCE_PRINTED, SOURCE_PROXY, SOURCE_LABEL, SOURCE_FILLER = "printed_name", "proxy", "line_label", "form_filler"
CLOSING_RE = re.compile(r"(mit freundlichen grüßen|mit freundlichen gruessen|freundliche grüße|hochachtungsvoll|viele grüße|"
                        r"herzliche grüße|kind regards|best regards|yours sincerely|yours faithfully)", re.IGNORECASE)
_PROXY_RE = re.compile(r"^\s*(i\.\s?[AV]\.|ppa\.|pp\.)\s*(.*)$", re.IGNORECASE)
_GEZ_RE = re.compile(r"^\s*gez\.\s*(.+?)\s*$", re.IGNORECASE)
_NAME_RE = re.compile(r"^(?:(?:Dr|Prof|Dipl)\.(?:-\w+\.?)?\s+)*[A-ZÄÖÜ][\wäöüß'-]*\.?(?:\s+[A-ZÄÖÜ][\wäöüß'-]*\.?){0,3}$")
_HANDWRITTEN_DATE_RE = re.compile(r"\d{1,2}\s*[./|]\s*\d{1,2}\s*[./|]\s*\d{2,4}")
_PLACE_DEN_RE = re.compile(r"^\s*([A-ZÄÖÜ][\wäöüß .-]{1,30}?)\s*,\s*(?:den\s+|am\s+)?(.*)$")
_PLACE_PREFIX_RE = re.compile(r"^\s*(?:[A-ZÄÖÜ][\wäöüß .-]{1,30}?\s*,\s*(?:den|am)?\s*)?$")
# Role words of signature line labels: (pattern, role, issuing side of the document).
ROLE_RULES: Tuple[Tuple["re.Pattern[str]", str, bool], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), role, issuing) for pattern, role, issuing in (
        (r"wohnungsgeber", "landlord", True), (r"vermieter", "landlord", True), (r"(?<!ver)mieter", "tenant", False),
        (r"arbeitgeber|employer", "employer", True), (r"arbeitnehmer|beschäftigte|employee", "employee", False),
        (r"verkäufer|seller", "seller", True), (r"(?<!ver)käufer|buyer", "buyer", False),
        (r"auftragnehmer|contractor", "contractor", True), (r"auftraggeber", "client", False),
        (r"aussteller|issuer", "issuer", True), (r"antragsteller|applicant", "applicant", False),
        (r"versicherungsnehmer|policyholder", "policyholder", False), (r"kontoinhaber|account holder", "account_holder", False),
        (r"erziehungsberechtigt|guardian", "guardian", False), (r"patient", "patient", False),
        (r"\bkunde|customer", "customer", False), (r"zahlungspflichtig|payer", "payer", False)))
_LEADING_SIGNATURE_RE = re.compile(r"^\W*(?:ort\s*,?\s*/?\s*)?(?:datum\s*[,/]?\s*)?(?:unterschrift|signature)\b", re.IGNORECASE)
_FILLER_WORDS = re.compile(r"\b(?:der|die|des|unterschrift|datum|ort|und|/|,|\(|\))\b", re.IGNORECASE)


@dataclass
class SignatureBlock:
    """A signature place found on a page (before the signer is attributed)."""

    page: int
    kind: str  # line (printed label under the space) | closing (gap of a closing block) | gez (typed 'gez. Name')
    label: str
    label_box: Box
    region: Optional[Box] = None  # the signature space in page points
    status: str = UNCLEAR
    ink: Optional[float] = None
    printed_name: Optional[str] = None
    proxy: Optional[str] = None  # 'i. A. Name' / 'i. V. Name' as printed
    role: Optional[str] = None
    issuing: Optional[bool] = None
    place: Optional[str] = None
    date: Optional[str] = None
    removed: List[str] = field(default_factory=list)


def role_of(label: str) -> Tuple[Optional[str], Optional[bool]]:
    """(role, issuing side) named by a signature label ('Unterschrift des Wohnungsgebers' -> landlord, True)."""
    for pattern, role, issuing in ROLE_RULES:
        if pattern.search(label):
            return role, issuing
    return None, None


def _role_only(text: str) -> bool:
    """A short label that only names a role ('Mieter', 'Der Vermieter', 'Datum, Unterschrift Kunde')."""
    clean = " ".join(text.split())
    if not clean or len(clean) > 45 or re.search(r"\d", clean):
        return False
    role, _ = role_of(clean)
    rest = _FILLER_WORDS.sub(" ", clean)
    return role is not None and len(rest.split()) <= 2


def _signature_label(text: str) -> bool:
    """
    A printed signature label: short, no sentence ('Unterschrift des Wohnungsgebers', 'Datum/Unterschrift', 'Mieter');
    a label that starts with 'Unterschrift' may be longer ('Unterschrift des Wohnungsgebers oder der beauftragten
    Person (Nachweis erforderlich)'). A form field label naming a role ('Name des Wohnungsgebers') is no signature label.
    """
    clean = " ".join(text.split())
    if not clean or re.search(r"[.!?]\s*$", clean.strip("()")):
        return False
    if _LEADING_SIGNATURE_RE.match(clean) and len(clean) <= 120:
        return True
    if len(clean) > 60 or len(clean.split()) > 6:
        return False
    return bool(SIGNATURE_LABEL_RE.search(clean)) or (_role_only(clean) and classify_label(clean) is None)


def name_like(text: str) -> bool:
    """A printed person / signer name: 1-4 capitalised words, no digits, no label words."""
    clean = " ".join(text.strip(" ,;()").split())
    if not clean or re.search(r"\d", clean) or len(clean) > 60 or clean.isupper() and len(clean) > 3:
        return False
    if SIGNATURE_LABEL_RE.search(clean) or CLOSING_RE.search(clean) or classify_label(clean) or not_a_party(clean):
        return False
    return _NAME_RE.match(clean) is not None and role_of(clean)[0] is None


def _date_like(text: str) -> bool:
    return bool(DATE_TOKEN_RE.search(text) or _HANDWRITTEN_DATE_RE.search(text))


def _date_span(text: str) -> Optional[Tuple[int, int]]:
    """
    Character span of the date in a line that also holds other writing: OCR
    joins a date and the signature written beside it into one line
    ('07.08.2026 <scribble read as words>'). A leading '<Ort>, den' belongs
    to the date. None for a line that holds only a date (and its place).
    """
    match = DATE_TOKEN_RE.search(text) or _HANDWRITTEN_DATE_RE.search(text)
    if match is None:
        return None
    start = 0 if _PLACE_PREFIX_RE.match(text[:match.start()]) else match.start()
    rest = text[:start] + text[match.end():]
    return (start, match.end()) if sum(c.isalpha() for c in rest) >= 2 else None


def _date_part(text: str, box: Box) -> Tuple[str, Box]:
    """The date of a date line and its share of the line box (by characters, one character of margin)."""
    span = _date_span(text)
    if span is None:
        return text, box
    step = (box[2] - box[0]) / max(1, len(text))
    return text[span[0]:span[1]].strip(), (box[0] + max(0, span[0] - 1) * step, box[1],
                                            box[0] + min(len(text), span[1] + 1) * step, box[3])


def _inside(box: Box, region: Box, tolerance: float = 1.5) -> bool:
    """True if the centre of a box lies inside a region."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return region[0] - tolerance <= cx <= region[2] + tolerance and region[1] - tolerance <= cy <= region[3] + tolerance


def _overlap_x(a: Box, b: Box) -> bool:
    return a[0] < b[2] and a[2] > b[0]


def _rule_above(overview: Optional[Image.Image], width: float, height: float, box: Box) -> bool:
    """A printed rule (a pixel row dark over most of the label width) just above a label."""
    if overview is None:
        return False
    from fluxsense.extract.fieldread import dark_rows
    sx, sy = overview.width / width, overview.height / height
    top, bottom = max(0, int((box[1] - 8) * sy)), max(1, int(box[1] * sy))
    crop = overview.crop((int(box[0] * sx), top, max(int(box[0] * sx) + 1, int(box[2] * sx)), max(top + 1, bottom)))
    return any(r > 0.5 for r in dark_rows(crop))


def _pen_ink(overview: Image.Image, width: float, height: float, region: Box, keep: Sequence[Box]) -> float:
    """Pen ink in a signature space: printed rules and the boxes of kept text (dates written next to it) do not count."""
    if keep:
        overview = overview.copy()
        draw = ImageDraw.Draw(overview)
        sx, sy = overview.width / width, overview.height / height
        for box in keep:
            draw.rectangle((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy), fill="white")
    return stroke_share(overview, width, height, region)


def _status(ink: Optional[float]) -> str:
    if ink is None:
        return UNCLEAR
    return SIGNED if ink >= SIGNED_INK else EMPTY if ink <= EMPTY_INK else UNCLEAR


def _closing_block(index: int, lines: Sequence[Tuple[str, Box]], page: int) -> Optional[SignatureBlock]:
    """
    The signature space of a closing: the first vertical gap of at least
    1.8 line heights in the closing's column (after the organisation line);
    OCR lines of the scribble (much taller than the closing line) do not
    close the gap. The line after the gap is the printed name.
    """
    text, box = lines[index]
    h = max(6.0, box[3] - box[1])
    column = [(t, b) for t, b in lines[index + 1:] if abs(b[0] - box[0]) < 60 and box[3] <= b[1] < box[3] + 140][:7]
    previous = box
    tall: List[Tuple[str, Box]] = []
    for t, b in column:
        if b[3] - b[1] > 1.6 * h and not _date_like(t):
            tall.append((t, b))
            continue
        gap = b[1] - previous[3]
        if gap >= max(14.0, 1.8 * h):
            region = (box[0] - 10, previous[3] + 0.5, box[0] + 230, b[1] - 0.5)
            block = SignatureBlock(page, CLOSING, " ".join(text.split()), box, region=region)
            rest = column[column.index((t, b)):]
            proxy = next((p for p, _ in rest[:2] if _PROXY_RE.match(p)), None)
            if proxy:
                match = _PROXY_RE.match(proxy)
                name = (match.group(2) if match else "").strip() or next((n for n, _ in rest[1:3] if name_like(n)), "")
                block.proxy, block.printed_name = " ".join(proxy.split()), name if name_like(name) else None
            elif name_like(t):
                block.printed_name = " ".join(t.split())
            return block
        if gap > 70:
            break
        previous = b
    if tall:  # the scribble's OCR line fills the gap: it is the signature space
        t, b = tall[0]
        after = [(n, nb) for n, nb in column if nb[1] >= b[3] - 1 and (n, nb) not in tall]
        region = (box[0] - 10, b[1] - 1, box[0] + 230, b[3] + 1)
        block = SignatureBlock(page, CLOSING, " ".join(text.split()), box, region=region)
        if after and name_like(after[0][0]):
            block.printed_name = " ".join(after[0][0].split())
        return block
    return None


def _line_block(index: int, lines: Sequence[Tuple[str, Box]], page: int, width: float) -> SignatureBlock:
    """The signature space above a printed signature label, up to the next label on the same row."""
    text, box = lines[index]
    h = max(6.0, box[3] - box[1])
    right = min(width, box[0] + max(210.0, box[2] - box[0] + 20))
    mid = (box[1] + box[3]) / 2
    beside = [b[0] for t, b in lines if b[0] > box[2] and abs((b[1] + b[3]) / 2 - mid) < h]
    if beside:
        right = min(right, min(beside) - 4)
    top = max(0.0, box[1] - max(32.0, 4 * h))
    # printed running text or a form label above the space ends it (a scribble's OCR line is short)
    above = [b[3] for t, b in lines if b[3] <= box[1] and b[3] > top and _overlap_x(b, (box[0] - 10, 0, right, 0))
             and (len(t.split()) >= 4 or len(t.strip()) >= 25 or classify_label(t)) and not _date_like(t)]
    if above:
        top = max(above) + 0.5
    region = (max(0.0, box[0] - 10), top, max(box[2], right), box[1] - 0.5)
    block = SignatureBlock(page, LINE, " ".join(text.split()), box, region=region)
    block.role, block.issuing = role_of(text)
    below = [(t, b) for t, b in lines if b[1] >= box[3] - 1 and b[1] - box[3] < 2.2 * h and _overlap_x(b, box)]
    if below:
        t = below[0][0]
        proxy = _PROXY_RE.match(t)
        if proxy and name_like(proxy.group(2)):
            block.proxy, block.printed_name = " ".join(t.split()), " ".join(proxy.group(2).split())
        elif name_like(t):
            block.printed_name = " ".join(t.split())
    return block


def find_blocks(page: PageText, overview: Optional[Image.Image] = None) -> List[SignatureBlock]:
    """
    Signature places of one page with their status.

    Args:
        page: Page with layout items or word boxes.
        overview: Low-resolution page render for the ink measurement (None: status unclear).

    Returns:
        The blocks in reading order.
    """
    width, height = page.width or 595.0, page.height or 842.0
    lines = sorted(page_lines(page), key=lambda tb: (round(tb[1][1]), tb[1][0]))
    blocks: List[SignatureBlock] = []
    for index, (text, box) in enumerate(lines):
        gez = _GEZ_RE.match(text)
        if gez:
            block = SignatureBlock(page.page, TYPED_SIGNATURE, " ".join(text.split()), box, region=box, status=SIGNED)
            block.printed_name = gez.group(1) if name_like(gez.group(1)) else None
            blocks.append(block)
        elif CLOSING_RE.search(text):
            closing = _closing_block(index, lines, page.page)
            if closing is not None:
                blocks.append(closing)
        elif _signature_label(text):
            blocks.append(_line_block(index, lines, page.page, width))
    # a label under the printed name of a closing ('(Unterschrift)') names the same signature
    closings = [b for b in blocks if b.kind == CLOSING and b.region]
    blocks = [b for b in blocks if b.kind != LINE or not any(
        c.region and b.region and _overlap_x(b.region, c.region) and b.region[1] < c.region[3] + 30 and b.label_box[1] > c.region[1]
        for c in closings)]
    for block in blocks:
        if block.kind == TYPED_SIGNATURE or block.region is None:
            continue
        keep = [b if classify_label(t) else _date_part(t, b)[1] for t, b in lines
                if _inside(b, block.region) and (_date_like(t) or classify_label(t))]
        block.ink = _pen_ink(overview, width, height, block.region, keep) if overview is not None else None
        block.status = _status(block.ink)
    # a bare role word ('Mieter') is a signature place only with ink above it or a signature rule
    return [b for b in blocks if not (b.kind == LINE and not SIGNATURE_LABEL_RE.search(b.label)
                                       and b.status == EMPTY and not _rule_above(overview, width, height, b.label_box))]


def scrub_page(page: PageText, blocks: Sequence[SignatureBlock]) -> List[str]:
    """
    Removes the OCR text of signature scribbles from a scanned page (in
    place): text lines inside a signed or unclear signature space that are no
    date and no form label leave the layout items, word boxes and every
    OCR-derived text source. Digital pages are never changed.

    Returns:
        The removed lines.
    """
    if not page.scanned():
        return []
    width, height = page.width or 595.0, page.height or 842.0
    removed: List[str] = []
    shortened: Dict[str, str] = {}  # dated scribble line -> its date
    for block in blocks:
        if block.region is None or block.kind == TYPED_SIGNATURE or block.status == EMPTY:
            continue
        for text, box in page_lines(page):
            if not text.strip() or not _inside(box, block.region) or classify_label(text) \
                    or " ".join(text.split()) == block.label:
                continue
            if not _date_like(text):
                block.removed.append(text.strip())
            elif _date_span(text) is not None:
                date_text = _date_part(text, box)[0]
                shortened[text.strip()] = date_text
                block.removed.append(text.strip().replace(date_text, "", 1).strip(" ,"))
        removed += block.removed
    if not removed:
        return []
    garbage = {r.strip() for r in removed} | set(shortened)
    regions = [b.region for b in blocks if b.region is not None and b.removed]
    kept_items = []
    for item in page.items or []:
        x0, y0, x1, y1 = (item.bbox[0] / 1000 * width, item.bbox[1] / 1000 * height, item.bbox[2] / 1000 * width,
                          item.bbox[3] / 1000 * height)
        content_lines = item.content.splitlines() or [item.content]
        step = (y1 - y0) / max(1, len(content_lines))
        rest = [shortened.get(line.strip(), "") if line.strip() in shortened else line
                for n, line in enumerate(content_lines)
                if not (line.strip() in garbage and line.strip() not in shortened
                        and any(_inside((x0, y0 + n * step, x1, y0 + (n + 1) * step), r) for r in regions))]
        if rest == content_lines:
            kept_items.append(item)
        elif rest:
            item.content = "\n".join(rest)
            kept_items.append(item)
    page.items = kept_items
    page.words = [w for w in page.words if not (any(_inside((w.x0, w.y0, w.x1, w.y1), r) for r in regions)
                                                 and not _date_like(w.text))]
    for key, text in list(page.sources.items()):
        if key in ("fieldread", "handread") or (key == SOURCE_NATIVE and not page.native_ocr):
            continue
        lines = text.splitlines()
        for line in removed:
            target = line.strip()
            for n, existing in enumerate(lines):
                if existing.strip() == target:
                    del lines[n]
                    break
        lines = [shortened.get(line.strip(), line) for line in lines]
        page.sources[key] = "\n".join(lines)
    logger.info(f"signatures p{page.page}: removed {len(removed)} OCR line(s) of signature scribbles")
    return removed


def place_and_date(block: SignatureBlock, page: PageText, form: Sequence[FormField]) -> None:
    """
    Place and date next to a signature: the nearest date / place-date field
    reading of the page on the signature's rows, else a printed '<Ort>, den
    <Datum>' / date line on those rows.
    """
    if block.region is None:
        return
    top, bottom = block.region[1] - 40, block.label_box[3] + 30
    readings = [f for f in form if f.page == block.page and f.kind in (DATE, PLACE_DATE) and f.entry and not f.empty
                and top <= (f.bbox[1] + f.bbox[3]) / 2 <= bottom]
    if readings:
        centre = (block.region[0] + block.region[2]) / 2
        nearest = min(readings, key=lambda f: abs((f.bbox[1] + f.bbox[3]) / 2 - block.label_box[1])
                      + 0.2 * abs((f.bbox[0] + f.bbox[2]) / 2 - centre))
        value = nearest.value if nearest.value and re.match(r"\d{4}-\d{2}-\d{2}$", nearest.value) else None
        block.date = value
        place_text = nearest.entry if nearest.kind == PLACE_DATE and nearest.entry else ""
        label_place = re.match(r"^\s*([A-ZÄÖÜ][\wäöüß .-]{1,30}?)\s*,\s*den\b", nearest.label)
        match = _PLACE_DEN_RE.match(place_text) if place_text else None
        if label_place:
            block.place = label_place.group(1).strip()
        elif match and not re.search(r"\d", match.group(1)):
            block.place = match.group(1).strip()
        if block.date:
            return
    for text, box in sorted(page_lines(page), key=lambda tb: abs(tb[1][1] - block.label_box[1])):
        if not (top <= (box[1] + box[3]) / 2 <= bottom) or not _date_like(text):
            continue
        token = DATE_TOKEN_RE.search(text)
        parsed = parse_date(token.group(0)) if token else None
        if parsed is None:
            continue
        block.date = block.date or parsed.isoformat()
        match = _PLACE_DEN_RE.match(text)
        if match and not re.search(r"\d", match.group(1)) and match.group(1).strip().lower() not in ("ort", "datum", "place"):
            block.place = block.place or match.group(1).strip()
        return


def _party_name(party: Any) -> Optional[str]:
    if not isinstance(party, dict):
        return None
    value = party.get("name") or party.get("company")
    return str(value) if value else None


def attribute(blocks: Sequence[SignatureBlock], fields: Dict[str, Any], party_case: Optional[str],
              width: float = 595.0, height: float = 842.0, sizes: Optional[Dict[int, Tuple[float, float]]] = None) -> List[SignatureOut]:
    """
    The signer of every signature place, from printed content only: (d) an
    'i. A.' / 'i. V.' name, (a) the printed name under the signature or in
    the closing, (b) the party named by the line label (issuing roles -
    landlord, employer, seller, issuer - are the sender, the counterpart roles
    the recipient; on a §2.4 form the other way round), (c) the filler of a
    §2.4 form (the sender). Without evidence - and for an empty signature
    place - the signer stays None.

    Args:
        blocks: Signature places of the document.
        fields: Final fields (parties).
        party_case: Party case of the contract type ('2.4' for forms filled in by a person).
        width: Default page width in points.
        height: Default page height in points.
        sizes: Page -> (width, height) in points.

    Returns:
        The output records.
    """
    meta = fields.get("meta_header") or {}
    form = party_case == "2.4"
    out: List[SignatureOut] = []
    for block in blocks:
        name: Optional[str] = None
        source: Optional[str] = None
        evidence: Optional[str] = None
        role = block.role
        if block.proxy and block.printed_name:
            name, source, evidence = block.printed_name, SOURCE_PROXY, f"'{block.proxy}' printed at the signature"
        elif block.printed_name:
            name, source = block.printed_name, SOURCE_PRINTED
            evidence = (f"typed signature '{block.label}'" if block.kind == TYPED_SIGNATURE else
                        f"printed name '{block.printed_name}' under the signature" + (f" of '{block.label}'" if block.label else ""))
        elif block.role is not None:
            side = "recipient" if (block.issuing is True) == form else "sender"
            name = _party_name(meta.get(side))
            if name:
                source, evidence = SOURCE_LABEL, f"line label '{block.label}' names the {block.role} ({side})"
        if name is None and form and block.kind != CLOSING:
            name = _party_name(meta.get("sender"))
            if name:
                source, evidence = SOURCE_FILLER, f"the filler of the form (§2.4) signs '{block.label}'"
                role = role or "form_filler"
        if block.kind == CLOSING and role is None:
            role = "sender"
        if block.status == EMPTY and name is not None:  # nobody signed: the printed evidence names the intended signer only
            evidence, name, source = f"not signed; intended signer '{name}' from {evidence}", None, None
        w, h = (sizes or {}).get(block.page, (width, height))
        bbox = [round(v, 1) for v in (block.region[0] / w * 1000, block.region[1] / h * 1000, block.region[2] / w * 1000,
                                      block.region[3] / h * 1000)] if block.region else None
        out.append(SignatureOut(signer_name=name, role=role, place=block.place, date=block.date, page=block.page, bbox=bbox,
                                status=block.status, evidence=evidence or f"signature place '{block.label}' (no printed signer)",
                                signer_source=source, label=block.label, removed_ocr=list(block.removed)))
    return out
