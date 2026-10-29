# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/layout/din5008.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    DIN 5008 letter geometry from word/item boxes (relative page
                coordinates, tolerant to skew and scaling): address window
                (recipient) with the small return-address line (sender),
                letterhead band, right information column, reference line
                ('Ihr Zeichen / Unser Zeichen / Datum') and the closing /
                signature area. Used as labelled prompt candidates and to
                validate sender/recipient after extraction.
------------------------------------------------------------------------------
"""

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fluxsense.catalog.contract import (is_address_only, is_department, is_document_title, is_name_fragment, not_a_party,
                                       party_problem)
from fluxsense.extract.base import PageText, Word
from fluxsense.grounding.normalize import normalize_text

# Address window, union of DIN 5008 form A (27 mm) and B (45 mm), 20 mm from the left, 85 x 45 mm - with margin.
WINDOW_X: Tuple[float, float] = (0.03, 0.60)
WINDOW_X0_MAX: float = 0.30
WINDOW_Y: Tuple[float, float] = (0.085, 0.36)
LETTERHEAD_Y_MAX: float = 0.12
INFO_COLUMN_X0: float = 0.58
INFO_COLUMN_Y_MAX: float = 0.40
REFERENCE_RE = re.compile(r"(ihr(?:e)?\s+zeichen|unser(?:e)?\s+zeichen|ihre\s+nachricht|unsere\s+nachricht|aktenzeichen|"
                          r"gesch[aä]ftszeichen|unser\s+az)", re.IGNORECASE)
SALUTATION_RE = re.compile(r"^\s*(sehr\s+geehrte|liebe[rs]?\b|hallo\b|guten\s+tag|dear\b|mit\s+freundlichen|freundliche\s+gr|"
                           r"viele\s+gr|beste\s+gr|herzliche\s+gr|kind\s+regards|best\s+regards)", re.IGNORECASE)
TITLE_ONLY_RE = re.compile(r"^\s*(herrn?|frau|firma|an|eheleute|familie|z\.?\s*hd\.?.*)\s*$", re.IGNORECASE)
ZIP_CITY_RE = re.compile(r"(?:^|\s)(?:D-)?\d{5}\s+[A-Za-zÄÖÜäöüß]")
ADDRESS_MARKER_RE = re.compile(r"^\s*(herrn?|frau|firma|postfach|familie|eheleute)\b", re.IGNORECASE)
NON_ADDRESS_RE = re.compile(r"\d+,\d{2}\b|^\s*[\w\s.-]*\d{1,2}\.\d{1,2}\.(\d{2,4})?\s*$|\d{1,2}\.\d{1,2}\.\d{4}\s*$|:\s")


@dataclass
class Line:
    """A text line with its box in points."""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def height(self) -> float:
        """Line height."""
        return max(0.1, self.y1 - self.y0)


@dataclass
class LayoutHints:
    """Letter regions of the first page (and the closing area of the last page)."""

    address_window: List[str] = field(default_factory=list)
    return_address: Optional[str] = None
    letterhead: List[str] = field(default_factory=list)
    info_column: List[str] = field(default_factory=list)
    reference_line: List[str] = field(default_factory=list)
    closing: List[str] = field(default_factory=list)

    def empty(self) -> bool:
        """True if nothing was detected."""
        return not (self.address_window or self.letterhead or self.reference_line)

    def prompt_block(self) -> str:
        """Labelled candidates for the extraction prompt."""
        parts = []
        if self.letterhead:
            parts.append("LETTERHEAD: " + " | ".join(self.letterhead[:4]))
        if self.return_address:
            parts.append("RETURN_ADDRESS (sender): " + self.return_address)
        if self.address_window:
            parts.append("ADDRESS_WINDOW (recipient): " + " | ".join(self.address_window[:6]))
        if self.info_column:
            parts.append("INFO_COLUMN (sender details): " + " | ".join(self.info_column[:6]))
        if self.reference_line:
            parts.append("REFERENCE_LINE: " + " | ".join(self.reference_line[:4]))
        return "\n".join(parts)


def group_lines(words: List[Word]) -> List[Line]:
    """
    Groups words into lines: words with the same vertical center (within half
    a line height) form a row; a row is split where the horizontal gap
    between neighbouring words exceeds 60 pt (an address-window line and an
    info-column line at almost the same height stay separate lines), and
    each line's text is in reading order.
    """
    rows: List[List[Word]] = []
    for word in sorted(words, key=lambda w: ((w.y0 + w.y1) / 2, w.x0)):
        if word.cell or not word.text.strip():
            continue
        center = (word.y0 + word.y1) / 2
        for row in rows[-3:]:
            top, bottom = min(w.y0 for w in row), max(w.y1 for w in row)
            if abs((top + bottom) / 2 - center) <= 0.5 * max(bottom - top, word.y1 - word.y0):
                row.append(word)
                break
        else:
            rows.append([word])
    segments: List[List[Word]] = []
    for row in rows:
        ordered = sorted(row, key=lambda w: w.x0)
        segments.append([ordered[0]])
        for word in ordered[1:]:
            if word.x0 - max(w.x1 for w in segments[-1]) >= 60:
                segments.append([])
            segments[-1].append(word)
    lines = [Line(min(w.x0 for w in s), min(w.y0 for w in s), max(w.x1 for w in s), max(w.y1 for w in s),
                  " ".join(w.text for w in s)) for s in segments]
    return sorted(lines, key=lambda l: ((l.y0 + l.y1) / 2, l.x0))


def analyze(first: Optional[PageText], last: Optional[PageText] = None) -> LayoutHints:
    """
    Detects DIN 5008 regions on the first page (closing area on the last page).

    Args:
        first: First page of the logical document (with word boxes).
        last: Last page (closing / signature area).

    Returns:
        LayoutHints (empty if no boxes are available).
    """
    hints = LayoutHints()
    if first is None or not first.words:
        return hints
    width, height = first.width or 595.0, first.height or 842.0
    lines = group_lines(first.words)
    for line in lines:
        rx0, rx1, ry = line.x0 / width, line.x1 / width, (line.y0 + line.y1) / 2 / height
        if ry < LETTERHEAD_Y_MAX:
            hints.letterhead.append(line.text)
        elif rx0 >= INFO_COLUMN_X0 and ry < INFO_COLUMN_Y_MAX:
            hints.info_column.append(line.text)
        if WINDOW_Y[0] <= ry <= WINDOW_Y[1] and WINDOW_X[0] <= rx0 <= WINDOW_X0_MAX and rx1 <= WINDOW_X[1] \
                and not REFERENCE_RE.search(line.text) and not SALUTATION_RE.match(line.text) \
                and not NON_ADDRESS_RE.search(line.text):
            hints.address_window.append(line)  # type: ignore[arg-type]
    window: List[Line] = sorted(hints.address_window, key=lambda l: l.y0)  # type: ignore[arg-type]
    hints.address_window = []
    if window:
        median = statistics.median(l.height for l in window)
        clusters: List[List[Line]] = [[window[0]]]
        for line in window[1:]:
            if line.y0 - clusters[-1][-1].y1 > 1.6 * median:
                clusters.append([])
            clusters[-1].append(line)
        is_address = [any(ZIP_CITY_RE.search(l.text) or ADDRESS_MARKER_RE.match(l.text) for l in c) for c in clusters]
        if any(is_address):
            block = [l for l in clusters[max(i for i, ok in enumerate(is_address) if ok)] if l.height <= 1.4 * median]
            if block and (block[0].height < 0.85 * median or re.search(r"\s[·•|]\s|,\s*\d{5}\s", block[0].text)) \
                    and len(block) > 2:
                hints.return_address = block[0].text
                block = block[1:]
            texts = [l.text for l in block]
            if 2 <= len(texts) <= 9:
                hints.address_window = texts
    for i, line in enumerate(lines):
        if REFERENCE_RE.search(line.text):
            hints.reference_line.append(line.text)
            if i + 1 < len(lines) and lines[i + 1].y0 - line.y1 < 2 * line.height:
                hints.reference_line.append(lines[i + 1].text)
    closing_page = last or first
    if closing_page.words:
        h = closing_page.height or 842.0
        hints.closing = [l.text for l in group_lines(closing_page.words) if (l.y0 + l.y1) / 2 / h > 0.55][-8:]
    return hints


def _tokens(text: str) -> List[str]:
    """Significant normalized tokens."""
    return [t for t in normalize_text(text).split() if len(t) >= 3]


def contained(value: Optional[str], texts: List[str]) -> bool:
    """True if most significant tokens of value appear in the texts."""
    tokens = _tokens(value or "")
    if not tokens:
        return False
    hay = set(_tokens(" ".join(texts)))
    return sum(1 for t in tokens if t in hay) / len(tokens) >= 0.6


def party_name(party: Optional[Dict[str, Any]]) -> Optional[str]:
    """Company or name of a party."""
    if not isinstance(party, dict):
        return None
    return party.get("company") or party.get("name")


def recipient_from_window(window: List[str]) -> Optional[Dict[str, str]]:
    """Parses the recipient (name/company, street, zip, city) from address-window lines."""
    from fluxsense.stages.identity import COMPANY_MARKERS
    lines = [l.strip() for l in window if l.strip() and not TITLE_ONLY_RE.match(l)]
    while lines and not_a_party(lines[0]):
        lines.pop(0)  # info-column labels / postal marks above the name
    if not lines:
        return None
    lines = [re.sub(r"^(herrn?|frau|firma)\s+", "", l, flags=re.IGNORECASE) for l in lines]
    if never_party_line(lines[0]) or ZIP_CITY_RE.match(lines[0]):
        return None  # the window shows no name line (only street / city): no recipient from the geometry
    party: Dict[str, str] = {"company" if COMPANY_MARKERS.search(lines[0]) else "name": lines[0]}
    zip_index = next((i for i, l in enumerate(lines) if ZIP_CITY_RE.search(l)), None)
    if zip_index is not None:
        match = re.search(r"\b(\d{5})\s+(.+)$", lines[zip_index])
        if match:
            party["zip_code"], party["city"] = match.group(1), match.group(2).strip()
        if zip_index >= 1 and zip_index - 1 > 0:
            party["street"] = lines[zip_index - 1]
    return party


def sender_candidate(hints: LayoutHints) -> Optional[str]:
    """Sender from the return-address line (first segment), else the first letterhead line."""
    if hints.return_address:
        first = re.split(r"\s[·•|–-]\s|,", hints.return_address)[0].strip()
        if len(_tokens(first)) >= 1 and not never_party_line(first):
            return first
    for line in hints.letterhead:
        if len(_tokens(line)) >= 1 and not REFERENCE_RE.search(line) and not never_party_line(line) \
                and not is_name_fragment(line):
            return line.strip()
    return None


def never_party_line(line: str) -> bool:
    """Lines that are never a party name: info-column labels, postal marks, document titles, street/city only, departments."""
    text = re.sub(r"^[#\s]+", "", line)
    return not_a_party(text) or is_document_title(text) or is_address_only(text) or is_department(text)


GEOMETRY_CONFLICT: str = "layout.geometry_conflict"


def _party_ok(party: Optional[Dict[str, Any]]) -> bool:
    """A party whose company/name passes the plausibility gate (a bare surname alone does not)."""
    if not isinstance(party, dict) or not party_name(party):
        return False
    if is_name_fragment(party_name(party)):
        return False  # a single layout word ('Velmora') may be a fragment of the real name
    company, name = party.get("company"), party.get("name")
    if company and party_problem(company):
        return False
    return not (name and party_problem(name, person=not company))


def decide_party(model: Optional[Dict[str, Any]], geometry: Optional[Dict[str, Any]], text: str) -> str:
    """
    What the geometry may do with a party the model extracted.

    Returns:
        'keep' (geometry unusable or nothing to do), 'agree' (both name the same party), 'fill' (model empty),
        'replace' (the model value is implausible or not printed) or 'conflict' (both plausible but different:
        the model value is kept and flagged).
    """
    if not _party_ok(geometry):
        return "keep"
    model_name, geometry_name = party_name(model), party_name(geometry)
    if not model_name:
        return "fill"
    if contained(model_name, [geometry_name]) or contained(geometry_name, [model_name]):
        return "agree"
    if _party_ok(model) and (not text or contained(model_name, [text])):
        return "conflict"
    return "replace"


def swapped_roles(meta: Dict[str, Any], hints: LayoutHints) -> bool:
    """
    The model swapped the roles: its sender is the name in the address
    window and its recipient is the letterhead / return-address organisation
    (both regions agree, so swapping is no replacement).
    """
    sender = meta.get("sender") if isinstance(meta.get("sender"), dict) else {}
    recipient = meta.get("recipient") if isinstance(meta.get("recipient"), dict) else {}
    letterhead = sender_candidate(hints)
    if not (party_name(sender) and party_name(recipient) and hints.address_window and letterhead):
        return False
    return contained(party_name(sender), hints.address_window) and contained(letterhead, [party_name(recipient)]) \
        and not contained(party_name(recipient), hints.address_window)


def enforce_parties(meta: Dict[str, Any], hints: LayoutHints, window_recipient: bool = True, letterhead_sender: bool = True,
                    text: str = "", conflicts: Optional[List[Tuple[str, str, str]]] = None) -> List[str]:
    """
    Validates sender/recipient against the letter geometry. Salutations and
    greetings are never parties. The geometry never silently replaces a
    plausible, printed model value: agreement keeps it, a conflict
    keeps the model value and is reported for review, and the geometry only
    fills an empty or implausible party. The address window only ever yields
    the recipient, the letterhead / return address only the sender, and only
    for the types whose policy allows it (catalog.contract.geometry_policy).

    Args:
        meta: meta_header (modified in place).
        hints: Layout hints.
        window_recipient: The address window may set the recipient.
        letterhead_sender: The letterhead / return address may set the sender.
        text: Text of the document's pages (is the model value printed?).
        conflicts: Receives (path, model value, geometry value) of geometry/model conflicts.

    Returns:
        Notes about the changes.
    """
    notes: List[str] = []
    for key in ("sender", "recipient"):
        party = meta.get(key)
        if isinstance(party, dict):
            for sub in ("company", "name"):
                if party.get(sub) and SALUTATION_RE.match(str(party[sub])):
                    notes.append(f"{key}.{sub} {party.pop(sub)!r} removed (salutation/greeting is not a party)")
    if window_recipient and letterhead_sender and swapped_roles(meta, hints):
        meta["sender"], meta["recipient"] = meta.get("recipient"), meta.get("sender")
        notes.append(f"sender/recipient swapped: the address window shows {party_name(meta['recipient'])!r}, the letterhead "
                     f"{party_name(meta['sender'])!r}")
    candidates: List[Tuple[str, Optional[Dict[str, Any]], str]] = []
    if window_recipient and hints.address_window:
        recipient = meta.get("recipient") if isinstance(meta.get("recipient"), dict) else {}
        if not contained(party_name(recipient), hints.address_window):
            candidates.append(("recipient", recipient_from_window(hints.address_window), "address window"))
    allowed = hints.letterhead + hints.info_column + hints.closing + ([hints.return_address] if hints.return_address else [])
    if letterhead_sender and hints.address_window and allowed:
        sender = meta.get("sender") if isinstance(meta.get("sender"), dict) else {}
        name = party_name(sender)
        if not name or not contained(name, allowed):
            candidate = sender_candidate(hints)
            if candidate and not contained(candidate, hints.address_window):
                candidates.append(("sender", {"company": candidate}, "letterhead/return address"))
    for key, geometry, source in candidates:
        current = meta.get(key) if isinstance(meta.get(key), dict) else {}
        decision = decide_party(current, geometry, text)
        if decision in ("fill", "replace") and geometry:
            notes.append(f"{key} {party_name(current)!r} -> {party_name(geometry)!r} ({source})")
            meta[key] = geometry
        elif decision == "conflict" and geometry:
            notes.append(f"{key} {party_name(current)!r} kept; {source} shows {party_name(geometry)!r} (geometry/model conflict)")
            if conflicts is not None:
                sub = "company" if current.get("company") else "name"
                conflicts.append((f"meta_header.{key}.{sub}", party_name(current), party_name(geometry)))
    return notes
