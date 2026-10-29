# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_layout.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    DIN 5008 regions from word boxes (address window with
                return-address line, letterhead, reference line) and party
                validation (salutations are never parties).
------------------------------------------------------------------------------
"""

import random
from typing import Any, Dict, List

from fluxsense.extract.base import PageText, Word
from fluxsense.layout.din5008 import analyze, enforce_parties
from fluxsense.testdata import build_letter


def line(x: float, y: float, text: str, size: float = 10.0) -> List[Word]:
    """Words of one line at x/y (points) with a font size."""
    words, cursor = [], x
    for token in text.split():
        width = len(token) * size * 0.5
        words.append(Word(cursor, y, cursor + width, y + size, token))
        cursor += width + size * 0.3
    return words


LETTER = build_letter(random.Random("din-letter-1"))  # generated DIN 5008 letter (seed recorded)


def din_page(skew: float = 0.0) -> PageText:
    """The generated DIN 5008 form-B letter page (A4 in points, native word boxes), optionally shifted."""
    page = LETTER.pages[0].page_text(1)
    page.words = [Word(w.x0, w.y0 + skew, w.x1, w.y1 + skew, w.text) for w in page.words]
    return page


def test_regions_detected_and_tolerant_to_shift() -> None:
    """Address window, return address, letterhead and reference line; a 15 pt shift (skewed scan) still works."""
    sender, recipient, reference = LETTER.truth["sender"], LETTER.truth["recipient"], LETTER.truth["doc_number"]
    zip_city = next(line for line in LETTER.pages[0].lines() if line.startswith(LETTER.facts["recipient_zip"]))
    for skew in (0.0, 15.0):
        hints = analyze(din_page(skew))
        assert hints.return_address and hints.return_address.startswith(sender)
        assert recipient in hints.address_window and zip_city in hints.address_window
        assert any(sender in l for l in hints.letterhead)
        assert any(f"Unser Zeichen: {reference}" in l for l in hints.reference_line)
        assert "ADDRESS_WINDOW (recipient)" in hints.prompt_block()


def test_salutation_is_never_a_party_and_window_wins() -> None:
    """The greeting ('Sehr geehrte Frau ...,') is removed; the recipient comes from the address window; the sender from
    the letterhead."""
    meta = {"sender": {"company": "CARDPAY3D"}, "recipient": {"name": LETTER.facts["greeting"]}}
    notes = enforce_parties(meta, analyze(din_page()), text="x")  # the model values are not printed on the page
    assert meta["recipient"]["name"] == LETTER.truth["recipient"] and meta["recipient"]["zip_code"] == LETTER.facts["recipient_zip"]
    assert meta["sender"] == {"company": LETTER.truth["sender"]}
    assert len(notes) == 3


def test_correct_parties_are_kept() -> None:
    """Parties that match the regions are not touched."""
    meta = {"sender": {"company": LETTER.truth["sender"]}, "recipient": {"name": LETTER.truth["recipient"]}}
    assert enforce_parties(meta, analyze(din_page())) == []


def test_no_window_without_address() -> None:
    """A table or heading in the window zone is no address block."""
    words = line(62, 150, "01.09. 01.09. REWE -7,35") + line(62, 165, "Buchung Valuta Vorgang")
    assert analyze(PageText(page=1, words=words)).address_window == []


def _hints(window: List[str], letterhead: List[str], return_address: str = "") -> Any:
    """Layout hints of a synthetic letter page."""
    from fluxsense.layout.din5008 import LayoutHints
    return LayoutHints(address_window=window, letterhead=letterhead, return_address=return_address or None)


def test_geometry_never_replaces_a_plausible_model_party() -> None:
    """Synthetic fixtures: junk geometry (heading with date, label word, code string, bare surname) never
    replaces the model value; a plausible conflict keeps the model value and is reported; empty parties are filled."""
    from fluxsense.layout.din5008 import decide_party
    text = "Beispiel Prüftechnik GmbH\nErika Beispiel\nHans Probe\nMusterwerk AG"
    cases = [
        ({"sender": {"company": "Beispiel Prüftechnik GmbH"}}, _hints(["Erika Beispiel", "00345 Musterstadt"],
                                                                     ["Inspection sheet - 2031-04-17"])),
        ({"sender": {"name": "Hans Probe"}}, _hints(["Erika Beispiel", "00345 Musterstadt"], ["Ihre Kundennummer"])),
        ({"recipient": {"name": "Erika Beispiel"}}, _hints(["Q 12 7AB4 902C D3 E117 AF05", "00345 Musterstadt"], [])),
        ({"recipient": {"name": "Erika Beispiel"}}, _hints(["Meier", "Lindenweg 5", "00345 Musterstadt"], [])),
    ]
    for meta, hints in cases:
        before = {k: dict(v) for k, v in meta.items()}
        enforce_parties(meta, hints, text=text)
        assert {k: meta[k] for k in before} == before, (before, meta)
    meta = {"recipient": {"name": "Hans Probe"}}
    conflicts: List[Any] = []
    notes = enforce_parties(meta, _hints(["Erika Beispiel", "Lindenweg 5", "00345 Musterstadt"], []), text=text,
                            conflicts=conflicts)
    assert meta["recipient"] == {"name": "Hans Probe"} and "geometry/model conflict" in notes[0]
    assert conflicts == [("meta_header.recipient.name", "Hans Probe", "Erika Beispiel")]
    meta = {}
    enforce_parties(meta, _hints(["Erika Beispiel", "Lindenweg 5", "00345 Musterstadt"], ["Musterwerk AG"]), text=text)
    assert meta["recipient"]["name"] == "Erika Beispiel" and meta["sender"] == {"company": "Musterwerk AG"}
    assert decide_party({"name": "Erika Beispiel"}, {"name": "Frau Erika Beispiel"}, text) == "agree"
    assert decide_party({"name": "Ihre Kundennummer"}, {"name": "Erika Beispiel"}, text) == "replace"


def test_role_mapping_by_type() -> None:
    """The window is the recipient of letters/invoices/statements only; forms keep their filler; enclosures, notes and
    certificates get no window geometry (certificates may use the letterhead)."""
    from fluxsense.catalog.contract import geometry_policy
    assert geometry_policy("OFFICIAL_LETTER") == geometry_policy("INVOICE") == geometry_policy("BANK_STATEMENT") == (True, True)
    assert geometry_policy("CERTIFICATE") == (False, True)
    assert geometry_policy("APPLICATION") == (True, False) and geometry_policy("APPLICATION", cover_sheet=True) == (True, True)
    assert geometry_policy("OTHER") == geometry_policy("NOTE") == (False, False)
    meta = {"sender": {"name": "Hans Probe"}}
    window, letterhead = geometry_policy("APPLICATION")
    enforce_parties(meta, _hints(["Musterkasse AG", "00345 Musterstadt"], ["Musterwerk AG"]), window, letterhead, text="x")
    assert meta["sender"] == {"name": "Hans Probe"} and meta["recipient"]["company"] == "Musterkasse AG"


def test_letterhead_fragment_is_never_a_sender() -> None:
    """A single letterhead word without legal form ('Velmora', an OCR fragment above 'Deutsche Beispielkasse') is not taken
    as sender; acronyms and names with a legal form still are (synthetic)."""
    from fluxsense.layout.din5008 import decide_party, sender_candidate
    assert sender_candidate(_hints(["Erika Beispiel"], ["Velmora", "Beispielkasse Nord AG"])) == "Beispielkasse Nord AG"
    assert sender_candidate(_hints(["Erika Beispiel"], ["BKN", "Beispielkasse Nord AG"])) == "BKN"
    assert decide_party({}, {"company": "Velmora"}, "") == "keep"
    meta: Dict[str, Any] = {}
    enforce_parties(meta, _hints(["Erika Beispiel", "Lindenweg 5", "00345 Musterstadt"], ["Velmora"]), text="x")
    assert "sender" not in meta and meta["recipient"]["name"] == "Erika Beispiel"


def test_swapped_roles_are_swapped_back() -> None:
    """The model took the addressee as sender and the letterhead organisation as recipient: both regions agree, so the
    roles are swapped (no replacement, no conflict)."""
    hints = _hints(["Familie", "Hans Probe", "Lindenweg 5", "00345 Musterstadt"], ["Musterkasse AG"])
    meta = {"sender": {"name": "Hans Probe"}, "recipient": {"company": "Musterkasse AG", "name": "Erika Beispiel"}}
    conflicts: List[Any] = []
    notes = enforce_parties(meta, hints, text="Hans Probe\nMusterkasse AG\nErika Beispiel", conflicts=conflicts)
    assert meta["sender"]["company"] == "Musterkasse AG" and meta["recipient"]["name"] == "Hans Probe"
    assert notes[0].startswith("sender/recipient swapped") and conflicts == []


def test_window_and_info_column_lines_stay_apart() -> None:
    """An info-column word slightly above a window line is not merged into it; line text is in reading order."""
    from fluxsense.layout.din5008 import group_lines
    words = [Word(360, 131, 420, 140, "Rech-Nr.:"), Word(424, 131, 470, 140, "4711"), Word(62, 133, 90, 143, "Herrn"),
             Word(120, 158, 160, 168, "Probe"), Word(62, 158, 100, 168, "Hans")]
    assert [l.text for l in group_lines(words)] == ["Rech-Nr.: 4711", "Herrn", "Hans Probe"]
