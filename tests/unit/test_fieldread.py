# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_fieldread.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Form fields read by their printed label (§12) on generated
                scans of hand-filled forms (handwriting-like font): labels
                under the entry line, an inline 'Ort, den ___', a header
                label with the entry below, empty fields, an ambiguous
                '1|10|26' date; the readings become the FORM FIELDS block and
                the 'fieldread' source, map to the schema and win over a
                differing full-page value. Negative: a printed letter does
                not trigger. The vision model is mocked.
------------------------------------------------------------------------------
"""

import random
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from fluxsense.extract import build_text_layer
from fluxsense.extract.base import PageText, Word
from fluxsense.extract.fieldread import (CHECK_AMBIGUOUS, CHECK_DISAGREE, EMPTY, SOURCE_FIELDREAD, FieldReader,
                                         date_candidates, entry_region, find_labels)
from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from fluxsense.testdata import build_letter
from fluxsense.testdata.layout import Page, render, signature
from fluxsense.testdata.parties import person
from tests.conftest import FakeBackend

LEFT = 62.0


def _scan(page: Page, target: Path, dpi: int = 300) -> Path:
    digital = render([page], target.with_suffix(".digital.pdf"))
    with pymupdf.open(digital) as src, pymupdf.open() as out:
        pix = src[0].get_pixmap(dpi=dpi)
        new = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
        out.save(target)
    return target


def _form(rng: random.Random, signing: str, empty_phone: bool = True) -> Tuple[Page, Dict[str, Any]]:
    """A landlord confirmation in three label styles: entry line above its label, inline 'Ort, den', header label with
    the entries below; handwriting-like entries; the phone field is left empty."""
    landlord, mover, flat = person(rng), person(rng), person(rng).address
    move_in = "3.2.24"
    page = Page()
    page.text(LEFT, 60, "Bestätigung des Wohnungsgebers", 15, bold=True)
    # style 1: entry written on a line, the printed label below it
    page.text(LEFT + 4, 120, move_in, 12, hand=True)
    page.rule(LEFT, 124, 300)
    page.text(LEFT, 134, "Einzug am", 7.5)
    page.text(LEFT + 4, 170, "EG links", 12, hand=True)
    page.rule(LEFT, 174, 300)
    page.text(LEFT, 184, "Geschoss / Lage", 7.5)
    if not empty_phone:
        page.text(LEFT + 4, 220, "030 23125042", 12, hand=True)
    page.rule(LEFT, 224, 300)
    page.text(LEFT, 234, "Telefon", 7.5)
    # style 3: a header label, the entries below it
    page.text(LEFT, 280, "Name und Anschrift des Wohnungsgebers lauten:", 9)
    page.text(LEFT + 4, 300, landlord.name, 12, hand=True)
    page.text(LEFT + 4, 318, f"{landlord.address.street}, {landlord.address.zip_code} {landlord.address.city}", 12, hand=True)
    page.text(LEFT, 360, "Folgende Person(en) ist/sind eingezogen:", 9)
    page.text(LEFT + 4, 380, mover.name, 12, hand=True)
    # style 2: inline 'Ort, den ___'
    page.text(LEFT, 460, f"{landlord.address.city}, den", 10)
    page.text(LEFT + 120, 460, signing, 12, hand=True)
    signature(page, 330, 455, 7)
    page.rule(320, 466, 533)
    page.text(330, 476, "Unterschrift des Wohnungsgebers", 7.5)
    facts = {"landlord": landlord.name, "landlord_street": landlord.address.street, "mover": mover.name,
             "move_in": move_in, "city": landlord.address.city}
    return page, facts


def _ovis_items(page: Page, printed_only: bool = True) -> List[Dict[str, Any]]:
    """The OCR layout: printed runs exactly, handwriting garbled (as the real OCR does)."""
    items = []
    for run in page.runs:
        content = run.text if not run.hand else ("71806" if "/" in run.text else run.text.replace("e", "c")[:6])
        if run.hand and printed_only:
            continue
        items.append({"bbox": [round(run.left() / page.width * 1000), round((run.y - run.size) / page.height * 1000),
                               round((run.left() + run.width()) / page.width * 1000), round((run.y + 2) / page.height * 1000)],
                      "category": "text", "content": content})
    return items


def _vision(facts: Dict[str, Any], signing: str, seen: List[LLMRequest]) -> Any:
    """Mocked crop reader: answers by the printed label of the request (never sees the full page)."""
    answers = {"Einzug am": (facts["move_in"], "2024-02-03"), "Geschoss / Lage": ("EG links", None), "Telefon": (EMPTY, None),
               "Name und Anschrift des Wohnungsgebers lauten:": (f"{facts['landlord']}\n{facts['landlord_street']}", None),
               "Folgende Person(en) ist/sind eingezogen:": (facts["mover"], None)}

    def read(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Fieldread checkbox"):
            return {"state": "no checkbox"}
        seen.append(request)
        label = request.prompt.split('The printed label of the field is: "')[1].split('"')[0]
        if label.endswith(", den"):
            iso = {"7/8/26": "2026-08-07", "1|10|26": "2026-10-01"}.get(signing)
            return {"entry": signing, "iso": iso}
        entry, iso = answers.get(label, (EMPTY, None))
        return {"entry": entry, "iso": iso}
    return read


def test_label_lexicon_and_regions_cover_the_entries() -> None:
    """Labels are found with their kind; the region of a label under its line covers the line above, the inline
    region covers the entry to the right, the header region covers the lines below."""
    page, _ = _form(random.Random("form-1"), "7/8/26")
    words = [Word(r.left(), r.y - r.size, r.left() + r.width(), r.y + 2, r.text) for r in page.runs if not r.hand]
    labels = {f.label: f for f in find_labels(PageText(page=1, words=words))}
    assert {"Einzug am", "Geschoss / Lage", "Telefon", "Name und Anschrift des Wohnungsgebers lauten:",
            "Folgende Person(en) ist/sind eingezogen:"} <= set(labels)
    assert labels["Einzug am"].kind == "date" and labels["Folgende Person(en) ist/sind eingezogen:"].kind == "persons"
    inline = next(f for name, f in labels.items() if name.endswith(", den"))
    assert inline.kind == "place_date" and inline.target == "doc_date"
    under = entry_region(labels["Einzug am"], 595, 842, below=False)
    assert under[1] <= 120 - 12 and under[3] >= 134       # the handwritten line above the label is inside
    beside = entry_region(inline, 595, 842, below=False)
    assert beside[2] >= LEFT + 120 + 40                    # the inline entry right of 'den' is inside
    header = entry_region(labels["Name und Anschrift des Wohnungsgebers lauten:"], 595, 842, below=True)
    assert header[3] >= 318                                # both entry lines below the header label


def test_date_candidates() -> None:
    """Bars may be separators or ones; slashes and dots are separators; future and invalid readings are dropped."""
    assert date_candidates("7/8/26", "2026-08-07", date(2026, 10, 30)) == ["2026-08-07"]
    assert set(date_candidates("1|10|26", "2026-10-01", date(2026, 10, 30))) == {"2026-10-01", "2026-10-11"}
    assert date_candidates("32.02.24", None, date(2026, 10, 30)) == []


def _pipeline(tmp_path: Path, signing: str, settings: Any, stage2_doc_date: str) -> Tuple[Any, Dict[str, Any], List[LLMRequest]]:
    page, facts = _form(random.Random("form-pipe"), signing)
    pdf = _scan(page, tmp_path / "form.pdf")
    seen: List[LLMRequest] = []
    vision = _vision(facts, signing, seen)
    stage2: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("OCR ovisocr3"):
            return _ovis_items(page, printed_only=False)
        if request.label.startswith("Fieldread"):
            return vision(request)
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["CERTIFICATE"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            stage2.append(request)
            return {"meta_header": {"doc_date": {"v": stage2_doc_date, "p": 1}},
                    "bodies": {"legal_body": {"document_title": "Bestätigung des Wohnungsgebers"}}}
        return None

    settings.extract.engine = "ovisocr3"
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf)
    facts["stage2"] = stage2
    return result, facts, seen


def test_form_fields_read_mapped_and_authoritative(settings: Any, tmp_path: Path) -> None:
    """End to end: crops are read per label (never with the full-page value), the FORM FIELDS block reaches the
    extraction, the readings map to doc_date / event / persons / sender / floor, the differing full-page doc_date is
    replaced and flagged with both values, the empty phone field yields nothing."""
    result, facts, seen = _pipeline(tmp_path, "7/8/26", settings, "2018-06-07")
    ld = result.logical_documents[0]
    assert seen and all("2018-06-07" not in r.prompt for r in seen) and all(len(r.images) == 1 for r in seen)
    assert "FORM FIELDS" in facts["stage2"][0].prompt and "Einzug am: 3.2.24 -> 2024-02-03" in facts["stage2"][0].prompt
    meta, legal = ld.fields["meta_header"], ld.fields["bodies"]["legal_body"]
    assert meta["doc_date"] == "2026-08-07" and meta["sender"]["name"] == facts["landlord"]
    assert legal["certified_event"] == "MOVE_IN" and legal["event_date"] == "2024-02-03"
    assert legal["certified_persons"] == [facts["mover"]] and legal["subject_address"]["location_detail"] == "EG links"
    assert "phone" not in (meta.get("sender") or {})  # the empty phone field yields no value
    disagree = [c for c in ld.plausibility.findings if c.check_id == CHECK_DISAGREE]
    assert disagree and "2018-06-07" in disagree[0].message and "2026-08-07" in disagree[0].message
    ev = next(e for e in ld.evidence if e.path == "meta_header.doc_date")
    assert ev.matched_source == SOURCE_FIELDREAD and not ev.auto_accepted
    event = next(e for e in ld.evidence if e.path == "bodies.legal_body.event_date")
    assert event.provenance == "label"


def test_ambiguous_bar_date_is_flagged_with_candidates(settings: Any, tmp_path: Path) -> None:
    """'1|10|26': the bars may be separators or ones - both readings are named, the field is flagged."""
    result, _, _ = _pipeline(tmp_path, "1|10|26", settings, "2026-10-01")
    ld = result.logical_documents[0]
    ambiguous = [c for c in ld.plausibility.findings if c.check_id == CHECK_AMBIGUOUS]
    assert ambiguous and "2026-10-01" in ambiguous[0].message and "2026-10-11" in ambiguous[0].message
    ev = next(e for e in ld.evidence if e.path == "meta_header.doc_date")
    assert not ev.auto_accepted and any("ambiguous" in r for r in ev.review_reasons)


def test_printed_letter_does_not_trigger(settings: Any, tmp_path: Path) -> None:
    """Negative: a digital letter (entries printed as text) gets no field reading."""
    doc = build_letter(random.Random("fieldread-letter"))
    pdf = render(doc.pages, tmp_path / "letter.pdf")
    from fluxsense.extract import build_text_layer as build
    layer = build(pdf, settings, "native")
    seen: List[LLMRequest] = []
    assert FieldReader(FakeBackend(lambda r: seen.append(r)), settings).run(pdf, layer.pages) == [] and seen == []


def _checkbox_form(rng: random.Random, move_in: str) -> Tuple[Page, Dict[str, Any]]:
    """Ticked inline 'Einzug am' with a handwritten date, directly above an unticked, empty 'Auszug am'; a name+address
    header block; a short floor entry the OCR reads too."""
    landlord, mover = person(rng), person(rng)
    page = Page()
    page.text(LEFT, 60, "Wohnungsgeberbestätigung", 15, bold=True)
    page.text(LEFT, 120, "[x] Einzug am", 10)
    page.text(LEFT + 110, 120, move_in, 12, hand=True)
    page.text(LEFT, 140, "[ ] Auszug am", 10)
    page.text(LEFT, 180, "Geschoss / Lage:", 10)
    page.text(LEFT + 110, 180, "DG", 12, hand=True)
    page.text(LEFT, 220, "Telefon:", 10)
    page.text(LEFT, 260, "Name und Anschrift des Wohnungsgebers lauten:", 9)
    page.text(LEFT + 4, 280, landlord.name, 12, hand=True)
    page.text(LEFT + 4, 298, f"{landlord.address.street}, {landlord.address.zip_code} {landlord.address.city}", 12, hand=True)
    page.text(LEFT, 360, "Folgende Person(en) ist/sind eingezogen:", 9)
    page.text(LEFT + 4, 380, mover.name, 12, hand=True)
    return page, {"landlord": landlord, "mover": mover.name}


def _checkbox_items(page: Page) -> List[Dict[str, Any]]:
    """OCR layout of the checkbox form: boxes as glyphs, the short floor entry read, other handwriting garbled."""
    items = []
    for run in page.runs:
        content = run.text.replace("[x]", "☒").replace("[ ]", "☐")
        if run.hand and run.text != "DG":
            content = "".join(chr(ord(c) + 1) if c.isalpha() else c for c in run.text)[:8]
        items.append({"bbox": [round(run.left() / page.width * 1000), round((run.y - run.size) / page.height * 1000),
                               round((run.left() + run.width()) / page.width * 1000), round((run.y + 2) / page.height * 1000)],
                      "category": "text", "content": content})
    return items


def _checkbox_run(tmp_path: Path, settings: Any, move_in: str, answer_in: Tuple[str, Optional[str]]) -> Tuple[Any, Dict[str, Any], List[LLMRequest]]:
    page, facts = _checkbox_form(random.Random("checkbox-form"), move_in)
    pdf = _scan(page, tmp_path / "boxes.pdf")
    landlord = facts["landlord"]
    block = f"{landlord.name}\n{landlord.address.street}, {landlord.address.zip_code} {landlord.address.city}"
    answers = {"Einzug am": answer_in, "Geschoss / Lage:": ("DG", None), "Telefon:": (EMPTY, None),
               "Name und Anschrift des Wohnungsgebers lauten:": (block, None),
               "Folgende Person(en) ist/sind eingezogen:": (facts["mover"], None)}
    seen: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("OCR ovisocr3"):
            return _checkbox_items(page)
        if request.label.startswith("Fieldread checkbox"):
            return {"state": "no checkbox"}
        if request.label.startswith("Fieldread"):
            seen.append(request)
            label = request.prompt.split('The printed label of the field is: "')[1].split('"')[0]
            entry, iso = answers.get(label, (EMPTY, None))
            return {"entry": entry, "iso": iso}
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["CERTIFICATE"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return {"meta_header": {}, "bodies": {"legal_body": {"document_title": "Wohnungsgeberbestätigung"}}}
        return None

    settings.extract.engine = "ovisocr3"
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf)
    return result, facts, seen


def test_unticked_option_is_ignored_and_inline_crop_stays_on_its_line(settings: Any, tmp_path: Path) -> None:
    """The unticked 'Auszug am' directly below the ticked 'Einzug am' is never read; the ticked one decides MOVE_IN with
    its own date; the inline crop of 'Einzug am' does not reach the row above or below."""
    result, facts, seen = _checkbox_run(tmp_path, settings, "3.2.24", ("3.2.24", "2024-02-03"))
    labels = [r.prompt.split('field is: "')[1].split('"')[0] for r in seen]
    assert "Einzug am" in labels and not any(label.startswith("Auszug") for label in labels)
    legal = result.logical_documents[0].fields["bodies"]["legal_body"]
    assert legal["certified_event"] == "MOVE_IN" and legal["event_date"] == "2024-02-03"
    from fluxsense.extract.fieldread import FieldReader as Reader
    layer = build_text_layer(_scan(_checkbox_form(random.Random("checkbox-form"), "3.2.24")[0], tmp_path / "again.pdf"),
                             settings, "ovisocr3", backend=FakeBackend(lambda r: _checkbox_items(
                                 _checkbox_form(random.Random("checkbox-form"), "3.2.24")[0])))
    fields = Reader(FakeBackend(lambda r: {"entry": EMPTY}), settings).run(tmp_path / "again.pdf", layer.pages)
    einzug = next(f for f in fields if f.label == "Einzug am")
    auszug_top = (140 - 10) / 842 * 842  # the row below starts at its text top (baseline 140, 10 pt)
    assert einzug.ticked is True and einzug.region[1] > 62 and einzug.region[3] < auszug_top and einzug.region[0] >= einzug.bbox[2]


def test_ambiguous_handwritten_date_is_flagged_not_accepted(settings: Any, tmp_path: Path) -> None:
    """The reader returns '11/10/26' for a written '1|10|26': both readings are named, the date is not accepted."""
    result, _, _ = _checkbox_run(tmp_path, settings, "1|10|26", ("11/10/26", "2026-10-11"))
    ld = result.logical_documents[0]
    ambiguous = [c for c in ld.plausibility.findings if c.check_id == CHECK_AMBIGUOUS]
    assert ambiguous and "2026-10-01" in ambiguous[0].message and "2026-10-11" in ambiguous[0].message
    ev = next(e for e in ld.evidence if e.path == "bodies.legal_body.event_date")
    assert not ev.auto_accepted


def test_single_reading_is_never_auto_accepted_but_agreement_may_be(settings: Any, tmp_path: Path) -> None:
    """A handwritten value read only from its crop is flagged 'single reading'; the floor 'DG' that the OCR text also
    reads is an independent second reading and may be accepted; the name+address block is split."""
    result, facts, _ = _checkbox_run(tmp_path, settings, "3.2.24", ("3.2.24", "2024-02-03"))
    ld = result.logical_documents[0]
    by_path = {e.path: e for e in ld.evidence}
    name = by_path["meta_header.sender.name"]
    assert not name.auto_accepted and "handwritten form field (single reading)" in name.review_reasons
    floor = by_path["bodies.legal_body.subject_address.location_detail"]
    assert "handwritten form field (single reading)" not in floor.review_reasons and floor.auto_accepted
    sender = ld.fields["meta_header"]["sender"]
    landlord = facts["landlord"]
    assert sender["name"] == landlord.name and sender["zip_code"] == landlord.address.zip_code
    assert sender["street"] != landlord.name and landlord.address.street.startswith(sender["street"])


def test_name_with_street_only_is_not_mapped_to_street() -> None:
    """'Name, Street' without number and postcode: the name maps, nothing becomes a street."""
    from fluxsense.extract.fieldread import split_name_address
    assert split_name_address("Erika Probe, Musterweg") == ("Erika Probe", {})


def test_signature_page_with_date_lines_only_is_read(settings: Any, tmp_path: Path) -> None:
    """A scanned contract signature page with only '<Ort>, den 7/8/24' (OCR: '71804') and an empty 'Ort, den': the date
    lines alone trigger the field reading; the reading replaces the full-page misreading and stays flagged."""
    from datetime import date as day
    from fluxsense.testdata import build_contract
    doc = build_contract(random.Random("signature-page"), day(2024, 10, 1), day(2024, 8, 7), signature_text="7/8/24", rent=False)
    page = doc.pages[1]
    page.text(330, 160, "Ort, den", 11)
    pdf = _scan(page, tmp_path / "signatures.pdf")
    seen: List[LLMRequest] = []

    def items() -> List[Dict[str, Any]]:
        out = _ovis_items(page)
        for item in out:
            item["content"] = item["content"].replace("7/8/24", "71804")
        return out

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("OCR ovisocr3"):
            return items()
        if request.label.startswith("Fieldread"):
            seen.append(request)
            label = request.prompt.split('field is: "')[1].split('"')[0]
            return {"entry": "7/8/24", "iso": "2024-08-07"} if label != "Ort, den" else {"entry": EMPTY}
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["CONTRACT"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return {"meta_header": {"doc_date": {"v": "2018-04-07", "p": 1}}, "bodies": {"legal_body": {}}}
        return None

    settings.extract.engine = "ovisocr3"
    ld = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf).logical_documents[0]
    labels = sorted(r.prompt.split('field is: "')[1].split('"')[0] for r in seen)
    assert len(labels) == 2 and all(label.endswith(", den") for label in labels)
    assert ld.fields["meta_header"]["doc_date"] == "2024-08-07"
    ev = next(e for e in ld.evidence if e.path == "meta_header.doc_date")
    assert not ev.auto_accepted and "handwritten form field (single reading)" in ev.review_reasons


def _layer_with(page: Page, items: List[Dict[str, Any]], tmp_path: Path, settings: Any, name: str) -> Tuple[Path, Any]:
    pdf = _scan(page, tmp_path / f"{name}.pdf")
    return pdf, build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(lambda r: items))


def _item(page: Page, text: str, x0: float, y: float, x1: float, size: float = 10) -> Dict[str, Any]:
    return {"bbox": [round(x0 / page.width * 1000), round((y - size) / page.height * 1000), round(x1 / page.width * 1000),
                     round((y + 2) / page.height * 1000)], "category": "text", "content": text}


def test_label_item_containing_its_entry(settings: Any, tmp_path: Path) -> None:
    """'Musterstadt, den 71806' as ONE layout item: the crop covers the item (the entry is inside), the prompt names
    only the label, and an EMPTY answer is no reading (no disagreement)."""
    from fluxsense.extract.fieldread import map_fields
    page = Page()
    page.text(LEFT, 400, "Musterstadt, den", 10)
    page.text(LEFT + 90, 400, "7/8/24", 12, hand=True)
    for y, label in ((440, "Ort, Datum"), (480, "Datum:")):
        page.text(LEFT, y, label, 10)
    items = [_item(page, "Musterstadt, den 71806", LEFT, 400, LEFT + 140), _item(page, "Ort, Datum", LEFT, 440, LEFT + 50),
             _item(page, "Datum:", LEFT, 480, LEFT + 30)]
    pdf, layer = _layer_with(page, items, tmp_path, settings, "merged")
    seen: List[LLMRequest] = []
    fields = FieldReader(FakeBackend(lambda r: seen.append(r) or {"entry": EMPTY}), settings).run(pdf, layer.pages)
    merged = next(f for f in fields if f.label == "Musterstadt, den")
    assert merged.inside and merged.region[0] <= LEFT and merged.region[2] >= LEFT + 140
    prompt = next(r.prompt for r in seen if '"Musterstadt, den"' in r.prompt)
    assert "71806" not in prompt and merged.entry is None
    meta = {"meta_header": {"doc_date": "2018-06-07"}, "bodies": {}}
    _, checks = map_fields(meta, [], fields)
    assert not [c for c in checks if c.check_id == CHECK_DISAGREE and "Musterstadt" in c.message]


def test_checkbox_glyphs_and_checkbox_question(settings: Any, tmp_path: Path) -> None:
    """'□ Auszug am' is unticked and never read; 'Einzug am' without a glyph gets the checkbox question - 'yes' reads it
    as MOVE_IN, 'no' skips it."""
    from fluxsense.extract.fieldread import map_fields
    page = Page()
    page.text(LEFT + 14, 120, "Einzug am", 10)
    page.text(LEFT + 120, 120, "3.2.24", 12, hand=True)
    page.text(LEFT + 14, 140, "Auszug am", 10)
    page.text(LEFT, 180, "Telefon:", 10)
    items = [_item(page, "Einzug am", LEFT + 14, 120, LEFT + 70), _item(page, "□ Auszug am", LEFT, 140, LEFT + 70),
             _item(page, "Telefon:", LEFT, 180, LEFT + 40)]
    pdf, layer = _layer_with(page, items, tmp_path, settings, "boxes")
    for state, read in (("yes", True), ("no", False)):
        seen: List[str] = []

        def answer(request: LLMRequest, state: str = state) -> Any:
            seen.append(request.label)
            if request.label.startswith("Fieldread checkbox"):
                return {"state": state}
            return {"entry": "3.2.24", "iso": "2024-02-03"}

        fields = FieldReader(FakeBackend(answer), settings).run(pdf, layer.pages)
        labels = [label for label in seen if not label.startswith("Fieldread checkbox")]
        assert not any("Auszug" in label for label in seen)
        assert any("Einzug" in label for label in labels) is read
        doc = {"meta_header": {}, "bodies": {"legal_body": {}}}
        map_fields(doc, [], fields)
        assert (doc["bodies"]["legal_body"].get("certified_event") == "MOVE_IN") is read


def test_dot_transcribed_bar_date_keeps_its_ambiguity() -> None:
    """The reader transcribes '1|10|26' as '11.10.26': the '1' + separator reading is added, so both dates remain."""
    assert set(date_candidates("11.10.26", "2026-10-11", date(2026, 10, 30))) == {"2026-10-11", "2026-10-01"}
    assert date_candidates("07.08.2026", None, date(2026, 10, 30)) == ["2026-08-07"]


def test_numbered_persons_reach_certified_persons_and_recipient() -> None:
    """'für folgende Person(en)' with numbered lines: one person per line (empty numbers dropped) into
    certified_persons, and the first person becomes the recipient of a move confirmation; never auto-accepted alone."""
    from fluxsense.extract.fieldread import FormField, PERSONS, DATE, map_fields, parse_persons
    persons = person(random.Random("p1")).name, person(random.Random("p2")).name
    entry = f"1. {persons[0]}\n2. {persons[1]}\n3."
    assert parse_persons(entry) == list(persons)
    form = [FormField(1, "für folgende Person(en)", PERSONS, "certified_persons", (62, 300, 200, 310), entry=entry, value=entry),
            FormField(1, "Einzug am", DATE, "event.MOVE_IN", (62, 120, 120, 130), entry="3.2.24", value="2024-02-03",
                      candidates=["2024-02-03"], ticked=True)]
    doc: Dict[str, Any] = {"meta_header": {}, "bodies": {}}
    evidence: List[Any] = []
    map_fields(doc, evidence, form)
    assert doc["bodies"]["legal_body"]["certified_persons"] == list(persons)
    assert doc["meta_header"]["recipient"]["name"] == persons[0]
    assert any(e.path == "meta_header.recipient.name" and e.matched_source == SOURCE_FIELDREAD for e in evidence)
