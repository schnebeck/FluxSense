# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_handwriting.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Handwriting levels and signatures (field contract §12) on
                generated documents: the level from the signature pass and
                the probe; 'signature' reads only the signature date and
                flags no printed value; 'entries' reads form-first from the
                vision locator (lexicon as kind hint, typed entries that the
                OCR text agrees with are accepted, handwritten ones not);
                'full' reads line bands twice and accepts only agreeing
                lines; AcroForm widgets are read without vision; signature
                places get their status from the pen ink, their signer from
                printed content only (printed name, i. A., line label, form
                filler, none), and the OCR text of a scribble never reaches
                the extraction. The vision model is mocked.
------------------------------------------------------------------------------
"""

import random
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from fluxsense.extract import build_text_layer
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.extract.fieldread import FOUND_ACROFORM, FOUND_LOCATOR, FieldReader, map_fields
from fluxsense.extract.handread import DISPUTED, SOURCE_HANDREAD, HandReader, compare, handwriting_block, line_bands
from fluxsense.extract.native import read_pages
from fluxsense.grounding.acceptance import assess
from fluxsense.llm.base import LLMRequest
from fluxsense.llm.images import render_page
from fluxsense.models.result import FieldEvidence, GroundingStatus
from fluxsense.pipeline import Analyzer
from fluxsense.stages import signers
from fluxsense.stages.signatures import _parse_signatures, probe_handwriting
from fluxsense.testdata import build_contract, build_letter
from fluxsense.testdata.documents import build_form, build_permission_letter
from fluxsense.testdata.layout import Page, render, signature
from fluxsense.testdata.parties import person
from tests.conftest import FakeBackend

LEFT = 62.0
SINGLE = "handwritten form field (single reading)"


def _scan(page: Page, target: Path, dpi: int = 200) -> Path:
    digital = render([page], target.with_suffix(".digital.pdf"))
    with pymupdf.open(digital) as src, pymupdf.open() as out:
        pix = src[0].get_pixmap(dpi=dpi)
        new = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
        out.save(target)
    return target


def _item(page: Page, text: str, x0: float, y: float, x1: float, size: float = 10, y0: Optional[float] = None) -> Dict[str, Any]:
    top = y - size if y0 is None else y0
    return {"bbox": [round(x0 / page.width * 1000), round(top / page.height * 1000), round(x1 / page.width * 1000),
                     round((y + 2) / page.height * 1000)], "category": "text", "content": text}


def _items(page: Page) -> List[Dict[str, Any]]:
    """The OCR layout of a generated page: every run read exactly."""
    return [_item(page, run.text, run.left(), run.y, run.left() + run.width(), run.size) for run in page.runs]


def _label_of(request: LLMRequest) -> str:
    return request.prompt.split('The printed label of the field is: "')[1].split('"')[0]


# ------------------------------------------------------------------------------------------------------- levels


def test_signature_pass_and_probe_report_the_handwriting_level(settings: Any, tmp_path: Path) -> None:
    """The signature pass carries a valid level per page (an unknown value is dropped); the probe asks only for the
    level and a failed call leaves it unknown."""
    sigs = _parse_signatures({"pages": [{"page": 1, "blank": False, "letterhead": False, "salutation": False, "closing": True,
                                         "form": True, "handwriting": "entries"},
                                        {"page": 2, "blank": False, "letterhead": False, "salutation": False, "closing": False,
                                         "form": False, "handwriting": "scribbles"}]}, [1, 2])
    assert [s.handwriting for s in sigs] == ["entries", None]
    pdf = render([Page()], tmp_path / "blank.pdf")
    backend = FakeBackend(lambda r: {"pages": [{"page": 1, "handwriting": "full"}]})
    assert probe_handwriting(backend, settings, pdf, [1]) == {1: "full"}
    assert backend.labels() == ["Stage 1.0 handwriting p1-1"] and "handwriting" in backend.requests[0].prompt
    assert probe_handwriting(FakeBackend(lambda r: None), settings, pdf, [1]) == {}


def _typed_form(rng: random.Random, signed: bool = True) -> Tuple[Page, Any, Dict[str, Any]]:
    """The generated typed form (§2.4) with an OCR layout; signed: the signature stroke and its OCR garbage."""
    doc = build_form(rng)
    page = doc.pages[0]
    if not signed:
        page.strokes.clear()
    items = _items(page)
    if signed:
        x0 = min(x for x, _ in page.strokes[0])
        y = sum(y for _, y in page.strokes[0]) / len(page.strokes[0])
        items.append(_item(page, "Mmhvfr Iwo", x0, y + 6, x0 + 100, 12, y0=y - 10))  # the scribble read as text
    return page, doc, {"items": items}


def test_signature_level_reads_only_the_signature_date_and_flags_no_printed_value(settings: Any, tmp_path: Path) -> None:
    """A typed form with only a handwritten signature (level 'signature'): the five lexicon labels do not trigger the
    form reading - only the place/date line is read; the printed values are not flagged as handwritten single readings;
    the signature is signed, its OCR garbage is removed before the extraction, the signer is the form filler (§2.4)."""
    page, doc, layout = _typed_form(random.Random("typed-form"))
    pdf = _scan(page, tmp_path / "form.pdf")
    filler = doc.truth["sender"]
    member = doc.truth["doc_number"]
    place_date = page.runs[[r.text for r in page.runs].index("Ort, Datum:") + 1].text
    seen: List[str] = []
    stage2: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("OCR ovisocr3"):
            return layout["items"]
        if request.label.startswith("Stage 1.0 handwriting"):
            return {"pages": [{"page": 1, "handwriting": "signature"}]}
        if request.label.startswith("Fieldread"):
            seen.append(_label_of(request))
            return {"entry": place_date, "iso": doc.truth["doc_date"], "script": "typed"}
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["APPLICATION"], "page_indices": [1], "direction": "OUTBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            stage2.append(request)
            return {"meta_header": {"sender": {"name": {"v": filler, "p": 1}}, "doc_number": {"v": member, "p": 1},
                                    "doc_date": {"v": doc.truth["doc_date"], "p": 1}}, "bodies": {}}
        return None

    settings.extract.engine = "ovisocr3"
    ld = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf).logical_documents[0]
    assert ld.handwriting == {1: "signature"}
    assert seen == ["Ort, Datum:"]  # only the signature's date line, never the printed fields
    assert "[typed] Ort, Datum:" in stage2[0].prompt and "Mmhvfr" not in stage2[0].prompt
    assert not any(SINGLE in e.review_reasons for e in ld.evidence)
    date_ev = next(e for e in ld.evidence if e.path == "meta_header.doc_date")
    assert date_ev.value == doc.truth["doc_date"] and SINGLE not in date_ev.review_reasons
    assert [f.label for f in ld.form_fields] == ["Ort, Datum:"] and ld.form_fields[0].script == "typed"
    assert len(ld.signatures) == 1
    sig = ld.signatures[0]
    assert sig.status == "signed" and sig.signer_name == filler and sig.signer_source == "form_filler"
    assert sig.date == doc.truth["doc_date"] and sig.removed_ocr == ["Mmhvfr Iwo"] and sig.bbox and sig.page == 1


def test_unknown_level_keeps_the_lexicon_trigger(settings: Any, tmp_path: Path) -> None:
    """Without a level (probe failed) the lexicon trigger is unchanged: the typed form's labels are all read, and typed
    readings that the OCR text agrees with are no single readings."""
    page, doc, layout = _typed_form(random.Random("typed-form"), signed=False)
    pdf = _scan(page, tmp_path / "unknown.pdf")
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(lambda r: layout["items"]))
    values = {r.text.rstrip(":"): page.runs[i + 1].text for i, r in enumerate(page.runs[:-1]) if r.text.endswith(":")}
    seen: List[str] = []

    def answer(request: LLMRequest) -> Any:
        label = _label_of(request)
        seen.append(label)
        return {"entry": values.get(label.rstrip(":"), "EMPTY"), "script": "typed"}

    fields = FieldReader(FakeBackend(answer), settings).run(pdf, layer.pages)
    assert {"Name, Vorname:", "Straße, Hausnummer:", "Ort, Datum:"} <= set(seen)
    assert all(f.script in ("typed", "empty") for f in fields)


# ------------------------------------------------------------------------------------------------- entries


def _entries_page() -> Tuple[Page, Dict[str, Any]]:
    """A hand-filled form whose labels the lexicon mostly does not know."""
    page = Page()
    page.text(LEFT, 60, "Meldung eines Fahrzeugschadens", 14, bold=True)
    page.text(LEFT, 120, "Amtliches Kennzeichen", 8)
    page.text(LEFT + 150, 120, "MU-XY 123", 12, hand=True)
    page.text(LEFT, 160, "Farbe des Fahrzeugs", 8)
    page.text(LEFT, 200, "Ort, Datum", 8)
    page.text(LEFT + 150, 200, "Musterstadt, 07.08.2024", 11, mono=True)
    boxes = {"Amtliches Kennzeichen": (LEFT + 145, 108, LEFT + 260, 124), "Farbe des Fahrzeugs": (LEFT + 145, 148, LEFT + 260, 164),
             "Ort, Datum": (LEFT + 145, 188, LEFT + 330, 204)}
    return page, boxes


def _locator_answer(page: Page, boxes: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, Any]:
    def norm(box: Tuple[float, float, float, float]) -> List[float]:
        return [box[0] / page.width * 1000, box[1] / page.height * 1000, box[2] / page.width * 1000, box[3] / page.height * 1000]
    scripts = {"Amtliches Kennzeichen": "handwritten", "Farbe des Fahrzeugs": "empty", "Ort, Datum": "typed"}
    return {"fields": [{"label": label, "entry_box": norm(box), "kind": "text", "script": scripts[label], "ticked": None}
                       for label, box in boxes.items()]}


def test_entries_level_reads_form_first_from_the_locator(settings: Any, tmp_path: Path) -> None:
    """Level 'entries': the vision locator lists the label / entry pairs (also labels the lexicon does not know); each
    filled entry is cropped and read alone, an entry the locator and the page show empty is not read; the lexicon only
    refines the kind and gives the target ('Ort, Datum' -> place_date / doc_date). Typed readings agreeing with the OCR
    text are accepted, the handwritten single reading stays flagged."""
    page, boxes = _entries_page()
    pdf = _scan(page, tmp_path / "entries.pdf")
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(lambda r: [i for i in _items(page)
                                                                                        if "MU-XY" not in i["content"]]))
    layer.pages[0].handwriting = "entries"
    seen: List[LLMRequest] = []

    def answer(request: LLMRequest) -> Any:
        seen.append(request)
        if request.label.startswith("Fieldread locate"):
            return _locator_answer(page, boxes)
        label = _label_of(request)
        if label == "Amtliches Kennzeichen":
            return {"entry": "MU-XY 123", "script": "handwritten"}
        return {"entry": "Musterstadt, 07.08.2024", "iso": "2024-08-07", "script": "typed"}

    fields = FieldReader(FakeBackend(answer), settings).run(pdf, layer.pages)
    labels = [r.label for r in seen]
    assert labels[0] == "Fieldread locate p1" and len(seen[0].images) == 1
    read = sorted(_label_of(r) for r in seen[1:])
    assert read == ["Amtliches Kennzeichen", "Ort, Datum"]  # the empty colour field is not read
    by_label = {f.label: f for f in fields}
    assert all(f.found == FOUND_LOCATOR for f in fields)
    assert by_label["Farbe des Fahrzeugs"].empty and by_label["Amtliches Kennzeichen"].kind == "text"
    assert by_label["Ort, Datum"].kind == "place_date" and by_label["Ort, Datum"].target == "doc_date"
    assert by_label["Ort, Datum"].value == "2024-08-07" and by_label["Ort, Datum"].candidates == ["2024-08-07"]
    from fluxsense.extract.fieldread import apply_fields
    apply_fields(layer.pages, fields)
    text = layer.pages[0].sources["fieldread"]
    assert "[typed] Ort, Datum: Musterstadt, 07.08.2024 -> 2024-08-07" in text and "[handwritten] Amtliches Kennzeichen" in text
    doc: Dict[str, Any] = {"meta_header": {}, "bodies": {}}
    evidence: List[Any] = []
    map_fields(doc, evidence, fields)
    evidence.append(FieldEvidence(path="meta_header.doc_number", kind="identifier", value="MU-XY 123", page=1,
                                  status=GroundingStatus.GROUNDED, matched_source="fieldread"))
    doc["meta_header"]["doc_number"] = "MU-XY 123"
    assess(doc, evidence, layer, [1], "OTHER", [])
    by_path = {e.path: e for e in evidence}
    assert SINGLE not in by_path["meta_header.doc_date"].review_reasons
    assert SINGLE in by_path["meta_header.doc_number"].review_reasons


def test_entries_level_falls_back_to_the_lexicon(settings: Any, tmp_path: Path) -> None:
    """A failed locator call: the lexicon labels of the page are read (all of them - the page has handwritten entries)."""
    page, _ = _entries_page()
    pdf = _scan(page, tmp_path / "fallback.pdf")
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(lambda r: _items(page)))
    seen: List[str] = []

    def answer(request: LLMRequest) -> Any:
        seen.append(request.label)
        return None if request.label.startswith("Fieldread locate") else {"entry": "EMPTY"}

    fields = FieldReader(FakeBackend(answer), settings).run(pdf, layer.pages, {1: "entries"})
    assert seen[0] == "Fieldread locate p1" and [f.label for f in fields] == ["Ort, Datum"]


def test_typed_handwritten_and_acroform_readings_in_the_acceptance() -> None:
    """The same value as a typed reading the OCR text prints is accepted; as a handwritten reading it is a single
    reading; a digital form field value never is one (even without any page text)."""
    name = person(random.Random("landlord")).name
    label = "Name des Wohnungsgebers"
    for line, ocr, single in ((f"[typed] {label}: {name}", f"{label} {name}", False),
                              (f"[handwritten] {label}: {name}", f"{label} {name}", True),
                              (f"{label}: {name}", f"{label} {name}", True),
                              (f"[typed] {label}: {name}", f"{label} {name[:-1]}3", True),
                              (f"[form field] {label}: {name}", "", False)):
        page = PageText(page=1, sources={"ovis": ocr, "fieldread": line})
        fields = {"meta_header": {"sender": {"name": name}}}
        evidence = [FieldEvidence(path="meta_header.sender.name", kind="text", value=name, page=1,
                                  status=GroundingStatus.GROUNDED, matched_source="fieldread")]
        assess(fields, evidence, TextLayer(pages=[page], engine="t"), [1], "CERTIFICATE", [])
        assert (SINGLE in evidence[0].review_reasons) is single, (line, evidence[0].review_reasons)


# ------------------------------------------------------------------------------------------------------ AcroForm


def _acroform(target: Path) -> Path:
    page_model = Page()
    page_model.text(LEFT, 100, "Name des Antragstellers:", 10)
    page_model.text(LEFT, 140, "Ort, Datum:", 10)
    page_model.text(LEFT + 20, 180, "Ich bin einverstanden", 10)
    render([page_model], target)
    applicant = person(random.Random("acro")).name
    with pymupdf.open(target) as doc:
        page = doc[0]
        for name, rect, value in (("applicant", (200, 88, 450, 104), applicant), ("place_date", (200, 128, 450, 144),
                                                                                    "Musterstadt, 07.08.2024")):
            widget = pymupdf.Widget()
            widget.field_type, widget.field_name, widget.field_value = pymupdf.PDF_WIDGET_TYPE_TEXT, name, value
            widget.rect = pymupdf.Rect(*rect)
            page.add_widget(widget)
        box = pymupdf.Widget()
        box.field_type, box.field_name, box.rect = pymupdf.PDF_WIDGET_TYPE_CHECKBOX, "consent", pymupdf.Rect(LEFT, 170, LEFT + 12, 182)
        box.field_value = True
        page.add_widget(box)
        doc.save(target.with_suffix(".form.pdf"))
    return target.with_suffix(".form.pdf")


def test_acroform_widgets_are_read_without_vision(settings: Any, tmp_path: Path) -> None:
    """A digital PDF form: the widget values are the entries (no model call), labelled by the printed text next to
    them; they map like field readings and are never single readings."""
    pdf = _acroform(tmp_path / "acro.pdf")
    pages = read_pages(pdf, "native")
    backend = FakeBackend(lambda r: None)
    fields = FieldReader(backend, settings).run(pdf, pages)
    assert backend.requests == []
    by_label = {f.label: f for f in fields}
    assert set(by_label) == {"Name des Antragstellers:", "Ort, Datum:", "Ich bin einverstanden"}
    assert all(f.found == FOUND_ACROFORM and f.script == "typed" for f in fields)
    assert by_label["Ort, Datum:"].value == "2024-08-07" and by_label["Ich bin einverstanden"].ticked is True
    doc: Dict[str, Any] = {"meta_header": {}, "bodies": {}}
    evidence: List[Any] = []
    map_fields(doc, evidence, fields)
    applicant = person(random.Random("acro")).name
    assert doc["meta_header"]["doc_date"] == "2024-08-07" and doc["meta_header"]["sender"]["name"] == applicant
    from fluxsense.extract.fieldread import apply_fields
    apply_fields(pages, fields)
    assert f"[form field] Name des Antragstellers:: {applicant}" in pages[0].sources["fieldread"]
    assess(doc, evidence, TextLayer(pages=pages, engine="t"), [1], "APPLICATION", [])
    assert not any(SINGLE in e.review_reasons for e in evidence)


# ---------------------------------------------------------------------------------------------------------- full


GREETED, SIGNER = person(random.Random("hand-greeted")).first, person(random.Random("hand-signer")).first
MEETING = "12.09.2024"
MISREAD = "17.09.2024"


def _handwritten_letter() -> Tuple[Page, List[str]]:
    """A handwritten note with generated first names."""
    lines = [f"Hallo {GREETED},", "vielen Dank für die Blumen.", f"Wir sehen uns am {MEETING}", "in Musterstadt.",
             "Bis bald", SIGNER]
    page = Page()
    for n, line in enumerate(lines):
        page.text(LEFT, 120 + 34 * n, line, 14, hand=True)
    return page, lines


def test_line_bands_cut_between_text_lines(tmp_path: Path) -> None:
    """The bands of a handwritten page cover all lines and are cut in the empty rows between them."""
    page, lines = _handwritten_letter()
    pdf = _scan(page, tmp_path / "hand.pdf")
    bands = line_bands(render_page(pdf, 0, dpi=100), lines_per_band=2)
    assert len(bands) == 3 and all(a[1] <= b[0] + 1e-9 for a, b in zip(bands, bands[1:]))
    for run in page.runs:
        top, bottom = (run.y - 0.7 * run.size) / page.height, (run.y + 0.2 * run.size) / page.height
        assert any(b[0] <= top and bottom <= b[1] for b in bands), run.text


def test_compare_keeps_agreeing_lines_and_marks_the_others() -> None:
    lines, agreed, disputed = compare([f"Hallo {GREETED},", f"Wir sehen uns am {MEETING}", "ja ja ja ja ja ja ja ja ja ja"],
                                      [f"hallo {GREETED}", f"Wir sehen uns am {MISREAD}", "ja ja ja ja ja ja ja ja ja ja"])
    assert agreed == [f"Hallo {GREETED},"]
    assert (f"Wir sehen uns am {MEETING}", f"Wir sehen uns am {MISREAD}") in disputed
    assert lines[1].startswith(DISPUTED) and lines[2].startswith(DISPUTED)  # a runaway repetition is never agreed


def test_full_page_is_read_twice_and_only_agreeing_lines_are_accepted(settings: Any, tmp_path: Path) -> None:
    """Level 'full': every band is read twice with different prompts (no OCR text shown); the agreed lines and the
    '[?] A | B' lines form the 'handread' source and the HANDWRITTEN TEXT block; a value that stands only in a disputed
    line is flagged, a value of an agreed line is not."""
    page, lines = _handwritten_letter()
    pdf = _scan(page, tmp_path / "hand.pdf")
    layer = build_text_layer(pdf, settings, "ovisocr3", backend=FakeBackend(lambda r: [_item(page, "Hallc xyz", LEFT, 120, 200)]))
    layer.pages[0].handwriting = "full"
    seen: List[LLMRequest] = []

    def answer(request: LLMRequest) -> Any:
        seen.append(request)
        band = int(request.label.split("band ")[1].split()[0])
        second = request.label.endswith("B")
        chunk = lines[(band - 1) * 3:band * 3]
        if second and band == 1:
            chunk = [line.replace(MEETING, MISREAD) for line in chunk]
        return {"lines": chunk}

    readings = HandReader(FakeBackend(answer), settings).run(pdf, layer.pages)
    assert readings and len(seen) == 2 * readings[0].bands
    assert {r.prompt for r in seen if r.label.endswith("A")} != {r.prompt for r in seen if r.label.endswith("B")}
    assert all("Hallc" not in r.prompt and len(r.images) == 1 for r in seen)
    source = layer.pages[0].sources[SOURCE_HANDREAD]
    assert "in Musterstadt." in source.splitlines() and f"{DISPUTED}Wir sehen uns am {MEETING} | Wir sehen uns am {MISREAD}" in source
    assert handwriting_block(layer.pages[0]).startswith("HANDWRITTEN TEXT")
    iso = date(2024, 9, 12).isoformat()
    fields = {"meta_header": {"doc_date": iso, "sender": {"name": SIGNER}}}
    evidence = [FieldEvidence(path="meta_header.doc_date", kind="date", value=iso, page=1, status=GroundingStatus.GROUNDED),
                FieldEvidence(path="meta_header.sender.name", kind="text", value=SIGNER, page=1, status=GroundingStatus.GROUNDED)]
    assess(fields, evidence, layer, [1], "NOTE", [])
    reason = "handwritten page: not in a line two independent readings agree on"
    assert reason in evidence[0].review_reasons and reason not in evidence[1].review_reasons


# ------------------------------------------------------------------------------------------------------ signatures


def _blocks(page: Page, tmp_path: Path, name: str) -> Tuple[List[signers.SignatureBlock], PageText]:
    pdf = render([page], tmp_path / f"{name}.pdf")
    text = read_pages(pdf, "native")[0]
    return signers.find_blocks(text, render_page(pdf, 0, dpi=100)), text


def test_closing_signer_is_the_printed_name(tmp_path: Path) -> None:
    """A letter: the closing's gap holds the signature (signed), the printed name under it is the signer - never the
    organisation line above it."""
    doc = build_letter(random.Random("sig-letter"))
    blocks, _ = _blocks(doc.pages[0], tmp_path, "letter")
    out = signers.attribute(blocks, {"meta_header": {"sender": {"company": doc.truth["sender"]}}}, None)
    assert len(out) == 1 and out[0].status == "signed" and out[0].signer_name == doc.facts["contact"]
    assert out[0].signer_source == "printed_name" and out[0].role == "sender"


def test_proxy_name_is_the_signer(tmp_path: Path) -> None:
    """'i. A. <Name>' under the signature: the named person signs on behalf of the sender."""
    clerk = person(random.Random("clerk")).name
    page = Page()
    page.text(LEFT, 300, "Mit freundlichen Grüßen", 10)
    page.text(LEFT, 314, "Muster Versand GmbH", 10)
    signature(page, LEFT, 340, 3)
    page.text(LEFT, 362, f"i. A. {clerk}", 9)
    blocks, _ = _blocks(page, tmp_path, "proxy")
    out = signers.attribute(blocks, {}, None)
    assert out[0].signer_name == clerk and out[0].signer_source == "proxy" and "i. A." in (out[0].evidence or "")


def test_line_labels_map_to_parties_and_place_date_come_from_the_signing_line(tmp_path: Path) -> None:
    """A contract's signature page: 'Vermieter' / 'Mieter' under the signatures name the landlord (sender) and the
    tenant (recipient); the printed '<Ort>, den <Datum>' line above gives place and date."""
    doc = build_contract(random.Random("sig-contract"), date(2024, 10, 1), date(2024, 8, 7), rent=False)
    blocks, text = _blocks(doc.pages[1], tmp_path, "contract")
    for block in blocks:
        signers.place_and_date(block, text, [])
    fields = {"meta_header": {"sender": {"name": doc.truth["sender"]}, "recipient": {"name": doc.truth["recipient"]}}}
    out = {s.label: s for s in signers.attribute(blocks, fields, None)}
    assert set(out) == {"Vermieter", "Mieter"} and all(s.status == "signed" for s in out.values())
    assert out["Vermieter"].signer_name == doc.truth["sender"] and out["Vermieter"].role == "landlord"
    assert out["Mieter"].signer_name == doc.truth["recipient"] and out["Mieter"].signer_source == "line_label"
    city = doc.facts["signing_line"].split(",")[0]
    assert out["Mieter"].date == "2024-08-07" and out["Mieter"].place == city


def test_empty_signature_place_and_no_evidence(tmp_path: Path) -> None:
    """The landlord's place on the permission letter is empty (no signer); a plain 'Unterschrift' line of a letter
    without printed name or role has no signer (no error)."""
    doc = build_permission_letter(random.Random("sig-permission"))
    blocks, _ = _blocks(doc.pages[0], tmp_path, "permission")
    out = {s.label: s for s in signers.attribute(blocks, {"meta_header": {"recipient": {"name": doc.facts["tenant"]}}}, None)}
    assert out["Datum/Unterschrift Vermieter"].status == "empty" and out["Datum/Unterschrift Vermieter"].signer_name is None
    assert out["Datum/Unterschrift Mieter"].status == "signed" and out["Datum/Unterschrift Mieter"].signer_name == doc.facts["tenant"]
    page = Page()
    page.text(LEFT, 200, "Wir bitten um Rücksendung.", 10)
    signature(page, 330, 300, 5)
    page.rule(320, 310, 533)
    page.text(330, 320, "Unterschrift", 7)
    blocks, _ = _blocks(page, tmp_path, "plain")
    out_plain = signers.attribute(blocks, {"meta_header": {"sender": {"company": "Muster Versand GmbH"}}}, None)
    assert len(out_plain) == 1 and out_plain[0].status == "signed" and out_plain[0].signer_name is None


def test_unsigned_form_is_empty_and_body_text_is_no_signature_label(tmp_path: Path) -> None:
    """An unsigned form: its signature place is empty; a sentence mentioning 'Unterschrift' is no signature label."""
    page, _, _ = _typed_form(random.Random("typed-form"), signed=False)
    page.text(LEFT, 700, "Bitte senden Sie das Formular mit Ihrer Unterschrift zurück.", 9)
    blocks, _ = _blocks(page, tmp_path, "unsigned")
    assert [(b.label, b.status) for b in blocks] == [("Unterschrift", "empty")]


def test_scribble_ocr_never_reaches_the_extraction(settings: Any, tmp_path: Path) -> None:
    """A scanned letter whose OCR reads the signature as a name-like line: the line is removed from the page text before
    Stage 2, the printed name under the signature is the signer, the scribble's reading is reported as removed."""
    doc = build_letter(random.Random("sig-scan"))
    page = doc.pages[0]
    pdf = _scan(page, tmp_path / "letter-scan.pdf")
    stroke = page.strokes[0]
    x0, y = min(x for x, _ in stroke), sum(v for _, v in stroke) / len(stroke)
    items = _items(page) + [_item(page, "Tbeo Mnster", x0, y + 8, x0 + 105, 12, y0=y - 12)]
    items.sort(key=lambda i: (i["bbox"][1], i["bbox"][0]))
    stage2: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("OCR ovisocr3"):
            return items
        if request.label.startswith("Stage 1.0 handwriting"):
            return {"pages": [{"page": 1, "handwriting": "signature"}]}
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["LETTER"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            stage2.append(request)
            return {"meta_header": {"sender": {"company": {"v": doc.truth["sender"], "p": 1}}}, "bodies": {}}
        return None

    settings.extract.engine = "ovisocr3"
    ld = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf).logical_documents[0]
    assert stage2 and "Tbeo Mnster" not in stage2[0].prompt and doc.facts["contact"] in stage2[0].prompt
    assert len(ld.signatures) == 1
    sig = ld.signatures[0]
    assert sig.removed_ocr == ["Tbeo Mnster"] and sig.signer_name == doc.facts["contact"] and sig.status == "signed"
