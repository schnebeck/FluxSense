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
    assert date_candidates("31.02.24", None, date(2026, 10, 30)) == []


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
