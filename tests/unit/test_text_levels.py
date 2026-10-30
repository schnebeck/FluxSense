# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_text_levels.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Text levels raw / clean / corrected on generated pages:
                dehyphenation and reflow of running text (addresses, tables,
                lists stay line-based); dictionary corrections only for a
                unique OCR-explained candidate and never for names, ALL-CAPS,
                digits, e-mails or URLs; digital text is never corrected;
                fixed-format fields corrected by position class + checksum
                and by crop re-reads that never show the old value (accept,
                document error, conflict); arithmetic re-reads; review
                hooks; the searchable text layer (words in reading order,
                corrected words, soft-hyphen pairs, form entries, nothing
                from signature regions, page images untouched). The vision
                model and the dictionary are mocked.
------------------------------------------------------------------------------
"""

import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pymupdf
import pytest

from fluxsense.extract.base import PageText, TextLayer
from fluxsense.formats import FormatSpec
from fluxsense.layout.items import LayoutItem
from fluxsense.llm.base import LLMRequest
from fluxsense.models.result import (CheckResult, CheckStatus, FieldEvidence, FormFieldOut, GroundingStatus, LogicalDocument,
                                     PageTextOut, SignatureOut, TextCorrection)
from fluxsense.pipeline import AnalyzeOptions, Analyzer
from fluxsense.testdata import ids
from fluxsense.testdata.layout import Page, render
from fluxsense.testdata.parties import person
from fluxsense.text import clean_text, confirm_field, confirm_text, page_levels, render_text_layer
from fluxsense.text.correct import (CHECK_ARITHMETIC_PRINTED, CHECK_CORRECTED, CHECK_PRINTED_INVALID, CHECK_REREAD_CONFLICT,
                                    FieldCorrector, format_corrections, ocr_explained, word_corrections)
from fluxsense.text.dictionary import Dictionary, parse_ispell
from tests.conftest import FakeBackend

LEFT = 62.0
IBAN_TRUE = "DE44990123450532013000"
IBAN_PRINTED = "DE44 9901 2345 0532 0130 00"


def _scan(page: Page, target: Path, dpi: int = 200) -> Path:
    digital = render([page], target.with_suffix(".digital.pdf"))
    with pymupdf.open(digital) as src, pymupdf.open() as out:
        pix = src[0].get_pixmap(dpi=dpi)
        new = out.new_page(width=src[0].rect.width, height=src[0].rect.height)
        new.insert_image(new.rect, stream=pix.tobytes("png"))
        out.save(target)
    return target


def _ocr_page(page: Page, number: int = 1) -> PageText:
    """The OCR view of a page model: its words and lines as an OCR source (no digital text)."""
    return PageText(page=number, width=page.width, height=page.height, sources={"ovis": "\n".join(page.lines())},
                    words=page.words(), words_source="ovis")


def _item(text: str, top: float, category: str = "text") -> LayoutItem:
    return LayoutItem(1, (100.0, top, 900.0, top + 40.0), category, text, "ovis")


# --------------------------------------------------------------------------------------------------------- clean


PARAGRAPH = ("Wir bedanken uns für Ihren Auftrag und stellen Ihnen die folgenden Leis-\n"
             "tungen in Rechnung. Bitte überweisen Sie den Betrag innerhalb von vier\n"
             "Wochen auf das unten genannte Konto.")


def test_clean_joins_hyphenation_and_reflows_running_text_only() -> None:
    """Running text becomes one line (line-break hyphenation joined only before a lowercase continuation, soft hyphens
    removed); an address block, a table row, a list and a hyphenated compound keep their lines."""
    address = "Musterfirma GmbH\nHauptstraße 1\n01067 Musterstadt"
    table = "| Pos | Menge | Preis |\n| 1 | 2 | 12,50 |"
    bullets = "- erste Leistung für den Kunden im laufenden Monat\n- zweite Leistung für den Kunden im Folgemonat"
    compound = "Die Filiale in Musterstadt-\nLinden ist ab sofort geschlossen."
    raw = "\n".join([address, PARAGRAPH, table, bullets, compound, "Rech­nung"])
    page = PageText(page=1, sources={"ovis": raw}, items=[_item(address, 50), _item(PARAGRAPH, 150), _item(table, 250, "table"),
                                                           _item(bullets, 350), _item(compound, 450)])
    clean = clean_text(page)
    assert "Leistungen in Rechnung. Bitte überweisen Sie den Betrag innerhalb von vier Wochen auf das" in clean
    assert address in clean and table in clean and bullets in clean
    assert "Musterstadt-\nLinden" in clean and "Rechnung" in clean and "­" not in clean
    no_layout = PageText(page=1, sources={"ovis": address + "\n\n" + PARAGRAPH})
    assert clean_text(no_layout).endswith("Wochen auf das unten genannte Konto.") and address in clean_text(no_layout)


# --------------------------------------------------------------------------------------------------------- words


def _lookup(known: Sequence[str], suggestions: Dict[str, List[str]]) -> Any:
    def lookup(words: Sequence[str]) -> Dict[str, Optional[List[str]]]:
        return {w: (None if w in known else suggestions.get(w, [])) for w in words}
    return lookup


def test_dictionary_corrections_need_a_unique_ocr_explained_candidate() -> None:
    """'Besteliung' -> 'Bestellung' and 'Ubersicht' -> 'Übersicht' (unique, OCR-explained); two explained candidates,
    names, ALL-CAPS, tokens next to digits, e-mails and URLs are left alone."""
    name = person(random.Random("protected")).last
    text = (f"Ubersicht der Besteliung von Herrn {name} und {name}s Kollegen. RECHNUNG Nr. Belag2 "
            f"kontakt@musterfirma.example www.musterfirma.example Zahlungsfrlst Kasse")
    suggestions = {"Ubersicht": ["Ufersicht", "Übersicht"], "Besteliung": ["Bestellung", "Besteigung"],
                   name: [name + "e"], f"{name}s": [name], "Zahlungsfrlst": ["Zahlungsfrist", "Zahlungsfrost"],
                   "Kasse": ["Masse"]}
    out = word_corrections(1, text, _lookup(["der", "von", "und", "Kollegen", "Herrn", "Kasse"], suggestions), {f"{name}s"})
    pairs = {(c.before, c.after) for c in out}
    assert pairs == {("Ubersicht", "Übersicht"), ("Besteliung", "Bestellung"), ("Zahlungsfrlst", "Zahlungsfrist")}
    assert all(text[c.offset:c.offset + len(c.before)] == c.before for c in out)  # type: ignore[operator]
    assert ocr_explained("Kundennurnmer", "Kundennummer") and not ocr_explained("Rechnnug", "Rechnung")


def test_ispell_output_and_a_missing_dictionary() -> None:
    output = "@(#) International Ispell Version 3.1.20\n*\n\n& Besteliung 2 0: Bestellung, Besteigung\n\n# Xqzt 0\n\n"
    assert parse_ispell(output, ["Haus", "Besteliung", "Xqzt"]) == {"Haus": None, "Besteliung": ["Bestellung", "Besteigung"],
                                                                     "Xqzt": []}
    missing = Dictionary(command=None)
    assert not missing.available and missing.lookup(["Haus"]) == {}


def test_page_levels_digital_text_is_never_corrected() -> None:
    """The same misspelling: corrected on an OCR page (with its word box), untouched on a page of digital text."""
    page = Page()
    page.text(LEFT, 100, "Ihre Besteliung vom Montag", 10)
    lookup = _lookup(["Ihre", "vom", "Montag"], {"Besteliung": ["Bestellung"]})
    scanned = page_levels(_ocr_page(page), lookup)
    assert scanned.text == "Ihre Besteliung vom Montag" and scanned.corrected == "Ihre Bestellung vom Montag"
    assert scanned.corrections[0].bbox and scanned.corrections[0].evidence_kind == "dictionary"
    digital = page_levels(page.page_text(1), lookup)
    assert digital.native and digital.corrected == digital.clean and not digital.corrections


def test_format_corrections_in_the_text() -> None:
    """A letter at a digit position of an IBAN is repaired in the text when exactly one repair passes mod 97; a VAT ID
    only after its label; an application's own format the same way."""
    rng = random.Random("text-vat")
    vat = next(v for v in (ids.fictional_vat_id(rng) for _ in range(200)) if "0" in v[3:-1])
    zero = vat.index("0", 3)
    misread_vat = vat[:zero] + "O" + vat[zero + 1:]
    text = f"IBAN DE44 99O1 2345 0532 0130 00\nUSt-IdNr.: {misread_vat}\nKunde KD-12345G"
    customer = FormatSpec.custom("customer_number", ["\\K\\D999999"])
    out = {c.before: c.after for c in format_corrections(1, text, [customer])}
    assert out["DE44 99O1 2345 0532 0130 00"] == IBAN_PRINTED and out["KD-12345G"] == "KD-123456"
    assert out[misread_vat] == vat
    assert format_corrections(1, f"Referenz {misread_vat}") == []  # a VAT ID only after its label


# ----------------------------------------------------------------------------------------------- field corrector


def _iban_page(printed: str) -> Page:
    page = Page()
    page.text(LEFT, 120, "Bankverbindung", 10, bold=True)
    page.text(LEFT, 140, f"IBAN: {printed}", 11, mono=True)
    return page


def _corrector_case(tmp_path: Path, settings: Any, ocr_value: str, reread: Optional[str],
                    image_value: str = IBAN_PRINTED) -> Tuple[Any, Dict[str, Any], FieldEvidence, TextLayer, List[LLMRequest]]:
    pdf = _scan(_iban_page(image_value), tmp_path / "iban.pdf")
    layer = TextLayer(pages=[_ocr_page(_iban_page(ocr_value))], engine="ovisocr3")
    fields: Dict[str, Any] = {"meta_header": {"sender": {"iban": ocr_value.replace(" ", "")}}, "bodies": {}}
    ev = FieldEvidence(path="meta_header.sender.iban", kind="iban", value=ocr_value.replace(" ", ""), page=1,
                       status=GroundingStatus.GROUNDED)
    seen: List[LLMRequest] = []

    def answer(request: LLMRequest) -> Any:
        seen.append(request)
        return {"printed": reread}
    checks, records = FieldCorrector(FakeBackend(answer), settings).run(pdf, layer, fields, [ev], [], lambda f: None)
    return (checks, records), fields, ev, layer, seen


def test_forced_repair_confirmed_by_the_reread_is_accepted(settings: Any, tmp_path: Path) -> None:
    """'99O1' (letter at a digit position): the re-read (that never sees the OCR value) reads the repaired IBAN - the
    field and the page text are corrected, nothing stays flagged."""
    (checks, records), fields, ev, layer, seen = _corrector_case(tmp_path, settings, "DE44 99O1 2345 0532 0130 00", IBAN_PRINTED)
    assert fields["meta_header"]["sender"]["iban"] == IBAN_TRUE and ev.original_value == "DE4499O12345053201300" + "0"
    assert ev.action == "corrected" and ev.resolution == "reread+check" and records and not checks
    assert seen and all("99O1" not in r.prompt and "DE44" not in r.prompt for r in seen)
    pending = layer.pages[0].corrections
    assert pending and pending[0].before == "DE44 99O1 2345 0532 0130 00" and pending[0].after == IBAN_PRINTED
    assert page_levels(layer.pages[0]).corrected.endswith(f"IBAN: {IBAN_PRINTED}")


def test_forced_repair_without_reread_stays_flagged(settings: Any, tmp_path: Path) -> None:
    settings.text.format_reread = False
    (checks, records), fields, ev, _, seen = _corrector_case(tmp_path, settings, "DE44 99O1 2345 0532 0130 00", None)
    assert fields["meta_header"]["sender"]["iban"] == IBAN_TRUE and records and not seen
    assert [c.check_id for c in checks] == [CHECK_CORRECTED]


def test_digit_confusion_is_decided_by_the_reread(settings: Any, tmp_path: Path) -> None:
    """A digit misread (0 -> 6, no forced position): a re-read that differs and passes mod 97 replaces it; a re-read
    that confirms the printed value keeps it as a document error; anything else is a conflict for review."""
    misread = IBAN_PRINTED[:15] + "6" + IBAN_PRINTED[16:]
    (checks, _), fields, ev, _, _ = _corrector_case(tmp_path, settings, misread, IBAN_PRINTED)
    assert fields["meta_header"]["sender"]["iban"] == IBAN_TRUE and not checks and ev.resolution == "reread+check"
    (checks, _), fields, _, _, _ = _corrector_case(tmp_path, settings, misread, misread, image_value=misread)
    assert [c.check_id for c in checks] == [CHECK_PRINTED_INVALID] and fields["meta_header"]["sender"]["iban"] != IBAN_TRUE
    (checks, _), _, _, _, _ = _corrector_case(tmp_path, settings, misread, "DE44 9901 2345")
    assert [c.check_id for c in checks] == [CHECK_REREAD_CONFLICT]


def test_digital_values_are_never_corrected(settings: Any, tmp_path: Path) -> None:
    pdf = render([_iban_page(IBAN_PRINTED[:15] + "6" + IBAN_PRINTED[16:])], tmp_path / "digital.pdf")
    from fluxsense.extract.native import read_pages
    layer = TextLayer(pages=read_pages(pdf, "native"), engine="native")
    value = (IBAN_PRINTED[:15] + "6" + IBAN_PRINTED[16:]).replace(" ", "")
    fields = {"meta_header": {"sender": {"iban": value}}}
    ev = FieldEvidence(path="meta_header.sender.iban", kind="iban", value=value, page=1)
    backend = FakeBackend(lambda r: {"printed": IBAN_PRINTED})
    assert FieldCorrector(backend, settings).run(pdf, layer, fields, [ev], [], lambda f: None) == ([], [])
    assert backend.requests == []


def _invoice_page(net: str, tax: str, gross: str, qty: str, price: str, total: str) -> Page:
    page = Page()
    page.text(LEFT, 100, "Pos  Menge  Einzelpreis  Gesamt", 10)
    page.text(LEFT, 120, f"1  Beratung  {qty}  {price}  {total}", 10)
    page.text(LEFT, 200, f"Nettobetrag {net}", 10)
    page.text(LEFT, 220, f"MwSt 19 % {tax}", 10)
    page.text(LEFT, 240, f"Gesamtbetrag {gross}", 10)
    return page


def _report(fields: Dict[str, Any]) -> Any:
    ms = fields["bodies"]["finance_body"]["monetary_summation"]
    rows = fields["bodies"]["finance_body"]["line_items"]
    bad = [i for i, r in enumerate(rows) if abs(r["quantity"] * r["unit_price"] - r["total_price"]) > 0.01]
    net_ok = abs(ms["tax_basis_total_amount"] + ms["tax_total_amount"] - ms["grand_total_amount"]) < 0.01
    return SimpleNamespace(checks=[
        CheckResult(check_id="invoice.net_plus_tax", rule="arithmetic", status=CheckStatus.PASS if net_ok else CheckStatus.FAIL,
                    field="bodies.finance_body.monetary_summation.grand_total_amount"),
        CheckResult(check_id="invoice.qty_x_price", rule="line_items", status=CheckStatus.FAIL if bad else CheckStatus.PASS,
                    rows=bad)])


def test_arithmetic_mismatches_are_reread(settings: Any, tmp_path: Path) -> None:
    """Net + tax != gross and qty x price != line total: the values are re-read on their crops; the single re-read that
    differs and makes the equation hold replaces the OCR value (also in the text); a re-read confirming every printed
    value keeps them and flags the document's arithmetic."""
    pdf = _scan(_invoice_page("100,00", "19,00", "119,00", "2", "12,50", "25,00"), tmp_path / "invoice.pdf")
    ocr = _invoice_page("100,00", "19,00", "179,00", "2", "12,50", "26,00")
    layer = TextLayer(pages=[_ocr_page(ocr)], engine="ovisocr3")
    fields: Dict[str, Any] = {"bodies": {"finance_body": {
        "monetary_summation": {"tax_basis_total_amount": 100.0, "tax_total_amount": 19.0, "grand_total_amount": 179.0},
        "line_items": [{"description": "Beratung", "quantity": 2, "unit_price": 12.5, "total_price": 26.0, "source_page": 1}]}}}
    ms = "bodies.finance_body.monetary_summation"
    evidence = [FieldEvidence(path=f"{ms}.{k}", kind="amount", value=v, page=1, status=GroundingStatus.GROUNDED)
                for k, v in fields["bodies"]["finance_body"]["monetary_summation"].items()]
    truth = {"net total": "100,00", "VAT amount": "19,00", "gross total": "119,00", "unit price": "12,50", "line total": "25,00"}

    def answer(request: LLMRequest) -> Any:
        return {"printed": next((v for k, v in truth.items() if k in request.prompt), None)}
    report = _report(fields)
    checks, records = FieldCorrector(FakeBackend(answer), settings).run(
        pdf, layer, fields, evidence, [c for c in report.checks if c.status == CheckStatus.FAIL], _report)
    assert fields["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] == 119.0
    assert fields["bodies"]["finance_body"]["line_items"][0]["total_price"] == 25.0 and len(records) == 2 and not checks
    assert {(c.before, c.after) for c in layer.pages[0].corrections} == {("179,00", "119,00"), ("26,00", "25,00")}
    printed = dict(fields)
    printed["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] = 179.0
    evidence[2].value = 179.0
    checks, records = FieldCorrector(FakeBackend(lambda r: {"printed": "179,00" if "gross" in r.prompt else
                                                            ("100,00" if "net" in r.prompt else "19,00")}), settings).run(
        pdf, TextLayer(pages=[_ocr_page(ocr)], engine="o"), printed, evidence,
        [c for c in _report(printed).checks if c.check_id == "invoice.net_plus_tax"], _report)
    assert [c.check_id for c in checks] == [CHECK_ARITHMETIC_PRINTED] and not records


# ------------------------------------------------------------------------------------------------------- hooks


def test_review_hooks_confirm_text_and_fields() -> None:
    page = PageTextOut(page=1, text="Ihre Besteliung", source="ovis", clean="Ihre Besteliung", corrected="Ihre Besteliung")
    confirm_text(page, "Besteliung", "Bestellung", note="reviewer")
    assert page.corrected == "Ihre Bestellung" and page.corrections[-1].evidence_kind == "human"
    doc = LogicalDocument(index=0, type="INVOICE", fields={"meta_header": {"doc_number": "RE-2O24"}},
                          evidence=[FieldEvidence(path="meta_header.doc_number", value="RE-2O24", review_reasons=["x"])])
    ev = confirm_field(doc, "meta_header.doc_number", "RE-2024")
    assert doc.fields["meta_header"]["doc_number"] == "RE-2024" and ev.auto_accepted and ev.provenance == "human"
    assert ev.original_value == "RE-2O24" and not ev.review_reasons and doc.corrections[-1].trigger_checks == ["review"]


# ------------------------------------------------------------------------------------------------- text layer


def _layer_page() -> Tuple[Page, Page]:
    """The image page and its OCR reading: a hyphenated pair, a misspelt word, a signature with OCR garbage."""
    image, ocr = Page(), Page()
    for model, word, scribble in ((image, "Bestellung", ""), (ocr, "Besteliung", "Mmhvfr")):
        model.text(LEFT, 100, "Wir danken für Ihre Bestel-", 11)
        model.text(LEFT, 116, f"lung und die {word} vom Montag.", 11)
        model.text(LEFT, 200, "Mit freundlichen Grüßen", 11)
        if scribble:
            model.text(LEFT, 236, scribble, 14)
        model.text(LEFT, 270, "Mustermann", 11)
        model.text(LEFT, 320, "Kennzeichen", 9)
    return image, ocr


def test_render_text_layer_reading_order_and_untouched_images(tmp_path: Path) -> None:
    """The searchable copy: the corrected word at its position, the hyphenated pair split with a soft hyphen in one
    ActualText span, the handwritten entry at its region, nothing from the signature region; the page image stream is
    the same as in the source."""
    image, ocr = _layer_page()
    pdf = _scan(image, tmp_path / "scan.pdf")
    page = _ocr_page(ocr)
    levels = page_levels(page, _lookup(["Wir", "danken", "Ihre", "lung", "und", "die", "vom", "Montag", "Mit", "freundlichen",
                                        "Grüßen", "Mustermann", "Kennzeichen", "Bestellung"], {"Besteliung": ["Bestellung"]}))
    assert levels.clean.startswith("Wir danken für Ihre Bestellung und die Besteliung vom Montag.")
    signature = SignatureOut(page=1, bbox=[LEFT / 595 * 1000 - 5, 220 / 842 * 1000, 400, 250 / 842 * 1000], status="signed")
    entry = FormFieldOut(page=1, label="Kennzeichen", entry="MU-XY 123", script="handwritten", found="locator",
                         entry_bbox=[300, 300 / 842 * 1000, 500, 325 / 842 * 1000])
    result = SimpleNamespace(text_layer=SimpleNamespace(pages=[levels]),
                             logical_documents=[SimpleNamespace(signatures=[signature], form_fields=[entry])])
    out = tmp_path / "searchable.pdf"
    source = pdf.read_bytes()
    stats = render_text_layer(pdf, TextLayer(pages=[page], engine="ovisocr3"), out, result)
    assert stats["pages"] == 1
    with pymupdf.open(out) as doc, pymupdf.open(pdf) as src:
        words = [w[4] for w in doc[0].get_text("words")]
        assert words == ["Wir", "danken", "für", "Ihre", "Bestel", "lung", "und", "die", "Bestellung", "vom", "Montag.", "Mit",
                         "freundlichen", "Grüßen", "Mustermann", "Kennzeichen", "MU-XY", "123"]
        assert b"ActualText" in doc.xref_stream(doc[0].get_contents()[-1]) and "Mmhvfr" not in doc[0].get_text()
        xref, src_xref = doc[0].get_images()[0][0], src[0].get_images()[0][0]
        assert doc.xref_stream_raw(xref) == src.xref_stream_raw(src_xref)
    assert pdf.read_bytes() == source  # the source is never written


def test_digital_pages_get_no_second_text_layer(tmp_path: Path) -> None:
    image, _ = _layer_page()
    pdf = render([image], tmp_path / "digital.pdf")
    from fluxsense.extract.native import read_pages
    assert render_text_layer(pdf, TextLayer(pages=read_pages(pdf, "native"), engine="native"), tmp_path / "o.pdf")["pages"] == 0


# --------------------------------------------------------------------------------------------------------- pipeline


def test_pipeline_text_levels_and_format_hint(settings: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """include_page_text: every page has raw / clean / corrected; a per-call format for doc_number repairs the
    OCR's letter at a digit position (confirmed by the re-read) in the field and in the text; the sender's name is never
    spell-corrected."""
    sender = person(random.Random("pipe-sender"))
    image, ocr = Page(), Page()
    for model, number, word in ((image, "KD-123456", "Bestellung"), (ocr, "KD-12345G", "Besteliung")):
        model.text(LEFT, 80, sender.name, 11)
        model.text(LEFT, 140, f"Kundennummer {number}", 11)
        model.text(LEFT, 170, f"Vielen Dank für Ihre {word}.", 11)
    pdf = _scan(image, tmp_path / "hint.pdf")
    items = [{"bbox": [round(r.left() / 595 * 1000), round((r.y - r.size) / 842 * 1000),
                       round((r.left() + r.width()) / 595 * 1000), round((r.y + 2) / 842 * 1000)], "category": "text",
              "content": r.text} for r in ocr.runs]

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("OCR ovisocr3"):
            return items
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["LETTER"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return {"meta_header": {"sender": {"name": {"v": sender.name, "p": 1}}, "doc_number": {"v": "KD-12345G", "p": 1}},
                    "bodies": {}}
        if request.label.startswith("Reread"):
            return {"printed": "KD-123456"}
        return None

    class FakeDictionary:
        available = True

        def lookup(self, words: Sequence[str]) -> Dict[str, Optional[List[str]]]:
            return {w: (["Bestellung"] if w == "Besteliung" else [w + "x"] if w in sender.name.split() else None) for w in words}

    monkeypatch.setattr(Dictionary, "system", classmethod(lambda cls, languages: FakeDictionary()))
    settings.extract.engine = "ovisocr3"
    customer = FormatSpec.custom("doc_number", ["\\K\\D999999"], description="customer number of this sender")
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf, AnalyzeOptions(include_page_text=True,
                                                                                            formats=[customer]))
    ld = result.logical_documents[0]
    assert ld.fields["meta_header"]["doc_number"] == "KD-123456"
    page = result.text_layer.pages[0]
    assert "KD-12345G" in page.text and "Besteliung" in page.text  # raw stays verbatim
    assert "Kundennummer KD-123456" in page.corrected and "Ihre Bestellung." in page.corrected
    assert sender.name in page.corrected
    kinds = {c.before: c.evidence_kind for c in page.corrections}
    assert kinds["KD-12345G"] == "reread" and kinds["Besteliung"] == "dictionary"
    assert any(r.note.startswith("meta_header.doc_number: KD-12345G -> KD-123456") for r in ld.corrections)


def test_corrections_are_located_and_overlaps_keep_the_strongest_evidence() -> None:
    """Corrections without an offset are located by whole word; of two overlapping ones the stronger evidence wins
    (human > re-read > format > dictionary); the list ends up in offset order."""
    from fluxsense.text.levels import apply_corrections
    corrections = [TextCorrection(page=1, before="Hous", after="Haus", evidence_kind="dictionary"),
                   TextCorrection(page=1, before="Hous", after="House", evidence_kind="human"),
                   TextCorrection(page=1, before="Tur", after="Tür", evidence_kind="dictionary")]
    assert apply_corrections("Tur am Hous", corrections) == "Tür am House"
    assert [(c.before, c.offset, c.evidence_kind) for c in corrections] == [("Tur", 0, "dictionary"), ("Hous", 7, "human")]
