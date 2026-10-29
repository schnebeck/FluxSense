# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_grounding.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for the grounding normalizer, matcher (exact / weak / ungrounded incl. OCR confusions), evidence wire schema and the harness (hallucinated value flagged and nulled).
------------------------------------------------------------------------------
"""

from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

from fluxsense.catalog import default_catalog
from fluxsense.catalog.builtin import LINE_ITEMS
from fluxsense.config import Settings
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.grounding.evidence import Unwrapped, build_plan, field_kind, unwrap, wire_schema
from fluxsense.grounding.harness import GroundingHarness, is_critical, parse_printed, path_matches
from fluxsense.grounding.matcher import build_index, ground_value
from fluxsense.grounding.normalize import (amount_candidates, compact, extract_amounts, extract_dates, fold_ocr,
                                           normalize_text, parse_date, to_decimal)
from fluxsense.llm.base import LLMRequest
from fluxsense.models.result import FieldEvidence, GroundingStatus
from fluxsense.models.semantic import FinanceBody
from tests.conftest import FakeBackend

G, W, U = GroundingStatus.GROUNDED, GroundingStatus.WEAKLY_GROUNDED, GroundingStatus.UNGROUNDED


@pytest.mark.parametrize("token,expected", [
    ("1.234,56", {Decimal("1234.56")}), ("1,234.56", {Decimal("1234.56")}), ("1234,56", {Decimal("1234.56")}),
    ("1234.56", {Decimal("1234.56")}), ("12,50-", {Decimal("12.50")}), ("-7,35", {Decimal("7.35")}),
    ("1.234", {Decimal("1.234"), Decimal("1234")}), ("2.450,00 EUR", {Decimal("2450.00")}), ("€ 99", {Decimal("99")}),
    ("1'234.50", {Decimal("1234.50")}),
])
def test_amount_candidates(token: str, expected: set) -> None:
    """German, English, Swiss and signed amount formats."""
    assert amount_candidates(token) == expected


def test_extract_amounts_from_text() -> None:
    """All amounts of a text, quantized; thousands with spaces are also split."""
    found = extract_amounts("Netto 1.234,56 EUR, USt 19 % 234,57, Summe 1 469,13 €")
    assert {Decimal("1234.56"), Decimal("234.57"), Decimal("1469.13"), Decimal("19.00")} <= found


def test_to_decimal_keeps_sign() -> None:
    """Model values keep their sign (also trailing minus and S marker)."""
    assert to_decimal("-10.00") == Decimal("-10.00")
    assert to_decimal("12,50-") == Decimal("-12.50")
    assert to_decimal("1.234,56 S") == Decimal("-1234.56")
    assert to_decimal(12.5) == Decimal("12.5") and to_decimal(None) is None and to_decimal(True) is None


@pytest.mark.parametrize("text,expected", [
    ("Datum 01.03.2024", date(2024, 3, 1)), ("am 1.3.24", date(2024, 3, 1)), ("2024-03-01", date(2024, 3, 1)),
    ("1. März 2024", date(2024, 3, 1)), ("01. Mar. 2024", date(2024, 3, 1)), ("March 1, 2024", date(2024, 3, 1)),
    ("1 March 2024", date(2024, 3, 1)), ("15. Dezember 2025", date(2025, 12, 15)),
])
def test_extract_dates_formats(text: str, expected: date) -> None:
    """German numeric/month-name, ISO and English date formats."""
    dates, _ = extract_dates(text)
    assert expected in dates


def test_partial_dates_and_parse_date() -> None:
    """Year-less dates are collected separately; parse_date accepts ISO and German."""
    _, partial = extract_dates("Buchung 05.03. Wert 06.03.")
    assert {(5, 3), (6, 3)} <= partial
    assert parse_date("2024-03-01") == date(2024, 3, 1)
    assert parse_date("01.03.2024") == date(2024, 3, 1)
    assert parse_date("31.02.2024") is None and parse_date("") is None


def test_text_normalization_and_folding() -> None:
    """Umlauts, punctuation, OCR confusion folding, compact identifiers."""
    assert normalize_text("Müller-Lüdenscheidt  GmbH & Co.") == "mueller luedenscheidt gmbh co"
    assert fold_ocr("DE8O 37O4 l2") == fold_ocr("DE80 3704 12")
    assert compact("DE44 9901-2345/0532") == "DE44990123450532"


def page_index(text: str, page: int = 1, source: str = "native") -> Dict[int, Any]:
    """Index for one page with one text source."""
    layer = TextLayer(pages=[PageText(page=page, sources={source: text})], engine="test")
    return build_index(layer)


TEXT = ("Muster Elektro GmbH · Industriestraße 7 · 00173 Stuttgart\nRechnungsnummer: RE-2026-0815\n"
        "Rechnungsdatum: 14.09.2026\nNettobetrag 1.300,00 EUR\nIBAN DE44 9901 2345 0532 0130 00")


@pytest.mark.parametrize("kind,value,status", [
    ("amount", 1300.0, G), ("amount", "1300.00", G), ("amount", -1300, G), ("amount", 1301.0, U),
    ("date", "2026-09-14", G), ("date", "2026-09-15", U),
    ("iban", "DE44990123450532013000", G), ("iban", "DE44990123450532013001", U),
    ("identifier", "RE-2026-0815", G), ("identifier", "RE 2026 0815", G), ("identifier", "RE-2026-0816", U),
    ("text", "Muster Elektro GmbH", G), ("text", "Industriestrasse", G), ("text", "Muster Elektrik GmbH", W),
    ("text", "Acme Corporation Ltd", U),
])
def test_ground_value_statuses(kind: str, value: Any, status: GroundingStatus) -> None:
    """Exact, weak and failed matches per kind."""
    assert ground_value(kind, value, page_index(TEXT)).status == status


def test_ocr_confusions_are_weak_matches() -> None:
    """O/0 and l/1 confusions in OCR text only give weak grounding."""
    ocr = "Rechnung RE-2O26-O8l5 IBAN DE44 99Ol 2345 O532 O13O OO Betrag l.3OO,OO"
    index = page_index(ocr, source="ocr")
    assert ground_value("identifier", "RE-2026-0815", index).status == W
    assert ground_value("iban", "DE44990123450532013000", index).status == W
    assert ground_value("amount", 1300, index).status == W


def test_yearless_date_is_weak_and_other_page_counts() -> None:
    """'14.09.' only weakly grounds 2026-09-14; values on another page of the document still count."""
    assert ground_value("date", "2026-09-14", page_index("Buchung 14.09. REWE")).status == W
    layer = TextLayer(pages=[PageText(page=1, sources={"native": "Seite 1"}), PageText(page=2, sources={"native": TEXT})], engine="t")
    match = ground_value("amount", 1300, build_index(layer), page_hint=1)
    assert match.status == G and match.page == 2


def test_field_kinds_and_critical_patterns() -> None:
    """Kind inference and '*' path patterns."""
    assert field_kind("invoice_date", str) == "date" and field_kind("iban", str) == "iban"
    assert field_kind("invoice_number", str) == "identifier" and field_kind("total_price", Decimal) == "amount"
    assert field_kind("currency", str) == "code" and field_kind("city", str) == "text"
    assert path_matches("bodies.finance_body.tax_breakdown.0.tax_amount", "bodies.finance_body.tax_breakdown.*.*")
    assert not path_matches("bodies.finance_body.invoice_date", "bodies.finance_body.*.x")
    assert is_critical("bodies.finance_body.monetary_summation.grand_total_amount", [default_catalog().get("INVOICE")])


def test_evidence_schema_and_unwrap_round_trip() -> None:
    """Wrapped scalars, tuple rows and small lists map back to clean fields plus evidence."""
    plan = build_plan(FinanceBody, tables={"line_items": LINE_ITEMS})
    schema = wire_schema(plan)
    assert set(schema["properties"]["invoice_number"]["properties"]) == {"v", "p", "e"}
    assert "prefixItems" in schema["properties"]["line_items"]["items"]
    assert "pattern" not in str(schema)
    single = wire_schema(plan, with_page=False)
    assert "p" not in single["properties"]["invoice_date"]["properties"]
    out = Unwrapped(None)
    clean = unwrap(plan, {"invoice_number": {"v": "R-1", "p": 1, "e": "Rechnung R-1"}, "currency": "EUR",
                          "line_items": [["1", "Widget", 2, None, 5, 10, 19, 1]],
                          "payment_accounts": [{"iban": "DE44990123450532013000", "_p": 2}]}, "bodies.finance_body", out)
    assert clean["invoice_number"] == "R-1" and clean["currency"] == "EUR"
    assert clean["line_items"] == [{"pos": "1", "description": "Widget", "quantity": 2, "unit_price": 5, "total_price": 10,
                                    "tax_rate": 19, "source_page": 1}]
    paths = {e.path: e for e in out.evidence}
    assert paths["bodies.finance_body.invoice_number"].evidence == "Rechnung R-1"
    assert paths["bodies.finance_body.payment_accounts.0.iban"].page == 2
    assert FinanceBody.model_validate(clean)


def test_parse_printed() -> None:
    """Printed readings of the zoom answer become normalized values."""
    assert parse_printed("amount", "1.234,56 EUR") == 1234.56
    assert parse_printed("amount", "-7,35") == -7.35
    assert parse_printed("date", "01.03.2024") == "2024-03-01"
    assert parse_printed("iban", "DE44 9901") == "DE449901"


def harness_case(zoom_answer: Optional[Dict[str, Any]]) -> tuple:
    """Runs the harness on a hallucinated grand total (not on the page)."""
    layer = TextLayer(pages=[PageText(page=1, sources={"native": TEXT})], engine="t")
    fields = {"bodies": {"finance_body": {"invoice_number": "RE-2026-0815",
                                          "monetary_summation": {"grand_total_amount": 9999.0}}}}
    evidence = [FieldEvidence(path="bodies.finance_body.invoice_number", kind="identifier", value="RE-2026-0815", page=1),
                FieldEvidence(path="bodies.finance_body.monetary_summation.grand_total_amount", kind="amount", value=9999.0, page=1)]

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Zoom locate"):
            return {"tiles": [{"id": "F1", "tile": "BR"}]}
        if request.label.startswith("Zoom verify"):
            return zoom_answer
        return None

    backend = FakeBackend(handler)
    settings = Settings()
    outcome = GroundingHarness(backend, settings).run(None, layer, [1], fields, evidence, [], [default_catalog().get("INVOICE")])
    return fields, evidence, outcome, backend


def test_hallucinated_critical_value_is_nulled(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value that is neither in the text nor confirmed on the crop is removed and reported."""
    monkeypatch.setattr("fluxsense.vision.zoom.render_crop", lambda *a, **k: (Image.new("RGB", (10, 10)), 300))
    monkeypatch.setattr("fluxsense.llm.images.page_images", lambda *a, **k: [])
    fields, evidence, outcome, backend = harness_case({"answers": [{"id": "F1", "visible": False, "printed": None, "matches": False}]})
    total = evidence[1]
    assert total.status == U and total.action == "nulled" and total.original_value == 9999.0
    assert "grand_total_amount" not in fields["bodies"]["finance_body"]["monetary_summation"]
    assert any(c.check_id == "grounding.ungrounded" and c.severity.value == "ERROR" for c in outcome.checks)
    assert evidence[0].status == G
    assert any(label.startswith("Zoom verify") for label in backend.labels())


def test_zoom_reading_replaces_misread_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the zoomed crop shows a different printed value, it replaces the hallucination."""
    monkeypatch.setattr("fluxsense.vision.zoom.render_crop", lambda *a, **k: (Image.new("RGB", (10, 10)), 300))
    monkeypatch.setattr("fluxsense.llm.images.page_images", lambda *a, **k: [])
    fields, evidence, _, _ = harness_case({"answers": [{"id": "F1", "visible": True, "printed": "1.300,00", "matches": False}]})
    total = evidence[1]
    assert total.action == "replaced" and total.value == 1300.0 and total.escalation_level == 2
    assert fields["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] == 1300.0


WRONG_IBAN = "DE19990123450000000000"
SCAN_TEXT = "Bitte zahlen Sie auf IBAN DE19 9901 2345 0000 0000 00"


def iban_case(monkeypatch: pytest.MonkeyPatch, answer: Dict[str, Any]) -> tuple:
    """Scan (OCR only) with an IBAN read identically wrong by OCR and VLM."""
    monkeypatch.setattr("fluxsense.vision.zoom.render_crop", lambda *a, **k: (Image.new("RGB", (10, 10)), 400))
    monkeypatch.setattr("fluxsense.llm.images.page_images", lambda *a, **k: [])
    layer = TextLayer(pages=[PageText(page=1, sources={"ocr": SCAN_TEXT})], engine="t")
    fields = {"meta_header": {"recipient": {"iban": WRONG_IBAN}}}
    evidence = [FieldEvidence(path="meta_header.recipient.iban", kind="iban", value=WRONG_IBAN, page=1)]
    seen: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Optional[Any]:
        seen.append(request)
        if request.label.startswith("Zoom locate"):
            return {"tiles": [{"id": "F1", "tile": "TL"}]}
        return answer if request.label.startswith("Zoom verify") else None

    outcome = GroundingHarness(FakeBackend(handler), Settings()).run(
        None, layer, [1], fields, evidence, [], [default_catalog().get("LETTER")],
        failed_fields=["meta_header.recipient.iban"])
    return fields, evidence[0], outcome, seen


def test_ocr_only_grounding_is_weak_on_scans_and_iban_is_rezoomed(monkeypatch: pytest.MonkeyPatch) -> None:
    """OCR agreeing with the VLM is not independent: the IBAN is re-read on a 400 dpi crop and corrected."""
    from fluxsense.validate.general import validate_iban
    correct = "DE44990123450532013000"
    fields, ev, outcome, seen = iban_case(monkeypatch, {"answers": [{"id": "F1", "visible": True, "printed": "DE44 9901 2345 0532 0130 00", "matches": False}]})
    assert ev.critical and ev.matched_source == "ocr"
    assert ev.action == "replaced" and ev.value == correct and validate_iban(fields["meta_header"]["recipient"]["iban"])
    assert not any(c.check_id == "grounding.iban_checksum" for c in outcome.checks)
    assert any(r.label.startswith("Zoom verify") for r in seen)


def test_invalid_iban_stays_flagged_when_zoom_agrees(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the zoom repeats the invalid reading, the checksum failure overrides 'verified'."""
    fields, ev, outcome, _ = iban_case(monkeypatch, {"answers": [{"id": "F1", "visible": True, "printed": None, "matches": True}]})
    assert ev.status == U and ev.action == "nulled" and "iban" not in fields["meta_header"]["recipient"]
    assert any(c.check_id == "grounding.iban_checksum" and c.severity.value == "ERROR" for c in outcome.checks)


def test_counts_need_their_evidence_snippet() -> None:
    """An invented count is not grounded just because the number occurs somewhere (e.g. in an amount)."""
    layer = TextLayer(pages=[PageText(page=1, sources={"native": "REWE -49,40 Saldo 1.240,00\nAnzahl Buchungen: 12"})], engine="t")
    harness = GroundingHarness(FakeBackend(lambda r: None), Settings())
    invented = FieldEvidence(path="bodies.statement_body.debit_count", kind="integer", value=40, page=1)
    printed = FieldEvidence(path="bodies.statement_body.credit_count", kind="integer", value=12, page=1, evidence="Anzahl Buchungen: 12")
    harness.ground(layer, [1], [invented, printed], [default_catalog().get("BANK_STATEMENT")])
    assert invented.status == U and invented.critical
    assert printed.status == G


def test_false_alarm_rules() -> None:
    """OCR-grounded values on scans stay grounded; ungrounded party sub-fields are removed, not flagged."""
    from fluxsense.grounding.harness import null_ungrounded_party_names
    scan = TextLayer(pages=[PageText(page=1, sources={"docling": "Rechnungsdatum 14.09.2026 Summe 357,00 EUR"})], engine="t")
    ev = FieldEvidence(path="bodies.finance_body.invoice_date", kind="date", value="2026-09-14", page=1)
    GroundingHarness(FakeBackend(lambda r: None), Settings()).ground(scan, [1], [ev], [default_catalog().get("INVOICE")])
    assert ev.status == G and ev.matched_source == "docling"
    fields = {"meta_header": {"sender": {"company": "Muster GmbH", "tax_id": "12/345/67890"}}}
    evidence = [FieldEvidence(path="meta_header.sender.tax_id", kind="identifier", value="12/345/67890", status=U),
                FieldEvidence(path="meta_header.sender.company", kind="text", value="Muster GmbH", status=G)]
    checks = null_ungrounded_party_names(evidence, fields)
    assert fields["meta_header"]["sender"] == {"company": "Muster GmbH"} and evidence[0].status.value == "removed"
    assert checks[0].severity.value == "INFO" and checks[0].field is None


def test_names_with_all_words_printed_are_grounded() -> None:
    """A name whose words are all printed is grounded; one near-miss word is tolerated."""
    index = page_index("Herrn Max Mustermann, Lindenweg 3")
    assert ground_value("text", "Max Mustermann", index).status == G
    assert ground_value("text", "Max Musterman", index).status == G


def test_stale_evidence_paths_are_never_edited() -> None:
    """Evidence pointing at removed rows or overlaid values does not crash and does not overwrite (regression)."""
    from fluxsense.grounding.harness import set_path, snap_ungrounded
    fields = {"bodies": {"finance_body": {"tax_breakdown": [], "invoice_number": "XML-1"}}}
    assert set_path(fields, "bodies.finance_body.tax_breakdown.0.tax_amount", 1) is False
    evidence = [FieldEvidence(path="bodies.finance_body.tax_breakdown.0.tax_amount", kind="amount", value=9.99, status=U),
                FieldEvidence(path="bodies.finance_body.invoice_number", kind="identifier", value="RE-2026-0816", status=U)]
    assert snap_ungrounded(TextLayer(pages=[PageText(page=1, sources={"native": TEXT})], engine="t"), [1], fields, evidence) == []
    assert fields["bodies"]["finance_body"]["invoice_number"] == "XML-1"


def test_scan_names_are_zoomed_before_removal(monkeypatch: pytest.MonkeyPatch) -> None:
    """On scans OCR can miss a printed name: it is removed only if the zoom does not confirm it."""
    monkeypatch.setattr("fluxsense.vision.zoom.render_crop", lambda *a, **k: (Image.new("RGB", (10, 10)), 300))
    monkeypatch.setattr("fluxsense.llm.images.page_images", lambda *a, **k: [])
    layer = TextLayer(pages=[PageText(page=1, sources={"ocr": "Sehr geehrte Frau G0ttfr1ed, ihr Antrag"})], engine="t")

    def run(confirm: bool) -> Dict[str, Any]:
        fields = {"meta_header": {"recipient": {"name": "Frau Erika Maria Musterfrau", "street": "Lindenweg"}}}
        evidence = [FieldEvidence(path="meta_header.recipient.name", kind="text", value="Frau Erika Maria Musterfrau", page=1),
                    FieldEvidence(path="meta_header.recipient.street", kind="text", value="Lindenweg", page=1)]

        def handler(request: LLMRequest) -> Optional[Any]:
            if request.label.startswith("Zoom locate"):
                return {"tiles": [{"id": "F1", "tile": "TL"}]}
            if request.label.startswith("Zoom verify"):
                return {"answers": [{"id": "F1", "visible": confirm, "printed": None, "matches": confirm}]}
            return None

        GroundingHarness(FakeBackend(handler), Settings()).run(None, layer, [1], fields, evidence, [], [default_catalog().get("LETTER")])
        return fields["meta_header"]["recipient"]

    assert run(True) == {"name": "Frau Erika Maria Musterfrau", "street": "Lindenweg"}
    assert run(False) == {"street": "Lindenweg"}


def test_dehyphenation_and_split_digit_groups() -> None:
    """'Rech-\\nnung' and soft hyphens are joined; real compounds still match; split digit groups match."""
    from fluxsense.grounding.normalize import dehyphenate
    assert dehyphenate("Rech-\nnung") == "Rechnung" and dehyphenate("Ver­siche­rung") == "Versicherung"
    assert dehyphenate("Musterstadt-\nLinden") == "Musterstadt-\nLinden"
    index = page_index("Ihre Rech-\nnung vom Ver­sicherungs-\nbüro in Musterstadt-\nLinden\nIBAN DE44 9901\n2345 0532 0130 00\nUSt-IdNr. DE 3444\n87914")
    assert ground_value("text", "Rechnung", index).status == G
    assert ground_value("text", "Versicherungsbüro", index).status == G
    assert ground_value("text", "Musterstadt-Linden", index).status == G
    assert ground_value("iban", "DE44990123450532013000", index).status == G
    assert ground_value("identifier", "DE344487914", index).status == G


def test_capped_identifier_is_completed_not_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A value cut at the schema cap is completed from the printed token, or flagged."""
    from fluxsense.grounding import evidence as evmod
    from fluxsense.grounding.harness import complete_capped_values
    monkeypatch.setitem(evmod.MAX_LENGTH_BY_KIND, "identifier", 12)
    layer = TextLayer(pages=[PageText(page=1, sources={"native": "Report-ID 2031-4-9_7B20D4F1 Seite 1"})], engine="t")
    fields = {"meta_header": {"doc_number": "2031-4-9_7B2"}}
    evidence = [FieldEvidence(path="meta_header.doc_number", kind="identifier", value="2031-4-9_7B2")]
    assert complete_capped_values(layer, [1], fields, evidence) == []
    assert fields["meta_header"]["doc_number"] == "2031-4-9_7B20D4F1" and evidence[0].action == "completed"
    fields = {"meta_header": {"doc_number": "ZZZZZZZZZZZZ"}}
    evidence = [FieldEvidence(path="meta_header.doc_number", kind="identifier", value="ZZZZZZZZZZZZ")]
    assert complete_capped_values(layer, [1], fields, evidence)[0].check_id == "grounding.possibly_truncated"


def test_iban_roles_from_the_text_layer() -> None:
    """§7: a statement's unique IBAN on a later page, a form's IBAN printed with spaced groups (also after the country
    code) become the statement / filler account; an issuer footer IBAN is no form account (synthetic, valid test IBANs)."""
    from fluxsense.catalog import default_catalog
    from fluxsense.extract.base import PageText, TextLayer
    from fluxsense.grounding.harness import snap_ibans
    cat = default_catalog()
    layer = TextLayer(pages=[PageText(page=1, sources={"native": "Kreditkartenabrechnung"}),
                             PageText(page=2, sources={"native": "Abrechnungskonto IBAN DE44 9901 2345 0532 0130 00"})], engine="t")
    fields = {"bodies": {"statement_body": {}}}
    evidence: list = []
    snap_ibans(layer, [1, 2], fields, evidence, [cat.get("CREDIT_CARD_STATEMENT")])
    assert fields["bodies"]["statement_body"]["iban"] == "DE44990123450532013000" and evidence[0].action == "filled_from_text"
    form = TextLayer(pages=[PageText(page=1, sources={"native": "Antrag\nIBAN DE 44 9901 2345 0532 0130 00\n"
                                                              "Beispielkasse AG - Vorstand: A. B. - IBAN DE02 1203 0000 0000 2020 51"})],
                     engine="t")
    fields = {"meta_header": {"sender": {"name": "Erika Muster"}}}
    snap_ibans(form, [1], fields, [], [cat.get("APPLICATION")])
    assert fields["meta_header"]["sender"]["iban"] == "DE44990123450532013000"


def test_irregularly_grouped_iban_is_found() -> None:
    """An IBAN printed in irregular groups ('DE44 990 123 45 0532013000') is cut to the country length and checked."""
    from fluxsense.grounding.snap import printed_ibans
    assert printed_ibans("Account DE44 990 123 45 0532013000 Swift Code: MUSTDEFF") == ["DE44990123450532013000"]
    assert printed_ibans("DE44 9901 2345 0532 0130 00") == ["DE44990123450532013000"]
    assert printed_ibans("DE12 3456 7890 1234 5678 90") == []
