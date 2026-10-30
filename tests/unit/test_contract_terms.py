# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_contract_terms.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Field contract 1.5 on generated documents: rental contract
                payment terms (components add up to the total, payee IBAN,
                deposit, rented object), a mutual termination waiver is a
                minimum term and not the notice period, parties read only
                from handwritten signatures are dropped, and a landlord's
                handwritten Wohnungsgeberbestätigung is a CERTIFICATE with
                its event, date, persons and address but no form print
                number as doc_number. Negatives: a contract without payment
                terms and a private letter whose printed sender signs it.
------------------------------------------------------------------------------
"""

import random
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional

from fluxsense.extract.base import TextLayer
from fluxsense.grounding.acceptance import checked_paths
from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from fluxsense.stages.classify import Entity, route_certificates
from fluxsense.stages.repair import printed_form_code
from fluxsense.testdata import (LogicalDoc, build_contract, build_landlord_confirmation, build_letter,
                                build_permission_letter)
from fluxsense.testdata.layout import render
from fluxsense.validate.contract_terms import CHECK_PAYMENT_SUM, check_contract_payment, fix_minimum_term
from tests.conftest import FakeBackend

START, SIGNED = date(2024, 10, 1), date(2024, 7, 15)


def _w(value: Any, page: int = 1) -> Dict[str, Any]:
    return {"v": value, "p": page}


def _payment(doc: LogicalDoc) -> Dict[str, Any]:
    f = doc.facts
    return {"amount": f["rent_total"], "currency": "EUR", "frequency": "MONTHLY", "due_rule": "monatlich im Voraus",
            "components": [{"label": label, "amount": amount} for label, amount in f["components"]],
            "payment_account": {"name": f["holder"], "iban": f["iban"], "bic": f["bic"]}, "payment_reference": f["reference"]}


def test_payment_components_add_up_to_the_total() -> None:
    """The printed rent components sum to the printed total (check + 'check' provenance); a wrong component fails."""
    doc = build_contract(random.Random("rent-1"), START, SIGNED)
    fields = {"bodies": {"legal_body": {"payment_terms": _payment(doc)}}}
    assert [c.status.value for c in check_contract_payment(fields)] == ["PASS"]
    assert "bodies.legal_body.payment_terms.amount" in checked_paths(fields)
    fields["bodies"]["legal_body"]["payment_terms"]["components"][0]["amount"] += 10
    result = check_contract_payment(fields)
    assert result[0].check_id == CHECK_PAYMENT_SUM and result[0].status.value == "FAIL" and result[0].severity.value == "WARNING"
    assert "bodies.legal_body.payment_terms.amount" not in checked_paths(fields)


def test_contract_without_payment_terms_has_no_payment_check() -> None:
    """Negative: a contract that prints no payment terms gets no payment check and no payment facts."""
    doc = build_contract(random.Random("rent-none"), START, SIGNED, rent=False)
    assert "rent_total" not in doc.facts and "Miete:" not in doc.text()
    assert check_contract_payment({"bodies": {"legal_body": {}}}) == []


def test_termination_waiver_is_a_minimum_term() -> None:
    """'verzichten wechselseitig ... für die Dauer von 3 Jahren ... mit gesetzlicher Frist' is a minimum term."""
    doc = build_contract(random.Random("waiver"), START, SIGNED, waiver_years=3)
    clause = doc.facts["waiver_clause"]
    legal = {"notice_period": {"value": 36, "unit": "MONTHS", "anchor_type": "START_OF", "anchor_scope": "YEAR",
                               "original_text": clause}}
    notes = fix_minimum_term({"bodies": {"legal_body": legal}})
    assert notes and legal["notice_period"] == {}
    assert legal["minimum_term"]["value"] == 3 and legal["minimum_term"]["unit"] == "YEARS" and legal["minimum_term"]["mutual_waiver"]
    plain = {"notice_period": {"value": 3, "unit": "MONTHS", "original_text": "Kündigungsfrist 3 Monate zum Quartalsende"}}
    assert fix_minimum_term({"bodies": {"legal_body": plain}}) == [] and plain["notice_period"]["value"] == 3


def test_rental_contract_end_to_end(settings: Any, tmp_path: Path) -> None:
    """Rent table, payee IBAN, deposit and rented flat come through; the waiver lands in minimum_term; the payment
    check passes."""
    doc = build_contract(random.Random("rent-e2e"), START, SIGNED, waiver_years=3)
    pdf = render(doc.pages, tmp_path / "contract.pdf")
    clause = doc.facts["waiver_clause"]

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["CONTRACT"], "page_indices": [1, 2], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            payment = _payment(doc)
            payment["amount"] = _w(payment["amount"])
            payment["payment_account"]["iban"] = _w(payment["payment_account"]["iban"])
            return {"meta_header": {"sender": {"company": _w(doc.truth["sender"])}, "recipient": {"name": _w(doc.truth["recipient"])},
                                    "doc_number": _w(doc.facts["contract_number"])},
                    "bodies": {"legal_body": {"document_title": "Mietvertrag über Wohnraum", "effective_date": _w(START.isoformat()),
                                              "payment_terms": payment, "deposit": _w(doc.facts["deposit"]),
                                              "contract_object": dict(doc.facts["object"], description="Wohnung"),
                                              "notice_period": {"value": 36, "unit": "MONTHS", "original_text": clause}}}}
        return None

    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf)
    ld = result.logical_documents[0]
    legal = ld.fields["bodies"]["legal_body"]
    terms = legal["payment_terms"]
    assert float(terms["amount"]) == doc.facts["rent_total"] and len(terms["components"]) == 3
    assert terms["payment_account"]["iban"] == doc.facts["iban"] and float(legal["deposit"]) == doc.facts["deposit"]
    assert legal["contract_object"]["location_detail"] == doc.facts["object"]["location_detail"]
    assert legal["minimum_term"]["value"] == 3 and not (legal.get("notice_period") or {}).get("value")
    assert any(c.check_id == CHECK_PAYMENT_SUM and c.status.value == "PASS" for c in ld.plausibility.checks)
    amount = next(e for e in ld.evidence if e.path == "bodies.legal_body.payment_terms.amount")
    assert amount.status.value == "grounded" and amount.provenance == "check"


def _letter_handler(doc: LogicalDoc, sender: Dict[str, Any], tags: Any = ("LETTER",)) -> Any:
    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": list(tags), "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return {"meta_header": {"sender": sender, "recipient": {"name": _w(doc.truth["recipient"])}},
                    "bodies": {}, "audit": {"stamps": [], "has_handwriting": True, "type_matches": True}}
        return None
    return handler


def test_signature_only_sender_is_dropped(settings: Any, tmp_path: Path) -> None:
    """A permission letter whose only party marks are signature lines: an invented sender read from the tenant's
    signature (it carries the tenant's surname) is removed; the printed tenant stays recipient."""
    doc = build_permission_letter(random.Random("permission-1"))
    pdf = render(doc.pages, tmp_path / "permission.pdf")
    invented = {"name": {"v": f"Richardo {doc.facts['tenant_surname']}", "p": 1, "e": "Datum/Unterschrift Mieter"}}
    ld = Analyzer(settings, backend=FakeBackend(_letter_handler(doc, invented))).analyze(pdf).logical_documents[0]
    meta = ld.fields["meta_header"]
    assert not (meta.get("sender") or {}).get("name") and meta["recipient"]["name"] == doc.facts["tenant"]


def test_drop_signature_parties_rule() -> None:
    """§2.9 directly: an unprinted name carrying the other party's surname, or with a signature-line snippet, is
    removed; a name printed only next to 'Unterschrift' is removed; a name printed as a party stays; forms are exempt."""
    from fluxsense.models.result import FieldEvidence, GroundingStatus
    from fluxsense.stages.repair import drop_signature_parties
    doc = build_permission_letter(random.Random("permission-2"))
    text = doc.text()
    for sender, snippet in ((f"Richardo {doc.facts['tenant_surname']}", None), ("Hans Probe", "Datum/Unterschrift Mieter")):
        fields = {"meta_header": {"sender": {"name": sender}, "recipient": {"name": doc.facts["tenant"]}}}
        evidence = [FieldEvidence(path="meta_header.sender.name", kind="text", value=sender, evidence=snippet,
                                  status=GroundingStatus.UNGROUNDED)]
        assert drop_signature_parties(fields, evidence, text, None, "LETTER") and "sender" not in fields["meta_header"]
        assert fields["meta_header"]["recipient"]["name"] == doc.facts["tenant"]
    signed = text + "\nMax Unbekannt\nUnterschrift"
    fields = {"meta_header": {"sender": {"name": "Max Unbekannt"}}}
    assert drop_signature_parties(fields, [], signed, None, "LETTER") and "sender" not in fields["meta_header"]
    fields = {"meta_header": {"sender": {"name": "Max Unbekannt"}}}
    assert drop_signature_parties(fields, [], signed, None, "APPLICATION") == []
    fields = {"meta_header": {"recipient": {"name": doc.facts["tenant"]}}}
    assert drop_signature_parties(fields, [], text, None, "LETTER") == []


def test_printed_sender_that_signs_is_kept(settings: Any, tmp_path: Path) -> None:
    """Negative: a private letter whose printed sender name matches its signature keeps the sender."""
    doc = build_letter(random.Random("private-1"), private=True)
    pdf = render(doc.pages, tmp_path / "letter.pdf")
    sender = {"name": _w(doc.truth["sender"])}
    ld = Analyzer(settings, backend=FakeBackend(_letter_handler(doc, sender))).analyze(pdf).logical_documents[0]
    assert ld.fields["meta_header"]["sender"]["name"] == doc.truth["sender"]
    assert not any(c.scope == "signature-only parties" for c in ld.corrections)


def test_landlord_confirmation_is_a_certificate_without_form_number(settings: Any, tmp_path: Path) -> None:
    """A handwritten Wohnungsgeberbestätigung answered as APPLICATION is re-typed CERTIFICATE; its event, date,
    persons and address come through; the footer form print number is no doc_number."""
    doc = build_landlord_confirmation(random.Random("wgb-1"))
    layer = TextLayer(pages=[doc.pages[0].page_text(1)], engine="t")
    assert route_certificates([Entity(type_tags=["APPLICATION"], pages=[1])], layer)[0].type_tags == ["CERTIFICATE"]
    assert printed_form_code(doc.facts["form_code"], layer.pages[0])
    pdf = render(doc.pages, tmp_path / "wgb.pdf")
    facts = doc.facts

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["APPLICATION"], "page_indices": [1], "direction": "OUTBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return {"meta_header": {"sender": {"name": _w(doc.truth["sender"])}, "doc_number": _w(facts["form_code"]),
                                    "doc_date": _w(doc.truth["doc_date"])},
                    "bodies": {"legal_body": {"document_title": "Wohnungsgeberbestätigung", "certified_persons": facts["persons"],
                                              "certified_event": facts["event"], "event_date": _w(facts["event_date"]),
                                              "subject_address": dict(facts["subject"], location_detail=facts["floor"])}}}
        return None

    ld = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf).logical_documents[0]
    assert ld.type == "CERTIFICATE" and "doc_number" not in ld.fields["meta_header"]
    legal = ld.fields["bodies"]["legal_body"]
    assert legal["certified_persons"] == facts["persons"] and legal["certified_event"] == facts["event"]
    assert legal["event_date"] == facts["event_date"] and legal["subject_address"]["street"] == facts["subject"]["street"]
    assert ld.fields["meta_header"]["sender"]["name"] == doc.truth["sender"]
