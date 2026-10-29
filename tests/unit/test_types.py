# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_types.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Output types use KPaperFlux's DocType
                vocabulary, the finer type is kept as subtype, UTILITY_BILL
                implies INVOICE, the catalog carries German cues.
------------------------------------------------------------------------------
"""

import json
from typing import Any, Dict, Optional

from fluxsense.catalog import default_catalog
from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from fluxsense.prompts import stage1
from tests.conftest import FakeBackend

KPAPERFLUX_DOCTYPES = {
    "QUOTE", "ORDER", "ORDER_CONFIRMATION", "DELIVERY_NOTE", "INVOICE", "CREDIT_NOTE", "RECEIPT", "DUNNING", "BANK_STATEMENT",
    "TAX_ASSESSMENT", "EXPENSE_REPORT", "UTILITY_BILL", "CONTRACT", "INSURANCE_POLICY", "PAYSLIP", "LEGAL_CORRESPONDENCE",
    "OFFICIAL_LETTER", "CERTIFICATE", "MEDICAL_DOCUMENT", "VEHICLE_REGISTRATION", "APPLICATION", "NOTE", "OTHER",
}


def test_every_output_type_is_a_kpaperflux_doctype() -> None:
    """The catalog never emits types unknown to KPaperFlux."""
    catalog = default_catalog()
    assert {catalog.get(n).output_type for n in catalog.names()} <= KPAPERFLUX_DOCTYPES
    assert catalog.output_tags(["CREDIT_CARD_STATEMENT", "LETTER", "TECHNICAL_DOC", "SICK_NOTE"]) == \
        ["BANK_STATEMENT", "OTHER", "MEDICAL_DOCUMENT"]


def test_german_cues_in_catalog_and_prompt() -> None:
    """Crisp German cues for the confusable types."""
    catalog = default_catalog()
    assert "Antrag" in catalog.get("APPLICATION").description and "Kündigung" in catalog.get("APPLICATION").description
    assert "Behördenschreiben" in catalog.get("OFFICIAL_LETTER").description
    assert "UTILITY_BILL" in catalog.get("INVOICE").description
    assert 'BOTH "INVOICE" and "UTILITY_BILL"' in stage1.PROMPT_STAGE_1_1_SYSTEM


def test_pipeline_maps_types_and_keeps_subtype(settings: Any, samples: Dict[str, Any]) -> None:
    """CREDIT_CARD_STATEMENT -> BANK_STATEMENT with subtype; UTILITY_BILL alone gets INVOICE."""
    answers = {"Stage 1.1": {"detected_entities": [
        {"type_tags": ["CREDIT_CARD_STATEMENT"], "page_indices": [1], "direction": "INBOUND", "tenant_context": "PRIVATE", "confidence": 0.9},
        {"type_tags": ["UTILITY_BILL"], "page_indices": [2, 3], "direction": "INBOUND", "tenant_context": "PRIVATE", "confidence": 0.9}]}}

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return answers["Stage 1.1"]
        if request.label.startswith("Stage 2"):
            return {"meta_header": {}, "bodies": {}}
        return None

    settings.pipeline.statement_chunk_rows = 1000
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["bank_statement_3p"].path)
    card, bill = result.logical_documents
    assert card.type == "BANK_STATEMENT" and card.subtype == "CREDIT_CARD_STATEMENT"
    assert card.fields["type_tags"] == ["BANK_STATEMENT"]
    assert bill.type_tags == ["INVOICE", "UTILITY_BILL"] and bill.type == "INVOICE"


FORM_WIRE = {"meta_header": {"sender": {"company": "Hausverwaltung Sonnenhof GmbH", "street": "Am Park 12", "city": "Musterstadt"},
                             "recipient": {"name": "Erika Beispiel", "street": "Lindenweg 3", "zip_code": "00115", "city": "Berlin"}},
             "bodies": {"correspondence_body": {"subject": "Kündigung"}}}


def form_handler(request: LLMRequest) -> Optional[Any]:
    """A cancellation form classified as APPLICATION; the model took the issuer as sender."""
    if request.label.startswith("Stage 1.1"):
        return {"detected_entities": [{"type_tags": ["APPLICATION"], "page_indices": [1], "direction": "INBOUND",
                                       "tenant_context": "PRIVATE", "confidence": 0.9}]}
    if request.label.startswith("Stage 2"):
        assert "Forms filled in by a person" in request.system
        return json.loads(json.dumps(FORM_WIRE))
    return None


def test_form_filled_by_user_is_outbound_without_identity(settings: Any, samples: Dict[str, Any]) -> None:
    """Without identity profiles, company-vs-person cues swap the parties."""
    doc = Analyzer(settings, backend=FakeBackend(form_handler)).analyze(samples["letter"].path).logical_documents[0]
    assert doc.direction == "OUTBOUND"
    assert doc.fields["meta_header"]["sender"]["name"] == "Erika Beispiel"
    assert doc.fields["meta_header"]["recipient"]["company"] == "Hausverwaltung Sonnenhof GmbH"


def test_form_filled_by_user_is_outbound_with_identity(settings: Any, samples: Dict[str, Any]) -> None:
    """With identity profiles, the user's block becomes the sender."""
    from fluxsense.models.identity import IdentityProfile
    settings.identity.private = IdentityProfile(name="Erika Beispiel", address_keywords=["00115", "Lindenweg"])
    doc = Analyzer(settings, backend=FakeBackend(form_handler)).analyze(samples["letter"].path).logical_documents[0]
    assert doc.direction == "OUTBOUND" and doc.fields["meta_header"]["sender"]["name"] == "Erika Beispiel"


def test_identity_not_in_extraction_prompt_and_invented_names_removed(settings: Any, samples: Dict[str, Any]) -> None:
    """The extraction model never sees the identity; names not printed on the page are removed."""
    from fluxsense.models.identity import IdentityProfile
    settings.identity.private = IdentityProfile(name="Zacharias Quux", address_keywords=["99999"], iban=["DE44990123450532013000"])
    systems = []

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["LETTER"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            systems.append(request.system)
            return {"meta_header": {"sender": {"company": "Hausverwaltung Sonnenhof GmbH"},
                                    "recipient": {"name": "Frau Martina Mustermann"}}, "bodies": {}}
        return None

    doc = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["letter"].path).logical_documents[0]
    assert systems and "Zacharias" not in systems[0] and "DE44990123450532013000" not in systems[0]
    # the invented name is gone; the recipient now comes from the address window (DIN 5008)
    assert doc.fields["meta_header"]["recipient"].get("name") == "Erika Beispiel"
    assert doc.fields["meta_header"]["sender"]["company"] == "Hausverwaltung Sonnenhof GmbH"
    removed = next(e for e in doc.evidence if e.path == "meta_header.recipient.name")
    assert removed.status.value == "removed" and removed.original_value == "Frau Martina Mustermann"
