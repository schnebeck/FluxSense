# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_zugferd.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for ZUGFeRD: CII parsing, classification short-circuit on generated sample invoices, visual comparison and XML overlay.
------------------------------------------------------------------------------
"""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from fluxsense.samples import build_cii_xml, embed_zugferd
from fluxsense.stages import zugferd
from tests.conftest import FakeBackend

XML = build_cii_xml("DEMO-2025-001", "20250102", "Quantum Dynamics", "Max Mustermann", "640.50", "121.69", "762.19")


def test_parse_cii() -> None:
    """Type code, totals, tax rows, IBAN, parties and dates are parsed."""
    data = zugferd.parse_cii_xml(XML)
    fin = data["finance_body"]
    assert data["type_tags"] == ["INVOICE"]
    assert fin["invoice_number"] == "DEMO-2025-001" and fin["invoice_date"] == "2025-01-02" and fin["currency"] == "EUR"
    assert fin["monetary_summation"]["grand_total_amount"] == "762.19"
    assert fin["tax_breakdown"] == [{"tax_basis_amount": "640.50", "tax_amount": "121.69", "tax_rate": "19", "tax_category": "S"}]
    assert fin["payment_accounts"][0]["iban"] == "DE44990123450532013000"
    assert data["meta_header"]["sender"]["vat_id"] == "DE123456789"
    assert zugferd.parse_cii_xml(build_cii_xml("C-1", "20250101", "S", "B", "1", "0", "1", type_code="381"))["type_tags"] == ["CREDIT_NOTE"]


def test_compare_counts_matches_and_differences() -> None:
    """The comparison yields an accuracy and lists differences."""
    data = zugferd.parse_cii_xml(XML)
    visual = {"meta_header": {"sender": {"company": "QUANTUM DYNAMICS"}},
              "bodies": {"finance_body": {"invoice_number": "DEMO-2025-001", "invoice_date": "2025-01-02",
                                          "monetary_summation": {"grand_total_amount": "762.20", "tax_total_amount": 121.69},
                                          "line_items": [{"description": "x"}]}}}
    result = zugferd.compare(data, visual)
    assert result.matched >= 4 and result.mismatched >= 1
    assert any(d["field"].endswith("grand_total_amount") and d["visual"] == "762.20" for d in result.differences)
    assert 0 < result.accuracy < 1
    merged = zugferd.overlay(visual, data)
    assert merged["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] == "762.19"
    assert merged["extraction_source"] == "ZUGFERD_NATIVE"


def test_short_circuit_skips_classification(settings: Any, samples: Dict[str, Any], tmp_path: Path) -> None:
    """With embedded XML, Stage 1 is skipped; the visual extraction still runs and differences become findings."""
    source = tmp_path / "invoice.pdf"
    shutil.copyfile(samples["invoice_wrong_vat"].path, source)
    pdf = embed_zugferd(source, tmp_path / "invoice_zugferd.pdf", XML)
    original = source.read_bytes()

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 2"):
            assert "ELECTRONIC INVOICE PRESENT" in request.prompt
            return {"meta_header": {"sender": {"company": "QUANTUM DYNAMICS"}},
                    "bodies": {"finance_body": {"invoice_number": {"v": "DEMO-2025-001"}, "invoice_date": {"v": "2025-01-02"},
                                                "monetary_summation": {"tax_basis_total_amount": {"v": 640.50},
                                                                       "tax_total_amount": {"v": 121.69},
                                                                       "grand_total_amount": {"v": 762.20}}}}}
        return None

    backend = FakeBackend(handler)
    result = Analyzer(settings, backend=backend).analyze(pdf)
    assert not any(label.startswith("Stage 1") for label in backend.labels())
    assert result.classification["source"] == "ZUGFERD"
    doc = result.logical_documents[0]
    assert doc.type == "INVOICE" and doc.zugferd_comparison is not None
    assert any(c.check_id == "zugferd.visual_mismatch" for c in doc.plausibility.findings)
    assert doc.fields["extraction_source"] == "ZUGFERD_NATIVE"
    assert doc.fields["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] == "762.19"
    assert doc.trust.zugferd_mismatches >= 1
    assert source.read_bytes() == original


def test_no_xml_in_plain_invoice(samples: Dict[str, Any]) -> None:
    """A plain generated invoice carries no XML."""
    assert zugferd.read_zugferd(samples["invoice_wrong_vat"].path) is None


def test_embedded_xml_values_are_protected(settings: Any, samples: Dict[str, Any], tmp_path: Path, monkeypatch: Any) -> None:
    """A post-check that replaces or nulls an XML value is reverted (the XML is authoritative); the parsed XML in
    result.zugferd is a copy that later field edits never mutate."""
    import fluxsense.pipeline as pipeline_module
    xml = build_cii_xml("DEMO-2026-002", "20310414", "Quantum Dynamics", "Max Mustermann", "10.00", "1.90", "11.90",
                        due_date="20310514")
    source = tmp_path / "invoice.pdf"
    shutil.copyfile(samples["invoice_wrong_vat"].path, source)
    pdf = embed_zugferd(source, tmp_path / "invoice_zugferd.pdf", xml)
    original = pipeline_module.sanitize_bank_identifiers

    def heuristic(fields: Dict[str, Any], keep: Any = None) -> List[str]:
        fields["bodies"]["finance_body"]["due_date"] = "2031-09-30"  # e.g. 'Ihr Tarif ist gültig bis 30.09.2031'
        fields["meta_header"].pop("doc_date", None)
        return original(fields, keep)

    monkeypatch.setattr(pipeline_module, "sanitize_bank_identifiers", heuristic)
    backend = FakeBackend(lambda r: {"bodies": {"finance_body": {"due_date": {"v": "2031-09-30"}}}} if r.label.startswith("Stage 2") else None)
    result = Analyzer(settings, backend=backend).analyze(pdf)
    doc = result.logical_documents[0]
    assert doc.fields["bodies"]["finance_body"]["due_date"] == "2031-05-14"
    assert doc.fields["meta_header"]["doc_date"] == "2031-04-14"
    assert any(c.scope == "embedded XML values" for c in doc.corrections)
    doc.fields["bodies"]["finance_body"]["due_date"] = "1999-01-01"
    assert result.zugferd["finance_body"]["due_date"] == "2031-05-14"
    restored = zugferd.protect_xml_values({"bodies": {"finance_body": {}}}, {"bodies": {"finance_body": {"due_date": "2031-05-14"}}})
    assert restored == [("bodies.finance_body.due_date", None, "2031-05-14")]
