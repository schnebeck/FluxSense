# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_pipeline.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    End-to-end pipeline tests with a fake backend: faulty-VAT
                invoice (plausibility findings, hallucinated value flagged
                and nulled by the grounding harness), multi-page statement
                (one merged table, balance checks), call budget.
------------------------------------------------------------------------------
"""

from typing import Any, Dict, Optional

from fluxsense.llm.base import LLMRequest
from fluxsense.models.result import GroundingStatus
from fluxsense.pipeline import Analyzer
from tests.conftest import FakeBackend


def ev(value: Any, page: int = 1, snippet: Optional[str] = None) -> Dict[str, Any]:
    """Wire evidence wrapper."""
    out: Dict[str, Any] = {"v": value, "p": page}
    if snippet:
        out["e"] = snippet
    return out


INVOICE_WIRE: Dict[str, Any] = {
    "meta_header": {"sender": {"company": ev("Muster Elektro GmbH"), "vat_id": ev("DE123456789")},
                    "recipient": {"name": ev("Erika Beispiel"), "city": ev("Berlin")},
                    "doc_date": ev("2026-09-14", 1, "14.09.2026"), "doc_number": ev("RE-2026-0815")},
    "bodies": {"finance_body": {
        "invoice_number": ev("RE-2026-0815", 1, "Rechnungsnummer: RE-2026-0815"),
        "invoice_date": ev("2026-09-14", 1, "14.09.2026"),
        "due_date": ev("2026-10-31", 1, "31.10.2026"),  # hallucinated: the page says 28.09.2026
        "currency": "EUR",
        "customer_id": ev("K-4711"),
        "line_items": [["1", "Installation Steckdosen", 4, None, 25.0, 100.0, 19, 1],
                       ["2", "Kabel NYM-J 3x1,5 (Meter)", 50, None, 2.0, 100.0, 19, 1],
                       ["3", "Arbeitszeit Elektriker (Std.)", 2, None, 50.0, 100.0, 19, 1]],
        "monetary_summation": {"tax_basis_total_amount": ev(300.0, 1, "300,00 EUR"), "tax_total_amount": ev(59.0, 1, "59,00 EUR"),
                               "grand_total_amount": ev(357.0, 1, "357,00 EUR")},
        "tax_breakdown": [{"tax_basis_amount": 300.0, "tax_amount": 59.0, "tax_rate": 19, "_p": 1}],
        "payment_accounts": [{"iban": "DE44990123450532013000", "bic": "XMPLDEFFXXX", "_p": 1}],
    }},
    "ai_confidence": 0.9,
    "audit": {"stamps": [], "has_handwriting": False, "type_matches": True},
}


def invoice_handler(request: LLMRequest) -> Optional[Any]:
    """Scripted answers for the faulty invoice."""
    if request.label.startswith("Stage 1.1"):
        return {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": [1], "direction": "INBOUND",
                                       "tenant_context": "PRIVATE", "confidence": 0.95}]}
    if request.label.startswith("Stage 2"):
        return INVOICE_WIRE
    if request.label.startswith("Zoom verify"):
        return {"answers": [{"id": "F1", "visible": False, "printed": None, "matches": False}]}
    if request.label.startswith("Zoom locate"):
        return {"tiles": [{"id": "F1", "tile": "MR"}]}
    if request.label.startswith("Correction totals"):
        return {"monetary_summation": {"tax_basis_total_amount": 300.0, "tax_total_amount": 59.0, "grand_total_amount": 357.0},
                "tax_breakdown": [{"tax_basis_amount": 300.0, "tax_amount": 59.0, "tax_rate": 19}]}
    return None


def test_faulty_invoice_end_to_end(settings: Any, samples: Dict[str, Any]) -> None:
    """Wrong VAT is reported; the hallucinated due date is flagged and removed."""
    backend = FakeBackend(invoice_handler)
    result = Analyzer(settings, backend=backend).analyze(samples["invoice_wrong_vat"].path)
    assert result.status in ("OK", "PARTIAL")
    assert len(result.logical_documents) == 1
    doc = result.logical_documents[0]
    codes = doc.plausibility.summary_codes()
    assert "invoice.net_plus_tax" in codes
    assert "invoice.vat_row" in codes
    assert doc.plausibility.status == "FAIL"
    due = next(e for e in doc.evidence if e.path.endswith("finance_body.due_date"))
    assert due.status == GroundingStatus.UNGROUNDED
    assert due.action == "nulled"
    assert "due_date" not in doc.fields["bodies"]["finance_body"]
    number = next(e for e in doc.evidence if e.path.endswith("finance_body.invoice_number"))
    assert number.status == GroundingStatus.GROUNDED
    assert doc.trust.nulled == 1 and doc.trust.ungrounded >= 1
    assert doc.trust.score < 1.0
    # printed values are kept: the correction pass must not "fix" the document
    assert doc.fields["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] in ("357.0", "357")
    assert doc.corrections and doc.corrections[0].applied is False
    # few calls per document: classification, extraction, zoom, one correction
    stage_labels = [label.split(" ")[0] for label in backend.labels()]
    assert stage_labels.count("Stage") == 2
    assert len(backend.requests) <= 5
    assert result.timings["ai_seconds_per_page"] >= 0
    assert doc.visual_audit and doc.visual_audit["meta_mode"] == "STAMP_ONLY"


def statement_wire(entries: list, opening: Any, closing: Any, sign_error_row: Optional[int] = None) -> Dict[str, Any]:
    """Wire answer for the synthetic statement."""
    rows = []
    for i, (day, cp, ref, amount, balance) in enumerate(entries):
        value = float(amount) * (-1 if i == sign_error_row else 1)
        rows.append([day.isoformat(), None, f"{cp} {ref}", None, value, float(balance), 1 + i // 18])
    return {"meta_header": {"sender": {"company": ev("Musterbank eG")}},
            "bodies": {"statement_body": {"iban": ev("DE44990123450532013000"), "period_start": ev("2026-09-01"),
                                          "period_end": ev("2026-09-30"), "opening_balance": ev(float(opening), 1),
                                          "closing_balance": ev(float(closing), 3), "transactions": rows}}}


def test_statement_merged_table_and_balance(settings: Any, samples: Dict[str, Any]) -> None:
    """A 3-page statement yields ONE table; balance and running balance pass."""
    truth = samples["bank_statement_3p"].data
    wire = statement_wire(truth["entries"], truth["opening"], truth["closing"])

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["BANK_STATEMENT"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return wire
        return None

    settings.pipeline.output_budget_tokens = 100000
    settings.pipeline.statement_chunk_rows = 1000
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["bank_statement_3p"].path)
    doc = result.logical_documents[0]
    assert doc.type == "BANK_STATEMENT"
    assert len(doc.tables) == 1 and doc.tables[0].name == "transactions"
    assert len(doc.tables[0].rows) == len(truth["entries"])
    assert doc.tables[0].source_pages == [1, 2, 3]
    by_id = {c.check_id: c for c in doc.plausibility.checks}
    assert by_id["statement.balance"].status.value == "PASS"
    assert by_id["statement.running_balance"].status.value == "PASS"
    assert by_id["grounding.table_rows"].status.value == "PASS"


def test_statement_sign_error_is_located(settings: Any, samples: Dict[str, Any]) -> None:
    """A flipped sign is reported with row and page; the targeted correction repairs it."""
    truth = samples["bank_statement_3p"].data
    bad = statement_wire(truth["entries"], truth["opening"], truth["closing"], sign_error_row=20)
    good = statement_wire(truth["entries"], truth["opening"], truth["closing"])
    page2_rows = [r for r in good["bodies"]["statement_body"]["transactions"] if r[-1] == 2]

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["BANK_STATEMENT"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return bad
        if request.label.startswith("Correction transactions"):
            return {"rows": page2_rows}
        return None

    settings.pipeline.output_budget_tokens = 100000
    settings.pipeline.statement_chunk_rows = 1000
    settings.grounding.max_escalations = 0
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["bank_statement_3p"].path)
    doc = result.logical_documents[0]
    record = doc.corrections[0]
    assert "statement.running_balance" in record.trigger_checks
    assert record.pages == [2]
    assert record.applied is True
    by_id = {c.check_id: c for c in doc.plausibility.checks}
    assert by_id["statement.balance"].status.value == "PASS"


def chunk_rows(entries: list, pages: list) -> list:
    """Tuple rows of the given pages (18 bookings per page in the sample)."""
    return [[e[0].isoformat(), None, f"{e[1]} {e[2]}", None, float(e[3]), float(e[4]), 1 + i // 18]
            for i, e in enumerate(entries) if 1 + i // 18 in pages]


def test_statement_chunked_upfront_and_coverage_gap_filled(settings: Any, samples: Dict[str, Any]) -> None:
    """>40 rows -> page-chunked; a chunk that silently drops a page is detected and that page re-extracted."""
    truth = samples["bank_statement_3p"].data
    header = statement_wire(truth["entries"], truth["opening"], truth["closing"])
    del header["bodies"]["statement_body"]["transactions"]
    labels: list = []

    def handler(request: LLMRequest) -> Optional[Any]:
        labels.append(request.label)
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["BANK_STATEMENT"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2 chunk transactions p1-3"):
            return {"rows": chunk_rows(truth["entries"], [1, 3])}  # page 2 silently missing
        if request.label.startswith("Stage 2 chunk transactions p2-2"):
            return {"rows": chunk_rows(truth["entries"], [2])}
        if request.label.startswith("Stage 2"):
            return header
        return None

    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["bank_statement_3p"].path)
    doc = result.logical_documents[0]
    assert "Stage 2 chunk transactions p2-2" in labels
    rows = doc.tables[0].rows
    assert len(rows) == len(truth["entries"]) and [r["source_page"] for r in rows] == sorted(r["source_page"] for r in rows)
    assert {c.check_id: c.status.value for c in doc.plausibility.checks}["statement.balance"] == "PASS"


def test_balance_failure_over_many_pages_is_reextracted(settings: Any, samples: Dict[str, Any]) -> None:
    """A balance failure without a specific page triggers a page-chunked re-extraction (never 'skipped')."""
    truth = samples["bank_statement_3p"].data
    wire = statement_wire(truth["entries"], truth["opening"], truth["closing"])
    wire["bodies"]["statement_body"]["transactions"] = [r for i, r in enumerate(wire["bodies"]["statement_body"]["transactions"])
                                                       if i not in (5, 25, 40)]
    for row in wire["bodies"]["statement_body"]["transactions"]:
        row[5] = None  # no running balance printed -> no page-specific check

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["BANK_STATEMENT"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2 chunk"):
            return {"rows": chunk_rows(truth["entries"], [1, 2, 3])}
        if request.label.startswith("Stage 2"):
            return wire
        return None

    settings.pipeline.statement_chunk_rows = 1000
    settings.pipeline.output_budget_tokens = 100000
    settings.grounding.max_escalations = 0
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["bank_statement_3p"].path)
    doc = result.logical_documents[0]
    record = doc.corrections[0]
    assert "page-chunked" in record.scope and record.applied
    assert {c.check_id: c.status.value for c in doc.plausibility.checks}["statement.balance"] == "PASS"


def test_aux_text_covers_exactly_the_imaged_pages(settings: Any, samples: Dict[str, Any]) -> None:
    """Every Stage 2 call sends text only for the pages whose images it carries (models follow the text)."""
    import re as _re
    seen = []

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["BANK_STATEMENT"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            seen.append(request)
            return {"meta_header": {}, "bodies": {"statement_body": {}}} if "chunk" not in request.label else {"rows": []}
        return None

    settings.llm.max_images_per_call = 2  # forces fewer images than pages
    settings.pipeline.signature_min_pages = 99
    Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["bank_statement_3p"].path)
    assert seen
    for request in seen:
        image_pages = sorted(int(img.label.split()[1]) for img in request.images)
        text_pages = sorted({int(p) for p in _re.findall(r"=== PAGE (\d+) ===", request.prompt)})
        assert text_pages == image_pages, (request.label, text_pages, image_pages)
