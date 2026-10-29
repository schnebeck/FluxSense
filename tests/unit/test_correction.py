# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_correction.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    The targeted correction is applied end to end
                when it resolves the failing check, and every decision is
                explained in the record.
------------------------------------------------------------------------------
"""

from typing import Any, Dict, Optional

from fluxsense.llm.base import LLMRequest
from fluxsense.models.result import CheckResult, CheckStatus, PlausibilityReport, Severity
from fluxsense.pipeline import Analyzer
from fluxsense.stages.correction import _Scope, decide
from tests.conftest import FakeBackend


def fail(check_id: str, severity: Severity = Severity.ERROR) -> CheckResult:
    """A failed check."""
    return CheckResult(check_id=check_id, rule="r", status=CheckStatus.FAIL, severity=severity, message=check_id)


def test_decide_accepts_resolved_trigger_even_with_equal_count() -> None:
    """Resolving the trigger without a new error is enough, even if a warning appears."""
    before = PlausibilityReport(checks=[fail("invoice.line_sum")])
    after = PlausibilityReport(checks=[fail("dates.order", Severity.WARNING)])
    applied, note = decide(before, after, [fail("invoice.line_sum")], {}, {}, _Scope("totals", [1]))
    assert applied and "resolved ['invoice.line_sum']" in note
    applied, note = decide(before, PlausibilityReport(checks=[fail("invoice.line_sum")]), [fail("invoice.line_sum")], {}, {},
                           _Scope("totals", [1]))
    assert not applied and "still failing" in note


def test_missing_line_item_is_corrected_end_to_end(settings: Any, samples: Dict[str, Any]) -> None:
    """A dropped line item breaks the line sum; the targeted re-extraction restores it and is applied."""
    truth = samples["invoice_multipage"].data
    split = truth["split"]
    rows = [[n, d, float(q), None, float(p), float(t), 19, 1 if i < split else 2] for i, (n, d, q, p, t) in enumerate(truth["lines"])]
    wire = {"meta_header": {"sender": {"company": "Muster Elektro GmbH"}},
            "bodies": {"finance_body": {"invoice_number": {"v": "RE-2026-1001", "p": 1}, "invoice_date": {"v": "2026-10-01", "p": 1},
                                        "line_items": [r for i, r in enumerate(rows) if i != 25],
                                        "monetary_summation": {"tax_basis_total_amount": {"v": float(truth["net"]), "p": 2},
                                                               "tax_total_amount": {"v": float(truth["tax"]), "p": 2},
                                                               "grand_total_amount": {"v": float(truth["gross"]), "p": 2}}}}}

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": [1, 2], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return wire
        if request.label.startswith("Correction line_items"):
            return {"rows": rows}
        return None

    settings.grounding.max_escalations = 0
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["invoice_multipage"].path)
    doc = result.logical_documents[0]
    record = next(c for c in doc.corrections if c.scope.startswith("line_items"))
    assert record.applied, record.note
    assert "invoice.line_sum" in record.trigger_checks and "rows 33->34" in record.note
    assert len(doc.fields["bodies"]["finance_body"]["line_items"]) == 34
    assert "invoice.line_sum" not in doc.plausibility.summary_codes()
    assert doc.trust.corrections_applied >= 1


def test_missing_closing_balance_is_reread(settings: Any, samples: Dict[str, Any]) -> None:
    """A statement without closing balance triggers a targeted balance re-read on first/last page."""
    from tests.unit.test_pipeline import statement_wire
    truth = samples["bank_statement_3p"].data
    wire = statement_wire(truth["entries"], truth["opening"], truth["closing"])
    del wire["bodies"]["statement_body"]["closing_balance"]

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["BANK_STATEMENT"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return wire
        if request.label == "Correction statement balances":
            assert request.images and len(request.images) == 2
            return {"opening_balance": float(truth["opening"]), "closing_balance": float(truth["closing"])}
        return None

    settings.pipeline.statement_chunk_rows = 1000
    settings.pipeline.output_budget_tokens = 100000
    settings.pipeline.signature_min_pages = 99
    settings.grounding.max_escalations = 0
    doc = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["bank_statement_3p"].path).logical_documents[0]
    record = next(c for c in doc.corrections if c.scope == "statement balances")
    assert record.applied, record.note
    assert float(doc.fields["bodies"]["statement_body"]["closing_balance"]) == float(truth["closing"])
