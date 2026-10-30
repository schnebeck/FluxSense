# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_date_consistency.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Field contract §9 on generated documents: contradictory dates
                (contract signed decades before its start, due date decades
                away, statement date years outside its period) are flagged
                on both fields with a reason naming the other one; an
                ambiguous handwritten / zoom-read year is re-read only when
                exactly one reading is consistent, and stays flagged; a zoom
                reading in a conflict is no visual support; consistent
                documents stay clean.
------------------------------------------------------------------------------
"""

import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image

from fluxsense.extract.base import TextLayer
from fluxsense.grounding.acceptance import assess
from fluxsense.llm.base import LLMRequest
from fluxsense.models.result import FieldEvidence, GroundingStatus
from fluxsense.pipeline import Analyzer
from fluxsense.testdata import LogicalDoc, build_contract, build_invoice, build_statement
from fluxsense.testdata.generate import BUILDERS, KINDS
from fluxsense.testdata.layout import render
from fluxsense.validate.dates import CHECK_CONSISTENCY, CHECK_YEAR, check_dates_consistency
from tests.conftest import FakeBackend

TODAY = date(2026, 10, 30)
G, V = GroundingStatus.GROUNDED, GroundingStatus.VERIFIED


def _ev(path: str, value: str, status: GroundingStatus = G, **extra: Any) -> FieldEvidence:
    return FieldEvidence(path=path, kind="date", value=value, page=1, status=status, **extra)


def _layer(doc: LogicalDoc) -> TextLayer:
    return TextLayer(pages=[page.page_text(i + 1) for i, page in enumerate(doc.pages)], engine="t")


def test_invoice_due_date_decades_away_is_flagged_on_both_fields() -> None:
    """A due date 30 years after the invoice date is a WARNING on both dates; neither is auto-accepted."""
    doc = build_invoice(random.Random("due-far"))
    far = (date.fromisoformat(doc.truth["due_date"]) + timedelta(days=365 * 30)).isoformat()
    fields = {"meta_header": {"doc_date": doc.truth["doc_date"]}, "bodies": {"finance_body": {"due_date": far}}}
    evidence = [_ev("meta_header.doc_date", doc.truth["doc_date"]), _ev("bodies.finance_body.due_date", far)]
    checks, notes = check_dates_consistency(fields, evidence, TODAY)
    assert notes == [] and {c.field for c in checks if c.check_id == CHECK_CONSISTENCY} == \
        {"meta_header.doc_date", "bodies.finance_body.due_date"} and all(c.severity.value == "WARNING" for c in checks)
    assert fields["bodies"]["finance_body"]["due_date"] == far  # never silently fixed
    assess(fields, evidence, _layer(doc), [1], "INVOICE", checks)
    due = evidence[1]
    assert not due.auto_accepted and not evidence[0].auto_accepted
    assert any(r.startswith(f"date conflict: due_date {far} vs doc_date {doc.truth['doc_date']}") for r in due.review_reasons)


def test_statement_date_years_outside_its_period_is_flagged() -> None:
    """A statement dated years after its period conflicts with period_start and period_end."""
    doc = build_statement(random.Random("st-outside"))
    period = doc.truth["statement"]
    wrong = (date.fromisoformat(period["period_end"]) + timedelta(days=365 * 5)).isoformat()
    fields = {"meta_header": {"doc_date": wrong}, "bodies": {"statement_body": {"period_start": period["period_start"],
                                                                                 "period_end": period["period_end"]}}}
    evidence = [_ev("meta_header.doc_date", wrong), _ev("bodies.statement_body.period_start", period["period_start"]),
                _ev("bodies.statement_body.period_end", period["period_end"])]
    checks, _ = check_dates_consistency(fields, evidence, TODAY)
    assert {c.field for c in checks} == {e.path for e in evidence}


def test_ambiguous_signing_year_is_chosen_by_consistency_and_stays_flagged() -> None:
    """A zoom reading '7.8.04' of a handwritten '7/8/24' contradicts the contract start 1.10.24: the only consistent
    reading 2024-08-07 is chosen, flagged, and the zoom reading no longer counts as support."""
    fields = {"meta_header": {"doc_date": "2004-08-07"}, "bodies": {"legal_body": {"effective_date": "2024-10-01"}}}
    evidence = [_ev("meta_header.doc_date", "2004-08-07", V, original_value="2018-07-04", escalation_level=2, action="replaced",
                    verification={"printed": "7.8.04"}),
                _ev("bodies.legal_body.effective_date", "2024-10-01", evidence="ab 1.10.24")]
    checks, notes = check_dates_consistency(fields, evidence, TODAY)
    assert fields["meta_header"]["doc_date"] == "2024-08-07" and notes
    assert [c.check_id for c in checks] == [CHECK_YEAR]
    assert checks[0].message == "year ambiguous, chosen by consistency with effective_date (2004-08-07 -> 2024-08-07)"
    assert evidence[0].status != V and evidence[0].original_value == "2018-07-04"


def test_several_consistent_readings_choose_nothing() -> None:
    """Two consistent year readings (2008 and 2026 for a contract started 2009): nothing is chosen, both fields flagged."""
    fields = {"meta_header": {"doc_date": "2006-06-01"}, "bodies": {"legal_body": {"effective_date": "2009-01-01"}}}
    evidence = [_ev("meta_header.doc_date", "2006-06-01", V, escalation_level=2), _ev("bodies.legal_body.effective_date", "2009-01-01")]
    checks, notes = check_dates_consistency(fields, evidence, TODAY)
    assert notes == [] and fields["meta_header"]["doc_date"] == "2006-06-01"
    year = next(c for c in checks if c.check_id == CHECK_YEAR)
    assert year.message.startswith("year ambiguous: several reading(s) consistent with effective_date")
    assert {c.field for c in checks if c.check_id == CHECK_CONSISTENCY} == {"meta_header.doc_date", "bodies.legal_body.effective_date"}
    assert evidence[0].status == GroundingStatus.UNGROUNDED  # the zoom reading in a conflict is no support


def test_unambiguous_conflict_is_not_resolved() -> None:
    """A grounded four-digit date is never re-read; an ungrounded reference never decides."""
    fields = {"meta_header": {"doc_date": "2006-08-07"}, "bodies": {"legal_body": {"effective_date": "2024-10-01"}}}
    evidence = [_ev("meta_header.doc_date", "2006-08-07", evidence="07.08.2006"), _ev("bodies.legal_body.effective_date", "2024-10-01")]
    checks, notes = check_dates_consistency(fields, evidence, TODAY)
    assert notes == [] and [c.check_id for c in checks] == [CHECK_CONSISTENCY, CHECK_CONSISTENCY]
    evidence = [_ev("meta_header.doc_date", "2004-08-07", V, escalation_level=2),
                _ev("bodies.legal_body.effective_date", "2024-10-01", GroundingStatus.UNGROUNDED)]
    fields["meta_header"]["doc_date"] = "2004-08-07"
    checks, notes = check_dates_consistency(fields, evidence, TODAY)
    assert notes == [] and fields["meta_header"]["doc_date"] == "2004-08-07"


@pytest.mark.parametrize("start,signed,amendment", [(date(2024, 10, 1), date(2024, 7, 1), None),
                                                    (date(2015, 3, 1), date(2015, 1, 15), date(2024, 5, 6))])
def test_consistent_contracts_stay_clean(start: date, signed: date, amendment: Optional[date]) -> None:
    """Signed 3 months before the start, or a later amendment years after the start: no date finding."""
    doc = build_contract(random.Random("clean-contract"), start, signed, amendment=amendment)
    fields = {"meta_header": {"doc_date": doc.truth["doc_date"]}, "bodies": {"legal_body": {"effective_date": start.isoformat()}}}
    evidence = [_ev("meta_header.doc_date", doc.truth["doc_date"]), _ev("bodies.legal_body.effective_date", start.isoformat())]
    assert check_dates_consistency(fields, evidence, TODAY) == ([], [])


def test_generated_documents_are_date_consistent() -> None:
    """The truth of every generated kind passes the windows (no false alarm on consistent documents)."""
    for kind in KINDS:
        for seed in range(3):
            for doc in BUILDERS[kind](random.Random(f"{kind}-{seed}")):
                truth, period = doc.truth, doc.truth["statement"]
                fields = {"meta_header": {"doc_date": truth["doc_date"]},
                          "bodies": {"finance_body": {"due_date": truth["due_date"]} if "statement" not in doc.kind else {},
                                     "statement_body": {"period_start": period["period_start"], "period_end": period["period_end"],
                                                        "payment_due_date": truth["due_date"] if "statement" in doc.kind else None},
                                     "legal_body": {"effective_date": doc.facts.get("effective_date")}}}
                checks, _ = check_dates_consistency(fields, [], TODAY)
                assert checks == [], (kind, seed, [c.message for c in checks])


def test_contract_with_handwritten_two_digit_signing_date(settings: Any, tmp_path: Path, monkeypatch: Any) -> None:
    """End to end: the OCR text garbles the handwritten '7/8/24', the model misreads it, the zoom reads '7.8.04';
    the consistency step chooses 2024-08-07 against the printed start 'ab 1.10.24' and keeps doc_date flagged with
    the reason."""
    monkeypatch.setattr("fluxsense.vision.zoom.render_crop", lambda *a, **k: (Image.new("RGB", (10, 10)), 300))
    doc = build_contract(random.Random("rental-1"), date(2024, 10, 1), date(2024, 8, 7), signature_text="71804")
    pdf = render(doc.pages, tmp_path / "contract.pdf")

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["CONTRACT"], "page_indices": [1, 2], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return {"meta_header": {"doc_date": {"v": "2018-07-04", "p": 2, "e": "71804"},
                                    "doc_number": {"v": doc.facts["contract_number"], "p": 1}},
                    "bodies": {"legal_body": {"effective_date": {"v": "2024-10-01", "p": 1, "e": doc.facts["start_text"]}}}}
        if request.label.startswith("Zoom locate"):
            return {"tiles": [{"id": "F1", "tile": "TL"}]}
        if request.label.startswith("Zoom verify"):
            return {"answers": [{"id": "F1", "visible": True, "printed": "7.8.04", "matches": False}]}
        return None

    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf)
    ld = result.logical_documents[0]
    assert ld.fields["meta_header"]["doc_date"] == "2024-08-07"
    assert ld.fields["bodies"]["legal_body"]["effective_date"] == "2024-10-01"
    ev = next(e for e in ld.evidence if e.path == "meta_header.doc_date")
    assert not ev.auto_accepted and ev.status != V
    assert "year ambiguous, chosen by consistency with effective_date (2004-08-07 -> 2024-08-07)" in ev.review_reasons
    assert any(c.check_id == CHECK_YEAR for c in ld.plausibility.findings)
    assert any(c.scope == "ambiguous year" for c in ld.corrections)
