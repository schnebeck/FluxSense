# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_statement_generated.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Generated bank and credit-card statements through the
                pipeline: a customer / account / card number is never the
                statement number (§4), period, IBAN and balances survive the
                statement path, and the printed period and IBAN are filled
                from the text layer when the model leaves them out.
------------------------------------------------------------------------------
"""

import random
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from fluxsense.catalog import default_catalog
from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from fluxsense.stages.repair import enforce_doc_semantics
from fluxsense.testdata import LogicalDoc, build_statement
from fluxsense.testdata.layout import render
from tests.conftest import FakeBackend


def test_statement_number_is_never_a_customer_account_or_card_number() -> None:
    """Without a printed statement number, a customer, account or card number taken as one is removed; a printed
    statement number is kept."""
    spec = default_catalog().get("CREDIT_CARD_STATEMENT")
    doc = build_statement(random.Random("card-nonumber"), card=True, print_number=False)
    for value in (doc.facts["customer_number"], doc.facts["account_number"]):
        fields = {"meta_header": {}, "bodies": {"statement_body": {"statement_number": value,
                                                                     "account_number": doc.facts["account_number"]}}}
        notes = enforce_doc_semantics(fields, [], spec, doc.text(), True)
        assert "statement_number" not in fields["bodies"]["statement_body"] and notes, value
    printed = build_statement(random.Random("card-number"), card=True)
    fields = {"meta_header": {}, "bodies": {"statement_body": {"statement_number": printed.truth["doc_number"],
                                                                 "account_number": printed.facts["account_number"]}}}
    enforce_doc_semantics(fields, [], spec, printed.text(), True)
    assert fields["bodies"]["statement_body"]["statement_number"] == printed.truth["doc_number"]


def _ev(value: Any, page: int = 1) -> Dict[str, Any]:
    return {"v": value, "p": page}


def _model(doc: LogicalDoc, omit_period_and_iban: bool) -> Any:
    """A scripted model reading the generated statement; it takes the customer number as statement number when none
    is printed and may leave out period and IBAN."""
    gold, st = doc.truth, doc.truth["statement"]
    rows = [[day, booked, text, None, amount, None, page] for day, booked, text, amount, _balance, page in doc.facts["transactions"]]
    types = doc.doc_types[-1:]

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": types, "page_indices": list(range(1, len(doc.pages) + 1)),
                                           "direction": "INBOUND", "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if "chunk" in request.label:
            pages = [int(i.label.split()[-1]) for i in request.images if i.label.split()[-1].isdigit()]
            return {"rows": [r for r in rows if r[-1] in pages]}
        if not request.label.startswith("Stage 2"):
            return None
        body: Dict[str, Any] = {"statement_number": _ev(gold["doc_number"] or doc.facts["customer_number"]),
                                "statement_date": _ev(gold["doc_date"]), "opening_balance": _ev(st["opening_balance"]),
                                "closing_balance": _ev(st["closing_balance"], len(doc.pages)), "transactions": rows}
        if doc.facts["card_number"]:
            body["card_number_masked"] = _ev(doc.facts["card_number"])
            body["account_number"] = _ev(doc.facts["account_number"])
        if gold["due_date"]:
            body["payment_due_date"] = _ev(gold["due_date"], len(doc.pages))
        if not omit_period_and_iban:
            body.update({"period_start": _ev(st["period_start"]), "period_end": _ev(st["period_end"])})
            if gold["iban"]:
                body["iban"] = _ev(gold["iban"])
        return {"meta_header": {"sender": {"company": _ev(gold["sender"])}, "recipient": {"name": _ev(gold["recipient"])},
                                "doc_date": _ev(gold["doc_date"])},
                "bodies": {"statement_body": body}, "ai_confidence": 0.9}
    return handler


@pytest.mark.parametrize("card,printed,omit,others", [(False, True, False, False), (False, True, True, False),
                                                      (False, True, True, True), (True, False, False, False),
                                                      (True, True, True, False)])
def test_generated_statement_core_fields(settings: Any, tmp_path: Path, card: bool, printed: bool, omit: bool,
                                         others: bool) -> None:
    """Period, IBAN, balances and dates of a generated statement come out exactly - also when the model leaves period
    and IBAN out, the bookings print counterparty IBANs and a sub-period ends on the statement date; no statement
    number is emitted when none is printed."""
    doc = build_statement(random.Random(f"st-{card}-{printed}-{others}"), card=card, print_number=printed, info_page=True,
                          counterparty_ibans=others)
    pdf = render(doc.pages, tmp_path / "statement.pdf")
    result = Analyzer(settings, backend=FakeBackend(_model(doc, omit))).analyze(pdf)
    assert [d.pages for d in result.logical_documents] == [list(range(1, len(doc.pages) + 1))]
    fields = result.logical_documents[0].fields
    body, meta, st = fields["bodies"]["statement_body"], fields["meta_header"], doc.truth["statement"]
    assert (body.get("period_start"), body.get("period_end")) == (st["period_start"], st["period_end"])
    assert float(body["opening_balance"]) == st["opening_balance"] and float(body["closing_balance"]) == st["closing_balance"]
    assert body.get("iban") == doc.truth["iban"] and meta.get("doc_date") == doc.truth["doc_date"]
    number = meta.get("doc_number") or body.get("statement_number")
    assert number == doc.truth["doc_number"]
    assert len(body["transactions"]) == doc.truth["table"]["row_count"]
