# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_long_invoice.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Long invoices whose item table runs over more pages
                than one extraction call sees - page-chunked parallel line
                items, totals from the totals page, completeness checks
                (position gaps, row anchors, unseen totals block). Synthetic
                10-page invoice (samples.make_invoice_long).
------------------------------------------------------------------------------
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from fluxsense.samples import make_invoice_long
from tests.conftest import FakeBackend


def _rows(lines: List[Any], page: int) -> List[List[Any]]:
    """Tuple rows of one page in the line-item column order."""
    return [[n, d, float(q), None, float(p), float(t), 19, page] for n, d, q, p, t, pg in lines if pg == page]


def _handler(truth: Dict[str, Any], totals_in_header: bool) -> Any:
    """Scripted model: header call without rows, one chunk call per page, totals call on request."""
    pages = truth["pages"]
    totals = {"tax_basis_total_amount": {"v": float(truth["net"]), "p": truth["totals_page"]},
              "tax_total_amount": {"v": float(truth["tax"]), "p": truth["totals_page"]},
              "grand_total_amount": {"v": float(truth["gross"]), "p": truth["totals_page"]},
              "due_payable_amount": {"v": float(truth["gross"]), "p": truth["totals_page"]}}
    header: Dict[str, Any] = {"meta_header": {"sender": {"company": "Muster Elektro GmbH"}},
                              "bodies": {"finance_body": {"invoice_number": {"v": "RE-2031-4711", "p": 1},
                                                          "invoice_date": {"v": "2031-02-10", "p": 1}}}}
    if totals_in_header:
        header["bodies"]["finance_body"]["monetary_summation"] = totals

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": list(range(1, pages + 1)),
                                           "direction": "INBOUND", "tenant_context": "BUSINESS", "confidence": 0.9}]}
        chunk = re.match(r"Stage 2 chunk line_items p(\d+)-(\d+)", request.label)
        if chunk:
            first, last = int(chunk.group(1)), int(chunk.group(2))
            return {"rows": [r for p in range(first, last + 1) for r in _rows(truth["lines"], p)]}
        if request.label.startswith("Stage 2 totals"):
            return {"bodies": {"finance_body": {"monetary_summation": totals}}}
        if request.label.startswith("Stage 2"):
            return header
        return None
    return handler


def test_long_invoice_is_chunked_per_page_with_totals_from_the_totals_page(settings: Any, tmp_path: Path) -> None:
    """49 items over 7 pages + 3 info pages: one header call (first, totals and last page) and one parallel chunk call
    per item page; all rows with their source page; the line sum and net + tax = gross hold."""
    truth = make_invoice_long(tmp_path / "long.pdf").data
    backend = FakeBackend(_handler(truth, totals_in_header=True))
    settings.grounding.max_escalations = 0
    result = Analyzer(settings, backend=backend).analyze(tmp_path / "long.pdf")
    doc = result.logical_documents[0]
    rows = doc.fields["bodies"]["finance_body"]["line_items"]
    assert len(rows) == 49 and [r["source_page"] for r in rows] == [line[5] for line in truth["lines"]]
    header = next(r for r in backend.requests if r.label == "Stage 2 (INVOICE)")
    assert [int(i.label.split()[-1]) for i in header.images] == [1, 7, 10]
    chunks = [r.label for r in backend.requests if r.label.startswith("Stage 2 chunk")]
    assert chunks == [f"Stage 2 chunk line_items p{p}-{p}" for p in range(1, 8)]
    codes = doc.plausibility.summary_codes()
    assert "invoice.line_sum" not in codes and "invoice.net_plus_tax" not in codes
    assert {c.check_id for c in doc.plausibility.checks if c.status.value == "PASS"} >= {"invoice.line_sum", "invoice.net_plus_tax"}


def test_unseen_totals_block_is_read_from_its_page(settings: Any, tmp_path: Path) -> None:
    """The header call returned no totals: the totals block printed on an unseen page is read by a targeted call."""
    truth = make_invoice_long(tmp_path / "long.pdf", per_page=(4, 4, 4, 4, 4, 4, 4, 2), info_pages=0).data
    backend = FakeBackend(_handler(truth, totals_in_header=False))
    settings.grounding.max_escalations = 0
    settings.llm.max_images_per_call = 1  # the header call sees page 1 only
    result = Analyzer(settings, backend=backend).analyze(tmp_path / "long.pdf")
    ms = result.logical_documents[0].fields["bodies"]["finance_body"]["monetary_summation"]
    assert float(ms["grand_total_amount"]) == float(truth["gross"])
    totals_call = next(r for r in backend.requests if r.label.startswith("Stage 2 totals"))
    assert [int(i.label.split()[-1]) for i in totals_call.images] == [8]


def test_position_gap_triggers_rows_of_the_missing_page(settings: Any, tmp_path: Path) -> None:
    """A single call that stops early (rows of page 3 missing) is completed from row anchors / position gaps."""
    truth = make_invoice_long(tmp_path / "short.pdf", per_page=(4, 4, 4), info_pages=0).data
    base = _handler(truth, totals_in_header=True)

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label == "Stage 2 (INVOICE)":
            answer = base(request)
            answer["bodies"]["finance_body"]["line_items"] = _rows(truth["lines"], 1) + _rows(truth["lines"], 2)
            return answer
        return base(request)

    backend = FakeBackend(handler)
    settings.grounding.max_escalations = 0
    result = Analyzer(settings, backend=backend).analyze(tmp_path / "short.pdf")
    rows = result.logical_documents[0].fields["bodies"]["finance_body"]["line_items"]
    assert len(rows) == 12 and [r["pos"] for r in rows] == [str(i) for i in range(1, 13)]
    assert [r.label for r in backend.requests if r.label.startswith("Stage 2 chunk")] == ["Stage 2 chunk line_items p3-3"]


def test_chunk_schema_and_rate_normalisation() -> None:
    """Chunk rows must carry a description and an amount (no all-null rows); fractional tax rates become percent."""
    from fluxsense.catalog import default_catalog
    from fluxsense.grounding.evidence import build_plan, wire_schema
    from fluxsense.models.semantic import BODY_MODELS
    from fluxsense.stages.extract import percent_rates, required_cells
    spec = default_catalog().get("INVOICE").tables[0]
    plan = build_plan(BODY_MODELS[spec.body_key], tables={spec.field_name: spec}).fields[spec.field_name]
    cells = required_cells(wire_schema(plan), spec)["items"]["prefixItems"]
    types = dict(zip(spec.columns, (c["type"] for c in cells)))
    assert "null" not in types["description"] and "null" not in types["total_price"] and "null" in types["quantity"]
    rows = [{"tax_rate": 0.19}, {"tax_rate": 19}, {"tax_rate": None}]
    assert percent_rates(rows) == 1 and [r["tax_rate"] for r in rows] == [19.0, 19, None]
