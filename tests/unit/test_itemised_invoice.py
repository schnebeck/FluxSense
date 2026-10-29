# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_itemised_invoice.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Multi-page invoices whose later pages are an itemised usage
                list (generated telecom invoices, totals on page 1 or 2):
                the usage rows are no line items and trigger no chunking,
                the header call keeps page 1 and the grand-total page, and a
                chunked header call without header or totals falls back to
                the single call. A generated distributor invoice is the
                control.
------------------------------------------------------------------------------
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from fluxsense.extract import build_text_layer
from fluxsense.extract.base import TextLayer
from fluxsense.llm import budget
from fluxsense.llm.base import LLMRequest
from fluxsense.llm.images import ImagePart
from fluxsense.pipeline import Analyzer
from fluxsense.stages.extract import grand_total_pages, header_pages, item_page_range, row_anchors, totals_pages
from fluxsense.testdata import LogicalDoc, build_invoice, build_telecom_invoice
from fluxsense.testdata.layout import render
from tests.conftest import FakeBackend


def _layer(doc: LogicalDoc, tmp_path: Path, settings: Any) -> TextLayer:
    pdf = render(doc.pages, tmp_path / f"{doc.kind}.pdf")
    return build_text_layer(pdf, settings, "native")


@pytest.mark.parametrize("totals_page", [1, 2])
def test_usage_list_rows_are_no_item_anchors(settings: Any, tmp_path: Path, totals_page: int) -> None:
    """Numbered call rows with clock times and no item-table header are no line-item anchors; the grand total is found."""
    doc = build_telecom_invoice(random.Random(f"telecom-{totals_page}"), totals_page=totals_page)
    layer = _layer(doc, tmp_path, settings)
    pages = list(range(1, len(doc.pages) + 1))
    assert row_anchors(layer, pages) == {} and item_page_range({}, totals_pages(layer, pages), pages) == []
    assert grand_total_pages(layer, pages) == [totals_page]
    assert totals_page in header_pages(pages, totals_pages(layer, pages), 2, grand_total_pages(layer, pages))


def test_distributor_item_pages_still_anchor(settings: Any, tmp_path: Path) -> None:
    """Control: a long distributor invoice keeps its item anchors on every item page (also without a repeated header)."""
    doc = build_invoice(random.Random("distributor-1"), long=True, items=100)
    layer = _layer(doc, tmp_path, settings)
    pages = list(range(1, len(doc.pages) + 1))
    anchors = row_anchors(layer, pages)
    assert sum(len(v) for v in anchors.values()) == 100 and sorted(anchors) == pages[:-1]
    grand = grand_total_pages(layer, pages)
    assert grand == [len(doc.pages) - 1] and header_pages(pages, totals_pages(layer, pages), 2, grand) == [1, grand[0]]


def test_budget_fit_keeps_protected_pages() -> None:
    """Dropping images for the slot context keeps the first page and the protected (grand-total) page."""
    images = [ImagePart(f"PAGE {p}", "data:" + "x" * 40000, 1000, 1400) for p in (1, 2, 3, 4)]
    request = LLMRequest(prompt="p", label="Stage 2", model="m", images=list(images))
    budget.fit(request, 4000, protect=[1, 2])
    assert [i.label for i in request.images][:2] == ["PAGE 1", "PAGE 2"]
    request = LLMRequest(prompt="p", label="Stage 2", model="m", images=list(images))
    budget.fit(request, 4000)
    assert request.images[0].label == "PAGE 1" and request.images[-1].label == "PAGE 4"


def _wire(value: Any, page: int = 1) -> Dict[str, Any]:
    return {"v": value, "p": page}


def _model(doc: LogicalDoc, header_answers: bool = True) -> Any:
    """A scripted model answering from what the attached page images show (generated truth)."""
    gold = doc.truth
    lines = {i + 1: page.lines() for i, page in enumerate(doc.pages)}

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": sorted(lines), "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if not request.label.startswith("Stage 2"):
            return None
        seen = [int(i.label.split()[-1]) for i in request.images if i.label.split()[-1].isdigit()]
        if "chunk" in request.label:
            rows = []
            for page in seen:
                for line in lines[page]:
                    parts = [p.strip() for p in line.split("  ") if p.strip()]
                    if len(parts) == 6 and parts[0].isdigit():
                        amount = float(parts[5].replace(".", "").replace(",", "."))
                        rows.append([parts[0], parts[2], float(parts[3]), None,
                                     float(parts[4].replace(".", "").replace(",", ".")), amount, 19, page])
            return {"rows": rows}
        if not header_answers and request.label == "Stage 2 (INVOICE)" and "line_items" not in json.dumps(request.schema):
            return {"meta_header": {}, "bodies": {"finance_body": {}}}
        totals_seen = any("Rechnungsbetrag" in line or "Gesamtbetrag" in line for p in seen for line in lines[p])
        finance: Dict[str, Any] = {"payment_accounts": [{"iban": gold["iban"], "_p": 1}]}
        meta: Dict[str, Any] = {}
        if 1 in seen:
            meta = {"sender": {"company": _wire(gold["sender"]), "vat_id": _wire(doc.facts["sender_vat_id"])},
                    "recipient": {"name": _wire(gold["recipient"])}, "doc_date": _wire(gold["doc_date"]),
                    "doc_number": _wire(gold["doc_number"])}
            finance.update({"invoice_number": _wire(gold["doc_number"]), "invoice_date": _wire(gold["doc_date"])})
        if totals_seen:
            finance.update({"due_date": _wire(gold["due_date"]),
                            "monetary_summation": {"tax_basis_total_amount": _wire(gold["total_net"]),
                                                   "tax_total_amount": _wire(gold["tax_amount"]),
                                                   "grand_total_amount": _wire(gold["total_gross"])},
                            "tax_breakdown": [{"tax_basis_amount": gold["total_net"], "tax_amount": gold["tax_amount"],
                                               "tax_rate": 19, "_p": 1}]})
        if "line_items" in json.dumps(request.schema or {}) and doc.facts.get("line_items"):
            finance["line_items"] = [[None, label, 1, None, amount, amount, 19, 1] for label, amount in doc.facts["line_items"]]
        return {"meta_header": meta, "bodies": {"finance_body": finance}, "ai_confidence": 0.9,
                "audit": {"stamps": [], "has_handwriting": False, "type_matches": True}}
    return handler


def _core(result: Any) -> Dict[str, Any]:
    fields = result.logical_documents[0].fields
    meta, finance = fields["meta_header"], fields["bodies"]["finance_body"]
    sums = finance.get("monetary_summation") or {}
    def amount(key: str) -> Optional[float]:
        return float(sums[key]) if sums.get(key) not in (None, "") else None

    return {"sender": (meta.get("sender") or {}).get("company"), "recipient": (meta.get("recipient") or {}).get("name"),
            "doc_date": meta.get("doc_date"), "doc_number": meta.get("doc_number"), "total_gross": amount("grand_total_amount"),
            "total_net": amount("tax_basis_total_amount"), "tax_amount": amount("tax_total_amount"),
            "due_date": finance.get("due_date"), "rows": len(finance.get("line_items") or [])}


@pytest.mark.parametrize("totals_page", [1, 2])
def test_telecom_invoice_keeps_header_totals_and_items(settings: Any, tmp_path: Path, totals_page: int) -> None:
    """End to end: header fields and totals come back, the charge summary stays the line-item table (no usage rows),
    and no chunk call is made."""
    doc = build_telecom_invoice(random.Random(f"telecom-{totals_page}"), totals_page=totals_page)
    pdf = render(doc.pages, tmp_path / "telecom.pdf")
    backend = FakeBackend(_model(doc))
    core = _core(Analyzer(settings, backend=backend).analyze(pdf))
    gold = doc.truth
    assert core == {"sender": gold["sender"], "recipient": gold["recipient"], "doc_date": gold["doc_date"],
                    "doc_number": gold["doc_number"], "total_gross": gold["total_gross"], "total_net": gold["total_net"],
                    "tax_amount": gold["tax_amount"], "due_date": gold["due_date"], "rows": gold["table"]["row_count"]}
    assert not [label for label in backend.labels() if "chunk" in label]


def test_incomplete_chunked_header_falls_back_to_the_single_call(settings: Any, tmp_path: Path) -> None:
    """A chunked invoice whose header call returns neither header nor totals is extracted again in one call."""
    settings.llm.max_images_per_call = 2  # item pages beyond one call's images: upfront chunking
    doc = build_invoice(random.Random("distributor-2"), long=True, items=70)
    pdf = render(doc.pages, tmp_path / "distributor.pdf")
    backend = FakeBackend(_model(doc, header_answers=False))
    core = _core(Analyzer(settings, backend=backend).analyze(pdf))
    labels: List[str] = [label for label in backend.labels() if label.startswith("Stage 2 (")]
    assert labels.count("Stage 2 (INVOICE)") == 2
    assert core["doc_number"] == doc.truth["doc_number"] and core["total_gross"] == doc.truth["total_gross"]
