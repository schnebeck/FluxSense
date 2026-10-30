# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_testdata.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    The synthetic test-document generator: determinism per seed,
                valid fictional identifiers, arithmetic that adds up, ground
                truth printed in the text layer, matching CII XML, scan
                variants without text layer, the privacy denylist check,
                the offline (no-LLM) pipeline stages on generated documents
                and a full analysis with a truth-answering backend.
------------------------------------------------------------------------------
"""

import json
import random
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from fluxsense.catalog import contract
from fluxsense.config import Settings
from fluxsense.extract import build_text_layer
from fluxsense.extract.native import has_font_resources
from fluxsense.grounding.snap import iban_valid, printed_ibans
from fluxsense.layout.din5008 import analyze, recipient_from_window
from fluxsense.llm.base import LLMRequest
from fluxsense.stages import zugferd
from fluxsense.testdata import (KINDS, build_invoice, build_letter, build_statement, denylist_hits, generate_file,
                                generate_set, ids, vocabulary)
from fluxsense.testdata.score import load_results, score

SEED = 11


@pytest.fixture(scope="module")
def docset(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """One digital file per kind (seed 11), plus its ground truth."""
    out = tmp_path_factory.mktemp("testdocs")
    files = generate_set(out, SEED, len(KINDS), scan_share=0.0)
    return {"dir": out, "files": files, "truth": json.loads((out / "ground_truth.json").read_text(encoding="utf-8"))}


def test_same_seed_same_documents(tmp_path: Path) -> None:
    """A seed fully determines documents and truth; another seed gives other documents."""
    runs = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        runs.append([generate_file(tmp_path / name, 5, i, kind, scan=False)
                     for i, kind in enumerate(("invoice", "card_statement", "stack"))])
    a, b = runs
    assert [f.entry for f in a] == [f.entry for f in b]
    assert [d.text() for f in a for d in f.docs] == [d.text() for f in b for d in f.docs]
    other = build_invoice(random.Random("another seed"))
    assert other.truth != a[0].docs[0].truth


def test_identifiers_are_valid_and_fictional() -> None:
    """IBANs pass mod-97 in the fictional 99xxxxxx bank-code range; VAT IDs carry a valid check digit in the DE9 range."""
    rng = random.Random(1)
    for _ in range(50):
        iban = ids.fictional_iban(rng)
        assert iban_valid(iban) and iban[4:6] == "99" and len(iban) == 22
        vat = ids.fictional_vat_id(rng)
        assert contract.valid_tax_id(vat, vat=True) and vat[2] == "9" and int(vat[-1]) == ids.vat_check_digit(vat[2:10])
        assert contract.valid_tax_id(ids.fictional_tax_number(rng))
    assert ids.vat_check_digit("13669597") == 6  # published ISO 7064 MOD 11,10 worked example


def test_amounts_add_up() -> None:
    """Invoice net + VAT = gross and the line totals sum to the net; statement opening + bookings = closing."""
    for seed in range(6):
        inv = build_invoice(random.Random(seed), long=seed % 2 == 0)
        t = inv.truth
        assert round(t["total_net"] + t["tax_amount"], 2) == t["total_gross"] and t["table"]["sum_line_totals"] == t["total_net"]
        for card in (False, True):
            st = build_statement(random.Random(seed), card=card)
            s, table = st.truth["statement"], st.truth["table"]
            assert round(s["opening_balance"] + table["sum_credits"] + table["sum_debits"], 2) == round(s["closing_balance"], 2)
            if card:
                assert s["closing_balance"] < 0 and st.truth["due_date"] > s["period_end"]


def test_long_invoice_layout() -> None:
    """Items across pages with carry-over rows, totals on the last item page, a terms page after it."""
    doc = build_invoice(random.Random(3), long=True, items=100)
    texts = ["\n".join(page.lines()) for page in doc.pages]
    assert len(doc.pages) >= 3 and "Allgemeine Geschäftsbedingungen" in texts[-1]
    assert "Übertrag auf Seite 2" in texts[0] and "Übertrag von Seite 1" in texts[1]
    assert "Gesamtbetrag" in texts[-2] and all("Gesamtbetrag" not in t for t in texts[:-2])
    bilingual = build_invoice(random.Random(3), long=True, bilingual=True)
    assert " / Invoice No.: " in bilingual.text() and " / Invoice date: " in bilingual.text()


def test_truth_is_printed_in_the_text_layer(docset: Dict[str, Any]) -> None:
    """Every core value of the ground truth is printed on the document's own pages (native text layer)."""
    settings = Settings()
    for generated, entry in zip(docset["files"], docset["truth"]):
        layer = build_text_layer(generated.path, settings, "native")
        for doc in entry["logical_documents"]:
            text = "\n".join(p.sources.get("native", "") for p in layer.pages if p.page in doc["pages"])
            flat = " ".join(text.split())
            for key in ("doc_date", "due_date"):
                if doc[key]:
                    assert ids.fmt_date_de(date.fromisoformat(doc[key])) in flat, (entry["path"], key)
            for key in ("total_gross", "total_net", "tax_amount"):
                if doc[key] is not None:
                    assert ids.fmt_de(Decimal(str(doc[key]))) in flat, (entry["path"], key)
            if doc["doc_number"]:
                assert doc["doc_number"] in flat, entry["path"]
            for key in ("sender", "recipient"):  # a form prints the filler as 'Last, First'
                if doc[key]:
                    assert all(token in flat for token in doc[key].split()), (entry["path"], key)
            if doc["iban"]:
                assert ids.group_iban(doc["iban"]) in flat


def test_offline_stages_on_generated_documents(docset: Dict[str, Any]) -> None:
    """No-LLM stages read the truth back: contract labels (doc_date, doc_number), the DIN 5008 window (recipient),
    printed IBANs and the embedded CII XML."""
    settings = Settings()
    for generated, entry in zip(docset["files"], docset["truth"]):
        layer = build_text_layer(generated.path, settings, "native")
        for doc in entry["logical_documents"]:
            type_name = doc["doc_types"][-1]
            pages = [p for p in layer.pages if p.page in doc["pages"]]
            lines = [line for p in pages for line in p.sources.get("native", "").splitlines()]
            if doc["doc_date"] and type_name == "APPLICATION":
                assert contract.form_signature_dates(lines) == [doc["doc_date"]]
            elif doc["doc_date"] and type_name == "CONTRACT":
                assert contract.date_line_dates(lines) == [doc["doc_date"]]
            elif doc["doc_date"] and contract.date_labels(type_name):
                hits = contract.find_labelled("date", contract.date_labels(type_name), lines)
                assert hits and hits[0].value == doc["doc_date"], (entry["path"], [h.value for h in hits])
            if doc["doc_number"]:
                assert contract.own_number_hits(type_name, lines)[0].value == doc["doc_number"], entry["path"]
            if doc["recipient"] and type_name not in ("APPLICATION", "CONTRACT"):  # no address window printed
                window = recipient_from_window(analyze(pages[0]).address_window)
                assert window and (window.get("name") or window.get("company")) == doc["recipient"], entry["path"]
            if doc["iban"]:
                assert doc["iban"] in [i for line in lines for i in printed_ibans(line)]
        xml = zugferd.read_zugferd(generated.path)
        if generated.kind == "invoice_zugferd":
            fin, gold = xml["finance_body"], entry["logical_documents"][0]
            assert (fin["invoice_number"], fin["invoice_date"], fin["due_date"]) == (gold["doc_number"], gold["doc_date"],
                                                                                  gold["due_date"])
            assert float(fin["monetary_summation"]["grand_total_amount"]) == gold["total_gross"]
            assert fin["payment_accounts"][0]["iban"] == gold["iban"]
        else:
            assert xml is None


def test_scan_variant_has_no_text_layer(tmp_path: Path) -> None:
    """A scan variant is image-only with the same pages and the same truth as its digital twin."""
    scan = generate_file(tmp_path, 2, 0, "invoice", scan=True)
    digital = generate_file(tmp_path, 2, 0, "invoice", scan=False)
    assert scan.path.name.endswith("_scan.pdf") and not has_font_resources(scan.path) and has_font_resources(digital.path)
    assert scan.manifest["pages"] == digital.manifest["pages"] and scan.manifest["chars"] == 0
    gold, twin = scan.entry["logical_documents"][0], digital.entry["logical_documents"][0]
    assert {k: v for k, v in gold.items() if k != "visual"} == {k: v for k, v in twin.items() if k != "visual"}


def test_vocabulary_contains_no_denylisted_entry() -> None:
    """Privacy by construction: no generator entry and no generated text matches a denylist (a synthetic denylist here;
    a project-external denylist can be checked with 'fluxsense gen-testdocs --denylist')."""
    denylist = ["Jane Roe", "Fictitia Holding", "DE00 1234 5678", "Hauptstraße 1", "4711-REAL-0815"]
    texts = vocabulary() + [build_letter(random.Random(s)).text() for s in range(5)]
    assert denylist_hits(texts, denylist) == []
    assert denylist_hits(texts + ["Brief an jane  roe"], denylist) == ["Jane Roe"]
    assert len(vocabulary()) > 150


def test_stack_segments_and_score(docset: Dict[str, Any], tmp_path: Path) -> None:
    """A stack is a letter with its terms page plus an enclosure; scoring a perfect result gives full marks."""
    entry = next(e for e in docset["truth"] if "_stack" in e["path"])
    assert [d["pages"] for d in entry["logical_documents"]] == [[1, 2], [3]]
    result = {"source": {"path": entry["path"]}, "logical_documents": []}
    for doc in entry["logical_documents"]:
        meta: Dict[str, Any] = {"doc_date": doc["doc_date"], "doc_number": doc["doc_number"]}
        if doc["sender"]:
            meta["sender"] = {"company": doc["sender"]}
        if doc["recipient"]:
            meta["recipient"] = {"name": doc["recipient"]}
        result["logical_documents"].append({"pages": doc["pages"], "type": doc["doc_types"][0], "fields": {
            "meta_header": meta, "bodies": {"correspondence_body": {"deadline": doc["due_date"]}}}})
    (tmp_path / "r.json").write_text(json.dumps(result), encoding="utf-8")
    summary = score([entry], load_results(tmp_path))
    assert summary["segmentation_exact"] == 1 and summary["type_correct"] == 2 and summary["field_accuracy"] == 1.0


def _wire(value: Any, snippet: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {"v": value, "p": 1}
    if snippet:
        out["e"] = snippet
    return out


def test_analysis_with_a_truth_answering_backend(settings: Settings, tmp_path: Path) -> None:
    """End to end without an LLM: a backend that answers with the ground truth of a generated invoice yields that
    truth, auto-accepted, with no failed plausibility check."""
    from fluxsense.pipeline import Analyzer
    from tests.conftest import FakeBackend
    generated = generate_file(tmp_path, 4, 0, "invoice", scan=False)
    gold = generated.entry["logical_documents"][0]
    doc = generated.docs[0]
    rows = [line for page in doc.pages for line in page.lines()]
    items = []
    for line in rows:
        parts = line.split("  ")
        if len(parts) == 6 and parts[0].isdigit():
            items.append([parts[0], parts[2], float(parts[3]), None, float(parts[4].replace(".", "").replace(",", ".")),
                          float(parts[5].replace(".", "").replace(",", ".")), 19, 1])
    assert len(items) == gold["table"]["row_count"]

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.95}]}
        if request.label.startswith("Stage 2"):
            due = ids.fmt_date_de(date.fromisoformat(gold["due_date"]))
            return {"meta_header": {"sender": {"company": _wire(gold["sender"]), "vat_id": _wire(doc.facts["sender_vat_id"])},
                                    "recipient": {"name": _wire(gold["recipient"])},
                                    "doc_date": _wire(gold["doc_date"]), "doc_number": _wire(gold["doc_number"])},
                    "bodies": {"finance_body": {
                        "invoice_number": _wire(gold["doc_number"]), "invoice_date": _wire(gold["doc_date"]),
                        "due_date": _wire(gold["due_date"], f"Zahlbar bis {due}"), "currency": "EUR", "line_items": items,
                        "monetary_summation": {"tax_basis_total_amount": _wire(gold["total_net"]),
                                               "tax_total_amount": _wire(gold["tax_amount"]),
                                               "grand_total_amount": _wire(gold["total_gross"])},
                        "tax_breakdown": [{"tax_basis_amount": gold["total_net"], "tax_amount": gold["tax_amount"],
                                           "tax_rate": 19, "_p": 1}],
                        "payment_accounts": [{"iban": gold["iban"], "_p": 1}]}},
                    "ai_confidence": 0.9, "audit": {"stamps": [], "has_handwriting": False, "type_matches": True}}
        return None

    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(generated.path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "r.json").write_text(result.model_dump_json(), encoding="utf-8")
    summary = score([generated.entry], load_results(tmp_path / "results"))
    wrong = {f: c for f, c in summary["fields"].items() if c["wrong"] or c["missed"] or c["extra"]}
    assert summary["field_accuracy"] == 1.0 and summary["table_rows_exact"] == 1, wrong
    ld = result.logical_documents[0]
    assert not [c for c in ld.plausibility.findings if c.status.value == "FAIL"]
    core = [e for e in ld.evidence if e.path in ("meta_header.doc_date", "meta_header.doc_number", "meta_header.sender.company",
                                                 "meta_header.recipient.name", "bodies.finance_body.due_date")]
    assert core and all(e.auto_accepted for e in core), [(e.path, e.review_reasons) for e in core if not e.auto_accepted]


def test_phone_and_card_numbers_are_reserved_or_invalid() -> None:
    """Every printed phone number lies in a Bundesnetzagentur drama-number block; every card number starts with the
    unused prefix 9999 and a full card number fails the Luhn check."""
    import re
    from fluxsense.testdata import generate as gen
    texts = [doc.text() for kind in gen.KINDS for seed in range(3) for doc in gen.BUILDERS[kind](random.Random(f"{kind}-{seed}"))]
    text = "\n".join(re.sub(r"\b[A-Z]{2}\d{2}(?: \d{4}){4,5}(?: \d{1,2})?\b", " ", t) for t in texts)  # IBANs aside
    phones = re.findall(r"\b0\d{2,4} \d{7,8}\b", text)
    assert phones and all(ids.is_drama_number(p) for p in phones)
    cards = re.findall(r"\b\d{4} (?:\d{4}|xxxx) (?:\d{4}|xxxx) \d{4}\b", text)
    full = [c for c in cards if "x" not in c]
    assert cards and full and all(c.startswith("9999") for c in cards) and not any(ids.luhn_valid(c) for c in full)
    assert ids.luhn_valid("4111 1111 1111 1111") and not ids.is_drama_number("0221 4711123")
