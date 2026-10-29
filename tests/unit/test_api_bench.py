# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_api_bench.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for the REST API (FastAPI TestClient) and the resumable bench runner with a fake backend.
------------------------------------------------------------------------------
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from fluxsense.api.app import create_app
from fluxsense.bench import BenchRunner, format_summary
from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from tests.conftest import FakeBackend
from tests.unit.test_pipeline import invoice_handler


def letter_or_invoice(request: LLMRequest) -> Optional[Any]:
    """Classifies everything as INVOICE and reuses the invoice answers."""
    return invoice_handler(request)


def test_api_endpoints(settings: Any, samples: Dict[str, Any]) -> None:
    """health, models, catalog and analyze."""
    client = TestClient(create_app(settings, backend=FakeBackend(letter_or_invoice)))
    assert client.get("/v1/health").json()["status"] == "ok"
    assert client.get("/v1/models").json()["served"] == ["fake"]
    assert "INVOICE" in client.get("/v1/catalog").json()
    pdf = samples["invoice_wrong_vat"].path
    resp = client.post("/v1/analyze", files={"file": ("rechnung.pdf", pdf.read_bytes(), "application/pdf")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"]["path"] == "rechnung.pdf"
    assert body["logical_documents"][0]["plausibility"]["status"] == "FAIL"
    assert client.post("/v1/analyze", files={"file": ("x.pdf", b"hello", "application/pdf")}).status_code == 415
    assert client.post("/v1/analyze?ocr_engine=bogus", files={"file": ("x.pdf", b"%PDF", "application/pdf")}).status_code == 422
    assert "/v1/analyze" in client.get("/openapi.json").json()["paths"]


def test_bench_resumable_with_summary(settings: Any, samples: Dict[str, Any], tmp_path: Path) -> None:
    """Results per file, summary CSV/JSON with budget and trust; a second run skips finished files."""
    source = tmp_path / "in"
    source.mkdir()
    shutil.copy(samples["invoice_wrong_vat"].path, source / "a.pdf")
    shutil.copy(samples["letter"].path, source / "b.pdf")
    backend = FakeBackend(letter_or_invoice)
    out = tmp_path / "out"
    summary = BenchRunner(settings, analyzer=Analyzer(settings, backend=backend)).run(source, out)
    assert sorted(p.name for p in (out / "results").iterdir()) == ["a.json", "b.json"]
    assert summary["files"] == 2 and summary["errors"] == 0 and summary["files_analyzed_this_run"] == 2
    assert {"wall_s_per_page", "ai_seconds_total", "ai_s_per_page", "ai_s_per_page_p95", "docs_per_hour"} <= set(summary["budget"])
    assert abs(summary["budget"]["wall_s_per_page"] - summary["wall_seconds"] / summary["pages"]) < 0.06
    assert summary["quality"]["trust_mean"] is not None
    rows = list(csv.DictReader((out / "summary.csv").open(encoding="utf-8")))
    assert rows[0]["plausibility"] == "FAIL" and "invoice.net_plus_tax" in rows[0]["finding_codes"]
    assert json.loads((out / "summary.json").read_text())["files"] == 2
    text = format_summary(summary)
    assert text.splitlines()[1].startswith("BUDGET wall-clock") and "AI-seconds" in text
    acceptance = summary["acceptance"]
    assert acceptance["auto_accepted"] + acceptance["flagged_for_review"] > 0 and 0 <= acceptance["auto_accept_rate"] <= 1
    assert "doc_date" in acceptance["per_field"] and "auto_accept_rate" in acceptance["per_field"]["doc_date"]
    assert rows[0]["auto_accept_rate"] != "" and "auto_accept_rate" in text
    calls = len(backend.requests)
    again = BenchRunner(settings, analyzer=Analyzer(settings, backend=backend)).run(source, out)
    assert again["files_analyzed_this_run"] == 0 and len(backend.requests) == calls


def test_bench_isolates_broken_pdf(settings: Any, samples: Dict[str, Any], tmp_path: Path) -> None:
    """A broken PDF is recorded as ERROR without stopping the run."""
    source = tmp_path / "in"
    source.mkdir()
    (source / "broken.pdf").write_bytes(b"%PDF-1.4 garbage")
    shutil.copy(samples["invoice_wrong_vat"].path, source / "ok.pdf")
    summary = BenchRunner(settings, analyzer=Analyzer(settings, backend=FakeBackend(letter_or_invoice))).run(source, tmp_path / "o")
    assert summary["errors"] == 1 and summary["files"] == 2
    assert json.loads((tmp_path / "o" / "results" / "broken.json").read_text())["status"] == "ERROR"


def test_acceptance_summary_per_field() -> None:
    """auto_accept_rate overall and per short field name; unfilled records are ignored."""
    from fluxsense.bench import acceptance_field, acceptance_summary
    from fluxsense.models.result import AnalysisResult, FieldEvidence, LogicalDocument
    evidence = [FieldEvidence(path="meta_header.doc_date", auto_accepted=True, provenance="label"),
                FieldEvidence(path="bodies.finance_body.invoice_date", auto_accepted=False, provenance="model"),
                FieldEvidence(path="bodies.finance_body.payment_accounts.0.iban", auto_accepted=True, provenance="check"),
                FieldEvidence(path="meta_header.sender.street", auto_accepted=False, provenance="model"),
                FieldEvidence(path="meta_header.recipient.name", auto_accepted=None)]
    result = AnalysisResult.model_construct(logical_documents=[LogicalDocument.model_construct(type="INVOICE", evidence=evidence)])
    summary = acceptance_summary([result])
    assert summary["auto_accept_rate"] == 0.6667 and summary["provenance"] == {"label": 1, "model": 1, "check": 1}
    assert summary["informational"]["filled"] == 1 and summary["informational"]["auto_accepted"] == 0
    assert summary["per_field"]["doc_date"] == {"filled": 2, "auto_accepted": 1, "auto_accept_rate": 0.5}
    assert summary["per_field"]["iban"]["auto_accept_rate"] == 1.0 and "recipient" not in summary["per_field"]
    assert acceptance_field("meta_header.sender.street") == "other" and "other" not in summary["per_field"]


def test_reassess_replays_acceptance_offline(settings: Any, samples: Dict[str, Any], tmp_path: Path) -> None:
    """The bench cache keeps word boxes and layout items; 'reassess' re-scores saved results without LLM calls."""
    from fluxsense.reassess import layer_from_json, reassess_run
    source = tmp_path / "in"
    source.mkdir()
    shutil.copy(samples["invoice_wrong_vat"].path, source / "a.pdf")
    backend = FakeBackend(letter_or_invoice)
    out = tmp_path / "out"
    BenchRunner(settings, analyzer=Analyzer(settings, backend=backend)).run(source, out)
    cache = json.loads((out / ".cache" / "a.json").read_text(encoding="utf-8"))
    assert cache["pages"][0]["words"] and layer_from_json(cache).pages[0].words
    calls = len(backend.requests)
    summary = reassess_run(out, tmp_path / "re")
    assert len(backend.requests) == calls and summary["files"] == 1
    original = json.loads((out / "results" / "a.json").read_text(encoding="utf-8"))
    again = json.loads((tmp_path / "re" / "results" / "a.json").read_text(encoding="utf-8"))

    def accepted(data: Dict[str, Any]) -> List[Any]:
        return sorted((e["path"], e["auto_accepted"]) for e in data["logical_documents"][0]["evidence"]
                      if e.get("auto_accepted") is not None)

    assert accepted(original) == accepted(again)


def test_concurrent_prepares_are_safe(settings: Any, samples: Dict[str, Any]) -> None:
    """PyMuPDF is not thread-safe (find_tables keeps a module global): concurrent prepares run under one lock."""
    from concurrent.futures import ThreadPoolExecutor
    analyzer = Analyzer(settings, backend=FakeBackend(lambda r: None))
    paths = [samples["invoice_multipage"].path, samples["bank_statement_3p"].path, samples["invoice_wrong_vat"].path] * 3
    with ThreadPoolExecutor(max_workers=6) as pool:
        prepared = list(pool.map(analyzer.prepare, paths))
    assert all(p.layer.pages and p.layer.pages[0].sources.get("native") for p in prepared)
    assert prepared[0].layer.tables() and len(prepared[1].layer.pages) == 3


def test_error_codes_and_page_text(settings: Any, samples: Dict[str, Any], tmp_path: Path) -> None:
    """Per-page text on request; machine-readable error codes for an unreachable backend and invalid input."""
    from fluxsense.llm.base import BackendUnreachable
    from fluxsense.pipeline import AnalyzeOptions

    def handler(request: Any) -> Any:
        return letter_or_invoice(request)

    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["letter"].path, AnalyzeOptions(include_page_text=True))
    assert result.text_layer.pages and result.text_layer.pages[0].page == 1 and result.text_layer.pages[0].text
    assert result.text_layer.pages[0].source in ("native", "stream", "ovis", "ocr", "docling") and result.error_code is None

    def down(request: Any) -> Any:
        raise BackendUnreachable("LLM backend unreachable at http://localhost:8080/v1")

    result = Analyzer(settings, backend=FakeBackend(down)).analyze(samples["letter"].path)
    assert result.status == "ERROR" and result.error_code == "BACKEND_UNREACHABLE"
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(b"not a pdf at all")
    result = Analyzer(settings, backend=FakeBackend(handler)).analyze(junk)
    assert result.status == "ERROR" and result.error_code == "INPUT_INVALID"


def test_rejected_api_key_is_backend_auth(settings: Any, samples: Dict[str, Any]) -> None:
    """A rejected API key is BACKEND_AUTH (not BACKEND_UNREACHABLE): an ERROR result, REST 502 and a degraded health."""
    from fluxsense.llm.base import BackendAuthError, auth_message

    class Rejecting(FakeBackend):
        def list_models(self) -> List[str]:
            raise BackendAuthError(auth_message(401, "http://ai.invalid/v1"))

    def reject(request: Any) -> Any:
        raise BackendAuthError(auth_message(401, "http://ai.invalid/v1"))

    result = Analyzer(settings, backend=FakeBackend(reject)).analyze(samples["letter"].path)
    assert result.status == "ERROR" and result.error_code == "BACKEND_AUTH" and "invalid API key" in result.errors[0]
    client = TestClient(create_app(settings, backend=Rejecting(reject)))
    resp = client.post("/v1/analyze", files={"file": ("brief.pdf", samples["letter"].path.read_bytes(), "application/pdf")})
    assert resp.status_code == 502 and resp.json()["detail"]["error_code"] == "BACKEND_AUTH"
    health = client.get("/v1/health").json()
    assert health["status"] == "degraded" and health["llm_backend_error"] == "BACKEND_AUTH"
    assert client.get("/v1/models").status_code == 502
