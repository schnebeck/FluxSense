# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_deferred.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Deferred verify phase: identical final results to inline mode
                with the same mocked verifier answers, verify calls only after
                all extraction calls (one model block), resume after a crash
                between the phases from the persistent JSONL queue.
------------------------------------------------------------------------------
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from fluxsense.bench import BenchRunner
from fluxsense.deferred import DeferredStore, fingerprint, request_from_json, request_to_json
from fluxsense.llm.base import LLMRequest
from fluxsense.llm.images import ImagePart
from fluxsense.pipeline import Analyzer
from tests.conftest import FakeBackend
from tests.unit.test_pipeline import invoice_handler


def comparable(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The parts of a result that must not depend on the verify mode."""
    docs = []
    for doc in result["logical_documents"]:
        docs.append({
            "fields": doc["fields"], "type": doc["type"], "pages": doc["pages"],
            "checks": sorted((c["check_id"], c["status"], c.get("field") or "") for c in doc["plausibility"]["checks"]),
            "evidence": sorted((e["path"], str(e.get("value")), e["status"], str(e.get("action"))) for e in doc["evidence"]),
            "corrections": sorted(c["note"] for c in doc["corrections"]), "trust": doc["trust"]["score"],
        })
    return docs


def run(settings: Any, source: Path, out: Path, mode: str, backend: FakeBackend, **kwargs: Any) -> Dict[str, Any]:
    """Runs the bench in a verify mode and returns the summary."""
    settings.pipeline.verify_mode = mode
    settings.models.vision = "verifier-27b"
    settings.llm.max_concurrency = 1
    return BenchRunner(settings, analyzer=Analyzer(settings, backend=backend)).run(source, out, **kwargs)


def corpus(samples: Dict[str, Any], tmp_path: Path) -> Path:
    """Two synthetic PDFs (one needs a zoom verification)."""
    source = tmp_path / "in"
    source.mkdir()
    shutil.copy(samples["invoice_wrong_vat"].path, source / "a.pdf")
    shutil.copy(samples["letter"].path, source / "b.pdf")
    return source


def results(out: Path) -> Dict[str, Any]:
    """Result JSON per file."""
    return {p.name: json.loads(p.read_text()) for p in sorted((out / "results").glob("*.json"))}


def test_request_serialization_and_fingerprint() -> None:
    """Queued requests round-trip; the fingerprint ignores labels but not content."""
    request = LLMRequest(prompt="p", label="Zoom verify p1", model="m", images=[ImagePart("CROP F1", "data:x", 1, 2)])
    again = request_from_json(json.loads(json.dumps(request_to_json(request))))
    assert fingerprint(again) == fingerprint(request) and again.images[0].label == "CROP F1"
    assert fingerprint(LLMRequest(prompt="q", label="Zoom verify p1", model="m")) != fingerprint(request)


def test_deferred_equals_inline_and_batches_verify_calls(settings: Any, samples: Dict[str, Any], tmp_path: Path) -> None:
    """Same verifier answers -> identical final results; all verify calls come after all extraction calls."""
    source = corpus(samples, tmp_path)
    inline_backend, deferred_backend = FakeBackend(invoice_handler), FakeBackend(invoice_handler)
    run(settings, source, tmp_path / "inline", "inline", inline_backend)
    summary = run(settings, source, tmp_path / "deferred", "deferred", deferred_backend)
    a, b = results(tmp_path / "inline"), results(tmp_path / "deferred")
    assert a.keys() == b.keys()
    for name in a:
        assert comparable(a[name]) == comparable(b[name]), name
        assert b[name]["verification_pending"] is False
    labels = deferred_backend.labels()
    first_verify = min(i for i, l in enumerate(labels) if l.startswith("Zoom"))
    assert all(not l.startswith(("Stage", "Correction")) for l in labels[first_verify:])
    assert summary["deferred"]["requests"] >= 2 and summary["deferred"]["pending_after"] == 0
    assert summary["deferred"]["verify_wall_s"] >= 0
    due = next(e for e in b["a.json"]["logical_documents"][0]["evidence"] if e["path"].endswith("due_date"))
    assert due["action"] == "nulled"


def test_resume_after_crash_between_phases(settings: Any, samples: Dict[str, Any], tmp_path: Path) -> None:
    """Phase 1 writes pending results + a JSONL queue; a new run completes them without re-extracting."""
    source = corpus(samples, tmp_path)
    out = tmp_path / "out"
    run(settings, source, out, "deferred", FakeBackend(invoice_handler), stop_after_phase1=True)
    pending = results(out)
    assert pending["a.json"]["verification_pending"] is True
    store = DeferredStore(out / ".cache" / "deferred")
    assert store.open_requests() and (out / ".cache" / "deferred" / "queue.jsonl").exists()
    resumed_backend = FakeBackend(invoice_handler)
    summary = run(settings, source, out, "deferred", resumed_backend)
    assert summary["files_analyzed_this_run"] == 0 and summary["deferred"]["pending_after"] == 0
    assert not any(l.startswith("Stage") for l in resumed_backend.labels())
    inline_out = tmp_path / "inline"
    run(settings, source, inline_out, "inline", FakeBackend(invoice_handler))
    assert comparable(results(out)["a.json"]) == comparable(results(inline_out)["a.json"])


def test_single_file_deferred_matches_inline(settings: Any, samples: Dict[str, Any]) -> None:
    """analyze() of one file stays inline by default; with verify_mode=deferred it gives the same result."""
    settings.models.vision = "verifier-27b"
    inline = Analyzer(settings, backend=FakeBackend(invoice_handler)).analyze(samples["invoice_wrong_vat"].path)
    settings.pipeline.verify_mode = "deferred"
    backend = FakeBackend(invoice_handler)
    deferred = Analyzer(settings, backend=backend).analyze(samples["invoice_wrong_vat"].path)
    assert comparable(json.loads(inline.model_dump_json())) == comparable(json.loads(deferred.model_dump_json()))
    assert deferred.verification_pending is False and "verify_phase_wall" in deferred.timings
