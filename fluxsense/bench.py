# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/bench.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Batch runner for `fluxsense bench`: resumable per-file JSON
                results, producer/consumer pipelining (text layer / Docling
                for document N+1 while the LLM works on document N), N
                parallel LLM pipelines (llama-server slots), optional text
                review as a separate final phase (no model ping-pong), and a
                summary (CSV + JSON) with throughput budget, trust,
                plausibility, escalation and ZUGFeRD accuracy aggregates.
------------------------------------------------------------------------------
"""

import csv
import json
import pickle
import statistics
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluxsense import __version__
from fluxsense.config import Settings
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.log import get_logger
from fluxsense.models.result import AnalysisResult
from fluxsense.pipeline import AnalyzeOptions, Analyzer, Prepared
from fluxsense.prompts import PROMPT_VERSION

logger = get_logger("bench")

BUDGET_MEAN_S_PER_PAGE: float = 15.0
BUDGET_MAX_S_PER_PAGE: float = 30.0
CSV_FIELDS: List[str] = [
    "file", "status", "pages", "engine", "documents", "types", "plausibility", "plausibility_score", "trust_score",
    "failed_checks", "finding_codes", "fields", "grounded", "weakly_grounded", "ungrounded", "verified_visual", "nulled",
    "replaced", "table_rows", "table_rows_ungrounded", "corrections_applied", "auto_accepted", "flagged_for_review",
    "auto_accept_rate", "zugferd_accuracy", "llm_calls",
    "prompt_tokens", "completion_tokens", "ai_seconds", "ai_s_per_page", "wall_s_per_page", "text_layer_s",
    "classification_s", "extraction_s", "grounding_s", "correction_s", "over_budget", "error",
]


def discover(directory: Path) -> List[Path]:
    """All PDFs below a directory (recursive, sorted)."""
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def result_name(pdf: Path, root: Path) -> str:
    """Stable result file name for a PDF (relative path flattened)."""
    rel = pdf.relative_to(root) if pdf.is_relative_to(root) else Path(pdf.name)
    return "__".join(rel.with_suffix("").parts) + ".json"


def _layer_to_json(layer: TextLayer) -> Dict[str, Any]:
    """Text-layer cache for the review phase and the offline acceptance replay (fluxsense reassess)."""
    from fluxsense.reassess import layer_to_json
    return layer_to_json(layer)


def _layer_from_json(data: Dict[str, Any]) -> TextLayer:
    """Restores a cached text layer."""
    return TextLayer(pages=[PageText(page=p["page"], sources=p["sources"]) for p in data.get("pages", [])],
                     engine=data.get("engine", "cache"))


def summarize_row(pdf: Path, result: Optional[AnalysisResult], error: Optional[str], concurrency: int = 1) -> Dict[str, Any]:
    """One summary CSV row."""
    row: Dict[str, Any] = {k: "" for k in CSV_FIELDS}
    row["file"] = str(pdf)
    if result is None:
        row.update({"status": "ERROR", "error": error or ""})
        return row
    docs = result.logical_documents
    order = {"PASS": 0, "WARN": 1, "FAIL": 2}
    worst = max((d.plausibility.status for d in docs), key=lambda s: order.get(s, 0), default="")
    t = result.timings
    zug = [d.zugferd_comparison.accuracy for d in docs if d.zugferd_comparison and d.zugferd_comparison.accuracy is not None]
    accepted = sum(d.trust.auto_accepted for d in docs)
    flagged = sum(d.trust.flagged_for_review for d in docs)
    row.update({
        "status": result.status, "pages": result.source.page_count, "engine": result.text_layer.engine, "documents": len(docs),
        "types": "+".join(d.type for d in docs), "plausibility": worst,
        "plausibility_score": min((d.plausibility.score for d in docs), default=""),
        "trust_score": min((d.trust.score for d in docs), default=""),
        "failed_checks": sum(len(d.plausibility.findings) for d in docs),
        "finding_codes": " ".join(sorted({c for d in docs for c in d.plausibility.summary_codes()})),
        "fields": sum(d.trust.fields_total for d in docs), "grounded": sum(d.trust.grounded for d in docs),
        "weakly_grounded": sum(d.trust.weakly_grounded for d in docs), "ungrounded": sum(d.trust.ungrounded for d in docs),
        "verified_visual": sum(d.trust.verified_visual for d in docs), "nulled": sum(d.trust.nulled for d in docs),
        "replaced": sum(d.trust.replaced for d in docs), "table_rows": sum(d.trust.table_rows_total for d in docs),
        "table_rows_ungrounded": sum(d.trust.table_rows_ungrounded for d in docs),
        "corrections_applied": sum(d.trust.corrections_applied for d in docs),
        "auto_accepted": accepted, "flagged_for_review": flagged,
        "auto_accept_rate": round(accepted / (accepted + flagged), 4) if accepted + flagged else "",
        "zugferd_accuracy": round(sum(zug) / len(zug), 4) if zug else "",
        "llm_calls": len(result.ai_calls), "prompt_tokens": sum(c.prompt_tokens for c in result.ai_calls),
        "completion_tokens": sum(c.completion_tokens for c in result.ai_calls), "ai_seconds": t.get("ai_seconds", 0),
        "ai_s_per_page": t.get("ai_seconds_per_page", 0), "wall_s_per_page": t.get("wall_seconds_per_page", 0),
        "text_layer_s": t.get("text_layer", 0), "classification_s": t.get("classification", 0),
        "extraction_s": t.get("extraction", 0), "grounding_s": t.get("grounding", 0), "correction_s": t.get("correction", 0),
        # Per-file budget: its share of the slots (AI-seconds / pages / parallel slots).
        "over_budget": t.get("ai_seconds_per_page", 0) / max(1, concurrency) > BUDGET_MAX_S_PER_PAGE,
        "error": "; ".join(result.errors)[:300],
    })
    return row


# Short field names of the acceptance summary (ground-truth vocabulary) -> evidence path suffixes.
ACCEPTANCE_FIELDS: Dict[str, tuple] = {
    "doc_date": ("meta_header.doc_date", "finance_body.invoice_date", "statement_body.statement_date"),
    "doc_number": ("meta_header.doc_number", "finance_body.invoice_number", "statement_body.statement_number"),
    "sender": ("meta_header.sender.company", "meta_header.sender.name"),
    "recipient": ("meta_header.recipient.company", "meta_header.recipient.name"),
    "total_gross": ("monetary_summation.grand_total_amount", "monetary_summation.due_payable_amount"),
    "total_net": ("monetary_summation.tax_basis_total_amount", "monetary_summation.line_total_amount"),
    "tax_amount": ("monetary_summation.tax_total_amount",),
    "iban": ("iban",),
    "due_date": ("finance_body.due_date", "statement_body.payment_due_date", "correspondence_body.deadline"),
    "period_start": ("statement_body.period_start",), "period_end": ("statement_body.period_end",),
    "opening_balance": ("statement_body.opening_balance",), "closing_balance": ("statement_body.closing_balance",),
}


def acceptance_field(path: str) -> str:
    """Short field name of an evidence path ('other' if it is no core field)."""
    for name, suffixes in ACCEPTANCE_FIELDS.items():
        if any(path.endswith(suffix) for suffix in suffixes):
            return name
    return "other"


def acceptance_summary(results: List[AnalysisResult]) -> Dict[str, Any]:
    """
    Field contract §0.5 KPI inputs: auto_accept_rate (auto-accepted / filled
    fields) overall and per field, plus provenance counts.
    """
    per_field: Dict[str, Dict[str, int]] = {}
    provenance: Dict[str, int] = {}
    reasons: Dict[str, int] = {}
    accepted = flagged = 0
    info = {"filled": 0, "auto_accepted": 0}
    for result in results:
        for doc in result.logical_documents:
            for ev in doc.evidence:
                if ev.auto_accepted is None:
                    continue
                name = acceptance_field(ev.path)
                if name == "other":  # informational: not a core field of the contract
                    info["filled"] += 1
                    info["auto_accepted"] += int(ev.auto_accepted)
                    continue
                stats = per_field.setdefault(name, {"filled": 0, "auto_accepted": 0})
                stats["filled"] += 1
                stats["auto_accepted"] += int(ev.auto_accepted)
                accepted += int(ev.auto_accepted)
                flagged += int(not ev.auto_accepted)
                if ev.provenance:
                    provenance[ev.provenance] = provenance.get(ev.provenance, 0) + 1
                for reason in ev.review_reasons:
                    key = reason.split(":")[0].split(" (")[0]
                    reasons[key] = reasons.get(key, 0) + 1
    for stats in per_field.values():
        stats["auto_accept_rate"] = round(stats["auto_accepted"] / stats["filled"], 4) if stats["filled"] else None  # type: ignore[assignment]
    order = list(ACCEPTANCE_FIELDS)
    return {
        "auto_accept_rate": round(accepted / (accepted + flagged), 4) if accepted + flagged else None,
        "auto_accepted": accepted, "flagged_for_review": flagged, "provenance": provenance,
        "review_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "per_field": {name: per_field[name] for name in order if name in per_field},
        "informational": {**info, "auto_accept_rate": round(info["auto_accepted"] / info["filled"], 4) if info["filled"] else None,
                          "note": "non-core fields (addresses, phones, line details): not part of the KPI"},
    }


def _percentile(values: List[float], q: float) -> float:
    """Simple percentile (nearest rank)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def aggregate(rows: List[Dict[str, Any]], results: List[AnalysisResult], wall_seconds: float) -> Dict[str, Any]:
    """Budget, quality and escalation aggregates."""
    ok = [r for r in rows if r["status"] != "ERROR"]
    per_page = [float(r["ai_s_per_page"]) for r in ok if r["ai_s_per_page"] != ""]
    latency = [float(r["wall_s_per_page"]) for r in ok if r["wall_s_per_page"] != ""]
    pages = sum(int(r["pages"] or 0) for r in ok)
    ai_total = sum(float(r["ai_seconds"] or 0) for r in ok)
    escalations: Dict[str, int] = {}
    for result in results:
        for doc in result.logical_documents:
            for level, count in doc.trust.escalations.items():
                escalations[level] = escalations.get(level, 0) + count
    trust = [float(r["trust_score"]) for r in ok if r["trust_score"] != ""]
    zug = [float(r["zugferd_accuracy"]) for r in ok if r["zugferd_accuracy"] != ""]
    status_counts: Dict[str, int] = {}
    for r in ok:
        status_counts[r["plausibility"] or "NONE"] = status_counts.get(r["plausibility"] or "NONE", 0) + 1
    return {
        "files": len(rows), "errors": len(rows) - len(ok), "pages": pages, "wall_seconds": round(wall_seconds, 1),
        "budget": {
            "wall_s_per_page": round(wall_seconds / pages, 2) if pages else 0.0,
            "metric": "wall_s_per_page = run wall-clock / pages (THE budget metric); AI-seconds are reported separately",
            "ai_seconds_total": round(ai_total, 1),
            "ai_s_per_page": round(ai_total / pages, 2) if pages else 0.0,
            "ai_s_per_page_mean": round(statistics.mean(per_page), 2) if per_page else 0.0,
            "ai_s_per_page_p50": round(_percentile(per_page, 0.5), 2), "ai_s_per_page_p95": round(_percentile(per_page, 0.95), 2),
            "file_latency_s_per_page_p50": round(_percentile(latency, 0.5), 2),
            "file_latency_s_per_page_p95": round(_percentile(latency, 0.95), 2),
            "docs_per_hour": round(len(ok) / wall_seconds * 3600, 1) if wall_seconds else 0.0,
            "pages_per_hour": round(pages / wall_seconds * 3600, 1) if wall_seconds else 0.0,
            "over_budget_files": [r["file"] for r in ok if r["over_budget"] is True],
            "target_mean_s_per_page": BUDGET_MEAN_S_PER_PAGE, "target_max_s_per_page": BUDGET_MAX_S_PER_PAGE,
        },
        "quality": {
            "trust_mean": round(statistics.mean(trust), 3) if trust else None,
            "plausibility_status": status_counts,
            "fields": sum(int(r["fields"] or 0) for r in ok), "grounded": sum(int(r["grounded"] or 0) for r in ok),
            "weakly_grounded": sum(int(r["weakly_grounded"] or 0) for r in ok),
            "ungrounded": sum(int(r["ungrounded"] or 0) for r in ok), "verified_visual": sum(int(r["verified_visual"] or 0) for r in ok),
            "nulled": sum(int(r["nulled"] or 0) for r in ok), "corrections_applied": sum(int(r["corrections_applied"] or 0) for r in ok),
            "zugferd_accuracy_mean": round(statistics.mean(zug), 4) if zug else None,
            "escalation_levels": escalations,
        },
        "acceptance": acceptance_summary(results),
        "llm": {"calls": sum(int(r["llm_calls"] or 0) for r in ok), "prompt_tokens": sum(int(r["prompt_tokens"] or 0) for r in ok),
                "completion_tokens": sum(int(r["completion_tokens"] or 0) for r in ok)},
    }


class BenchRunner:
    """Runs a resumable benchmark over a directory."""

    def __init__(self, settings: Settings, analyzer: Optional[Analyzer] = None, options: Optional[AnalyzeOptions] = None) -> None:
        """
        Args:
            settings: Settings.
            analyzer: Analyzer (default: built from settings).
            options: Analysis options.
        """
        self.settings = settings
        self.analyzer = analyzer or Analyzer(settings)
        self.options = options or AnalyzeOptions()
        self._lock = threading.Lock()
        self._record = None
        self._replay = None

    def run(self, source: Path, out_dir: Path, force: bool = False, limit: Optional[int] = None,
            stop_after_phase1: bool = False) -> Dict[str, Any]:
        """
        Executes the benchmark.

        Args:
            source: Directory with PDFs (or a single PDF).
            out_dir: Output directory.
            force: Re-analyze files that already have a result.
            limit: Max number of files.
            stop_after_phase1: Deferred mode, tests only: stop before the verify phase (simulated crash).

        Returns:
            The summary dict (also written to out_dir/summary.json).
        """
        from contextlib import nullcontext
        from fluxsense.wol import lease_for
        lease = lease_for(self.settings.hooks)  # None unless a WOL-Server is configured
        with lease.job() if lease is not None else nullcontext():
            return self._run(source, out_dir, force, limit, stop_after_phase1)

    def _run(self, source: Path, out_dir: Path, force: bool, limit: Optional[int], stop_after_phase1: bool) -> Dict[str, Any]:
        """run() without the optional WOL job bracket."""
        deferred = self.settings.pipeline.verify_mode == "deferred"
        self._deferred_setup(out_dir) if deferred else None
        files = [source] if source.is_file() else discover(source)
        root = source.parent if source.is_file() else source
        if limit:
            files = files[:limit]
        results_dir, cache_dir = out_dir / "results", out_dir / ".cache"
        results_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bench_meta.json").write_text(json.dumps({
            "fluxsense_version": __version__, "prompt_version": PROMPT_VERSION, "source": str(source),
            "settings": self.settings.public_dump(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2), encoding="utf-8")
        todo, done = [], []
        for pdf in files:
            target = results_dir / result_name(pdf, root)
            if not force and target.exists() and (self._is_complete(target) or self._is_pending(target)):
                done.append(pdf)  # pending ones are completed in the deferred phase (resume)
            else:
                todo.append(pdf)
        logger.info(f"bench: {len(files)} files, {len(done)} already done, {len(todo)} to analyze")
        start = time.monotonic()
        prep_pool = ThreadPoolExecutor(max_workers=max(1, self.settings.extract.docling_concurrency), thread_name_prefix="prep")
        llm_pool = ThreadPoolExecutor(max_workers=max(1, self.settings.llm.max_concurrency), thread_name_prefix="llm")
        prepared: Dict[Path, Future] = {pdf: prep_pool.submit(self._prepare, pdf) for pdf in todo}
        futures = [llm_pool.submit(self._analyze, pdf, prepared[pdf], results_dir / result_name(pdf, root),
                                   cache_dir / result_name(pdf, root)) for pdf in todo]
        for future in futures:
            future.result()
        prep_pool.shutdown()
        llm_pool.shutdown()
        deferred_stats: Dict[str, Any] = {}
        if deferred and not stop_after_phase1:
            deferred_stats = self._deferred_phase(files, root, results_dir, cache_dir)
        review_text = self.settings.review.text if self.options.review_text is None else self.options.review_text
        if review_text:
            self._review_phase(files, root, results_dir, cache_dir)
        wall = time.monotonic() - start
        rows, results = [], []
        for pdf in files:
            target = results_dir / result_name(pdf, root)
            result, error = None, None
            if target.exists():
                data = json.loads(target.read_text(encoding="utf-8"))
                if "fluxsense_version" in data:
                    result = AnalysisResult.model_validate(data)
                    results.append(result)
                else:
                    error = data.get("error")
            rows.append(summarize_row(pdf, result, error, self.settings.llm.max_concurrency))
        with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        summary = aggregate(rows, results, wall if todo else sum(float(r["ai_seconds"] or 0) for r in rows))
        summary["files_analyzed_this_run"] = len(todo)
        if deferred:
            summary["deferred"] = deferred_stats
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @staticmethod
    def _is_complete(target: Path) -> bool:
        """True if a stored result is a finished (non-error) analysis."""
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.info(f"Re-analyzing {target.name}: unreadable result ({e})")
            return False
        return data.get("status") in ("OK", "PARTIAL") and not data.get("verification_pending")

    @staticmethod
    def _is_pending(target: Path) -> bool:
        """True for an analysed result whose deferred verification is still open."""
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.info(f"unreadable result {target.name}: {e}")
            return False
        return bool(data.get("verification_pending")) and data.get("status") in ("OK", "PARTIAL")

    def _prepare(self, pdf: Path) -> Prepared:
        """Producer: text layer + XML."""
        logger.info(f"prepare {pdf.name}")
        return self.analyzer.prepare(pdf, self.options.engine)

    def _analyze(self, pdf: Path, prepared_future: Future, target: Path, cache: Path) -> None:
        """Consumer: LLM analysis of one file, written immediately (resumable)."""
        try:
            prepared = prepared_future.result()
            cache.write_text(json.dumps(_layer_to_json(prepared.layer), ensure_ascii=False), encoding="utf-8")
            if self._record is not None:
                self._record.set_doc(target.name)
                result = self._record_analyzer.analyze_prepared(prepared, self.options)
                result.verification_pending = self._record.pending
                if result.verification_pending:
                    cache.with_suffix(".prepared.pkl").write_bytes(pickle.dumps(prepared))
            else:
                result = self.analyzer.analyze_prepared(prepared, self.options)
            target.write_text(result.model_dump_json(indent=1), encoding="utf-8")
            logger.info(f"done {pdf.name}: {result.status}, {result.timings.get('ai_seconds_per_page')} s/page AI")
        except Exception as e:  # noqa: BLE001 - one broken PDF must not stop the benchmark; the error is persisted
            logger.error(f"bench: {pdf.name} failed: {e}")
            target.write_text(json.dumps({"status": "ERROR", "file": str(pdf), "error": f"{e.__class__.__name__}: {e}",
                                          "traceback": traceback.format_exc()[-4000:]}), encoding="utf-8")

    def _deferred_setup(self, out_dir: Path) -> None:
        """Creates the persistent store and the record/replay analyzers."""
        from fluxsense.deferred import RECORD, REPLAY, CachingBackend, DeferredStore
        self._store = DeferredStore(out_dir / ".cache" / "deferred")
        self._real = self.analyzer.backend
        self._record = CachingBackend(self._real, self._store, self.settings, RECORD)
        self._replay = CachingBackend(self._real, self._store, self.settings, REPLAY)
        self._record_analyzer = Analyzer(self.settings, backend=self._record, catalog=self.analyzer.catalog)
        self._replay_analyzer = Analyzer(self.settings, backend=self._replay, catalog=self.analyzer.catalog)

    def _deferred_phase(self, files: List[Path], root: Path, results_dir: Path, cache_dir: Path) -> Dict[str, Any]:
        """
        Runs queued verify requests (grouped by model) and replays pending
        documents with cached answers until no new request appears.

        Returns:
            Phase statistics (rounds, requests, verify wall seconds, still pending).
        """
        from fluxsense.deferred import run_queue
        stats = {"rounds": 0, "requests": 0, "verify_wall_s": 0.0, "replay_wall_s": 0.0, "pending_after": 0}
        for _ in range(max(1, self.settings.pipeline.deferred_rounds)):
            pending = []
            for pdf in files:
                target = results_dir / result_name(pdf, root)
                if target.exists() and json.loads(target.read_text(encoding="utf-8")).get("verification_pending"):
                    pending.append((pdf, target, cache_dir / result_name(pdf, root)))
            if not pending:
                break
            count, seconds = run_queue(self._store, self._real, self.settings.llm.max_concurrency)
            stats["rounds"] += 1
            stats["requests"] += count
            stats["verify_wall_s"] = round(stats["verify_wall_s"] + seconds, 3)
            start = time.monotonic()
            for pdf, target, cache in pending:
                pickled = cache.with_suffix(".prepared.pkl")
                if not pickled.exists():
                    logger.warning(f"deferred: no prepared input for {pdf.name}; re-preparing")
                    prepared = self.analyzer.prepare(pdf, self.options.engine)
                else:
                    prepared = pickle.loads(pickled.read_bytes())
                self._replay.set_doc(target.name)
                result = self._replay_analyzer.analyze_prepared(prepared, self.options)
                result.verification_pending = self._replay.pending
                result.timings["verify_phase_wall"] = stats["verify_wall_s"]
                target.write_text(result.model_dump_json(indent=1), encoding="utf-8")
                if not result.verification_pending:
                    pickled.unlink(missing_ok=True)
            stats["replay_wall_s"] = round(stats["replay_wall_s"] + time.monotonic() - start, 3)
        stats["pending_after"] = sum(1 for pdf in files if (results_dir / result_name(pdf, root)).exists()
                                     and json.loads((results_dir / result_name(pdf, root)).read_text(encoding="utf-8"))
                                     .get("verification_pending"))
        return stats

    def _review_phase(self, files: List[Path], root: Path, results_dir: Path, cache_dir: Path) -> None:
        """Text review with the second model as a separate, final batch phase."""
        for pdf in files:
            target = results_dir / result_name(pdf, root)
            cache = cache_dir / result_name(pdf, root)
            if not target.exists() or not cache.exists():
                continue
            data = json.loads(target.read_text(encoding="utf-8"))
            if "fluxsense_version" not in data:
                continue
            result = AnalysisResult.model_validate(data)
            if all(d.review and d.review.text_verdict for d in result.logical_documents):
                continue
            layer = _layer_from_json(json.loads(cache.read_text(encoding="utf-8")))
            self.analyzer.review_text(result, layer)
            target.write_text(result.model_dump_json(indent=1), encoding="utf-8")


def _acceptance_line(acceptance: Dict[str, Any]) -> str:
    """Terminal line of the acceptance summary."""
    if not acceptance or acceptance.get("auto_accept_rate") is None:
        return "auto-accept: no filled fields assessed"
    fields = ", ".join(f"{name} {stats['auto_accept_rate']} ({stats['auto_accepted']}/{stats['filled']})"
                       for name, stats in acceptance["per_field"].items())
    info = acceptance.get("informational") or {}
    return (f"auto_accept_rate {acceptance['auto_accept_rate']} ({acceptance['auto_accepted']} accepted, "
            f"{acceptance['flagged_for_review']} flagged; provenance {acceptance['provenance']}); per field: {fields}; "
            f"informational (not in KPI): {info.get('auto_accepted')}/{info.get('filled')}")


def format_summary(summary: Dict[str, Any]) -> str:
    """Human-readable budget summary for the terminal."""
    b, q = summary["budget"], summary["quality"]
    lines = [
        f"files {summary['files']} (errors {summary['errors']}), pages {summary['pages']}, wall {summary['wall_seconds']} s",
        f"BUDGET wall-clock: {b['wall_s_per_page']} s/page (run wall / pages; target <= {b['target_mean_s_per_page']}), "
        f"{b['docs_per_hour']} docs/h, {b['pages_per_hour']} pages/h",
        f"AI-seconds (sum of call latencies, overlapping under concurrency): {b['ai_seconds_total']} s total, "
        f"{b['ai_s_per_page']} s/page; per file p50 {b['ai_s_per_page_p50']} | p95 {b['ai_s_per_page_p95']}",
        f"over budget: {len(b['over_budget_files'])} files",
        (f"deferred verify phase: {summary['deferred'].get('requests')} requests in {summary['deferred'].get('rounds')} rounds, "
         f"verify wall {summary['deferred'].get('verify_wall_s')} s, replay {summary['deferred'].get('replay_wall_s')} s, "
         f"still pending {summary['deferred'].get('pending_after')}") if summary.get("deferred") else "deferred verify: off",
        f"trust mean {q['trust_mean']}, plausibility {q['plausibility_status']}, fields {q['fields']} "
        f"(grounded {q['grounded']}, weak {q['weakly_grounded']}, ungrounded {q['ungrounded']}, verified {q['verified_visual']}, "
        f"nulled {q['nulled']}), corrections {q['corrections_applied']}, escalation levels {q['escalation_levels']}, "
        f"ZUGFeRD accuracy {q['zugferd_accuracy_mean']}",
        _acceptance_line(summary.get("acceptance") or {}),
        f"LLM: {summary['llm']['calls']} calls, {summary['llm']['prompt_tokens']} prompt / {summary['llm']['completion_tokens']} completion tokens",
    ]
    return "\n".join(lines)
