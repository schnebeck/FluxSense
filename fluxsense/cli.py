# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/cli.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Command line: analyze, bench, serve, config (show / path /
                init / import-kpaperflux), doctor, catalog, samples,
                gen-testdocs, score-testdocs.
------------------------------------------------------------------------------
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import httpx

from fluxsense import __version__
from fluxsense.config import Settings, config_path, load_settings, resolve_api_key, save_identity, save_settings
from fluxsense.extract import ENGINES
from fluxsense.log import get_logger, setup_logging
from fluxsense.llm.base import AUTH_STATUS, auth_message

logger = get_logger("cli")


def _options(args: argparse.Namespace) -> "AnalyzeOptions":
    """Builds AnalyzeOptions from CLI flags."""
    from fluxsense.pipeline import AnalyzeOptions
    return AnalyzeOptions(engine=args.ocr_engine, include_page_text=bool(getattr(args, "page_text", False)),
                          review_vision=True if args.review_vision else None,
                          review_text=True if args.review_text else (False if args.no_review else None),
                          grounding=False if args.no_grounding else None, audit=False if args.no_audit else None)


def _apply_common(settings: Settings, args: argparse.Namespace) -> None:
    """Applies flags that override settings."""
    if getattr(args, "no_review", False):
        settings.review.vision = False
        settings.review.text = False
    if getattr(args, "concurrency", None):
        settings.llm.max_concurrency = args.concurrency
    if getattr(args, "model", None):
        settings.models.primary = settings.models.vision = args.model


def cmd_analyze(args: argparse.Namespace, settings: Settings) -> int:
    """fluxsense analyze <pdf|dir>."""
    from fluxsense.bench import discover
    from fluxsense.pipeline import Analyzer
    _apply_common(settings, args)
    target = Path(args.path)
    files = discover(target) if target.is_dir() else [target]
    analyzer = Analyzer(settings)
    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in files:
        result = analyzer.analyze(pdf, _options(args))
        payload = result.model_dump_json(indent=2)
        if out_dir:
            (out_dir / f"{pdf.stem}.json").write_text(payload, encoding="utf-8")
        if args.json:
            print(payload)
        else:
            print(f"{pdf.name}: {result.status}, {len(result.logical_documents)} document(s), "
                  f"AI {result.timings.get('ai_seconds')} s ({result.timings.get('ai_seconds_per_page')} s/page), "
                  f"{len(result.ai_calls)} LLM calls")
            for doc in result.logical_documents:
                print(f"  [{doc.index}] {doc.type} pages {doc.pages} {doc.direction}/{doc.tenant_context} "
                      f"plausibility {doc.plausibility.status} ({doc.plausibility.score}) trust {doc.trust.score}")
                for finding in doc.plausibility.findings:
                    print(f"      {finding.severity.value:7} {finding.check_id}: {finding.message}")
    return 0


def cmd_bench(args: argparse.Namespace, settings: Settings) -> int:
    """fluxsense bench <dir> --out DIR."""
    from fluxsense.bench import BenchRunner, format_summary
    _apply_common(settings, args)
    runner = BenchRunner(settings, options=_options(args))
    summary = runner.run(Path(args.path), Path(args.out), force=args.force, limit=args.limit)
    print(format_summary(summary))
    print(f"results: {Path(args.out) / 'results'}  summary: {Path(args.out) / 'summary.csv'}, {Path(args.out) / 'summary.json'}")
    return 0


def cmd_reassess(args: argparse.Namespace, settings: Settings) -> int:
    """fluxsense reassess <run dir> --out DIR: offline acceptance replay of saved bench results (no LLM calls)."""
    from fluxsense.bench import _acceptance_line
    from fluxsense.reassess import reassess_run
    summary = reassess_run(Path(args.run), Path(args.out), Path(args.pdf_root) if args.pdf_root else None,
                           layout=not args.no_layout_replay)
    print(f"files {summary['files']}")
    print(_acceptance_line(summary["acceptance"]))
    print(f"results: {Path(args.out) / 'results'}")
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    """fluxsense serve."""
    import uvicorn
    from fluxsense.api.app import create_app
    uvicorn.run(create_app(settings), host=args.host or settings.server.host, port=args.port or settings.server.port)
    return 0


def cmd_config(args: argparse.Namespace, settings: Settings) -> int:
    """fluxsense config show|path|init|import-kpaperflux."""
    if args.action == "path":
        print(config_path())
    elif args.action == "show":
        data = settings.model_dump(mode="json")
        if not args.with_identity:
            data["identity"] = {k: ("configured" if not getattr(settings.identity, k).is_empty() else "empty")
                                for k in ("private", "business")}
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.action == "init":
        path = config_path()
        if path.exists() and not args.force:
            print(f"{path} exists (use --force to overwrite)")
            return 1
        print(f"written {save_settings(settings, path)}")
    elif args.action == "import-kpaperflux":
        from fluxsense.config_import import KPAPERFLUX_CONF, import_kpaperflux_identities
        source = Path(args.conf).expanduser() if args.conf else KPAPERFLUX_CONF
        private, business = import_kpaperflux_identities(source)
        if private:
            settings.identity.private = private
        if business:
            settings.identity.business = business
        print(f"read {source} (read-only): private={'yes' if private else 'no'}, business={'yes' if business else 'no'}")
        if args.dry_run:
            return 0
        print(f"written [identity] to {save_identity(settings.identity)}")
    return 0


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    """Checks endpoints, models, engines and local tools."""
    import shutil
    from fluxsense.extract import docling_engine
    problems = 0

    def report(ok: bool, label: str, detail: str = "") -> None:
        """Prints one check line."""
        nonlocal problems
        problems += 0 if ok else 1
        print(f"[{'ok' if ok else '!!'}] {label}{': ' + detail if detail else ''}")

    print(f"FluxSense {__version__}, config {config_path()} ({'found' if config_path().exists() else 'defaults'})")
    key = resolve_api_key(settings.llm)
    report(key is not None, "API key", "resolved" if key else f"none (env {settings.llm.api_key_env} or api_key_command)")
    keepalive = Path(settings.hooks.keepalive_file) if settings.hooks.keepalive_file else None
    if keepalive:
        print(f"[--] keep-awake file {keepalive}: {'present' if keepalive.exists() else 'absent on this host'}")
    print(f"[--] wake hook: {'configured' if settings.hooks.wake_command.strip() else 'not configured'}")
    from fluxsense.grounding.blz import USER_BLZ_FILE, load_blz
    print(f"[--] German bank-code list: {len(load_blz())} codes ({USER_BLZ_FILE})" if load_blz()
          else "[--] German bank-code list: not installed (optional; 'fluxsense blz-update' downloads it)")
    from fluxsense.wol import WolError, lease_for
    lease = lease_for(settings.hooks)
    if lease is None:
        print("[--] WOL: not configured")
    else:
        try:
            requests = lease.status()
            report(True, "WOL-Server", f"{settings.hooks.wol_url} device '{settings.hooks.wol_device}', "
                                       f"open requests: {len(requests.get('requests') or [])}")
        except (WolError, httpx.HTTPError) as e:
            report(False, "WOL-Server", f"{settings.hooks.wol_url}: {e}")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        resp = httpx.get(f"{settings.llm.base_url.rstrip('/')}/models", headers=headers, timeout=10.0)
        if resp.status_code in AUTH_STATUS:
            report(False, "llm endpoint", auth_message(resp.status_code, settings.llm.base_url))
        resp.raise_for_status()
        models = {m.get("id"): m for m in resp.json().get("data", [])}
        report(True, "llm endpoint", settings.llm.base_url)
        for role in ("primary", "vision", "review_text", "docling_vlm", "ovis_ocr"):
            name = getattr(settings.models, role, "")
            if not name:
                continue
            state = (models.get(name) or {}).get("status", {}).get("value", "")
            report(name in models, f"model {role}", f"{name} {state}".strip() if name in models else
                   f"{name} is not served by {settings.llm.base_url} - set [models] {role} to one of: "
                   f"{', '.join(sorted(m for m in models if m)) or '(none)'}")
    except httpx.HTTPError as e:
        report(False, "llm endpoint", f"{settings.llm.base_url}: {e}")
    if not settings.extract.docling_serve_url:
        print("[--] docling-serve: not configured")
    else:
        try:
            resp = httpx.get(f"{settings.extract.docling_serve_url.rstrip('/')}/openapi.json", timeout=10.0)
            has_convert = resp.status_code == 200 and "/v1/convert/file" in resp.json().get("paths", {})
            report(has_convert, "docling-serve", settings.extract.docling_serve_url)
        except (httpx.HTTPError, ValueError) as e:
            report(False, "docling-serve", f"{settings.extract.docling_serve_url}: {e.__class__.__name__}")
    report(docling_engine.docling_available(), "local docling package", "installed" if docling_engine.docling_available() else "pip install -e .[docling]")
    for tool in (settings.extract.ocrmypdf_binary, "tesseract"):
        report(shutil.which(tool) is not None, f"tool {tool}")
    from fluxsense.extract import resolve_engine
    print(f"[--] text engine '{settings.extract.engine}' resolves to '{resolve_engine(settings.extract.engine, settings)}'")
    return 1 if problems else 0


def cmd_catalog(args: argparse.Namespace, settings: Settings) -> int:
    """Lists the document type catalog."""
    from fluxsense.catalog import default_catalog
    catalog = default_catalog()
    for name in catalog.names():
        spec = catalog.get(name)
        print(f"{name}: {spec.description} -> {', '.join(spec.body_keys())}; rules {', '.join(spec.rules)}")
        if args.verbose:
            for q in spec.questions:
                print(f"    ? {q}")
    return 0


def cmd_blz_update(args: argparse.Namespace, settings: Settings) -> int:
    """Refreshes the German bank-code list from the Bundesbank."""
    from fluxsense.grounding.blz import USER_BLZ_FILE, update_blz
    count = update_blz(args.url)
    print(f"{count} bank codes written to {USER_BLZ_FILE}")
    return 0


def cmd_samples(args: argparse.Namespace, settings: Settings) -> int:
    """Writes the synthetic sample PDFs."""
    from fluxsense.samples import make_all
    for name, truth in make_all(Path(args.out)).items():
        print(f"{name}: {truth.path}")
    return 0


def cmd_gen_testdocs(args: argparse.Namespace, settings: Settings) -> int:
    """Writes a generated test-document set (or the public smoke set) with its ground truth."""
    from fluxsense.testdata import KINDS, denylist_hits, generate_set, smoke_set
    out = Path(args.out)
    if args.smoke:
        files = smoke_set(out)
    else:
        kinds = args.kinds.split(",") if args.kinds else list(KINDS)
        files = generate_set(out, args.seed, args.count, args.scan_share, kinds)
    for generated in files:
        docs = ", ".join(f"{d.kind} {len(d.pages)}p" for d in generated.docs)
        print(f"{generated.path.name}: {generated.manifest['pages']} pages, {len(generated.docs)} document(s) ({docs})")
    print(f"ground truth: {out / 'ground_truth.json'}")
    if args.denylist:
        entries = [line.strip() for line in Path(args.denylist).expanduser().read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#")]
        hits = denylist_hits([d.text() for f in files for d in f.docs], entries)
        print(f"denylist: {len(hits)} of {len(entries)} entries found in the generated text")
        return 1 if hits else 0
    return 0


def cmd_score_testdocs(args: argparse.Namespace, settings: Settings) -> int:
    """Scores a bench run against the ground truth of a generated set."""
    import json

    from fluxsense.testdata.score import format_score, load_results, score
    truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    results_dir = Path(args.results)
    if (results_dir / "results").is_dir():
        results_dir = results_dir / "results"
    summary = score(truth, load_results(results_dir))
    print(json.dumps(summary, indent=2) if args.json else format_score(summary))
    return 0


def _analysis_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by analyze and bench."""
    parser.add_argument("--ocr-engine", choices=ENGINES, default=None, help="auxiliary text engine (default from config)")
    parser.add_argument("--no-review", action="store_true", help="disable all LLM reviews")
    parser.add_argument("--review-vision", action="store_true", help="enable the per-field vision review")
    parser.add_argument("--review-text", action="store_true", help="enable the text second opinion (second model)")
    parser.add_argument("--no-grounding", action="store_true", help="disable grounding escalation (deterministic grounding still runs)")
    parser.add_argument("--no-audit", action="store_true", help="disable the folded visual audit")
    parser.add_argument("--page-text", action="store_true", help="include per-page text (text_layer.pages) in the result")
    parser.add_argument("--model", default=None, help="override the primary/vision model id")
    parser.add_argument("--concurrency", type=int, default=None, help="parallel LLM pipelines")


def build_parser() -> argparse.ArgumentParser:
    """Creates the argument parser."""
    parser = argparse.ArgumentParser(prog="fluxsense", description="Semantic PDF analysis on local AI")
    parser.add_argument("--version", action="version", version=f"fluxsense {__version__}")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--config", default=None, help="config file (default ~/.config/fluxsense/config.toml)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyze", help="analyze a PDF or a directory")
    p.add_argument("path")
    p.add_argument("--out", default=None, help="write <stem>.json per file into this directory")
    p.add_argument("--json", action="store_true", help="print the full JSON result")
    _analysis_flags(p)
    p.set_defaults(func=cmd_analyze)
    p = sub.add_parser("bench", help="resumable batch analysis with summary")
    p.add_argument("path")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="re-analyze files with existing results")
    p.add_argument("--limit", type=int, default=None)
    _analysis_flags(p)
    p.set_defaults(func=cmd_bench)
    p = sub.add_parser("reassess", help="offline acceptance replay of a bench run (no LLM calls)")
    p.add_argument("run", help="bench output directory (results/ and .cache/)")
    p.add_argument("--out", required=True)
    p.add_argument("--pdf-root", default=None, help="source PDF directory (default: from bench_meta.json)")
    p.add_argument("--no-layout-replay", action="store_true", help="do not replay the geometry decision on parties")
    p.set_defaults(func=cmd_reassess)
    p = sub.add_parser("serve", help="run the REST API")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.set_defaults(func=cmd_serve)
    p = sub.add_parser("config", help="configuration")
    p.add_argument("action", choices=["show", "path", "init", "import-kpaperflux"])
    p.add_argument("--force", action="store_true")
    p.add_argument("--with-identity", action="store_true", help="show identity data")
    p.add_argument("--conf", default=None, help="KPaperFlux config file (import)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_config)
    p = sub.add_parser("doctor", help="check endpoints, models and tools")
    p.set_defaults(func=cmd_doctor)
    p = sub.add_parser("catalog", help="list document types")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_catalog)
    p = sub.add_parser("blz-update", help="refresh the German bank-code list (Bundesbank)")
    p.add_argument("--url", default=None, help="direct URL of blz-aktuell-txt-data.txt")
    p.set_defaults(func=cmd_blz_update)
    p = sub.add_parser("samples", help="write synthetic sample PDFs")
    p.add_argument("out")
    p.set_defaults(func=cmd_samples)
    p = sub.add_parser("gen-testdocs", help="generate fictional test documents with exact ground truth")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--count", type=int, default=12)
    p.add_argument("--scan-share", type=float, default=0.25, help="share of image-only scan variants")
    p.add_argument("--kinds", default=None, help="comma-separated kinds (default: all)")
    p.add_argument("--smoke", action="store_true", help="the public smoke benchmark set (ignores seed/count/kinds)")
    p.add_argument("--denylist", default=None, help="file with entries (one per line) that must not occur in the output")
    p.set_defaults(func=cmd_gen_testdocs)
    p = sub.add_parser("score-testdocs", help="score a bench run against a generated ground truth")
    p.add_argument("--truth", required=True, help="ground_truth.json of the generated set")
    p.add_argument("--results", required=True, help="bench output directory (or its results/ folder)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_score_testdocs)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(Path(args.config).expanduser() if args.config else None)
    return int(args.func(args, settings) or 0)


if __name__ == "__main__":
    sys.exit(main())
