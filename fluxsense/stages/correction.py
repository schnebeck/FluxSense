# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/correction.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    ONE targeted LLM correction pass after failed table/arithmetic
                checks: re-extracts only the affected scope (rows of the
                affected pages, or the totals block) from higher-resolution
                page images, re-runs the checks and keeps the new values only
                if fewer checks fail (otherwise reverts).
------------------------------------------------------------------------------
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fluxsense.catalog.registry import TableSpec, TypeSpec
from fluxsense.config import Settings
from fluxsense.extract.base import TextLayer
from fluxsense.grounding.evidence import Unwrapped, build_plan, unwrap, wire_schema
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import page_images
from fluxsense.log import get_logger
from fluxsense.models.result import CheckResult, CorrectionRecord, PlausibilityReport, Severity
from fluxsense.models.semantic import BODY_MODELS, MonetarySummation, TaxBreakdownRow
from fluxsense.prompts import stage2 as prompts
from fluxsense.tables.stitch import drop_artefact_rows
from fluxsense.validate.core import SEVERITY_WEIGHT

logger = get_logger("stages.correction")

TABLE_CHECKS: Dict[str, str] = {
    "statement.balance": "transactions", "statement.running_balance": "transactions", "statement.total_debits": "transactions",
    "statement.total_credits": "transactions", "statement.debit_count": "transactions", "statement.credit_count": "transactions",
    "statement.credit_card_balance": "transactions", "invoice.qty_x_price": "line_items", "invoice.line_sum": "line_items",
    "invoice.line_sum_per_rate": "line_items",
}
TOTALS_CHECKS = frozenset({"invoice.net_plus_tax", "invoice.breakdown_basis_sum", "invoice.breakdown_tax_sum", "invoice.vat_row"})
STATEMENT_BALANCE_FIELDS = ("bodies.statement_body.opening_balance", "bodies.statement_body.closing_balance")
MAX_CORRECTION_PAGES: int = 2

Evaluator = Callable[[Dict[str, Any]], PlausibilityReport]
Reextractor = Callable[[str, List[int]], Optional[List[Dict[str, Any]]]]


@dataclass
class _Scope:
    """What to re-extract."""

    kind: str
    pages: List[int]
    table: Optional[TableSpec] = None


def _correctable(report: PlausibilityReport) -> List[CheckResult]:
    """Failed checks that a targeted pass can address."""
    return [c for c in report.findings if c.check_id in TABLE_CHECKS or c.check_id in TOTALS_CHECKS
            or (c.check_id == "completeness.required" and c.field in STATEMENT_BALANCE_FIELDS)
            or c.check_id == "statement.sign_convention"]


def choose_scope(failed: List[CheckResult], specs: Sequence[TypeSpec], pages: List[int],
                 evidence_pages: Dict[str, int], trusted: Optional[Dict[str, bool]] = None) -> Optional[_Scope]:
    """
    Picks the scope of the correction pass (tables first).

    Args:
        failed: Correctable failed checks.
        specs: Type specs.
        pages: Pages of the logical document.
        evidence_pages: field path -> page where the value was printed.
        trusted: field path -> True if confirmed by the native text layer or a zoomed crop.

    Returns:
        The scope, or None if nothing sensible can be re-extracted (or the
        values are confirmed as printed, i.e. the document itself is inconsistent).
    """
    if any(c.check_id == "statement.sign_convention" or (c.check_id == "completeness.required" and c.field in STATEMENT_BALANCE_FIELDS)
           for c in failed):
        return _Scope("statement_totals", sorted({pages[0], pages[-1]}))
    tables = {t.name: t for s in specs for t in s.tables}
    table_checks = [c for c in failed if TABLE_CHECKS.get(c.check_id) in tables]
    if table_checks:
        # The most specific check (fewest affected pages, e.g. a running-balance break) wins.
        check = min(table_checks, key=lambda c: len(set(c.pages)) if c.pages else len(pages) + 1)
        affected = sorted(set(check.pages)) or list(pages)
        if len(affected) > MAX_CORRECTION_PAGES:
            # Never skip a failed balance/sum over many pages: re-extract the whole table page-chunked.
            return _Scope("table_full", list(pages), tables[TABLE_CHECKS[check.check_id]])
        return _Scope("table", affected, tables[TABLE_CHECKS[check.check_id]])
    if any(c.check_id in TOTALS_CHECKS for c in failed):
        involved = {p: ok for p, ok in (trusted or {}).items() if "monetary_summation" in p or "tax_breakdown" in p}
        if involved and all(involved.values()):
            return None
        found = sorted({p for path, p in evidence_pages.items() if "monetary_summation" in path or "tax_breakdown" in path})
        return _Scope("totals", found or [pages[-1]])
    return None


def penalty(report: PlausibilityReport) -> float:
    """Severity-weighted sum of failed checks."""
    return sum(SEVERITY_WEIGHT[c.severity] for c in report.findings)


def _row_count(fields: Dict[str, Any], scope: _Scope) -> Optional[int]:
    """Rows of the scope's table (None for totals)."""
    if scope.table is None:
        return None
    return len(((fields.get("bodies") or {}).get(scope.table.body_key) or {}).get(scope.table.field_name) or [])


def decide(before: PlausibilityReport, after: PlausibilityReport, triggers: List[CheckResult],
           old_fields: Dict[str, Any], new_fields: Dict[str, Any], scope: _Scope) -> Tuple[bool, str]:
    """
    Accepts a correction if the weighted penalty drops, or if a trigger check
    is resolved without introducing a new ERROR. Returns (applied, explanation).
    """
    trigger_ids = {c.check_id for c in triggers}
    after_failed = {c.check_id: c for c in after.findings}
    resolved = sorted(trigger_ids - set(after_failed))
    old_errors = {c.check_id for c in before.findings if c.severity == Severity.ERROR}
    new_errors = sorted({c.check_id for c in after.findings if c.severity == Severity.ERROR} - old_errors)
    p_before, p_after = penalty(before), penalty(after)
    applied = p_after < p_before or (bool(resolved) and not new_errors)
    still = "; ".join(f"{c.check_id}: {c.message}" for c in after.findings if c.check_id in trigger_ids)[:400]
    note = (f"penalty {p_before:.2f}->{p_after:.2f}, rows {_row_count(old_fields, scope)}->{_row_count(new_fields, scope)}, "
            f"resolved {resolved or 'none'}, new errors {new_errors or 'none'}"
            + (f"; still failing: {still}" if still else ""))
    if not applied and not still:
        note += "; values kept as printed"
    return applied, note


class TargetedCorrector:
    """Runs at most one targeted correction per logical document."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: LLM backend.
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings

    def _text(self, layer: TextLayer, pages: List[int]) -> str:
        """Auxiliary text of the pages."""
        return "\n\n".join(f"=== PAGE {p} ===\n{layer.page(p).best_text() if layer.page(p) else ''}" for p in pages)[:60000]

    def run(self, pdf_path: Path, layer: TextLayer, pages: List[int], fields: Dict[str, Any], report: PlausibilityReport,
            specs: Sequence[TypeSpec], evidence_pages: Dict[str, int], evaluate: Evaluator,
            trusted: Optional[Dict[str, bool]] = None, reextract: Optional[Reextractor] = None) -> tuple:
        """
        Executes the correction pass if a correctable check failed.

        Args:
            pdf_path: PDF file.
            layer: Text layer.
            pages: Pages of the logical document.
            fields: Current clean fields.
            report: Current plausibility report.
            specs: Type specs.
            evidence_pages: field path -> page.
            evaluate: Re-runs the plausibility checks on candidate fields.
            trusted: field path -> confirmed-as-printed flag.
            reextract: Page-chunked re-extraction of a whole table (table name, pages) -> rows.

        Returns:
            Tuple (fields, report, CorrectionRecord or None).
        """
        failed = _correctable(report)
        if not failed:
            return fields, report, None
        scope = choose_scope(failed, specs, pages, evidence_pages, trusted)
        if scope is None:
            record = CorrectionRecord(trigger_checks=[c.check_id for c in failed], failed_before=len(report.findings),
                                      failed_after=len(report.findings), scope="none",
                                      note="skipped: values confirmed as printed, or too many pages affected for a targeted pass")
            return fields, report, record
        record = CorrectionRecord(trigger_checks=[c.check_id for c in failed], pages=scope.pages,
                                  scope=scope.table.name if scope.table else "totals", failed_before=len(report.findings))
        candidate = copy.deepcopy(fields)
        if scope.kind == "table_full":
            ok = False
            if reextract is not None:
                rows = reextract(scope.table.name, scope.pages)
                if rows:
                    candidate.setdefault("bodies", {}).setdefault(scope.table.body_key, {})[scope.table.field_name] = rows
                    ok = True
            record.scope = f"{scope.table.name} (page-chunked re-extraction)"
        elif scope.kind == "statement_totals":
            ok = self._fix_statement_totals(pdf_path, layer, scope, candidate, failed)
            record.scope = "statement balances"
        elif scope.kind == "table":
            ok = self._fix_table(pdf_path, layer, scope, candidate, failed)
        else:
            ok = self._fix_totals(pdf_path, layer, scope, candidate, failed)
        if not ok:
            record.note = "correction call failed or returned no data"
            logger.info(f"Correction ({record.scope}, pages {record.pages}): {record.note}")
            return fields, report, record
        new_report = evaluate(candidate)
        record.failed_after = len(new_report.findings)
        applied, record.note = decide(report, new_report, failed, fields, candidate, scope)
        logger.info(f"Correction ({record.scope}, pages {record.pages}): {'APPLIED' if applied else 'not applied'} - {record.note}")
        if applied:
            record.applied = True
            return candidate, new_report, record
        return fields, report, record

    def _images(self, pdf_path: Path, pages: List[int]) -> list:
        """High-resolution page images (escalation above the medium level)."""
        px = int(self.settings.llm.max_image_px * 1.5)
        return page_images(pdf_path, pages, [f"PAGE {p} (high resolution)" for p in pages], px,
                           self.settings.llm.image_format, dpi=max(self.settings.llm.render_dpi, 250))

    def _call(self, prompt: str, label: str, pdf_path: Path, pages: List[int], schema: Dict[str, Any], max_tokens: int) -> Optional[Dict[str, Any]]:
        """One correction call."""
        result = self.backend.generate_json(LLMRequest(prompt=prompt, label=label, model=self.settings.models.primary,
                                                       images=self._images(pdf_path, pages), schema=schema,
                                                       schema_name="correction", max_tokens=max_tokens))
        return result if isinstance(result, dict) else None

    def _fix_table(self, pdf_path: Path, layer: TextLayer, scope: _Scope, fields: Dict[str, Any], failed: List[CheckResult]) -> bool:
        """Re-extracts the rows of the affected pages and splices them in place."""
        spec = scope.table
        plan = build_plan(BODY_MODELS[spec.body_key], tables={spec.field_name: spec}).fields[spec.field_name]
        schema = {"type": "object", "additionalProperties": False, "required": ["rows"], "properties": {"rows": wire_schema(plan)}}
        with_totals = spec.body_key == "finance_body"
        if with_totals:
            schema["properties"]["monetary_summation"] = wire_schema(build_plan(MonetarySummation, in_row=True))
        body = fields.setdefault("bodies", {}).setdefault(spec.body_key, {})
        rows = body.get(spec.field_name) or []
        current = [r for r in rows if r.get("source_page") in scope.pages]
        prompt = prompts.PROMPT_TARGETED_CORRECTION.format(
            failed_checks="\n".join(f"- {c.check_id}: {c.message}" for c in failed),
            pages=", ".join(str(p) for p in scope.pages),
            scope=(f"all rows of table '{spec.name}' printed on these pages, as arrays {spec.column_hint()}"
                   + ("; plus the printed totals (monetary_summation). Discounts (Rabatt), shipping (Versand), Skonto and "
                      "other charges/allowances are NOT line items unless printed as numbered positions: put them into "
                      "charge_total_amount / allowance_total_amount" if with_totals else "")),
            current_values=json.dumps([[r.get(c) for c in spec.columns] for r in current], ensure_ascii=False, default=str),
            document_text=self._text(layer, scope.pages))
        result = self._call(prompt, f"Correction {spec.name} p{scope.pages}", pdf_path, scope.pages, schema,
                            800 + self.settings.pipeline.tokens_per_row * max(40, 2 * len(current)))
        if result is None:
            return False
        new_rows = unwrap(plan, result.get("rows") or [], spec.path, Unwrapped(None)) or []
        new_rows, _ = drop_artefact_rows(new_rows, ("description", "counterparty"))
        if len(scope.pages) == 1:
            for row in new_rows:
                row["source_page"] = scope.pages[0]
        first_index = next((i for i, r in enumerate(rows) if r.get("source_page") in scope.pages), len(rows))
        if with_totals and isinstance(result.get("monetary_summation"), dict):
            totals = {k: v for k, v in result["monetary_summation"].items() if v is not None}
            if totals:
                body["monetary_summation"] = {**(body.get("monetary_summation") or {}), **totals}
        if not new_rows:
            return False
        before = [r for i, r in enumerate(rows) if i < first_index and r.get("source_page") not in scope.pages]
        after = [r for i, r in enumerate(rows) if i >= first_index and r.get("source_page") not in scope.pages]
        body[spec.field_name] = before + new_rows + after
        return True

    def _fix_statement_totals(self, pdf_path: Path, layer: TextLayer, scope: _Scope, fields: Dict[str, Any],
                              failed: List[CheckResult]) -> bool:
        """Re-reads opening/closing balance (signed as printed) from the first and last page."""
        money = {"type": ["number", "null"]}
        schema = {"type": "object", "additionalProperties": False, "properties": {
            "opening_balance": money, "closing_balance": money, "statement_date": {"type": ["string", "null"], "maxLength": 12}}}
        body = fields.setdefault("bodies", {}).setdefault("statement_body", {})
        current = {k: body.get(k) for k in ("opening_balance", "closing_balance", "statement_date")}
        prompt = prompts.PROMPT_TARGETED_CORRECTION.format(
            failed_checks="\n".join(f"- {c.check_id}: {c.message}" for c in failed),
            pages=", ".join(str(p) for p in scope.pages),
            scope=("opening balance (previous/old balance, 'Alter Kontostand', 'Saldo letzte Abrechnung') and closing balance "
                   "(new balance, 'Neuer Kontostand', 'Neuer Saldo'), each signed as printed ('-', 'S', 'Soll' = negative), "
                   "plus the statement date (YYYY-MM-DD)"),
            current_values=json.dumps(current, ensure_ascii=False, default=str), document_text=self._text(layer, scope.pages))
        result = self._call(prompt, "Correction statement balances", pdf_path, scope.pages, schema, 200)
        if result is None:
            return False
        changed = False
        for key in ("opening_balance", "closing_balance", "statement_date"):
            if result.get(key) is not None:
                body[key] = result[key]
                changed = True
        return changed

    def _fix_totals(self, pdf_path: Path, layer: TextLayer, scope: _Scope, fields: Dict[str, Any], failed: List[CheckResult]) -> bool:
        """Re-reads monetary summation and tax breakdown."""
        ms_plan = build_plan(MonetarySummation, in_row=True)
        tax_plan = build_plan(TaxBreakdownRow, in_row=True)
        schema = {"type": "object", "additionalProperties": False, "properties": {
            "monetary_summation": wire_schema(ms_plan), "tax_breakdown": {"type": "array", "items": wire_schema(tax_plan)}}}
        finance = fields.setdefault("bodies", {}).setdefault("finance_body", {})
        current = {"monetary_summation": finance.get("monetary_summation"), "tax_breakdown": finance.get("tax_breakdown")}
        prompt = prompts.PROMPT_TARGETED_CORRECTION.format(
            failed_checks="\n".join(f"- {c.check_id}: {c.message}" for c in failed),
            pages=", ".join(str(p) for p in scope.pages),
            scope="monetary_summation (printed totals) and tax_breakdown (one row per printed tax rate)",
            current_values=json.dumps(current, ensure_ascii=False, default=str), document_text=self._text(layer, scope.pages))
        result = self._call(prompt, "Correction totals", pdf_path, scope.pages, schema, 600)
        if result is None:
            return False
        if isinstance(result.get("monetary_summation"), dict):
            finance["monetary_summation"] = {k: v for k, v in result["monetary_summation"].items() if v is not None}
        if isinstance(result.get("tax_breakdown"), list):
            finance["tax_breakdown"] = [r for r in result["tax_breakdown"] if isinstance(r, dict)
                                        and all(r.get(k) is not None for k in ("tax_basis_amount", "tax_amount", "tax_rate"))]
        return True
