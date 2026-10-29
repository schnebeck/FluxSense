# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/review/reviewer.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Optional LLM plausibility review (both off by default):
                (a) vision review with the primary VLM: per-field verdicts on
                the page images; (b) text-only second opinion with the
                review_text model, run as a separate batch phase to avoid model swaps.
------------------------------------------------------------------------------
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from fluxsense.config import Settings
from fluxsense.extract.base import TextLayer
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import page_images
from fluxsense.models.result import (CheckResult, CheckStatus, FieldEvidence, FieldVerdict, LogicalDocument,
                                     ReviewResult, Severity)
from fluxsense.prompts import verify as prompts

VISION_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["field", "verdict"],
        "properties": {"field": {"type": "string"}, "verdict": {"type": "string", "enum": ["MATCH", "MISMATCH", "NOT_VISIBLE"]},
                       "observed": {"type": ["string", "null"], "maxLength": 80}}}}},
}
TEXT_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["verdict", "issues"],
    "properties": {"verdict": {"type": "string", "enum": ["PLAUSIBLE", "SUSPICIOUS", "IMPLAUSIBLE"]},
                   "issues": {"type": "array", "items": {"type": "object", "additionalProperties": False, "properties": {
                       "field": {"type": "string"}, "severity": {"type": "string", "enum": ["INFO", "WARNING", "ERROR"]},
                       "explanation": {"type": "string", "maxLength": 300}}}}},
}


class Reviewer:
    """Runs the optional review passes."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: LLM backend.
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings

    def vision(self, pdf_path: Path, doc: LogicalDocument, evidence: List[FieldEvidence]) -> ReviewResult:
        """Vision review of the critical values."""
        review = doc.review or ReviewResult()
        values = [e for e in evidence if e.critical and e.value is not None][:25]
        if not values:
            return review
        pages = (doc.pages[:1] + doc.pages[-1:]) if len(doc.pages) > 1 else doc.pages
        images = page_images(pdf_path, pages[: self.settings.review.vision_max_pages], [f"PAGE {p}" for p in pages],
                             self.settings.llm.max_image_px, self.settings.llm.image_format)
        listing = "\n".join(f"- {e.path}: {e.value}" for e in values)
        result = self.backend.generate_json(LLMRequest(prompt=prompts.PROMPT_REVIEW_VISION.format(values=listing),
                                                       label="Review (vision)", model=self.settings.models.vision, images=images,
                                                       schema=VISION_SCHEMA, schema_name="review_vision", max_tokens=1500))
        review.vision_model = self.settings.models.vision
        if isinstance(result, dict):
            by_path = {e.path: e for e in values}
            for verdict in result.get("verdicts") or []:
                if isinstance(verdict, dict) and verdict.get("field") in by_path:
                    ev = by_path[verdict["field"]]
                    review.vision.append(FieldVerdict(field=ev.path, extracted=str(ev.value), verdict=verdict.get("verdict", "NOT_VISIBLE"),
                                                      observed=verdict.get("observed")))
        else:
            review.errors.append("vision review failed")
        for verdict in review.vision:
            if verdict.verdict == "MISMATCH":
                doc.plausibility.checks.append(CheckResult(
                    check_id="review.vision_mismatch", rule="review", status=CheckStatus.FAIL, severity=Severity.WARNING,
                    message=f"vision review sees '{verdict.observed}' instead of '{verdict.extracted}'", field=verdict.field,
                    source="review_vision"))
        return review

    def text(self, layer: TextLayer, doc: LogicalDocument) -> ReviewResult:
        """Text-only second opinion with the strong text model."""
        review = doc.review or ReviewResult()
        findings = "\n".join(f"- {c.check_id}: {c.message}" for c in doc.plausibility.findings) or "(none)"
        text = "\n\n".join(f"=== PAGE {p} ===\n{layer.page(p).best_text() if layer.page(p) else ''}" for p in doc.pages)[:50000]
        prompt = prompts.PROMPT_REVIEW_TEXT.format(doc_type=doc.type, findings=findings, document_text=text,
                                                   extraction=json.dumps(doc.fields, ensure_ascii=False, default=str)[:30000])
        result = self.backend.generate_json(LLMRequest(prompt=prompt, label="Review (text)", model=self.settings.models.review_text,
                                                       schema=TEXT_SCHEMA, schema_name="review_text", max_tokens=1500))
        review.text_model = self.settings.models.review_text
        if isinstance(result, dict):
            review.text_verdict = result.get("verdict")
            review.text_issues = [i for i in result.get("issues") or [] if isinstance(i, dict)]
            for issue in review.text_issues:
                severity = Severity(issue.get("severity", "INFO")) if issue.get("severity") in ("INFO", "WARNING", "ERROR") else Severity.INFO
                doc.plausibility.checks.append(CheckResult(
                    check_id="review.text_issue", rule="review", status=CheckStatus.FAIL,
                    severity=Severity.INFO if severity == Severity.INFO else Severity.WARNING,
                    message=str(issue.get("explanation", ""))[:300], field=issue.get("field"), source="review_text"))
        else:
            review.errors.append("text review failed")
        return review
