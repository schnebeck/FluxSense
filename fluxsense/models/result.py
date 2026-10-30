# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/models/result.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Output models: AnalysisResult -> logical_documents[] with
                fields (KPaperFlux SemanticExtraction), stitched tables,
                plausibility checks, reviews, timings and AI-call stats.
------------------------------------------------------------------------------
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    """Severity of a plausibility finding."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class CheckStatus(str, Enum):
    """Outcome of one plausibility check."""

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class CheckResult(BaseModel):
    """One executed plausibility check (passed, failed or skipped)."""

    check_id: str = Field(..., description="Stable identifier, e.g. 'invoice.net_plus_tax'")
    rule: str = Field(..., description="Rule group, e.g. 'arithmetic', 'iban', 'dates'")
    status: CheckStatus
    severity: Severity = Severity.INFO
    message: str = ""
    field: Optional[str] = Field(None, description="Dotted path of the field concerned")
    expected: Optional[str] = None
    actual: Optional[str] = None
    rows: List[int] = Field(default_factory=list, description="0-based row indices of the table concerned")
    pages: List[int] = Field(default_factory=list, description="1-based PDF pages concerned")
    table: Optional[str] = None
    source: str = Field("deterministic", description="deterministic | review_vision | review_text")


class PlausibilityReport(BaseModel):
    """Scored result of all plausibility checks of one logical document."""

    status: str = Field("PASS", description="PASS | WARN | FAIL")
    score: float = Field(1.0, description="1.0 = fully plausible, 0.0 = implausible")
    checks: List[CheckResult] = Field(default_factory=list)

    @property
    def findings(self) -> List[CheckResult]:
        """Failed checks only."""
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    def summary_codes(self) -> List[str]:
        """Check IDs of all failed checks (for CSV summaries)."""
        return [c.check_id for c in self.findings]


class TableView(BaseModel):
    """A logical (page-independent) table of a document."""

    name: str
    path: str = Field(..., description="Dotted path of the list inside the fields")
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    source_pages: List[int] = Field(default_factory=list)
    strategy: str = Field("single_call", description="single_call | chunked | zugferd")


class CorrectionRecord(BaseModel):
    """One targeted LLM correction pass after a failed check."""

    trigger_checks: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)
    scope: str = ""
    applied: bool = False
    failed_before: int = 0
    failed_after: int = 0
    note: str = ""


class FieldVerdict(BaseModel):
    """Per-field verdict of the vision review."""

    field: str
    extracted: Optional[str] = None
    verdict: str = Field("NOT_VISIBLE", description="MATCH | MISMATCH | NOT_VISIBLE")
    observed: Optional[str] = None
    comment: Optional[str] = None


class ReviewResult(BaseModel):
    """Result of the optional LLM review passes."""

    vision: List[FieldVerdict] = Field(default_factory=list)
    vision_model: Optional[str] = None
    text_verdict: Optional[str] = Field(None, description="PLAUSIBLE | SUSPICIOUS | IMPLAUSIBLE")
    text_issues: List[Dict[str, Any]] = Field(default_factory=list)
    text_model: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class GroundingStatus(str, Enum):
    """Grounding status of one extracted value."""

    GROUNDED = "grounded"
    WEAKLY_GROUNDED = "weakly_grounded"
    UNGROUNDED = "ungrounded"
    VERIFIED = "verified_visual"
    REMOVED = "removed"
    NOT_CHECKED = "not_checked"


class FieldEvidence(BaseModel):
    """Evidence layer for one extracted scalar value (kept apart from the clean fields)."""

    path: str = Field(..., description="Dotted path inside fields, e.g. bodies.finance_body.invoice_number")
    kind: str = Field("text", description="amount | date | iban | identifier | text | integer | code")
    value: Optional[Any] = None
    page: Optional[int] = None
    evidence: Optional[str] = Field(None, description="Verbatim snippet as printed (model claim)")
    bbox: Optional[List[float]] = Field(None, description="Model-reported box [x0,y0,x1,y1], 0-1000 normalized")
    line: Optional[str] = Field(None, description="Pointer to the numbered text line (pointer extraction, e.g. p1:L37)")
    critical: bool = False
    status: GroundingStatus = GroundingStatus.NOT_CHECKED
    matched_source: Optional[str] = Field(None, description="Text source that grounded the value (native|ocr|docling)")
    matched_page: Optional[int] = None
    matched_text: Optional[str] = None
    verification: Optional[Dict[str, Any]] = Field(None, description="Visual re-verification answer, if any")
    action: Optional[str] = Field(None, description="kept | nulled | replaced | flagged")
    original_value: Optional[Any] = None
    escalation_level: int = Field(1, description="0 thumbnail, 1 medium page, 2 region crop, 3 second model")
    resolution: Optional[str] = Field(None, description="Resolution that settled the value, e.g. page@1280px, crop@300dpi")
    provenance: Optional[str] = Field(None, description="Field contract §0.5: label | geometry | check | model (None = not filled)")
    auto_accepted: Optional[bool] = Field(None, description="True: provenance label/geometry/check, grounded or zoom-verified and "
                                                            "no failing check; False: flagged for review; None: not filled")
    review_reasons: List[str] = Field(default_factory=list, description="Why the value is not auto-accepted")


class TrustReport(BaseModel):
    """Hallucination-guard summary of one logical document."""

    score: float = 1.0
    fields_total: int = 0
    grounded: int = 0
    weakly_grounded: int = 0
    ungrounded: int = 0
    verified_visual: int = 0
    nulled: int = 0
    replaced: int = 0
    table_rows_total: int = 0
    table_rows_ungrounded: int = 0
    failed_checks: int = 0
    corrections_applied: int = 0
    self_consistency_disagreements: int = 0
    zugferd_mismatches: int = 0
    escalations: Dict[str, int] = Field(default_factory=dict, description="Escalation level -> number of fields settled there")
    auto_accepted: int = Field(0, description="Filled core fields accepted without review (field contract §0.5)")
    flagged_for_review: int = Field(0, description="Filled core fields flagged for review (auto_accepted false)")
    informational_accepted: int = Field(0, description="Non-core fields (addresses, phones, ...) accepted - not part of the KPI")
    informational_flagged: int = Field(0, description="Non-core fields flagged - not part of the KPI")
    provenance: Dict[str, int] = Field(default_factory=dict, description="Provenance -> number of filled fields")


class ZugferdComparison(BaseModel):
    """Comparison between the embedded ZUGFeRD XML (ground truth) and the visual extraction."""

    compared: int = 0
    matched: int = 0
    mismatched: int = 0
    missing_in_visual: int = 0
    accuracy: Optional[float] = None
    differences: List[Dict[str, Any]] = Field(default_factory=list)


class FormFieldOut(BaseModel):
    """One form field read by its label (field contract §12)."""

    page: int
    label: str = Field(..., description="Printed label of the field")
    kind: str = Field("text", description="date | place_date | name | persons | address | location | phone | text | checkbox")
    target: Optional[str] = Field(None, description="Schema target of an unambiguous label (e.g. doc_date, event.MOVE_IN)")
    entry: Optional[str] = Field(None, description="The entry as read ('EMPTY' for an empty field, None: not read)")
    value: Optional[str] = Field(None, description="Normalised value (ISO date for date kinds)")
    candidates: List[str] = Field(default_factory=list, description="All valid date readings (more than one: ambiguous)")
    script: Optional[str] = Field(None, description="typed | handwritten | empty as the reader saw it (None: not reported)")
    found: str = Field("label", description="label (lexicon label in the layout) | locator (vision form locator) | "
                                            "acroform (digital form widget, no vision)")
    ticked: Optional[bool] = Field(None, description="Checkbox state (None: no checkbox / unknown)")
    bbox: Optional[List[float]] = Field(None, description="Label box [x0,y0,x1,y1], 0-1000 normalized")
    entry_bbox: Optional[List[float]] = Field(None, description="Entry region [x0,y0,x1,y1], 0-1000 normalized")


class SignatureOut(BaseModel):
    """One signature place of a logical document; the signer comes from printed content only, never from the scribble."""

    signer_name: Optional[str] = Field(None, description="Signer derived from printed content (None: no evidence)")
    role: Optional[str] = Field(None, description="Role of the signer: the role named by the line label (landlord, tenant, "
                                                  "employer, applicant, ...), sender (letter closing) or form_filler")
    place: Optional[str] = Field(None, description="Place written / printed next to the signature")
    date: Optional[str] = Field(None, description="Date next to the signature (ISO)")
    page: int
    bbox: Optional[List[float]] = Field(None, description="Signature region [x0,y0,x1,y1], 0-1000 normalized")
    status: str = Field("unclear", description="signed | empty | unclear")
    evidence: Optional[str] = Field(None, description="Printed text the signature place and its signer were derived from")
    signer_source: Optional[str] = Field(None, description="printed_name | proxy (i. A. / i. V.) | line_label | form_filler")
    label: Optional[str] = Field(None, description="Printed label of the signature line, or the closing")
    removed_ocr: List[str] = Field(default_factory=list, description="OCR text of the scribble removed from the page text")


class LogicalDocument(BaseModel):
    """One logical document found in the PDF (possibly one of a stack)."""

    index: int
    type: str = Field(..., description="Primary document type (KPaperFlux DocType vocabulary)")
    subtype: Optional[str] = Field(None, description="Finer FluxSense type, e.g. CREDIT_CARD_STATEMENT or LETTER")
    type_tags: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list, description="1-based PDF pages")
    direction: str = "UNKNOWN"
    tenant_context: str = "UNKNOWN"
    confidence: float = 0.0
    fields: Dict[str, Any] = Field(default_factory=dict, description="KPaperFlux-compatible SemanticExtraction")
    tables: List[TableView] = Field(default_factory=list)
    plausibility: PlausibilityReport = Field(default_factory=PlausibilityReport)
    corrections: List[CorrectionRecord] = Field(default_factory=list)
    review: Optional[ReviewResult] = None
    visual_audit: Optional[Dict[str, Any]] = None
    evidence: List[FieldEvidence] = Field(default_factory=list)
    trust: TrustReport = Field(default_factory=TrustReport)
    zugferd_comparison: Optional[ZugferdComparison] = None
    pointer_stats: Optional[Dict[str, int]] = Field(None, description="Pointer extraction (experiment): pointed/resolved/changed")
    handwriting: Dict[int, str] = Field(default_factory=dict, description="Page -> handwriting level none | signature | "
                                                                          "entries | full (pages with a known level)")
    form_fields: List[FormFieldOut] = Field(default_factory=list, description="Form fields read by their label (§12)")
    signatures: List[SignatureOut] = Field(default_factory=list, description="Signature places and their signers")
    errors: List[str] = Field(default_factory=list)


class CallRecord(BaseModel):
    """Statistics of one LLM request."""

    label: str
    model: str
    ok: bool = True
    attempts: int = 1
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    images: int = 0
    structured: bool = Field(False, description="True if response_format json_schema was used")
    truncated: bool = Field(False, description="True if the output hit max_tokens")
    degenerate: bool = Field(False, description="True if runaway repetition was detected (and retried)")
    error: Optional[str] = None


class SourceInfo(BaseModel):
    """Information about the analyzed input file."""

    path: str
    sha256: str
    size_bytes: int
    page_count: int


class PageTextOut(BaseModel):
    """Text of one page (optional output, e.g. for a full-text index)."""

    page: int
    text: str
    source: str = Field(..., description="native | ovis | ocr | docling | stream")


class TextLayerInfo(BaseModel):
    """How the text of the PDF was obtained."""

    engine: str
    has_text_layer: bool = False
    ocr_applied: bool = False
    chars_per_page: List[int] = Field(default_factory=list)
    tables_detected: int = 0
    sources: List[str] = Field(default_factory=list, description="Text sources available (native, ocr, docling)")
    warnings: List[str] = Field(default_factory=list)
    pages: List[PageTextOut] = Field(default_factory=list, description="Per-page text (only with include_page_text)")


class AnalysisResult(BaseModel):
    """Complete result of analyzing one PDF."""

    fluxsense_version: str
    schema_version: str
    prompt_version: str
    status: str = Field("OK", description="OK | PARTIAL | ERROR")
    error_code: Optional[str] = Field(None, description="BACKEND_UNREACHABLE | BACKEND_AUTH | BACKEND_TIMEOUT | "
                                                        "CONTEXT_OVERFLOW | INPUT_INVALID (None when OK)")
    verification_pending: bool = Field(False, description="Deferred verify mode: verify answers not applied yet")
    source: SourceInfo
    text_layer: TextLayerInfo
    preflight: Dict[str, Any] = Field(default_factory=dict)
    classification: Dict[str, Any] = Field(default_factory=dict)
    zugferd: Optional[Dict[str, Any]] = None
    logical_documents: List[LogicalDocument] = Field(default_factory=list)
    timings: Dict[str, float] = Field(default_factory=dict)
    ai_calls: List[CallRecord] = Field(default_factory=list)
    engines: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)

    def ai_summary(self) -> Dict[str, Dict[str, float]]:
        """Aggregates AI calls per model (count, tokens, seconds)."""
        out: Dict[str, Dict[str, float]] = {}
        for call in self.ai_calls:
            entry = out.setdefault(call.model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0})
            entry["calls"] += 1
            entry["prompt_tokens"] += call.prompt_tokens
            entry["completion_tokens"] += call.completion_tokens
            entry["seconds"] = round(entry["seconds"] + call.seconds, 3)
        return out
