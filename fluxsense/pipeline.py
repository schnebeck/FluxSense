# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/pipeline.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Orchestration (order ported from KPaperFlux canonizer):
                prepare (text layer, ZUGFeRD XML) -> ZUGFeRD short-circuit of
                classification or Stage 1 -> per logical document: holistic
                Stage 2 (+ folded Stage 1.5 audit) -> ZUGFeRD comparison and
                overlay -> plausibility -> grounding/zoom -> targeted
                correction -> optional reviews -> hydrated, KPaperFlux-
                compatible fields, trust report, timings, AI-call stats.
------------------------------------------------------------------------------
"""

import copy
import hashlib
import time
from datetime import date
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from fluxsense import SCHEMA_VERSION, __version__
from fluxsense.catalog import Catalog, default_catalog
from fluxsense.catalog.contract import contract_for, form_filler, geometry_policy, subject_context
from fluxsense.catalog.registry import load_custom_types
from fluxsense.config import Settings
from fluxsense.extract.fieldread import FieldReader, apply_fields, map_fields
from fluxsense.extract import build_text_layer
from fluxsense.extract.base import TextLayer
from fluxsense.grounding.acceptance import assess
from fluxsense.grounding.harness import CHECK_IBAN_PRINTED_INVALID, GroundingHarness, holds, parse_printed, trust_report
from fluxsense.grounding.normalize import compact, to_decimal
from fluxsense.layout import din5008
from fluxsense.llm.base import BackendAuthError, BackendUnreachable, LLMBackend
from fluxsense.llm.openai_client import OpenAICompatClient
from fluxsense.log import get_logger
from fluxsense.models.identity import IdentityProfile
from fluxsense.models.result import (AnalysisResult, CallRecord, CheckResult, CheckStatus, CorrectionRecord, GroundingStatus,
                                     LogicalDocument, ReviewResult,
                                     PageTextOut, Severity, SourceInfo, TextLayerInfo)
from fluxsense.models.semantic import SemanticExtraction
from fluxsense.prompts import PROMPT_VERSION
from fluxsense.review.reviewer import Reviewer
from fluxsense.stages import zugferd
from fluxsense.stages.classify import ClassificationResult, Classifier, Entity
from fluxsense.stages.correction import TargetedCorrector
from fluxsense.stages.extract import Extractor
from fluxsense.stages.identity import (apply_form_semantics, unfilled_form, direction_from_parties, harmonize_parties,
                                      remove_role_labels)
from fluxsense.stages.repair import (clean_document_numbers, drop_signature_parties, complete_statement_years, drop_bank_statement_due_date,
                                    drop_form_deadline, enforce_doc_semantics, fix_transfer_confirmation,
                                    fill_tax_basis, sanitize_bank_identifiers, stamps_from_text,
                                    fill_row_amounts, fix_document_dates, fix_document_number, order_payee_accounts,
                                    fix_statement_semantics,
                                    is_form_code,
                                    sender_only_in_table,
                                    normalize_dates,
                                    repair_checks, zoom_targets_for_rows)
from fluxsense.vision.zoom import ZoomTarget
from fluxsense.validate import run_plausibility
from fluxsense.validate.core import CheckContext, score_report
from fluxsense.validate.contract_terms import check_contract_payment, fix_minimum_term
from fluxsense.validate.dates import check_dates_consistency
from fluxsense.wol import lease_for

logger = get_logger("pipeline")


@dataclass
class AnalyzeOptions:
    """Per-call overrides of the settings."""

    engine: Optional[str] = None
    review_vision: Optional[bool] = None
    review_text: Optional[bool] = None
    grounding: Optional[bool] = None
    audit: Optional[bool] = None
    include_page_text: bool = False


@dataclass
class Prepared:
    """CPU/IO part of an analysis (can run ahead of the LLM part)."""

    path: Path
    source: SourceInfo
    layer: TextLayer
    zugferd: Optional[Dict[str, Any]]
    seconds: float
    error: Optional[str] = None


@dataclass
class _DocWork:
    """Intermediate state of one logical document."""

    entity: Entity
    doc: LogicalDocument
    evidence: List[Any] = field(default_factory=list)


def swap_party_evidence(evidence: List[Any]) -> None:
    """Swaps sender/recipient evidence paths after a party swap."""
    for ev in evidence:
        if ev.path.startswith("meta_header.sender."):
            ev.path = "meta_header.recipient." + ev.path[len("meta_header.sender."):]
        elif ev.path.startswith("meta_header.recipient."):
            ev.path = "meta_header.sender." + ev.path[len("meta_header.recipient."):]


def sha256(path: Path) -> str:
    """SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


ERROR_BACKEND_UNREACHABLE = "BACKEND_UNREACHABLE"
ERROR_BACKEND_AUTH = "BACKEND_AUTH"
ERROR_BACKEND_TIMEOUT = "BACKEND_TIMEOUT"
ERROR_CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"
ERROR_INPUT_INVALID = "INPUT_INVALID"


def _readable_pdf(path: Path) -> bool:
    """True if PyMuPDF can open the file and it has pages."""
    from fluxsense.extract.native import page_count
    try:
        return page_count(path) > 0
    except (RuntimeError, ValueError, OSError):
        return False


def error_result(path: Path, code: str, message: str) -> AnalysisResult:
    """A result that carries only a machine-readable error (no analysis happened)."""
    size = path.stat().st_size if path.is_file() else 0
    digest = sha256(path) if path.is_file() else ""
    return AnalysisResult(
        fluxsense_version=__version__, schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION, status="ERROR",
        error_code=code, errors=[message],
        source=SourceInfo(path=str(path), sha256=digest, size_bytes=size, page_count=0),
        text_layer=TextLayerInfo(engine="none"))


def error_code_of(result: AnalysisResult) -> Optional[str]:
    """Machine-readable cause of a PARTIAL / ERROR result from its call records."""
    errors = " ".join(str(c.error or "") for c in result.ai_calls).lower()
    if "context overflow" in errors or "exceed" in errors:
        return ERROR_CONTEXT_OVERFLOW
    if "timeout" in errors:
        return ERROR_BACKEND_TIMEOUT
    if "unreachable" in errors:
        return ERROR_BACKEND_UNREACHABLE
    if result.source.page_count == 0:
        return ERROR_INPUT_INVALID
    return None


def _profiles(settings: Settings) -> tuple:
    """Configured identities (None if empty)."""
    private = settings.identity.private if not settings.identity.private.is_empty() else None
    business = settings.identity.business if not settings.identity.business.is_empty() else None
    return private, business


class Analyzer:
    """High-level API: analyze(pdf) -> AnalysisResult."""

    def __init__(self, settings: Settings, backend: Optional[LLMBackend] = None, catalog: Optional[Catalog] = None) -> None:
        """
        Args:
            settings: Settings.
            backend: LLM backend (default: OpenAI-compatible client).
            catalog: Type catalog (default: built-in plus custom types file).
        """
        self.settings = settings
        self.backend: LLMBackend = backend or OpenAICompatClient(settings.llm, settings.hooks)
        self.catalog = catalog or default_catalog()
        if settings.pipeline.custom_types_file:
            load_custom_types(self.catalog, Path(settings.pipeline.custom_types_file).expanduser())
        self.classifier = Classifier(self.backend, settings, self.catalog)
        self.extractor = Extractor(self.backend, settings, self.catalog)
        self.harness = GroundingHarness(self.backend, settings)
        self.corrector = TargetedCorrector(self.backend, settings)
        self.reviewer = Reviewer(self.backend, settings)

    # ---------------------------------------------------------------- prepare

    def prepare(self, pdf_path: Path, engine: Optional[str] = None) -> Prepared:
        """
        Builds the text layer and reads ZUGFeRD XML (no LLM involved).

        Args:
            pdf_path: PDF file (never modified).
            engine: Text engine override.

        Returns:
            Prepared input.
        """
        start = time.monotonic()
        path = Path(pdf_path)
        layer = build_text_layer(path, self.settings, engine, backend=self.backend)  # OvisOCR3 calls share the client
        source = SourceInfo(path=str(path), sha256=sha256(path), size_bytes=path.stat().st_size, page_count=layer.page_count)
        return Prepared(path=path, source=source, layer=layer, zugferd=zugferd.read_zugferd(path),
                        seconds=round(time.monotonic() - start, 3))

    # ---------------------------------------------------------------- analyze

    def analyze(self, pdf_path: Path, options: Optional[AnalyzeOptions] = None) -> AnalysisResult:
        """
        Analyzes one PDF end to end.

        Args:
            pdf_path: PDF file.
            options: Overrides.

        Returns:
            The AnalysisResult.
        """
        options = options or AnalyzeOptions()
        lease = lease_for(self.settings.hooks)
        with lease.job() if lease is not None else nullcontext():
            return self._analyze_file(Path(pdf_path), options)

    def _analyze_file(self, pdf_path: Path, options: AnalyzeOptions) -> AnalysisResult:
        """analyze() without the optional WOL job bracket."""
        try:
            prepared = self.prepare(Path(pdf_path), options.engine)
            if self.settings.pipeline.verify_mode == "deferred":
                result = self._analyze_deferred(prepared, options)
            else:
                result = self.analyze_prepared(prepared, options)
            review_text = self.settings.review.text if options.review_text is None else options.review_text
            if review_text:
                self.review_text(result, prepared.layer)
        except BackendUnreachable as e:
            logger.error(str(e))
            return error_result(Path(pdf_path), ERROR_BACKEND_UNREACHABLE, str(e))
        except BackendAuthError as e:
            logger.error(str(e))
            return error_result(Path(pdf_path), ERROR_BACKEND_AUTH, str(e))
        except (RuntimeError, ValueError, OSError) as e:  # pymupdf.FileDataError & co: not a readable PDF
            if Path(pdf_path).is_file() and _readable_pdf(Path(pdf_path)):
                raise
            return error_result(Path(pdf_path), ERROR_INPUT_INVALID, f"{e.__class__.__name__}: {e}")
        return result

    def _analyze_deferred(self, prepared: Prepared, options: AnalyzeOptions) -> AnalysisResult:
        """Single-file deferred verification (in memory): record, run the queue per model, replay to a fixpoint."""
        from fluxsense.deferred import RECORD, REPLAY, CachingBackend, DeferredStore, run_queue
        store = DeferredStore(None)
        record = CachingBackend(self.backend, store, self.settings, RECORD)
        record.set_doc(prepared.source.path)
        result = Analyzer(self.settings, backend=record, catalog=self.catalog).analyze_prepared(prepared, options)
        pending, verify_wall = record.pending, 0.0
        replay = CachingBackend(self.backend, store, self.settings, REPLAY)
        replay_analyzer = Analyzer(self.settings, backend=replay, catalog=self.catalog)
        for _ in range(max(1, self.settings.pipeline.deferred_rounds)):
            if not pending:
                break
            _, seconds = run_queue(store, self.backend, self.settings.llm.max_concurrency)
            verify_wall += seconds
            replay.set_doc(prepared.source.path)
            result = replay_analyzer.analyze_prepared(prepared, options)
            pending = replay.pending
        result.verification_pending = pending
        result.timings["verify_phase_wall"] = round(verify_wall, 3)
        return result

    def analyze_prepared(self, prepared: Prepared, options: Optional[AnalyzeOptions] = None) -> AnalysisResult:
        """
        LLM part of the analysis (classification, extraction, checks).

        Args:
            prepared: Output of prepare().
            options: Overrides.

        Returns:
            The AnalysisResult.
        """
        options = options or AnalyzeOptions()
        calls: List[CallRecord] = []
        collect = getattr(self.backend, "collect", None)
        start = time.monotonic()
        if collect is not None:
            with collect(calls):
                result = self._analyze(prepared, options)
        else:
            before = len(self.backend.calls)
            result = self._analyze(prepared, options)
            calls = self.backend.calls[before:]
        result.ai_calls = list(calls)
        result.timings["text_layer"] = prepared.seconds
        result.timings["analysis_wall"] = round(time.monotonic() - start, 3)
        ai_seconds = round(sum(c.seconds for c in calls), 3)
        result.timings["ai_seconds"] = ai_seconds
        pages = max(1, prepared.source.page_count)
        result.timings["ai_seconds_per_page"] = round(ai_seconds / pages, 3)
        result.timings["wall_seconds_per_page"] = round((result.timings["analysis_wall"] + prepared.seconds) / pages, 3)
        if options.include_page_text:
            result.text_layer.pages = [PageTextOut(page=p.page, text=p.best_text(), source=p.best_source() or "native")
                                       for p in prepared.layer.pages]
        if result.status != "OK" and result.error_code is None:
            result.error_code = error_code_of(result)
        return result

    def _analyze(self, prepared: Prepared, options: AnalyzeOptions) -> AnalysisResult:
        """Implementation of analyze_prepared()."""
        layer = prepared.layer
        private_id, business_id = _profiles(self.settings)
        engines = {"llm_base_url": self.settings.llm.base_url, "primary_model": self.settings.models.primary,
                   "vision_model": self.settings.models.vision, "text_engine": layer.engine}
        result = AnalysisResult(
            fluxsense_version=__version__, schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION,
            source=prepared.source,
            text_layer=TextLayerInfo(engine=layer.engine, has_text_layer=layer.has_text_layer, ocr_applied=layer.ocr_applied,
                                     chars_per_page=[len(t) for t in layer.texts()], tables_detected=len(layer.tables()),
                                     sources=layer.sources(), warnings=layer.warnings),
            engines=engines,
        )
        if layer.page_count == 0:
            result.status = "ERROR"
            result.errors.append("PDF has no pages")
            return result
        timings: Dict[str, float] = {k: 0.0 for k in ("classification", "extraction", "grounding", "plausibility", "correction", "review")}
        xml = prepared.zugferd
        t0 = time.monotonic()
        if xml:
            sender, recipient = xml["meta_header"].get("sender"), xml["meta_header"].get("recipient")
            direction, context = direction_from_parties(sender, recipient, private_id, business_id)
            entities = [Entity(type_tags=xml["type_tags"], pages=list(range(1, layer.page_count + 1)), direction=direction,
                               tenant_context=context, confidence=1.0)]
            result.classification = {"source": "ZUGFERD", "type_tags": xml["type_tags"]}
            result.zugferd = {"present": True, "type_tags": xml["type_tags"], "finance_body": copy.deepcopy(xml.get("finance_body")),
                              "meta_header": copy.deepcopy(xml.get("meta_header"))}
        else:
            classification: ClassificationResult = self.classifier.run(prepared.path, layer, private_id, business_id)
            entities = classification.entities
            result.preflight = classification.preflight
            result.classification = {"source": "AI", "strategy": classification.strategy, "language": classification.language,
                                     "raw": classification.raw, "validation_errors": classification.errors,
                                     "page_signatures": classification.signatures, "signature_starts": classification.starts}
        timings["classification"] = round(time.monotonic() - t0, 3)

        for index, entity in enumerate(entities):
            doc = self._analyze_entity(prepared, entity, index, options, private_id, business_id, xml, timings)
            result.logical_documents.append(doc)
        failed_docs = sum(1 for d in result.logical_documents if d.errors and not d.fields)
        if failed_docs == len(result.logical_documents):
            result.status = "ERROR"
        elif failed_docs or any(d.errors for d in result.logical_documents):
            result.status = "PARTIAL"
        result.timings.update(timings)
        if hasattr(self.backend, "fingerprints"):
            result.engines["llm_fingerprints"] = dict(getattr(self.backend, "fingerprints"))
        return result

    def _evaluate(self, fields: Dict[str, Any], entity: Entity, specs: List[Any]) -> Any:
        """Runs the deterministic plausibility rules on fields."""
        private_id, business_id = _profiles(self.settings)
        ctx = CheckContext(fields=fields, spec=specs[0], settings=self.settings.plausibility, direction=entity.direction,
                           tenant_context=entity.tenant_context, private_id=private_id, business_id=business_id)
        return run_plausibility(ctx, specs)

    def _analyze_entity(self, prepared: Prepared, entity: Entity, index: int, options: AnalyzeOptions,
                        private_id: Optional[IdentityProfile], business_id: Optional[IdentityProfile],
                        xml: Optional[Dict[str, Any]], timings: Dict[str, float]) -> LogicalDocument:
        """Runs Stage 2 and all checks for one logical document."""
        layer, pages = prepared.layer, entity.pages
        specs = self.catalog.specs_for(entity.type_tags)
        doc = LogicalDocument(index=index, type=specs[0].output_type, subtype=specs[0].name,
                              type_tags=self.catalog.output_tags(entity.type_tags), pages=pages,
                              direction=entity.direction, tenant_context=entity.tenant_context, confidence=entity.confidence)
        audit = self.settings.pipeline.visual_audit if options.audit is None else options.audit
        if xml and not self.settings.pipeline.zugferd_visual_audit:
            audit = False
        form_fields: List[Any] = []
        if self.settings.pipeline.fieldread:
            t0 = time.monotonic()
            form_pages = [layer.page(p) for p in pages if layer.page(p) is not None]
            form_fields = FieldReader(self.backend, self.settings).run(prepared.path, form_pages)  # type: ignore[arg-type]
            apply_fields(form_pages, form_fields)  # type: ignore[arg-type]
            if form_fields:
                timings["fieldread"] = timings.get("fieldread", 0.0) + round(time.monotonic() - t0, 3)
        t0 = time.monotonic()
        outcome = self.extractor.run(prepared.path, layer, pages, entity.type_tags, private_id, business_id,
                                     zugferd_present=bool(xml), audit=audit)
        timings["extraction"] += round(time.monotonic() - t0, 3)
        if outcome is None:
            doc.errors.append("Stage 2 extraction failed")
            if xml:
                fields = zugferd.overlay({"meta_header": {}, "bodies": {}}, xml)
                doc.fields = fields
            return doc
        fields = outcome.fields
        evidence = outcome.evidence
        if outcome.pointer_stats is not None:
            doc.pointer_stats = outcome.pointer_stats
        doc_number = next((e for e in evidence if e.path == "meta_header.doc_number"), None)
        if doc_number is not None and is_form_code(doc_number.value, doc_number.evidence,
                                                   layer.page(doc_number.page or pages[0])):
            (fields.get("meta_header") or {}).pop("doc_number", None)
            doc_number.original_value, doc_number.action = doc_number.value, "nulled"
            doc.errors.append(f"doc_number '{doc_number.value}' looks like a print/form code in the footer/margin - removed")
        statement = (fields.get("bodies") or {}).get("statement_body")
        if isinstance(statement, dict):
            complete_statement_years(statement)
            statement_notes = fix_statement_semantics(
                statement, evidence, "\n".join(layer.page(p).best_text() for p in pages if layer.page(p)))
            if statement_notes:
                doc.corrections.append(CorrectionRecord(trigger_checks=["statement.semantics"], scope="statement header",
                                                        applied=True, note="; ".join(statement_notes)))
        normalized = normalize_dates(fields)
        doc_text = "\n".join(layer.page(p).best_text() for p in pages if layer.page(p))
        first_hints = din5008.analyze(layer.page(pages[0]))
        has_letterhead = bool(first_hints.address_window or first_hints.return_address or entity.letterhead)
        hygiene = clean_document_numbers(fields) + drop_bank_statement_due_date(fields, entity.type_tags) \
            + enforce_doc_semantics(fields, evidence, specs[0], doc_text, has_letterhead)
        if hygiene:
            doc.corrections.append(CorrectionRecord(trigger_checks=["hygiene"], scope="numbers and dates", applied=True,
                                                    note="; ".join(hygiene)))
        contract_type = next((spec.name for spec in specs if contract_for(spec.name).date_group
                              or contract_for(spec.name).number_group), specs[0].name)
        if not xml:
            number_notes = fix_document_number(fields, evidence, doc_text, any(spec.body_key == "finance_body" for spec in specs),
                                               contract_type)
            if number_notes:
                doc.corrections.append(CorrectionRecord(trigger_checks=["doc_number"], scope="document number", applied=True,
                                                        note="; ".join(number_notes)))
        if contract_for(contract_type).party_case == "2.8":
            transfer_notes = fix_transfer_confirmation(fields, doc_text)
            if transfer_notes:
                doc.corrections.append(CorrectionRecord(trigger_checks=["transfer"], scope="bank transfer confirmation",
                                                        applied=True, note="; ".join(transfer_notes)))
        term_notes = fix_minimum_term(fields)
        if term_notes:
            doc.corrections.append(CorrectionRecord(trigger_checks=["contract"], scope="minimum term", applied=True,
                                                    note="; ".join(term_notes)))
        if not xml:
            basis_notes = fill_tax_basis(fields)
            if basis_notes:
                doc.corrections.append(CorrectionRecord(trigger_checks=["amounts"], scope="tax basis", applied=True,
                                                        note="; ".join(basis_notes)))
        form_notes = drop_form_deadline(fields, contract_type)
        if form_notes:
            doc.corrections.append(CorrectionRecord(trigger_checks=["dates"], scope="due date", applied=True,
                                                    note="; ".join(form_notes)))
        if not xml:
            payee_notes = order_payee_accounts(fields, evidence, doc_text, "CREDIT_NOTE" in entity.type_tags)
            if payee_notes:
                doc.corrections.append(CorrectionRecord(trigger_checks=["iban"], scope="payment accounts", applied=True,
                                                        note="; ".join(payee_notes)))
        if (contract_for(contract_type).date_group or any(spec.body_key == "finance_body" for spec in specs)):
            date_notes = fix_document_dates(fields, evidence, doc_text, outcome.audit, contract_type,
                                            layer.page(pages[0]).best_text() if layer.page(pages[0]) else None)
            if date_notes:
                doc.corrections.append(CorrectionRecord(trigger_checks=["dates"], scope="document dates", applied=True,
                                                        note="; ".join(date_notes)))
        if normalized:
            logger.info(f"normalized {len(normalized)} date fields to ISO 8601")
        doc.errors += [f"schema: {e}" for e in outcome.errors[:10]]
        role_notes = remove_role_labels(fields.setdefault("meta_header", {}))
        if role_notes:
            doc.corrections.append(CorrectionRecord(trigger_checks=["parties"], scope="role labels", applied=True,
                                                    note="; ".join(role_notes)))
        cover_sheet = "APPLICATION" in entity.type_tags and unfilled_form(outcome.audit)
        if cover_sheet:
            doc.corrections.append(CorrectionRecord(trigger_checks=["parties"], scope="form", applied=True,
                                                    note="pre-addressed form without signature or handwriting: contract §2.5 "
                                                         "(issuer = sender, printed address = recipient)"))
        if "APPLICATION" in entity.type_tags and not cover_sheet:
            meta = fields.setdefault("meta_header", {})
            sender = meta.get("sender") or {}
            lines = doc_text.splitlines()
            current = sender.get("name") or sender.get("company")
            subject = bool(current) and any(subject_context(lines, i) for i, line in enumerate(lines)
                                            if din5008.contained(str(current), [line]))
            filler = form_filler(lines) if not current or subject else None
            if filler and not (current and din5008.contained(filler[1], [str(current)])):
                if subject:
                    sender = {}  # the insured / deceased person named in the form is its subject, not the filler
                meta["sender"] = {**sender, "name": filler[1]}
                doc.corrections.append(CorrectionRecord(trigger_checks=["parties"], scope="meta_header.sender", applied=True,
                                                        note=f"form filler '{filler[1]}' taken from the field '{filler[0]}' (§2.4)"))
            swapped, reason = apply_form_semantics(fields.setdefault("meta_header", {}), private_id, business_id)
            if swapped:
                swap_party_evidence(evidence)
                logger.info(f"form semantics: sender/recipient swapped ({reason})")
            entity.direction = doc.direction = "OUTBOUND"
        if entity.tenant_context == "BUSINESS":
            # a person-name match alone never decides BUSINESS: re-derive the context from the parties
            meta = fields.get("meta_header") or {}
            _direction, context = direction_from_parties(meta.get("sender"), meta.get("recipient"), private_id, business_id)
            if context == "PRIVATE":
                entity.tenant_context = doc.tenant_context = "PRIVATE"
        if entity.direction in ("UNKNOWN", "") or entity.tenant_context in ("UNKNOWN", ""):
            meta = fields.get("meta_header") or {}
            direction, context = direction_from_parties(meta.get("sender"), meta.get("recipient"), private_id, business_id)
            if entity.direction in ("UNKNOWN", "") and direction != "UNKNOWN":
                entity.direction = doc.direction = direction
            if entity.tenant_context in ("UNKNOWN", "") and context != "UNKNOWN":
                entity.tenant_context = doc.tenant_context = context
        meta = fields.setdefault("meta_header", {})
        window_recipient, letterhead_sender = geometry_policy(contract_type, cover_sheet)
        if "APPLICATION" in entity.type_tags and not cover_sheet:
            letterhead_sender = False  # the filler of a form is never taken from the geometry
        geometry_conflicts: List[Tuple[str, str, str]] = []
        layout_notes = din5008.enforce_parties(meta, din5008.analyze(layer.page(pages[0]), layer.page(pages[-1])),
                                               window_recipient, letterhead_sender, doc_text, geometry_conflicts)
        if layout_notes:
            if any(n.startswith("sender/recipient swapped") for n in layout_notes):
                swap_party_evidence(evidence)
            doc.corrections.append(CorrectionRecord(trigger_checks=["layout"], scope="parties (DIN 5008 geometry)", applied=True,
                                                    note="; ".join(layout_notes)))
            for ev in evidence:
                if ev.path.startswith(("meta_header.sender.", "meta_header.recipient.")) and ev.value is not None \
                        and not holds(fields, ev):
                    ev.original_value, ev.action, ev.status = ev.value, "replaced_by_layout", GroundingStatus.REMOVED
        # forms filled in by a person and confirmations on printed forms carry the issuer in a form field (table cell)
        if contract_for(contract_type).party_case != "2.4" and contract_type != "CERTIFICATE" \
                and sender_only_in_table(meta.get("sender"), layer, pages):
            removed = meta.pop("sender")
            doc.corrections.append(CorrectionRecord(trigger_checks=["parties"], scope="meta_header.sender", applied=True,
                                                    note=f"sender {removed!r} removed: printed only in a table cell, not as letterhead/sender"))
            for ev in evidence:
                if ev.path.startswith("meta_header.sender.") and ev.status != GroundingStatus.REMOVED:
                    ev.original_value, ev.action, ev.status = ev.value, "nulled", GroundingStatus.REMOVED
        party_notes, direction = harmonize_parties(fields.setdefault("meta_header", {}), entity.direction, entity.type_tags,
                                                   private_id, business_id)
        entity.direction = doc.direction = direction
        if any("swapped" in n for n in party_notes):
            swap_party_evidence(evidence)
        if party_notes:
            doc.corrections.append(CorrectionRecord(trigger_checks=["parties"], scope="meta_header parties", applied=True,
                                                    note="; ".join(party_notes)))
        if any(contract_for(spec.name).party_case == "2.7" for spec in specs):
            # private notices, cards, memos (§2.7) have no parties - also when NOTE is only one of the type tags; applied
            # after the geometry and identity steps, which could otherwise set them again
            meta = fields.setdefault("meta_header", {})
            dropped = [key for key in ("sender", "recipient") if meta.pop(key, None)]
            if dropped:
                doc.corrections.append(CorrectionRecord(trigger_checks=["parties"], scope="parties", applied=True,
                                                        note=f"{'/'.join(dropped)} removed: private notice/card/memo has no parties (§2.7)"))
                for ev in evidence:
                    if ev.path.startswith(("meta_header.sender.", "meta_header.recipient.")) and ev.status != GroundingStatus.REMOVED:
                        ev.original_value, ev.action, ev.status = ev.value, "nulled", GroundingStatus.REMOVED
        extra_checks: List[CheckResult] = [
            CheckResult(check_id=din5008.GEOMETRY_CONFLICT, rule="layout", status=CheckStatus.FAIL, severity=Severity.INFO,
                        field=path, actual=model_value, expected=geometry_value,
                        message=f"geometry/model conflict: the model read {model_value!r}, the layout region shows "
                                f"{geometry_value!r} (model value kept, review)")
            for path, model_value, geometry_value in geometry_conflicts]
        if xml:
            comparison = zugferd.compare(xml, fields)
            doc.zugferd_comparison = comparison
            for diff in comparison.differences:
                # Diagnostic about the visual reading: the output value comes from the XML (ground truth).
                extra_checks.append(CheckResult(check_id="zugferd.visual_mismatch", rule="zugferd", status=CheckStatus.FAIL,
                                                severity=Severity.INFO, expected=diff["xml"], actual=diff.get("visual"),
                                                message=f"visual reading of {diff['field']}: {diff.get('visual')} vs XML {diff['xml']} "
                                                        f"(output uses the XML value)"))
            fields = zugferd.overlay(fields, xml)
        extra_checks += self._repair_rows(prepared, fields, pages)
        # Types without a folded visual audit (statements: folding it in degraded their extraction) still get the
        # stamps the text layer prints as stamp lines.
        audit = outcome.audit if outcome.audit is not None else {"layer_stamps": [], "audit_summary": {}, "meta_mode": "TEXT_ONLY"}
        added = stamps_from_text(audit, doc_text)
        if added:
            logger.info(f"visual audit: {added} stamp(s) taken from the text layer")
            outcome.audit = audit
        if outcome.audit:
            doc.visual_audit = outcome.audit
            integrity = outcome.audit.get("integrity") or {}
            if integrity.get("is_type_match") is False and integrity.get("suggested_types"):
                extra_checks.append(CheckResult(check_id="audit.type_mismatch", rule="audit", status=CheckStatus.FAIL,
                                                severity=Severity.INFO, message=f"visual audit suggests {integrity['suggested_types']}"))

        if form_fields:
            field_notes, field_checks = map_fields(fields, evidence, form_fields)
            extra_checks += field_checks
            if field_notes:
                doc.corrections.append(CorrectionRecord(trigger_checks=["fieldread"], scope="form fields", applied=True,
                                                        note="; ".join(field_notes)))
        t0 = time.monotonic()
        report = self._evaluate(fields, entity, specs)
        timings["plausibility"] += round(time.monotonic() - t0, 3)

        grounding_on = self.settings.grounding.enabled if options.grounding is None else options.grounding
        grounding_checks: List[CheckResult] = []
        if grounding_on:
            t0 = time.monotonic()
            failed_fields = [c.field for c in report.findings if c.field]
            grounding = self.harness.run(prepared.path, layer, pages, fields, evidence, outcome.tables, specs, failed_fields)
            grounding_checks = grounding.checks
            doc.corrections += grounding.snaps
            timings["grounding"] += round(time.monotonic() - t0, 3)
            report = self._evaluate(fields, entity, specs)

        if self.settings.pipeline.targeted_correction:
            t0 = time.monotonic()
            evidence_pages = {e.path: e.matched_page or e.page for e in evidence if (e.matched_page or e.page)}
            trusted = {e.path: e.status.value in ("verified_visual", "grounded") for e in evidence}
            fields, report, record = self.corrector.run(prepared.path, layer, pages, fields, report, specs, evidence_pages,
                                                        lambda f: self._evaluate(f, entity, specs), trusted,
                                                        lambda name, pg: self.extractor.reextract_table(
                                                            prepared.path, layer, pg, entity.type_tags, name))
            if record:
                doc.corrections.append(record)
            timings["correction"] += round(time.monotonic() - t0, 3)

        disagreements = 0
        if self.settings.grounding.self_consistency:
            disagreements = self._self_consistency(prepared, entity, evidence, extra_checks, private_id, business_id)

        signature_notes = drop_signature_parties(fields, evidence, doc_text, outcome.audit, contract_type)
        if signature_notes:
            doc.corrections.append(CorrectionRecord(trigger_checks=["parties"], scope="signature-only parties", applied=True,
                                                    note="; ".join(signature_notes)))
        extra_checks += check_contract_payment(fields)
        # §9: cross-field date consistency (after grounding / zoom / correction: their readings are final here)
        date_checks, date_notes = check_dates_consistency(fields, evidence, date.today(),
                                                          self.settings.plausibility.future_days_tolerance)
        extra_checks += date_checks
        if date_notes:
            doc.corrections.append(CorrectionRecord(trigger_checks=["dates.consistency"], scope="ambiguous year", applied=True,
                                                    note="; ".join(date_notes)))
        doc.plausibility = score_report(report.checks + grounding_checks + extra_checks)
        for table in outcome.tables:
            spec_rows = ((fields.get("bodies") or {}).get(table.path.split(".")[1]) or {}).get(table.path.split(".")[2])
            if isinstance(spec_rows, list):
                table.rows = spec_rows
                table.source_pages = sorted({int(r["source_page"]) for r in spec_rows if isinstance(r.get("source_page"), int)})
        doc.tables = outcome.tables
        doc.evidence = evidence

        review_vision = self.settings.review.vision if options.review_vision is None else options.review_vision
        if review_vision:
            t0 = time.monotonic()
            doc.review = self.reviewer.vision(prepared.path, doc, evidence)
            doc.plausibility = score_report(doc.plausibility.checks)
            timings["review"] += round(time.monotonic() - t0, 3)

        printed_invalid = {str(c.actual) for c in doc.plausibility.checks if c.check_id == CHECK_IBAN_PRINTED_INVALID and c.actual}
        bank_notes = sanitize_bank_identifiers(fields, printed_invalid)
        if bank_notes:
            doc.corrections.append(CorrectionRecord(trigger_checks=["identifiers"], scope="bank identifiers", applied=True,
                                                    note="; ".join(bank_notes)))
        xml_fields = zugferd.overlay({"meta_header": {}, "bodies": {}}, xml) if xml else None
        if xml_fields is not None:
            restored = zugferd.protect_xml_values(fields, xml_fields)
            if restored:
                by_path = {e.path: e for e in evidence}
                for path, changed, value in restored:
                    ev = by_path.get(path)
                    if ev is not None:
                        ev.original_value, ev.value, ev.action, ev.status = changed, value, "kept", GroundingStatus.GROUNDED
                doc.corrections.append(CorrectionRecord(
                    trigger_checks=["zugferd"], scope="embedded XML values", applied=True,
                    note="; ".join(f"{path}: {changed!r} -> XML {value!r} (the embedded XML is authoritative)"
                                   for path, changed, value in restored)))
        accepted = assess(fields, evidence, layer, pages, contract_type, doc.plausibility.checks, outcome.audit, xml_fields)
        logger.info(f"acceptance: {accepted['auto_accepted']} auto-accepted, {accepted['flagged']} flagged for review")
        doc.fields = self._hydrate(fields, entity, outcome, doc)
        comparison_mismatches = doc.zugferd_comparison.mismatched if doc.zugferd_comparison else 0
        doc.trust = trust_report(evidence, doc.tables, grounding_checks, doc.plausibility.score, len(doc.plausibility.findings),
                                 sum(1 for c in doc.corrections if c.applied), comparison_mismatches, disagreements)
        return doc

    def _repair_rows(self, prepared: Prepared, fields: Dict[str, Any], pages: List[int]) -> List[CheckResult]:
        """Fills transactions without amount (balance diff, text line, else zoomed crop of the row)."""
        body = (fields.get("bodies") or {}).get("statement_body")
        if not isinstance(body, dict) or not body.get("transactions"):
            return []
        rows = body["transactions"]
        result = fill_row_amounts(rows, body.get("opening_balance"), prepared.layer)
        targets = []
        for index, row in zoom_targets_for_rows(rows, result.unresolved, self.settings.grounding.max_escalations):
            label = f"signed amount of the booking '{str(row.get('description') or '')[:50]}' dated {row.get('booking_date')}"
            targets.append((index, ZoomTarget(id=f"R{index}", page=row["source_page"], label=label, value="?", kind="amount",
                                              evidence=str(row.get("description") or "")[:40] or None)))
        if targets:
            answers = self.harness.zoomer.verify(prepared.path, prepared.layer, [t for _, t in targets])
            for index, target in targets:
                answer = answers.get(target.id)
                value = parse_printed("amount", answer.printed) if answer and answer.visible else None
                if value is not None:
                    rows[index]["amount"] = str(value)
                    result.filled.append(index)
                    result.methods[index] = "zoom"
                    result.unresolved.remove(index)
        checks = repair_checks("transactions", result, rows)
        if result.unresolved:
            checks.append(CheckResult(check_id="table.amount_missing", rule="repair", status=CheckStatus.FAIL,
                                      severity=Severity.WARNING, table="transactions", rows=result.unresolved,
                                      message=f"{len(result.unresolved)} transactions without amount could not be filled"))
        return checks

    def _hydrate(self, fields: Dict[str, Any], entity: Entity, outcome: Any, doc: LogicalDocument) -> Dict[str, Any]:
        """Validates the final fields into the KPaperFlux-compatible SemanticExtraction."""
        payload = {
            "meta_header": fields.get("meta_header") or {},
            "bodies": fields.get("bodies") or {},
            "type_tags": self.catalog.output_tags(entity.type_tags),
            "direction": entity.direction,
            "tenant_context": entity.tenant_context,
            "ai_confidence": outcome.ai_confidence,
            "extraction_source": fields.get("extraction_source") or "AI",
            "visual_audit": outcome.audit,
        }
        try:
            model = SemanticExtraction.model_validate(payload)
            return model.model_dump(mode="json", exclude_none=True)
        except ValidationError as e:
            doc.errors.append(f"hydration: {e.error_count()} validation errors, raw fields kept")
            logger.warning(f"Hydration failed: {e}")
            return payload

    def _self_consistency(self, prepared: Prepared, entity: Entity, evidence: List[Any], checks: List[CheckResult],
                          private_id: Optional[IdentityProfile], business_id: Optional[IdentityProfile]) -> int:
        """Second images-only extraction; disagreeing critical values become findings."""
        second = self.extractor.run(prepared.path, prepared.layer, entity.pages, entity.type_tags, private_id, business_id,
                                    audit=False, aux_text=False)
        if second is None:
            return 0
        other = {e.path: e.value for e in second.evidence}
        disagreements = 0
        for ev in evidence:
            if not ev.critical or ev.path not in other:
                continue
            a, b = ev.value, other[ev.path]
            da, db = to_decimal(a) if ev.kind == "amount" else None, to_decimal(b) if ev.kind == "amount" else None
            same = (da is not None and db is not None and abs(da - db) < to_decimal("0.01")) or compact(str(a)) == compact(str(b))
            if not same:
                disagreements += 1
                checks.append(CheckResult(check_id="consistency.disagreement", rule="consistency", status=CheckStatus.FAIL,
                                          severity=Severity.WARNING, field=ev.path, expected=str(a), actual=str(b),
                                          message=f"images+text read {a!r}, images-only read {b!r}"))
        return disagreements

    # ---------------------------------------------------------------- reviews

    def review_text(self, result: AnalysisResult, layer: TextLayer) -> None:
        """Text-only second opinion (separate model; run as a batch phase in bench)."""
        calls: List[CallRecord] = []
        collect = getattr(self.backend, "collect", None)
        for doc in result.logical_documents:
            if not doc.fields:
                continue
            if collect is not None:
                with collect(calls):
                    doc.review = self.reviewer.text(layer, doc)
            else:
                doc.review = self.reviewer.text(layer, doc)
            doc.plausibility = score_report(doc.plausibility.checks)
            if doc.review is None:
                doc.review = ReviewResult()
        result.ai_calls.extend(calls)
        result.timings["review_text"] = round(sum(c.seconds for c in calls), 3)
