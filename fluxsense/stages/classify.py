# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/classify.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Stage 1 (ported from KPaperFlux core/ai/stage1.py): adaptive
                pre-flight routing SANDWICH / HEADER_SCAN / FULL_READ,
                classification & stack segmentation with page thumbnails,
                orphan-page validation, identity validation + plausibility
                arbiter, correction retries and a deterministic coverage
                repair. Fixes: page markers keep real page numbers in
                SANDWICH/HEADER modes.
------------------------------------------------------------------------------
"""

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from fluxsense.catalog.contract import is_bank_transfer_confirmation
from fluxsense.catalog.registry import Catalog
from fluxsense.config import Settings
from fluxsense.extract.base import TextLayer
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import ImagePart, page_images
from fluxsense.log import get_logger
from fluxsense.models.identity import IdentityProfile
from fluxsense.models.types import PREFLIGHT_TYPES, SANDWICH_TYPES
from fluxsense.prompts import stage1 as prompts
from fluxsense.stages.identity import validate_structure
from fluxsense.stages.signatures import (PageSignature, apply_layout, collect_signatures, document_starts, has_start_evidence, is_attachable,
                                         signatures_prompt)

logger = get_logger("stages.classify")

FULL_READ: str = "FULL_READ_MODE"
SANDWICH: str = "SANDWICH_MODE"
HEADER_SCAN: str = "HEADER_SCAN_MODE"
MAX_THUMBNAILS: int = 60

PREFLIGHT_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {"primary_type": {"type": "string", "enum": PREFLIGHT_TYPES},
                   "is_hybrid_suspicion": {"type": "boolean"}, "looks_like_stack": {"type": "boolean"},
                   "confidence": {"type": "number"}},
    "required": ["primary_type", "is_hybrid_suspicion", "looks_like_stack", "confidence"],
}
ARBITER_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {"is_plausible": {"type": "boolean"}, "reasoning": {"type": "string", "maxLength": 200},
                   "suggested_context": {"type": "string", "enum": ["PRIVATE", "BUSINESS", "UNKNOWN"]}},
    "required": ["is_plausible", "suggested_context"],
}


def classification_schema(type_names: List[str]) -> Dict[str, Any]:
    """JSON schema for Stage 1.1 with the catalog types as enum."""
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "source_file_summary": {"type": "object", "additionalProperties": False,
                                    "properties": {"primary_language": {"type": "string", "maxLength": 5}}},
            "detected_entities": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "type_tags": {"type": "array", "items": {"type": "string", "enum": type_names}},
                    "page_indices": {"type": "array", "items": {"type": "integer"}},
                    "direction": {"type": "string", "enum": ["INBOUND", "OUTBOUND", "INTERNAL", "UNKNOWN"]},
                    "tenant_context": {"type": "string", "enum": ["PRIVATE", "BUSINESS", "UNKNOWN"]},
                    "confidence": {"type": "number"},
                },
                "required": ["type_tags", "page_indices", "direction", "tenant_context", "confidence"]}},
        },
        "required": ["detected_entities"],
    }


@dataclass
class Entity:
    """A logical document found by Stage 1."""

    type_tags: List[str]
    pages: List[int]
    direction: str = "UNKNOWN"
    tenant_context: str = "UNKNOWN"
    confidence: float = 0.0
    letterhead: bool = False


@dataclass
class ClassificationResult:
    """Outcome of Stage 1."""

    entities: List[Entity]
    strategy: str = FULL_READ
    preflight: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    language: Optional[str] = None
    signatures: List[Dict[str, Any]] = field(default_factory=list)
    starts: List[int] = field(default_factory=list)


def extract_headers_footers(pages: List[str], header_ratio: float = 0.15, footer_ratio: float = 0.10) -> List[str]:
    """
    Reduces each page to its top and bottom region (KPaperFlux HEADER_SCAN).

    Args:
        pages: Page texts.
        header_ratio: Share of lines kept from the top.
        footer_ratio: Share of lines kept from the bottom.

    Returns:
        Reduced page texts.
    """
    out: List[str] = []
    for text in pages:
        lines = text.split("\n")
        total = len(lines)
        cut_top, cut_bottom = int(total * header_ratio), int(total * footer_ratio)
        if total < 10 or cut_top + cut_bottom >= total:
            out.append(text)
        else:
            out.append(f"[HEADER AREA]\n" + "\n".join(lines[:cut_top]) + "\n...\n[FOOTER AREA]\n" + "\n".join(lines[-cut_bottom:]))
    return out


def route(total_pages: int, preflight: Dict[str, Any], header_scan_min_pages: int) -> tuple:
    """
    Selects the scan strategy (KPaperFlux run_stage_1_adaptive phase B).

    Args:
        total_pages: Number of pages.
        preflight: Pre-flight answer ({} if skipped).
        header_scan_min_pages: Page count from which HEADER_SCAN is used.

    Returns:
        Tuple (strategy, 1-based page numbers to show).
    """
    primary = str(preflight.get("primary_type", "OTHER")).upper()
    if primary in SANDWICH_TYPES:
        indices = sorted({p for p in (1, 2, 3, total_pages) if 1 <= p <= total_pages})
        return SANDWICH, indices
    if total_pages >= header_scan_min_pages or bool(preflight.get("looks_like_stack")):
        return HEADER_SCAN, list(range(1, total_pages + 1))
    return FULL_READ, list(range(1, total_pages + 1))


def normalize_page_indices(result: Dict[str, Any], total_pages: int) -> None:
    """Shifts 0-based page_indices (contain 0, max < total) to 1-based in place."""
    entities = [e for e in result.get("detected_entities") or [] if isinstance(e, dict)]
    pages = [p for e in entities for p in e.get("page_indices") or [] if isinstance(p, int)]
    if pages and 0 in pages and max(pages) < total_pages:
        for ent in entities:
            ent["page_indices"] = [p + 1 for p in ent.get("page_indices") or [] if isinstance(p, int)]


def repair_coverage(entities: List[Entity], total_pages: int) -> List[Entity]:
    """
    Deterministic coverage repair after the AI retries: drops invalid and
    duplicate page claims, assigns orphan pages to the preceding entity (the
    KPaperFlux 'OTHER sink' rule) and orders entities by first page.

    Args:
        entities: Entities from the model.
        total_pages: Page count.

    Returns:
        Entities covering every page exactly once.
    """
    all_pages = [p for e in entities for p in e.pages]
    if all_pages and 0 in all_pages and max(all_pages) < total_pages:
        # The model answered 0-based page indices.
        entities = [Entity(e.type_tags, [p + 1 for p in e.pages], e.direction, e.tenant_context, e.confidence) for e in entities]
    claimed: set = set()
    clean: List[Entity] = []
    for ent in sorted(entities, key=lambda e: min(e.pages) if e.pages else total_pages + 1):
        pages = sorted({p for p in ent.pages if 1 <= p <= total_pages and p not in claimed})
        if pages:
            claimed.update(pages)
            clean.append(Entity(ent.type_tags or ["OTHER"], pages, ent.direction, ent.tenant_context, ent.confidence))
    if not clean:
        first = entities[0] if entities else None
        if first is not None:  # keep the model's types even if its page numbers were unusable
            return [Entity(first.type_tags or ["OTHER"], list(range(1, total_pages + 1)), first.direction,
                           first.tenant_context, first.confidence)]
        return [Entity(["OTHER"], list(range(1, total_pages + 1)))]
    for page in range(1, total_pages + 1):
        if page in claimed:
            continue
        owner = None
        for ent in clean:
            if ent.pages[0] < page:
                owner = ent
        owner = owner or clean[0]
        owner.pages = sorted(owner.pages + [page])
    return sorted(clean, key=lambda e: e.pages[0])


def enforce_starts(entities: List[Entity], starts: List[int]) -> List[Entity]:
    """
    Splits entities at signature-based document starts that the model merged
    (the split-off part keeps the entity's tags; context and direction too).

    Args:
        entities: Entities with full coverage.
        starts: Start pages decided from page signatures.

    Returns:
        Entities split at every start page.
    """
    result: List[Entity] = []
    for ent in entities:
        current: List[int] = []
        for page in ent.pages:
            if current and page in starts:
                result.append(Entity(list(ent.type_tags), current, ent.direction, ent.tenant_context, ent.confidence))
                current = []
            current.append(page)
        result.append(Entity(list(ent.type_tags), current, ent.direction, ent.tenant_context, ent.confidence))
    if len(result) != len(entities):
        logger.info(f"Stage 1: split merged documents at signature starts -> {[e.pages for e in result]}")
    return result


LETTER_TYPES = frozenset({"LETTER", "OFFICIAL_LETTER", "LEGAL_CORRESPONDENCE"})


def retype_enclosures(entities: List[Entity], signatures: List[PageSignature]) -> List[Entity]:
    """
    A 'letter' whose first page has no letterhead, no salutation and no address
    block is not a letter: it is an enclosure (screenshot, log, protocol) -> OTHER.

    Args:
        entities: Entities.
        signatures: Page signatures.

    Returns:
        Entities with re-typed enclosures.
    """
    by_page = {s.page: s for s in signatures}
    for ent in entities:
        first = by_page.get(ent.pages[0])
        if first is None or not ent.type_tags or not set(ent.type_tags) <= LETTER_TYPES:
            continue
        if not (first.letterhead or first.salutation or first.addressee or first.closing or first.form):
            logger.info(f"Stage 1: pages {ent.pages} re-typed {ent.type_tags} -> OTHER (enclosure without letterhead)")
            ent.type_tags = ["OTHER"]
    return entities


ROUTE_TO_TRANSFER: frozenset = frozenset({"RECEIPT", "OTHER", "ORDER_CONFIRMATION"})


def route_transfer_confirmations(entities: List[Entity], layer: TextLayer) -> List[Entity]:
    """
    Field contract §2.8: a bank's confirmation of a single transaction that
    the classifier called RECEIPT / OTHER / ORDER_CONFIRMATION is
    re-typed TRANSFER_CONFIRMATION when bank-transfer vocabulary is printed.
    """
    for ent in entities:
        if not ent.type_tags or not set(ent.type_tags) <= ROUTE_TO_TRANSFER:
            continue
        text = "\n".join(layer.page(p).best_text() for p in ent.pages if layer.page(p))
        if is_bank_transfer_confirmation(text):
            logger.info(f"Stage 1: pages {ent.pages} re-typed {ent.type_tags} -> TRANSFER_CONFIRMATION (bank transfer vocabulary)")
            ent.type_tags = ["TRANSFER_CONFIRMATION"]
    return entities


ROUTE_TO_NOTE: frozenset = frozenset({"LETTER", "OTHER"})
# Vocabulary of private notices and cards (§2.7): death notices, thank-you cards after a funeral, greeting cards.
NOTE_VOCABULARY_RE = re.compile(
    r"(in\s+stiller\s+trauer|in\s+tiefer\s+trauer|in\s+liebevoller\s+erinnerung|wir\s+nehmen\s+abschied|traueranzeige|"
    r"trauerfeier|beisetzung|urnenbeisetzung|entschlief|danksagung|herzlichen\s+glückwunsch|herzliche\s+glückwünsche|"
    r"frohe\s+weihnachten|frohe\s+ostern|ein\s+gutes\s+neues\s+jahr|alles\s+gute\s+zum|zum\s+geburtstag)", re.IGNORECASE)
NOTE_MAX_CHARS: int = 1500


def route_private_notes(entities: List[Entity], layer: TextLayer) -> List[Entity]:
    """
    Field contract §2.7: a short LETTER / OTHER without letterhead or address
    block that prints death-notice or greeting-card vocabulary is re-typed
    NOTE (private notices and cards have no parties).
    """
    for ent in entities:
        if not ent.type_tags or not set(ent.type_tags) <= ROUTE_TO_NOTE or ent.letterhead or len(ent.pages) > 2:
            continue
        text = "\n".join(layer.page(p).best_text() for p in ent.pages if layer.page(p))
        if len(text) <= NOTE_MAX_CHARS and NOTE_VOCABULARY_RE.search(text):
            logger.info(f"Stage 1: pages {ent.pages} re-typed {ent.type_tags} -> NOTE (private notice / card vocabulary)")
            ent.type_tags = ["NOTE"]
    return entities


def attach_boilerplate(entities: List[Entity], signatures: List[PageSignature]) -> List[Entity]:
    """
    Merges an entity into the preceding one when its first page is blank or
    boilerplate (e.g. a trailing 'Wichtige Informationen' page of a statement,
    terms). Enclosures without letterhead (logs, protocols) stay separate.

    Args:
        entities: Entities in page order.
        signatures: Page signatures.

    Returns:
        Entities with boilerplate pages attached to their predecessor.
    """
    by_page = {s.page: s for s in signatures}
    result: List[Entity] = []
    for ent in entities:
        first = by_page.get(ent.pages[0])
        if result and first is not None and is_attachable(first):
            result[-1].pages = sorted(result[-1].pages + ent.pages)
            logger.info(f"Stage 1: pages {ent.pages} attached to the preceding document (no start evidence)")
            continue
        result.append(ent)
    return result


class Classifier:
    """Stage 1 controller."""

    def __init__(self, backend: LLMBackend, settings: Settings, catalog: Catalog) -> None:
        """
        Args:
            backend: LLM backend.
            settings: Settings.
            catalog: Type catalog.
        """
        self.backend = backend
        self.settings = settings
        self.catalog = catalog

    def _thumbnails(self, pdf_path: Path, pages: List[int]) -> List[ImagePart]:
        """Small page thumbnails (coarse level 0)."""
        shown = pages[:MAX_THUMBNAILS]
        return page_images(pdf_path, shown, [f"PAGE {p} (thumbnail)" for p in shown],
                           self.settings.llm.classify_image_px, self.settings.llm.image_format, dpi=100)

    def preflight(self, pdf_path: Path, texts: List[str]) -> Dict[str, Any]:
        """Stage 1.0 pre-flight on the first three pages."""
        first = list(range(1, min(3, len(texts)) + 1))
        content = "\n".join(f"--- PAGE {p} ---\n{texts[p - 1][:4000]}" for p in first)
        request = LLMRequest(prompt=prompts.PROMPT_STAGE_1_0_PREFLIGHT.format(content=content, total_pages=len(texts)),
                             label="Stage 1.0 (Pre-Flight)", model=self.settings.models.primary,
                             images=self._thumbnails(pdf_path, first), schema=PREFLIGHT_SCHEMA,
                             schema_name="preflight", max_tokens=200)
        result = self.backend.generate_json(request)
        return result if isinstance(result, dict) else {}

    def _system_prompt(self, private_id: Optional[IdentityProfile], business_id: Optional[IdentityProfile]) -> str:
        """Static Stage 1.1 system prompt (cacheable)."""
        identity = json.dumps({"PRIVATE_ENTITY": private_id.summary() if private_id else "Not Configured",
                               "BUSINESS_ENTITY": business_id.summary() if business_id else "Not Configured"}, indent=2)
        names = self.catalog.names()
        hints = "\n".join(f"- {n}: {self.catalog.get(n).description}" for n in names)
        return prompts.PROMPT_STAGE_1_1_SYSTEM.format(identity_json_str=identity, allowed_types=json.dumps(names),
                                                      type_hints=hints)

    def run(self, pdf_path: Path, layer: TextLayer, private_id: Optional[IdentityProfile],
            business_id: Optional[IdentityProfile]) -> ClassificationResult:
        """
        Runs pre-flight (long documents only), routing and classification.

        Args:
            pdf_path: PDF file.
            layer: Auxiliary text layer.
            private_id: Private identity.
            business_id: Business identity.

        Returns:
            ClassificationResult with full page coverage.
        """
        texts = layer.texts()
        total = len(texts)
        cfg = self.settings.pipeline
        pre: Dict[str, Any] = {}
        if total >= cfg.preflight_min_pages:
            pre = self.preflight(pdf_path, texts)
        strategy, shown = route(total, pre, cfg.header_scan_min_pages)
        shown_texts = [texts[p - 1] for p in shown]
        if strategy == HEADER_SCAN:
            shown_texts = extract_headers_footers(shown_texts)
        analysis_text = "\n\n".join(f"--- PAGE {p} ---\n{t}" for p, t in zip(shown, shown_texts))
        signatures: List[PageSignature] = []
        starts: List[int] = []
        if total >= cfg.signature_min_pages:
            signatures = collect_signatures(self.backend, self.settings, pdf_path, layer)
            apply_layout(signatures, layer)
            starts = document_starts(signatures) if signatures else []
        system = self._system_prompt(private_id, business_id)
        user = prompts.PROMPT_STAGE_1_1_USER.format(mode=strategy, total_pages=total,
                                                    analysis_text=analysis_text[:cfg.max_text_chars])
        if signatures and len(signatures) == total:
            user = signatures_prompt(signatures, starts) + user
        schema = classification_schema(self.catalog.names())
        images = self._thumbnails(pdf_path, shown)
        request = LLMRequest(prompt=user, system=system, label="Stage 1.1 (Classification)",
                             model=self.settings.models.primary, images=images, schema=schema,
                             schema_name="classification", max_tokens=1500)
        result = self.backend.generate_json(request)
        errors: List[str] = []
        attempt = 0
        while isinstance(result, dict) and attempt < cfg.ai_retries:
            normalize_page_indices(result, total)
            errors = self.validate(result, texts, strategy, private_id, business_id)
            if not any(e.startswith("SEGMENTATION_ERROR") for e in errors):
                break  # identity doubts are settled by the arbiter, not by re-classifying
            attempt += 1
            logger.info(f"Stage 1.1 validation failed (attempt {attempt}): {errors}")
            correction = prompts.PROMPT_REFINEMENT_CORRECTION.format(
                error_summary="\n".join(f"- {e}" for e in errors),
                validation_checks="Ensure all pages are covered and context is correct.")
            retry_request = replace(request, label=f"Stage 1.1 Correction {attempt}",
                                    followups=[{"role": "assistant", "content": json.dumps(result)},
                                               {"role": "user", "content": correction}])
            retry = self.backend.generate_json(retry_request)
            if not isinstance(retry, dict):
                break
            result = retry
        raw = result if isinstance(result, dict) else {}
        entities = self._entities(raw, total)
        doubted = {int(m.group(1)) for e in errors if e.startswith("CONTEXT_ERROR") for m in [re.search(r"Page (\d+)", e)] if m}
        for ent in entities:
            if ent.pages and ent.pages[0] in doubted:
                ent.tenant_context = "UNKNOWN"
        if strategy == SANDWICH and len(entities) == 1:
            entities[0].pages = list(range(1, total + 1))
        entities = repair_coverage(entities, total)
        if starts:
            entities = enforce_starts(entities, starts)
        if signatures:
            entities = retype_enclosures(attach_boilerplate(entities, signatures), signatures)
            by_page = {sig.page: sig for sig in signatures}
            for ent in entities:
                first = by_page.get(ent.pages[0])
                ent.letterhead = bool(first and (first.letterhead or first.addressee or first.salutation))
        entities = route_private_notes(route_transfer_confirmations(entities, layer), layer)
        language = (raw.get("source_file_summary") or {}).get("primary_language")
        result = ClassificationResult(entities=entities, strategy=strategy, preflight=pre, raw=raw, errors=errors,
                                      language=language)
        result.signatures = [s.__dict__ for s in signatures]
        result.starts = starts
        return result

    def _entities(self, raw: Dict[str, Any], total: int) -> List[Entity]:
        """Converts the raw answer into Entity objects (unknown types -> OTHER)."""
        entities: List[Entity] = []
        for ent in raw.get("detected_entities") or []:
            if not isinstance(ent, dict):
                continue
            tags = [str(t).upper() for t in ent.get("type_tags") or [] if self.catalog.has(str(t))]
            if "UTILITY_BILL" in tags and "INVOICE" not in tags:
                tags.insert(0, "INVOICE")
            pages = [int(p) for p in ent.get("page_indices") or [] if isinstance(p, (int, float))]
            entities.append(Entity(type_tags=list(dict.fromkeys(tags)) or ["OTHER"], pages=pages,
                                   direction=str(ent.get("direction") or "UNKNOWN"),
                                   tenant_context=str(ent.get("tenant_context") or "UNKNOWN"),
                                   confidence=float(ent.get("confidence") or 0.0)))
        return entities

    def validate(self, result: Dict[str, Any], texts: List[str], strategy: str,
                 private_id: Optional[IdentityProfile], business_id: Optional[IdentityProfile]) -> List[str]:
        """
        Validates the classification (orphans, identity; arbiter second opinion).

        Args:
            result: Stage 1.1 answer.
            texts: Page texts.
            strategy: Scan strategy.
            private_id: Private identity.
            business_id: Business identity.

        Returns:
            Remaining errors.
        """
        errors: List[str] = []
        entities = [e for e in result.get("detected_entities") or [] if isinstance(e, dict)]
        total = len(texts)
        if strategy != SANDWICH:
            claimed = {int(p) for e in entities for p in e.get("page_indices") or [] if isinstance(p, (int, float))}
            missing = sorted(set(range(1, total + 1)) - claimed)
            if missing:
                errors.append(f"SEGMENTATION_ERROR: Missing pages {missing}. Every page must be assigned to an entity.")
        for err in validate_structure(result, texts, private_id, business_id):
            overridden = False
            for ent in entities:
                pages = ent.get("page_indices") or []
                if pages and f"Page {pages[0]}" in err:
                    overridden = self.arbiter(str(ent.get("tenant_context")), str(ent.get("direction")), int(pages[0]),
                                              texts, private_id, business_id)
                    break
            if not overridden:
                errors.append(err)
        return errors

    def arbiter(self, context: str, direction: str, page: int, texts: List[str],
                private_id: Optional[IdentityProfile], business_id: Optional[IdentityProfile]) -> bool:
        """
        Plausibility arbiter (KPaperFlux): AI second opinion on context/direction.

        Returns:
            True if the classification is plausible.
        """
        if page < 1 or page > len(texts):
            return False
        profile = business_id if context == "BUSINESS" else private_id
        info = "None"
        if profile:
            info = f"Name: {profile.name}, Company: {profile.company_name}, Keywords: {', '.join(profile.address_keywords)}"
        request = LLMRequest(prompt=prompts.PROMPT_PLAUSIBILITY_ARBITER.format(
            context=f"{context} ({direction})", profile_info=info, page_text=texts[page - 1][:2000]),
            label=f"Arbiter Check (P{page})", model=self.settings.models.primary, schema=ARBITER_SCHEMA,
            schema_name="arbiter", max_tokens=200)
        result = self.backend.generate_json(request)
        return isinstance(result, dict) and result.get("is_plausible") is True
