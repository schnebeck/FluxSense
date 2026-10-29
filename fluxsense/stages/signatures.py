# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/signatures.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Per-page signatures for stack segmentation: one cheap VLM call
                on page thumbnails (batched) records letterhead/sender, date,
                'page x of y', salutation, closing/signature, form header and
                blank pages. Document starts are then decided
                deterministically and enforced on the classification.
------------------------------------------------------------------------------
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fluxsense.config import Settings
from fluxsense.extract.base import TextLayer
from fluxsense.grounding.normalize import normalize_text
from fluxsense.layout.items import FOOTER, HEADER, PAGE_NUMBER
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.llm.images import page_images
from fluxsense.log import get_logger

logger = get_logger("stages.signatures")

BATCH_PAGES: int = 16
_PAGE_OF_RE = re.compile(r"(\d+)\s*(?:/|von|of|v\.)\s*(\d+)", re.IGNORECASE)
_PAGE_ONLY_RE = re.compile(r"(?:seite|page|blatt)\s*(\d+)", re.IGNORECASE)
LAYOUT_ONLY_FIELDS = frozenset({"header", "layout_page", "toc", "mentions"})  # not shown to the classifier
_BARE_PAGE_RE = re.compile(r"(?:-\s*)?(?:seite|page|s\.)?\s*(\d{1,3})(?:\s*-)?", re.IGNORECASE)
_TOC_TITLE_RE = re.compile(r"(?m)^\W*(inhaltsverzeichnis|inhalt|contents|table of contents)\W*$", re.IGNORECASE)
_TOC_LINE_RE = re.compile(r"(?m)^.{3,80}?(?:\.{3,}|\s{3,}|\t)\s*\d{1,3}\s*$")
_LAYOUT_PAGE_RE = re.compile(r"(?:seite|page|blatt)\s*\d+\s*(?:/|von|of)\s*\d+|^\s*\d+\s*/\s*\d+\s*$", re.IGNORECASE)

PROMPT_SIGNATURES = """
You see thumbnails of pages {first}-{last} of a scanned file that may contain SEVERAL documents (a stack).
For EACH page report what is visible, as printed. Do not guess.
- blank: the page is empty or nearly empty (scanner back side)
- letterhead: a sender letterhead/logo block is printed at the top
- sender: the sender/organisation name of the letterhead (short), else null
- date: the letter/document date printed near the top (YYYY-MM-DD), else null
- page_number: page numbering as printed (e.g. "1/3", "Seite 2 von 4"), else null
- salutation: the page starts a letter body with a salutation ("Sehr geehrte...", "Dear ...")
- closing: the page contains a closing / signature ("Mit freundlichen Grüßen", signature)
- form: the page is (the first page of) a form with a form title/header
- title: the document title printed on the page (e.g. "Rechnung", "Kontoauszug", "Antrag"), else null
- addressee: the page carries a recipient address block (address window)
- boilerplate: the page is general information / terms / AGB / Hinweise / Datenschutz / "Wichtige Informationen"
  without its own date, addressee or salutation

Short text start of each page (auxiliary):
{texts}

Return ONLY JSON: {{"pages": [{{"page": 1, "blank": false, "letterhead": true, "sender": "...", "date": null, "page_number": null,
"salutation": true, "closing": false, "form": false, "title": null, "addressee": true, "boilerplate": false}}]}}
"""

SIGNATURE_SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["pages"],
    "properties": {"pages": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["page", "blank", "letterhead", "salutation", "closing", "form"],
        "properties": {
            "page": {"type": "integer"}, "blank": {"type": "boolean"}, "letterhead": {"type": "boolean"},
            "sender": {"type": ["string", "null"], "maxLength": 40}, "date": {"type": ["string", "null"], "maxLength": 12},
            "page_number": {"type": ["string", "null"], "maxLength": 20}, "salutation": {"type": "boolean"},
            "closing": {"type": "boolean"}, "form": {"type": "boolean"}, "title": {"type": ["string", "null"], "maxLength": 40},
            "addressee": {"type": "boolean"}, "boilerplate": {"type": "boolean"}}}}},
}


@dataclass
class PageSignature:
    """Visual signature of one page."""

    page: int
    blank: bool = False
    letterhead: bool = False
    sender: Optional[str] = None
    date: Optional[str] = None
    page_number: Optional[str] = None
    salutation: bool = False
    closing: bool = False
    form: bool = False
    title: Optional[str] = None
    addressee: bool = False
    boilerplate: bool = False
    header: Optional[str] = None  # running page header from the layout (not asked from the model)
    layout_page: Optional[int] = None  # bare printed page number of a layout page-number item ('3')
    toc: bool = False  # the page is a table of contents
    mentions: Tuple[str, ...] = ()  # running document references (footer/header codes) printed on this page

    @property
    def page_index(self) -> Optional[int]:
        """The 'x' of 'page x of y' (None if not printed)."""
        if not self.page_number:
            return None
        match = _PAGE_OF_RE.search(self.page_number) or _PAGE_ONLY_RE.search(self.page_number)
        return int(match.group(1)) if match else None

    @property
    def page_total(self) -> Optional[int]:
        """The 'y' of 'page x of y' (None if not printed)."""
        match = _PAGE_OF_RE.search(self.page_number or "")
        return int(match.group(2)) if match else None


def similar_titles(a: Optional[str], b: Optional[str]) -> bool:
    """Titles of the same document (shared words >= 50 % of the shorter title, or prefix)."""
    na, nb = normalize_text(a or ""), normalize_text(b or "")
    if not na or not nb:
        return False
    ta, tb = set(na.split()), set(nb.split())
    return na.startswith(nb[:10]) or nb.startswith(na[:10]) or len(ta & tb) / max(1, min(len(ta), len(tb))) >= 0.5


def _same(a: Optional[str], b: Optional[str]) -> bool:
    """Fuzzy equality of short names (normalized, prefix tolerant)."""
    na, nb = normalize_text(a or ""), normalize_text(b or "")
    if not na or not nb:
        return True
    return na == nb or na.startswith(nb[:8]) or nb.startswith(na[:8])


def apply_layout(signatures: List[PageSignature], layer: TextLayer) -> None:
    """
    Adds layout evidence to the signatures (in place): the page header text
    (layout items of category 'header', e.g. OvisOCR3 / Docling page
    headers) and the page numbering printed in page-number / footer / header
    items when the model reported none.
    """
    for sig in signatures:
        page = layer.page(sig.page)
        items = getattr(page, "items", None) or []
        headers = [i.content for i in items if getattr(i, "category", "") == HEADER and i.content.strip()
                   and not i.content.lstrip().startswith("<img")]
        if headers:
            sig.header = " ".join(normalize_text(" ".join(headers)).split())[:80] or None
        if not sig.page_number:
            for item in items:
                if getattr(item, "category", "") in (PAGE_NUMBER, FOOTER, HEADER):
                    match = _LAYOUT_PAGE_RE.search(item.content)
                    if match:
                        sig.page_number = match.group(0)[:20]
                        break
        for item in items:
            bare = _BARE_PAGE_RE.fullmatch(item.content.strip()) if getattr(item, "category", "") == PAGE_NUMBER else None
            if bare:
                sig.layout_page = int(bare.group(1))
                break
        text = page.best_text() if page else ""
        sig.toc = bool(_TOC_TITLE_RE.search(text[:400])) or len(_TOC_LINE_RE.findall(text)) >= 3
    _apply_references(signatures, layer)


def _apply_references(signatures: List[PageSignature], layer: TextLayer) -> None:
    """
    Running document references: codes with digits printed in the header or
    footer of at least two pages (e.g. an offer number in every footer). Each
    page records which of them it prints anywhere (also in its body text).
    """
    found: Dict[str, int] = {}
    for sig in signatures:
        page = layer.page(sig.page)
        for item in getattr(page, "items", None) or []:
            if getattr(item, "category", "") in (HEADER, FOOTER):
                for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9./-]{4,}", item.content):
                    if sum(ch.isdigit() for ch in token) >= 3 and not _LAYOUT_PAGE_RE.search(token):
                        found[token] = found.get(token, 0) + 1
    references = [ref for ref, count in found.items() if count >= 2]
    if not references:
        return
    for sig in signatures:
        page = layer.page(sig.page)
        furniture = " ".join(i.content for i in getattr(page, "items", None) or [])
        flat = re.sub(r"\s+", "", (page.best_text() if page else "") + " " + furniture)
        sig.mentions = tuple(ref for ref in references if ref in flat)


def document_starts(signatures: List[PageSignature]) -> List[int]:
    """
    Decides deterministic document starts from page signatures. A page only
    starts a new document on POSITIVE evidence: page numbering restarting at 1,
    a salutation, a recipient address block with letterhead or date, a
    letterhead with a different date or sender, a different form with its
    own letterhead/date/addressee, or the first page after a letter's closing
    (an enclosure is a separate item, field contract §1). Blank pages,
    boilerplate pages (terms, information) and pages 'x > 1 of y' never start
    a document. Continuity signals - the same (or truncated) running layout
    header, a printed page-number run reaching back over the boundary, a
    table of contents earlier in the document - leave only the positive
    evidence (numbering restart, salutation, new address block, new
    letterhead date or sender) to split; the weaker cues (title, form,
    closing) do not.

    Args:
        signatures: Signatures in page order.

    Returns:
        1-based start pages.
    """
    starts: List[int] = []
    previous: Optional[PageSignature] = None
    ordered = sorted(signatures, key=lambda s: s.page)
    numbered = numbered_continuation(ordered)
    toc_seen = False
    doc_date: Optional[str] = None
    doc_sender: Optional[str] = None
    doc_mentions: set = set()
    for sig in ordered:
        ahead = [n for n in ordered if sig.page <= n.page <= sig.page + 2]
        shared = bool(doc_mentions) and any(set(n.mentions) & doc_mentions for n in ahead)
        continuity = (previous is not None and same_running_header(sig.header, previous.header)) or sig.page in numbered \
            or toc_seen or shared
        if previous is None:
            start = True
        elif sig.blank or sig.boilerplate or (sig.page_index is not None and sig.page_index > 1):
            start = False
        elif sig.page_index == 1 or sig.salutation:
            start = True
        elif sig.addressee and not previous.addressee and (sig.letterhead or sig.date):
            start = True  # a new address block appears
        elif sig.letterhead and sig.date and (
                (doc_date and sig.date != doc_date) or (sig.sender and doc_sender and not _same(sig.sender, doc_sender))):
            start = True  # a new letterhead with a new date or a new sender (compared with the current document)
        elif continuity:
            start = False  # running header / page numbering / table of contents continue: only positive evidence splits
        elif sig.letterhead and sig.title and previous.title and not similar_titles(sig.title, previous.title):
            start = True  # a new letterhead page with a different document title
        elif sig.form and sig.title and (sig.letterhead or sig.addressee or sig.date) and \
                (not previous.form or (previous.title and not similar_titles(sig.title, previous.title))):
            start = True  # letter -> form transition, or a different form with its own header/date/addressee
        elif previous.closing:
            start = True  # a letter ends with its closing: the next item is a separate document (field contract §1)
        else:
            start = False
        if start:
            starts.append(sig.page)
            toc_seen, doc_date, doc_sender, doc_mentions = False, None, None, set()
        toc_seen = toc_seen or sig.toc
        doc_date = doc_date or sig.date
        doc_sender = doc_sender or sig.sender
        doc_mentions |= set(sig.mentions)
        if not sig.blank:
            previous = sig
    return starts


def same_running_header(a: Optional[str], b: Optional[str]) -> bool:
    """The same running header on two pages (equal, or one a truncated form of the other: all its words in the other)."""
    if not a or not b:
        return False
    ta, tb = set(a.split()), set(b.split())
    return a == b or (min(len(ta), len(tb)) >= 2 and (ta <= tb or tb <= ta))


def numbered_continuation(signatures: List[PageSignature]) -> set:
    """
    Pages inside a printed page-number run: a page printed as number n (>= 2)
    implies that its document began n - 1 pages earlier, so the pages after
    that implied first page up to it continue the same document.
    """
    pages = {s.page for s in signatures}
    inside: set = set()
    for sig in signatures:
        number = sig.page_index if sig.page_index is not None else sig.layout_page
        if number is None or number < 2:
            continue
        first = sig.page - number + 1
        if first >= min(pages, default=1):
            inside |= {p for p in range(first + 1, sig.page + 1)}
    return inside


def has_start_evidence(sig: PageSignature) -> bool:
    """True if a page shows any sign of being the first page of a document."""
    return (not sig.blank and not sig.boilerplate) and bool(
        sig.page_index == 1 or sig.salutation or sig.addressee or (sig.letterhead and sig.date) or sig.form)


def is_attachable(sig: PageSignature) -> bool:
    """True for pages that belong to the preceding document by themselves: blank or boilerplate (terms, information)."""
    return sig.blank or (sig.boilerplate and not sig.salutation and not sig.addressee and sig.page_index in (None, 2, 3, 4, 5, 6))


def collect_signatures(backend: LLMBackend, settings: Settings, pdf_path: Path, layer: TextLayer) -> List[PageSignature]:
    """
    Runs the signature pass (one call per 16 pages, thumbnails + text starts).

    Args:
        backend: LLM backend.
        settings: Settings.
        pdf_path: PDF file.
        layer: Text layer.

    Returns:
        Signatures for the pages the model answered.
    """
    result: List[PageSignature] = []
    pages = [p.page for p in layer.pages]
    for start in range(0, len(pages), BATCH_PAGES):
        batch = pages[start:start + BATCH_PAGES]
        images = page_images(pdf_path, batch, [f"PAGE {p} (thumbnail)" for p in batch], settings.llm.classify_image_px,
                             settings.llm.image_format, dpi=100)
        texts = "\n".join(f"--- PAGE {p} ---\n{(layer.page(p).best_text() if layer.page(p) else '')[:300]}" for p in batch)
        answer = backend.generate_json(LLMRequest(
            prompt=PROMPT_SIGNATURES.format(first=batch[0], last=batch[-1], texts=texts), label=f"Stage 1.0 signatures p{batch[0]}-{batch[-1]}",
            model=settings.models.primary, images=images, schema=SIGNATURE_SCHEMA, schema_name="page_signatures",
            max_tokens=80 * len(batch) + 100))
        got = _parse_signatures(answer, batch)
        missing = [p for p in batch if p not in {s.page for s in got}]
        if missing:
            logger.info(f"signature pass described {len(got)}/{len(batch)} pages; asking again for {missing}")
            images = page_images(pdf_path, missing, [f"PAGE {p} (thumbnail)" for p in missing], settings.llm.classify_image_px,
                                 settings.llm.image_format, dpi=100)
            texts = "\n".join(f"--- PAGE {p} ---\n{(layer.page(p).best_text() if layer.page(p) else '')[:300]}" for p in missing)
            retry = backend.generate_json(LLMRequest(
                prompt=PROMPT_SIGNATURES.format(first=missing[0], last=missing[-1], texts=texts) +
                f"\nDescribe EXACTLY these pages, one entry each: {missing}",
                label=f"Stage 1.0 signatures retry p{missing[0]}-{missing[-1]}", model=settings.models.primary, images=images,
                schema=SIGNATURE_SCHEMA, schema_name="page_signatures", max_tokens=80 * len(missing) + 100))
            got += _parse_signatures(retry, missing)
        result += got
    return sorted(result, key=lambda s: s.page)


def _parse_signatures(answer: Any, batch: List[int]) -> List[PageSignature]:
    """Signature entries for the requested pages (one per page)."""
    out: Dict[int, PageSignature] = {}
    if isinstance(answer, dict):
        for entry in answer.get("pages") or []:
            if isinstance(entry, dict) and entry.get("page") in batch and entry["page"] not in out:
                out[entry["page"]] = PageSignature(**{k: entry.get(k) for k in PageSignature.__dataclass_fields__ if k in entry})
    return list(out.values())


def signatures_prompt(signatures: List[PageSignature], starts: List[int]) -> str:
    """Compact text block for the classification prompt."""
    lines = [json.dumps({k: v for k, v in s.__dict__.items() if v not in (None, False, ()) and k not in LAYOUT_ONLY_FIELDS},
                        ensure_ascii=False) for s in signatures]
    return ("### PAGE SIGNATURES (from the page images)\n" + "\n".join(lines)
            + f"\n### DOCUMENT STARTS (decided from the signatures, each MUST start a new entity; other pages MAY also "
              f"start one if their content shows a new document): {starts}\n")
