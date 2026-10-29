# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/docling_engine.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    IBM Docling as auxiliary layout/table/OCR source, three modes:
                'docling' (local standard pipeline: layout + TableFormer +
                tesseract OCR, CPU), 'docling-vlm' (local VLM pipeline using
                granite-docling through the OpenAI-compatible llama-swap API)
                and 'docling-serve' (remote REST, /v1/convert/file).
------------------------------------------------------------------------------
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from fluxsense.config import ExtractSettings, LlmSettings, ModelSettings, resolve_api_key
from fluxsense.extract.base import SOURCE_DOCLING, PageTable, Word
from fluxsense.llm.budget import compact_markdown
from fluxsense.log import get_logger

logger = get_logger("extract.docling")

MODE_LOCAL: str = "docling"
MODE_VLM: str = "docling-vlm"
MODE_SERVE: str = "docling-serve"
SERVE_ENDPOINT: str = "/v1/convert/file"
GRANITE_PROMPT: str = "Convert this page to docling."

_TESSERACT_LANG_MAP: Dict[str, str] = {"deu": "deu", "eng": "eng", "de": "deu", "en": "eng"}
_CONVERTERS: Dict[str, Any] = {}
_LOCK = threading.Lock()


class DoclingUnavailable(Exception):
    """Raised when Docling (or docling-serve) cannot be used."""


class DoclingPages:
    """Per-page Docling output."""

    def __init__(self, markdown: Dict[int, str], tables: List[PageTable], version: str,
                 items: Optional[Dict[int, List[Word]]] = None, layout: Optional[Dict[int, List[Any]]] = None) -> None:
        """
        Args:
            markdown: 1-based page -> markdown.
            tables: Table grids with page numbers.
            version: Docling version string.
            items: 1-based page -> text items with boxes (top-left origin, points).
        """
        self.markdown = markdown
        self.tables = tables
        self.version = version
        self.items = items or {}
        self.layout = layout or {}


def docling_available() -> bool:
    """Returns True if the docling package is importable."""
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def _docling_version() -> str:
    """Returns the installed docling version."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("docling")
    except PackageNotFoundError:
        return "unknown"


def _local_converter(settings: ExtractSettings) -> Any:
    """Builds (and caches) the local standard-pipeline converter."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (EasyOcrOptions, PdfPipelineOptions, RapidOcrOptions,
                                                    TesseractCliOcrOptions)
    from docling.document_converter import DocumentConverter, PdfFormatOption

    key = f"local:{settings.docling_ocr_engine}:{','.join(settings.ocr_languages)}"
    with _LOCK:
        if key in _CONVERTERS:
            return _CONVERTERS[key]
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        if settings.docling_ocr_engine == "easyocr":
            opts.ocr_options = EasyOcrOptions(lang=[lang[:2] for lang in settings.ocr_languages])
        elif settings.docling_ocr_engine == "rapidocr":
            opts.ocr_options = RapidOcrOptions()
        else:
            opts.ocr_options = TesseractCliOcrOptions(lang=[_TESSERACT_LANG_MAP.get(lang, lang) for lang in settings.ocr_languages])
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        _CONVERTERS[key] = converter
        return converter


def _vlm_converter(llm: LlmSettings, models: ModelSettings) -> Any:
    """Builds the VLM-pipeline converter that calls granite-docling via the OpenAI API."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    key = f"vlm:{llm.base_url}:{models.docling_vlm}"
    with _LOCK:
        if key in _CONVERTERS:
            return _CONVERTERS[key]
        headers: Dict[str, str] = {}
        api_key = resolve_api_key(llm)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        vlm_options = ApiVlmOptions(
            url=f"{llm.base_url.rstrip('/')}/chat/completions",
            params={"model": models.docling_vlm, "max_tokens": 8192},
            headers=headers,
            prompt=GRANITE_PROMPT,
            timeout=llm.timeout_s,
            scale=2.0,
            temperature=0.0,
            response_format=ResponseFormat.DOCTAGS,
        )
        opts = VlmPipelineOptions(enable_remote_services=True, vlm_options=vlm_options)
        converter = DocumentConverter(format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=opts)})
        _CONVERTERS[key] = converter
        return converter


def pages_from_document(doc: Any, version: str) -> DoclingPages:
    """
    Converts a DoclingDocument into per-page markdown and table grids.

    Args:
        doc: docling_core DoclingDocument.
        version: Docling version label.

    Returns:
        DoclingPages.
    """
    markdown: Dict[int, str] = {}
    for page_no in sorted(doc.pages.keys()):
        markdown[int(page_no)] = compact_markdown(doc.export_to_markdown(page_no=int(page_no), image_placeholder=""))
    tables: List[PageTable] = []
    for table in doc.tables:
        if not table.prov:
            continue
        page_no = int(table.prov[0].page_no)
        grid = [[" ".join((cell.text or "").split()) for cell in row] for row in table.data.grid]
        grid = [r for r in grid if any(c for c in r)]
        if len(grid) >= 2:
            tables.append(PageTable(page=page_no, rows=grid, source=SOURCE_DOCLING))
    items: Dict[int, List[Word]] = {}
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        for prov in getattr(item, "prov", None) or []:
            if not text:
                continue
            page = doc.pages.get(prov.page_no)
            height = page.size.height if page is not None else 842.0
            box = prov.bbox.to_top_left_origin(page_height=height)
            items.setdefault(int(prov.page_no), []).append(Word(box.l, box.t, box.r, box.b, text))
    for table in doc.tables:
        if not table.prov:
            continue
        page_no = int(table.prov[0].page_no)
        page = doc.pages.get(page_no)
        height = page.size.height if page is not None else 842.0
        for cell in table.data.table_cells:
            if cell.bbox is None or not (cell.text or "").strip():
                continue
            box = cell.bbox.to_top_left_origin(page_height=height)
            items.setdefault(page_no, []).append(Word(box.l, box.t, box.r, box.b, " ".join(cell.text.split()), cell=True))
    for words in items.values():
        words.sort(key=lambda w: (round(w.y0 / 4), w.x0))
    return DoclingPages(markdown, tables, version, items, docling_layout(doc))


def docling_layout(doc: Any) -> Dict[int, List[Any]]:
    """Layout items (labels mapped to engine-agnostic categories, bbox 0-1000) in reading order."""
    from fluxsense.layout.items import DOCLING_CATEGORIES, TEXT, LayoutItem
    layout: Dict[int, List[Any]] = {}
    for item, _level in doc.iterate_items(included_content_layers=None) if _accepts_layers(doc) else doc.iterate_items():
        label = str(getattr(getattr(item, "label", None), "value", getattr(item, "label", "")) or "")
        category = DOCLING_CATEGORIES.get(label, TEXT)
        if label == "table":
            try:
                content = item.export_to_markdown(doc=doc)
            except (TypeError, ValueError, AttributeError) as e:
                logger.debug(f"table export failed: {e}")
                content = ""
        else:
            content = getattr(item, "text", "") or ""
        for prov in (getattr(item, "prov", None) or [])[:1]:
            page = doc.pages.get(prov.page_no)
            width = page.size.width if page is not None else 595.0
            height = page.size.height if page is not None else 842.0
            box = prov.bbox.to_top_left_origin(page_height=height)
            rel = (box.l / width * 1000, box.t / height * 1000, box.r / width * 1000, box.b / height * 1000)
            layout.setdefault(int(prov.page_no), []).append(LayoutItem(int(prov.page_no), rel, category, content, "docling"))
    return layout


def _accepts_layers(doc: Any) -> bool:
    """True if iterate_items supports content layers (furniture included)."""
    import inspect
    try:
        return "included_content_layers" in inspect.signature(doc.iterate_items).parameters
    except (TypeError, ValueError):
        return False


def convert_local(pdf_path: Path, settings: ExtractSettings) -> DoclingPages:
    """
    Runs the local Docling standard pipeline.

    Raises:
        DoclingUnavailable: If docling is not installed or conversion fails.
    """
    if not docling_available():
        raise DoclingUnavailable("docling is not installed (pip install -e .[docling])")
    result = _local_converter(settings).convert(str(pdf_path))
    if result.document is None:
        raise DoclingUnavailable(f"Docling conversion failed: {result.status}")
    return pages_from_document(result.document, _docling_version())


def convert_vlm(pdf_path: Path, llm: LlmSettings, models: ModelSettings) -> DoclingPages:
    """
    Runs the local Docling VLM pipeline (granite-docling via llama-swap).

    Raises:
        DoclingUnavailable: If docling is not installed or conversion fails.
    """
    if not docling_available():
        raise DoclingUnavailable("docling is not installed (pip install -e .[docling])")
    result = _vlm_converter(llm, models).convert(str(pdf_path))
    if result.document is None:
        raise DoclingUnavailable(f"Docling VLM conversion failed: {result.status}")
    return pages_from_document(result.document, f"{_docling_version()}+{models.docling_vlm}")


def serve_form_data(settings: ExtractSettings) -> List[tuple]:
    """Builds the multipart form fields for docling-serve (v1 API)."""
    fields: List[tuple] = [("to_formats", "json"), ("to_formats", "md"), ("do_ocr", "true"),
                           ("image_export_mode", "placeholder"), ("pipeline", settings.docling_serve_pipeline),
                           ("do_table_structure", "true"), ("abort_on_error", "false")]
    if settings.docling_serve_pipeline == "vlm":
        fields.append(("vlm_pipeline_preset", settings.docling_serve_vlm_preset))
    for lang in settings.docling_serve_ocr_lang:
        fields.append(("ocr_lang", lang))
    return fields


def _form_dict(fields: List[tuple]) -> Dict[str, Any]:
    """Converts repeated form fields into the dict-of-lists form httpx expects."""
    data: Dict[str, Any] = {}
    for key, value in fields:
        if key in data:
            data[key] = data[key] if isinstance(data[key], list) else [data[key]]
            data[key].append(value)
        else:
            data[key] = value
    return data


_SERVE_HEALTH: Dict[str, bool] = {}


def serve_available(settings: ExtractSettings) -> bool:
    """Checks (and caches) whether docling-serve answers /health."""
    url = settings.docling_serve_url.rstrip("/")
    if not url:
        return False  # not configured
    if url not in _SERVE_HEALTH:
        try:
            _SERVE_HEALTH[url] = httpx.get(f"{url}/health", timeout=5.0).status_code == 200
        except httpx.HTTPError as e:
            logger.info(f"docling-serve not reachable at {url}: {e.__class__.__name__}")
            _SERVE_HEALTH[url] = False
    return _SERVE_HEALTH[url]


def convert_serve(pdf_path: Path, settings: ExtractSettings) -> DoclingPages:
    """
    Converts via a remote docling-serve instance.

    Raises:
        DoclingUnavailable: On connection errors or failed conversions.
    """
    url = f"{settings.docling_serve_url.rstrip('/')}{SERVE_ENDPOINT}"
    try:
        with pdf_path.open("rb") as handle:
            resp = httpx.post(url, files={"files": (pdf_path.name, handle, "application/pdf")},
                              data=_form_dict(serve_form_data(settings)),
                              timeout=httpx.Timeout(settings.docling_timeout_s, connect=10.0))
    except httpx.HTTPError as e:
        raise DoclingUnavailable(f"docling-serve unreachable: {e}") from e
    if resp.status_code != 200:
        raise DoclingUnavailable(f"docling-serve HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    document: Optional[Dict[str, Any]] = payload.get("document")
    if not document or payload.get("status") not in ("success", "partial_success"):
        raise DoclingUnavailable(f"docling-serve conversion failed: {payload.get('status')} {payload.get('errors')}")
    json_content = document.get("json_content")
    if not json_content:
        raise DoclingUnavailable("docling-serve returned no json_content")
    from docling_core.types.doc import DoclingDocument
    return pages_from_document(DoclingDocument.model_validate(json_content), "docling-serve")
