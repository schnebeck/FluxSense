# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/api/app.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    FastAPI REST service: POST /v1/analyze (multipart PDF),
                GET /v1/health, GET /v1/models, GET /v1/catalog. OpenAPI docs
                at /docs. Uploads go to a temporary file that is deleted
                after the analysis; concurrency is bounded by the LLM slots.
------------------------------------------------------------------------------
"""

import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from fluxsense import SCHEMA_VERSION, __version__
from fluxsense.llm.base import BackendAuthError, BackendUnreachable
from fluxsense.config import Settings, load_settings
from fluxsense.extract import ENGINES
from fluxsense.llm.base import LLMBackend
from fluxsense.log import get_logger
from fluxsense.models.result import AnalysisResult
from fluxsense.pipeline import AnalyzeOptions, Analyzer
from fluxsense.prompts import PROMPT_VERSION

logger = get_logger("api")


def create_app(settings: Optional[Settings] = None, backend: Optional[LLMBackend] = None) -> FastAPI:
    """
    Creates the FastAPI application.

    Args:
        settings: Settings (default: loaded from config/env).
        backend: Optional LLM backend (tests).

    Returns:
        The app.
    """
    settings = settings or load_settings()
    analyzer = Analyzer(settings, backend=backend)
    slots = threading.BoundedSemaphore(max(1, settings.llm.max_concurrency))
    app = FastAPI(title="FluxSense", version=__version__,
                  description="Semantic PDF analysis: classification, holistic extraction, grounding and plausibility.")

    @app.get("/v1/health")
    def health() -> Dict[str, Any]:
        """Service and backend status."""
        backend_ok = True
        backend_error = None
        try:
            analyzer.backend.list_models()
        except BackendAuthError as e:
            logger.info(f"health: {e}")
            backend_ok, backend_error = False, "BACKEND_AUTH"
        except (httpx.HTTPError, OSError) as e:
            logger.info(f"health: backend unreachable: {e}")
            backend_ok, backend_error = False, "BACKEND_UNREACHABLE"
        return {"status": "ok" if backend_ok else "degraded", "version": __version__, "schema_version": SCHEMA_VERSION,
                "prompt_version": PROMPT_VERSION, "llm_backend": backend_ok, "llm_backend_error": backend_error}

    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        """Configured model roles and the models served by the backend."""
        try:
            served = analyzer.backend.list_models()
        except BackendAuthError as e:
            raise HTTPException(status_code=502, detail={"error_code": "BACKEND_AUTH", "message": str(e)}) from e
        except (httpx.HTTPError, OSError) as e:
            raise HTTPException(status_code=502, detail=f"LLM backend unreachable: {e}") from e
        return {"roles": settings.models.model_dump(), "served": served}

    @app.get("/v1/catalog")
    def catalog() -> Dict[str, Any]:
        """Document types with questions and rules."""
        return {name: {"description": analyzer.catalog.get(name).description, "bodies": analyzer.catalog.get(name).body_keys(),
                       "questions": list(analyzer.catalog.get(name).questions), "rules": list(analyzer.catalog.get(name).rules)}
                for name in analyzer.catalog.names()}

    @app.post("/v1/analyze", response_model=AnalysisResult)
    async def analyze(file: UploadFile = File(..., description="PDF file"),
                      ocr_engine: Optional[str] = Query(None, description=f"one of {ENGINES}"),
                      review_vision: Optional[bool] = Query(None), review_text: Optional[bool] = Query(None),
                      grounding: Optional[bool] = Query(None),
                      include_page_text: bool = Query(False, description="add text_layer.pages[] (page, raw text, clean, corrected, corrections)")
                      ) -> AnalysisResult:
        """Analyzes an uploaded PDF."""
        if ocr_engine is not None and ocr_engine not in ENGINES:
            raise HTTPException(status_code=422, detail=f"ocr_engine must be one of {ENGINES}")
        data = await file.read()
        if len(data) > settings.server.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="file too large")
        if not data.startswith(b"%PDF"):
            raise HTTPException(status_code=415, detail="not a PDF")
        options = AnalyzeOptions(engine=ocr_engine, review_vision=review_vision, review_text=review_text, grounding=grounding,
                                 include_page_text=include_page_text)

        def run() -> AnalysisResult:
            """Blocking analysis in a worker thread."""
            with slots, tempfile.TemporaryDirectory(prefix="fluxsense-api-") as tmp:
                path = Path(tmp) / (Path(file.filename or "upload.pdf").name or "upload.pdf")
                path.write_bytes(data)
                result = analyzer.analyze(path, options)
                result.source.path = file.filename or "upload.pdf"
                return result

        try:
            result = await run_in_threadpool(run)
        except BackendUnreachable as e:
            raise HTTPException(status_code=503, detail={"error_code": "BACKEND_UNREACHABLE", "message": str(e)}) from e
        except BackendAuthError as e:
            raise HTTPException(status_code=502, detail={"error_code": "BACKEND_AUTH", "message": str(e)}) from e
        # BACKEND_AUTH is 502 (Bad Gateway): the upstream AI server rejected FluxSense's key; a 401 would blame the caller
        status = {"BACKEND_UNREACHABLE": 503, "BACKEND_AUTH": 502, "BACKEND_TIMEOUT": 504,
                  "INPUT_INVALID": 422}.get(result.error_code or "")
        if status and result.status == "ERROR":
            raise HTTPException(status_code=status, detail={"error_code": result.error_code,
                                                            "message": "; ".join(result.errors)[:500]})
        return result

    return app
