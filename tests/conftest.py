# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/conftest.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Shared fixtures: isolated settings (no config file, no key
                command, no network), a scriptable fake LLM backend and the
                synthetic sample PDFs.
------------------------------------------------------------------------------
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

from fluxsense.config import Settings
from fluxsense.llm.base import CallCollector, LLMRequest
from fluxsense.models.result import CallRecord
from fluxsense.samples import make_all

Handler = Callable[[LLMRequest], Optional[Any]]


class FakeBackend(CallCollector):
    """LLM backend returning scripted answers chosen by the request label."""

    def __init__(self, handler: Handler) -> None:
        """
        Args:
            handler: Function mapping a request to a JSON answer (or None).
        """
        super().__init__()
        self.handler = handler
        self.requests: List[LLMRequest] = []

    def generate_json(self, request: LLMRequest) -> Optional[Any]:
        """Answers via the handler and records stats."""
        self.requests.append(request)
        start = time.monotonic()
        answer = self.handler(request)
        self.record(CallRecord(label=request.label, model=request.model, ok=answer is not None, images=len(request.images),
                               seconds=round(time.monotonic() - start, 4), prompt_tokens=len(request.prompt) // 3,
                               completion_tokens=10, structured=request.schema is not None))
        return answer

    def list_models(self) -> List[str]:
        """Model ids."""
        return ["fake"]

    def labels(self) -> List[str]:
        """Labels of all requests in order."""
        return [r.label for r in self.requests]


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    """Settings without config file, key command or network engines."""
    monkeypatch.setenv("FLUXSENSE_CONFIG", str(tmp_path / "none.toml"))
    s = Settings()
    s.llm.api_key_command = ""
    s.extract.engine = "native"
    s.extract.scan_fallback = "none"
    s.llm.logical_retries = 0
    return s


@pytest.fixture(scope="session")
def samples(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Synthetic sample PDFs (generated once per session)."""
    return make_all(tmp_path_factory.mktemp("samples"))
