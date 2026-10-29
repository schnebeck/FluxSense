# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/llm/base.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Backend protocol and request type shared by all LLM backends
                (the OpenAI-compatible client and test fakes).
------------------------------------------------------------------------------
"""

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Protocol, Sequence

from fluxsense.llm.images import ImagePart
from fluxsense.models.result import CallRecord


@dataclass
class LLMRequest:
    """
    One logical JSON request.

    Attributes:
        prompt: The user prompt (first user turn).
        label: Stage label for logs and stats.
        model: Model name (llama-swap id).
        images: Images appended after the prompt text.
        schema: Optional JSON schema for constrained decoding.
        schema_name: Name of the schema (OpenAI json_schema.name).
        followups: Additional turns after the first user turn
            (e.g. assistant answer + user correction).
        max_tokens: Output limit (None = backend default).
        system: Static system prompt (rules, catalog, schema) sent first so the
            server can reuse its KV cache across documents.
    """

    prompt: str
    label: str
    model: str
    images: Sequence[ImagePart] = ()
    schema: Optional[Dict[str, Any]] = None
    schema_name: str = "response"
    followups: List[Dict[str, Any]] = field(default_factory=list)
    max_tokens: Optional[int] = None
    system: Optional[str] = None
    truncated: bool = False
    overflow: bool = False
    sampling: Dict[str, Any] = field(default_factory=dict)
    verbatim: bool = False  # send the prompt exactly as given (no minify hint, no response_format)


class BackendUnreachable(Exception):
    """The LLM endpoint refused the connection or did not answer the connect: the analysis fails fast (no retries)."""


class BackendAuthError(Exception):
    """The LLM endpoint rejected the request (HTTP 401 / 403: missing or invalid API key): the analysis fails fast."""


AUTH_STATUS = (401, 403)


def auth_message(status: int, base_url: str) -> str:
    """User-facing text of a rejected request."""
    return f"AI server rejected the request: missing or invalid API key (HTTP {status} from {base_url})"


class LLMBackend(Protocol):
    """Interface every backend implements."""

    calls: List[CallRecord]

    def generate_json(self, request: LLMRequest) -> Optional[Any]:
        """Returns the parsed JSON response, or None after all retries failed."""
        ...

    def list_models(self) -> List[str]:
        """Returns the model ids the backend serves."""
        ...


class CallCollector:
    """
    Mixin that records CallRecords globally and into an optional per-thread
    sink, so concurrent document pipelines each get their own statistics.
    """

    def __init__(self) -> None:
        """Initializes the record lists."""
        self.calls: List[CallRecord] = []
        self._collector_lock = threading.Lock()
        self._sink = threading.local()

    @contextmanager
    def collect(self, sink: List[CallRecord]) -> Iterator[List[CallRecord]]:
        """
        Collects the records of all calls made by the current thread.

        Args:
            sink: List receiving the records.

        Yields:
            The sink.
        """
        previous = getattr(self._sink, "records", None)
        self._sink.records = sink
        try:
            yield sink
        finally:
            self._sink.records = previous

    def record(self, record: CallRecord) -> None:
        """Stores a record globally and in the thread sink."""
        with self._collector_lock:
            self.calls.append(record)
        sink = getattr(self._sink, "records", None)
        if sink is not None:
            sink.append(record)
