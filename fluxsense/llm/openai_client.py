# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/llm/openai_client.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    OpenAI-compatible chat client for llama-swap / llama.cpp:
                JSON-schema constrained output (with automatic fallback for
                servers that reject response_format), image parts as data
                URIs, thinking disabled, transport retries, logical retries
                with error feedback, tiered JSON repair and per-call stats.
------------------------------------------------------------------------------
"""

import re
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from fluxsense.config import HookSettings, LlmSettings, resolve_api_key
from fluxsense.llm import budget
from fluxsense.llm.base import AUTH_STATUS, BackendAuthError, BackendUnreachable, CallCollector, LLMRequest, auth_message
from fluxsense.llm.json_repair import parse_json_response
from fluxsense.log import get_logger
from fluxsense.models.result import CallRecord
from fluxsense.wol import WolError, lease_for, wait_for

logger = get_logger("llm.openai")

RETRYABLE_STATUS: frozenset = frozenset({408, 429, 500, 502, 503, 504})
SCHEMA_REJECT_MARKERS: Tuple[str, ...] = ("response_format", "json_schema", "grammar", "schema")
# Grammar-constrained output still allows whitespace; Qwen pretty-prints by default, which costs
# ~30 % (tables: up to 2x) more output tokens. Measured on the reference setup.
MINIFY_INSTRUCTION: str = "\nOutput minified JSON on ONE line: no newlines, no indentation, no spaces between tokens."


class BackendError(Exception):
    """Raised for non-recoverable HTTP errors."""


class ContextOverflow(BackendError):
    """Raised when the server rejects a prompt that exceeds the slot context."""


OVERFLOW_MARKERS: Tuple[str, ...] = ("exceed_context_size", "exceeds the available context", "context size", "too many tokens")
MAX_OVERFLOW_RETRIES: int = 3
REPETITION_RE = re.compile(r"(.{1,16}?)\1{24,}", re.DOTALL)
ANTI_REPEAT_SAMPLING: Dict[str, Any] = {"repeat_penalty": 1.15, "repeat_last_n": 256, "frequency_penalty": 0.4,
                                         "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 3}


def is_degenerate(content: str) -> bool:
    """True if the output contains a runaway repetition (same 1-16 chars repeated 25+ times)."""
    return REPETITION_RE.search(content) is not None


def collapse_repetition(content: str) -> str:
    """Collapses runaway repetitions to a single occurrence (last-resort repair)."""
    return REPETITION_RE.sub(lambda m: m.group(1), content)

class OpenAICompatClient(CallCollector):
    """Client for /v1/chat/completions on an OpenAI-compatible server."""

    def __init__(self, llm: LlmSettings, hooks: Optional[HookSettings] = None,
                 transport: Optional[httpx.BaseTransport] = None) -> None:
        """
        Args:
            llm: Endpoint settings.
            hooks: Optional wake hook settings.
            transport: Optional httpx transport (tests).
        """
        super().__init__()
        self.settings = llm
        self.hooks = hooks or HookSettings()
        self.fingerprints: Dict[str, str] = {}
        self._no_schema_models: Set[str] = set()
        self._ready = False
        self._reacquired = False
        self._ready_lock = threading.Lock()
        headers = {"Content-Type": "application/json"}
        key = resolve_api_key(llm)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        timeout = httpx.Timeout(llm.timeout_s, connect=llm.connect_timeout_s)
        self._http = httpx.Client(base_url=llm.base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport)

    # ------------------------------------------------------------------ setup

    def close(self) -> None:
        """Closes the HTTP connection pool."""
        self._http.close()

    def list_models(self) -> List[str]:
        """
        Returns the model ids served by the endpoint.

        Raises:
            httpx.HTTPError: On connection or HTTP errors.
        """
        resp = self._http.get("/models", timeout=httpx.Timeout(15.0, connect=self.settings.connect_timeout_s))
        if resp.status_code in AUTH_STATUS:
            raise BackendAuthError(auth_message(resp.status_code, self.settings.base_url))
        resp.raise_for_status()
        return [m.get("id", "") for m in resp.json().get("data", [])]

    def ensure_ready(self) -> bool:
        """
        Checks the endpoint once (thread-safe); if unreachable and a wake hook
        is configured, runs it and polls until the server answers.

        Returns:
            True if the endpoint is reachable.

        Raises:
            BackendUnreachable: The endpoint does not answer (after the wake hook, if any).
        """
        with self._ready_lock:
            if self._ensure_ready():
                return True
        raise BackendUnreachable(f"LLM backend unreachable at {self.settings.base_url} - is the server running?")

    def _probe(self) -> bool:
        """True if the endpoint answers /models."""
        try:
            self.list_models()
            return True
        except httpx.HTTPError as e:
            logger.debug(f"LLM endpoint not reachable ({e.__class__.__name__}).")
            return False

    def _ensure_ready(self) -> bool:
        """ensure_ready() without locking."""
        if self._ready:
            return True
        lease = lease_for(self.hooks)
        if lease is not None and lease.request_id is None:
            try:  # optional WOL-Server device request: wakes the server and keeps it awake while we work
                lease.acquire()
            except WolError as e:
                logger.warning(f"{e}; continuing with the plain health check")
        if self._probe():
            self._ready = True
            return True
        if lease is not None and lease.request_id is not None and wait_for(self._probe, self.hooks.wake_timeout_s):
            self._ready = True
            return True
        logger.info("LLM endpoint not reachable.")
        if not self.hooks.wake_command.strip():
            return False
        logger.info("Running wake hook ...")
        try:
            subprocess.run(shlex.split(self.hooks.wake_command), check=False, timeout=60, capture_output=True)
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Wake hook failed: {e}")
            return False
        deadline = time.monotonic() + self.hooks.wake_timeout_s
        while time.monotonic() < deadline:
            try:
                self.list_models()
                self._ready = True
                logger.info("LLM endpoint is up.")
                return True
            except httpx.HTTPError as e:
                logger.debug(f"Waiting for endpoint: {e.__class__.__name__}")
                time.sleep(5)
        return False

    # ---------------------------------------------------------------- payload

    def build_messages(self, request: LLMRequest) -> List[Dict[str, Any]]:
        """
        Builds the chat messages: optional static system turn, one user turn
        with the prompt text and the images (each preceded by its label), then
        optional follow-up turns.

        Args:
            request: The request.

        Returns:
            OpenAI chat messages.
        """
        minify = self.settings.minify_output and not request.verbatim
        if request.images:
            content: List[Dict[str, Any]] = [{"type": "text", "text": request.prompt}]
            for image in request.images:
                content.append({"type": "text", "text": f"[IMAGE CONTEXT: {image.label}]"})
                content.append({"type": "image_url", "image_url": {"url": image.data_uri}})
            if minify:
                content.append({"type": "text", "text": MINIFY_INSTRUCTION.strip()})
            messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": request.prompt + (MINIFY_INSTRUCTION if minify else "")}]
        if request.system:
            messages.insert(0, {"role": "system", "content": request.system})
        messages.extend(request.followups)
        return messages

    def build_payload(self, request: LLMRequest, messages: List[Dict[str, Any]], use_schema: bool) -> Dict[str, Any]:
        """
        Builds the /chat/completions JSON body.

        Args:
            request: The request.
            messages: Chat messages.
            use_schema: Whether to send response_format json_schema.

        Returns:
            The payload dict.
        """
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "seed": self.settings.seed,
            "max_tokens": request.max_tokens or self.settings.max_output_tokens,
            "stream": False,
            "cache_prompt": True,
        }
        if self.settings.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        payload.update(request.sampling)
        if request.verbatim:
            return payload
        if use_schema and request.schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "strict": True, "schema": request.schema},
            }
        elif request.schema is not None or "json" in request.prompt.lower():
            payload["response_format"] = {"type": "json_object"}
        return payload

    # ---------------------------------------------------------------- calling

    def _reacquire_once(self) -> bool:
        """The backend went away mid-run: with a WOL-Server lease, request the device again once and wait for it."""
        lease = lease_for(self.hooks)
        if lease is None or self._reacquired:
            return False
        self._reacquired = True
        try:
            lease.reacquire()
        except WolError as e:
            logger.warning(f"{e}")
            return False
        self._ready = wait_for(self._probe, self.hooks.wake_timeout_s)
        return self._ready

    def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        """POSTs with transport retries on connection errors and retryable status codes."""
        last_error: Optional[Exception] = None
        for attempt in range(self.settings.transport_retries + 1):
            try:
                resp = self._http.post("/chat/completions", json=payload)
                if resp.status_code in AUTH_STATUS:  # a missing / wrong API key never heals by retrying
                    raise BackendAuthError(auth_message(resp.status_code, self.settings.base_url))
                if resp.status_code in RETRYABLE_STATUS and attempt < self.settings.transport_retries:
                    logger.warning(f"HTTP {resp.status_code} from backend, retrying ({attempt + 1})")
                    time.sleep(2 * (attempt + 1))
                    continue
                return resp
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                # connection refused / no answer to the connect: the server is down - fail fast, no retries
                self._ready = False
                if self._reacquire_once():
                    return self._post(payload)  # once: _reacquired is now set
                raise BackendUnreachable(f"LLM backend unreachable at {self.settings.base_url}: {e.__class__.__name__}") from e
            except httpx.TransportError as e:
                last_error = e
                logger.warning(f"Transport error ({e.__class__.__name__}), retrying ({attempt + 1})")
                time.sleep(2 * (attempt + 1))
        raise BackendError(f"Backend unreachable: {last_error}")

    def _complete(self, request: LLMRequest, messages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], bool]:
        """
        Executes one completion, falling back from json_schema if rejected.

        Returns:
            Tuple (content, response json, structured flag).

        Raises:
            BackendError: On non-recoverable HTTP errors.
        """
        use_schema = (self.settings.structured_output and request.schema is not None
                      and request.model not in self._no_schema_models)
        payload = self.build_payload(request, messages, use_schema)
        resp = self._post(payload)
        if resp.status_code == 400 and "response_format" in payload:
            body = resp.text.lower()
            if any(marker in body for marker in SCHEMA_REJECT_MARKERS):
                logger.warning(f"{request.model} rejected response_format; falling back to prompt-only JSON.")
                self._no_schema_models.add(request.model)
                use_schema = False
                payload = self.build_payload(request, messages, False)
                payload.pop("response_format", None)
                resp = self._post(payload)
        if resp.status_code == 400 and any(marker in resp.text.lower() for marker in OVERFLOW_MARKERS):
            raise ContextOverflow(f"HTTP 400 context overflow: {resp.text[:300]}")
        if resp.status_code != 200:
            raise BackendError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content") or ""
        request.truncated = choice.get("finish_reason") == "length"
        if request.truncated:
            logger.warning(f"[{request.label}] output truncated at max_tokens; attempting repair.")
        if data.get("system_fingerprint"):
            self.fingerprints[request.model] = str(data["system_fingerprint"])
        return content, data, use_schema

    def generate_json(self, request: LLMRequest) -> Optional[Any]:
        """
        Runs the request with logical retries: if the answer cannot be parsed
        as JSON, the parse error is fed back to the model.

        Args:
            request: The request.

        Returns:
            Parsed JSON (dict or list), or None after all retries.
        """
        self.ensure_ready()
        fit_notes = budget.fit(request, self.settings.context_tokens)
        if fit_notes:
            logger.info(f"[{request.label}] request reduced to fit the context: {fit_notes}")
        messages = self.build_messages(request)
        record = CallRecord(label=request.label, model=request.model, images=len(request.images))
        start = time.monotonic()
        result: Optional[Any] = None
        overflow_retries = 0
        attempt = 0
        while attempt < self.settings.logical_retries + 1:
            attempt += 1
            record.attempts = attempt
            try:
                content, data, structured = self._complete(request, messages)
            except ContextOverflow as e:
                request.overflow = True
                if overflow_retries >= MAX_OVERFLOW_RETRIES:
                    record.error = str(e)[:300]
                    logger.error(f"[{request.label}] still too large after {overflow_retries} reductions")
                    break
                overflow_retries += 1
                attempt -= 1
                budget.shrink_for_retry(request)
                messages = self.build_messages(request)
                logger.warning(f"[{request.label}] context overflow, retrying with reduced payload "
                               f"({len(request.images)} images, {len(request.prompt)} chars)")
                continue
            except BackendError as e:
                record.error = str(e)[:300]
                logger.error(f"[{request.label}] {e}")
                break
            usage = data.get("usage") or {}
            record.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            record.completion_tokens += int(usage.get("completion_tokens") or 0)
            record.structured = structured
            if is_degenerate(content):
                if not record.degenerate:
                    record.degenerate = True
                    logger.warning(f"[{request.label}] runaway repetition detected, retrying with repetition penalties")
                    request.sampling = {**request.sampling, **ANTI_REPEAT_SAMPLING}
                    request.max_tokens = max(1024, int((request.max_tokens or self.settings.max_output_tokens) * 0.7))
                    attempt -= 1
                    continue
                content = collapse_repetition(content)
            parsed, error = parse_json_response(content)
            if parsed is not None:
                result = parsed
                record.error = None
                break
            record.error = error
            logger.info(f"[{request.label}] logical retry {attempt} due to: {error}")
            messages = messages + [
                {"role": "assistant", "content": content[:4000]},
                {"role": "user", "content": f"### PREVIOUS ATTEMPT FAILED WITH ERROR:\n{error}\n\nPLEASE FIX THE JSON STRUCTURE! Return ONLY valid JSON."},
            ]
        record.ok = result is not None
        record.truncated = request.truncated
        record.seconds = round(time.monotonic() - start, 3)
        self.record(record)
        logger.info(f"[{request.label}] {request.model} {record.seconds:.1f}s "
                    f"in={record.prompt_tokens} out={record.completion_tokens} ok={record.ok}")
        return result
