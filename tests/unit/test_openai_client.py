# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_openai_client.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for the OpenAI-compatible client: payloads (json_schema, images, thinking off, cache_prompt), schema fallback, logical retries, stats.
------------------------------------------------------------------------------
"""

import json
from typing import Any, Dict, List

import httpx
import pytest

from fluxsense.config import LlmSettings
from fluxsense.llm.base import LLMRequest
from fluxsense.llm.images import ImagePart
from fluxsense.llm.openai_client import OpenAICompatClient

SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def make_client(responses: List[httpx.Response], seen: List[Dict[str, Any]]) -> OpenAICompatClient:
    """Client with a mock transport answering /models and /chat/completions."""
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        seen.append(json.loads(request.content))
        return queue.pop(0)

    llm = LlmSettings(base_url="http://test/v1", api_key_command="", transport_retries=0, logical_retries=1)
    return OpenAICompatClient(llm, transport=httpx.MockTransport(handler))


def answer(content: str, finish: str = "stop") -> httpx.Response:
    """A chat completion response."""
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                     "usage": {"prompt_tokens": 11, "completion_tokens": 3}, "system_fingerprint": "b1"})


def test_payload_with_schema_images_and_system() -> None:
    """json_schema response_format, image parts, system turn, thinking off, cache_prompt, seed."""
    seen: List[Dict[str, Any]] = []
    client = make_client([answer('{"a": 1}')], seen)
    image = ImagePart(label="PAGE 1", data_uri="data:image/png;base64,AAAA", width=10, height=10)
    result = client.generate_json(LLMRequest(prompt="extract", label="t", model="m", images=[image], schema=SCHEMA,
                                             schema_name="x", system="static rules"))
    assert result == {"a": 1}
    payload = seen[0]
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"] == SCHEMA
    assert payload["response_format"]["json_schema"]["name"] == "x"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["cache_prompt"] is True and payload["temperature"] == 0.0 and payload["seed"] == 42
    assert payload["messages"][0] == {"role": "system", "content": "static rules"}
    parts = payload["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "extract"}
    assert {"type": "text", "text": "[IMAGE CONTEXT: PAGE 1]"} in parts
    assert any(p.get("type") == "image_url" and p["image_url"]["url"].startswith("data:image/png") for p in parts)
    assert "minified" in parts[-1]["text"]
    record = client.calls[0]
    assert record.ok and record.prompt_tokens == 11 and record.completion_tokens == 3 and record.structured
    assert client.fingerprints["m"] == "b1"


def test_schema_rejection_falls_back_to_prompt_only() -> None:
    """A 400 mentioning response_format disables json_schema for that model."""
    seen: List[Dict[str, Any]] = []
    client = make_client([httpx.Response(400, text="response_format not supported"), answer('{"a": 2}')], seen)
    assert client.generate_json(LLMRequest(prompt="x", label="t", model="m", schema=SCHEMA)) == {"a": 2}
    assert "response_format" in seen[0] and "response_format" not in seen[1]
    assert "m" in client._no_schema_models


def test_logical_retry_feeds_error_back() -> None:
    """Unparseable output triggers one retry with the parse error in a follow-up turn."""
    seen: List[Dict[str, Any]] = []
    client = make_client([answer("I cannot comply"), answer('{"a": 3}')], seen)
    assert client.generate_json(LLMRequest(prompt="x", label="t", model="m", schema=SCHEMA)) == {"a": 3}
    assert seen[1]["messages"][-1]["role"] == "user"
    assert "PREVIOUS ATTEMPT FAILED" in seen[1]["messages"][-1]["content"]
    assert client.calls[0].attempts == 2


def test_truncation_flag_and_repair() -> None:
    """finish_reason=length marks the request and record as truncated; JSON is repaired."""
    seen: List[Dict[str, Any]] = []
    client = make_client([answer('{"rows": [[1], [2', finish="length")], seen)
    request = LLMRequest(prompt="x", label="t", model="m", schema=SCHEMA)
    assert client.generate_json(request) == {"rows": [[1], [2]]}
    assert request.truncated and client.calls[0].truncated


def test_backend_error_returns_none() -> None:
    """Non-retryable HTTP errors yield None and an error record."""
    seen: List[Dict[str, Any]] = []
    client = make_client([httpx.Response(500, text="boom")], seen)
    assert client.generate_json(LLMRequest(prompt="x", label="t", model="m")) is None
    assert client.calls[0].ok is False and "500" in client.calls[0].error


def test_collect_scopes_records_per_thread() -> None:
    """collect() receives only the calls made inside the context."""
    seen: List[Dict[str, Any]] = []
    client = make_client([answer('{"a": 1}'), answer('{"a": 2}')], seen)
    client.generate_json(LLMRequest(prompt="x", label="outside", model="m"))
    sink: list = []
    with client.collect(sink):
        client.generate_json(LLMRequest(prompt="x", label="inside", model="m"))
    assert [r.label for r in sink] == ["inside"] and len(client.calls) == 2


def test_degenerate_repetition_is_retried_with_penalties() -> None:
    """Runaway repetition (e.g. hundreds of zeros) -> one retry with repeat/DRY penalties and fewer tokens."""
    seen: List[Dict[str, Any]] = []
    runaway = '{"iban": "1234500000' + "0" * 400 + '"}'
    client = make_client([answer(runaway), answer('{"iban": "DE44990123450532013000"}')], seen)
    request = LLMRequest(prompt="x", label="t", model="m", schema=SCHEMA, max_tokens=4000)
    assert client.generate_json(request) == {"iban": "DE44990123450532013000"}
    assert "repeat_penalty" not in seen[0] and seen[1]["repeat_penalty"] > 1.0 and seen[1]["dry_multiplier"] > 0
    assert seen[1]["max_tokens"] < 4000 and client.calls[0].degenerate


def test_repetition_collapsed_if_retry_also_degenerates() -> None:
    """If the retry repeats too, the run is collapsed before parsing."""
    seen: List[Dict[str, Any]] = []
    runaway = '{"a": "x' + "ab" * 100 + '"}'
    client = make_client([answer(runaway), answer(runaway)], seen)
    result = client.generate_json(LLMRequest(prompt="x", label="t", model="m", schema=SCHEMA))
    assert result == {"a": "xab"}


def test_string_fields_are_length_capped() -> None:
    """Wire schema caps identifier and text lengths."""
    from fluxsense.grounding.evidence import build_plan, wire_schema
    from fluxsense.models.semantic import FinanceBody
    schema = wire_schema(build_plan(FinanceBody))
    assert schema["properties"]["invoice_number"]["properties"]["v"]["maxLength"] == 96
    assert schema["properties"]["payment_terms"]["maxLength"] == 400


def test_unreachable_backend_fails_fast() -> None:
    """A refused connection raises BackendUnreachable at once (no transport retries, no sleeps)."""
    import time
    from fluxsense.llm.base import BackendUnreachable

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    llm = LlmSettings(base_url="http://down.invalid/v1", api_key_command="", transport_retries=3, logical_retries=2)
    client = OpenAICompatClient(llm, transport=httpx.MockTransport(refuse))
    start = time.monotonic()
    try:
        client.generate_json(LLMRequest(prompt="x", label="t", model="m"))
        raised = False
    except BackendUnreachable as e:
        raised = "unreachable" in str(e)
    assert raised and time.monotonic() - start < 1.0


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_api_key_fails_fast_as_auth_error(status: int) -> None:
    """HTTP 401 / 403 raises BackendAuthError at once (no retries), on the health probe and on a completion."""
    import time
    from fluxsense.llm.base import BackendAuthError
    calls: List[str] = []

    def reject(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(status, json={"error": "invalid api key"})

    llm = LlmSettings(base_url="http://ai.invalid/v1", api_key_command="", transport_retries=3, logical_retries=2)
    client = OpenAICompatClient(llm, transport=httpx.MockTransport(reject))
    start = time.monotonic()
    with pytest.raises(BackendAuthError, match="missing or invalid API key"):
        client.generate_json(LLMRequest(prompt="x", label="t", model="m"))
    assert calls == ["/v1/models"] and time.monotonic() - start < 1.0
    client._ready = True
    with pytest.raises(BackendAuthError):
        client.generate_json(LLMRequest(prompt="x", label="t", model="m"))
    assert calls[-1] == "/v1/chat/completions" and len(calls) == 2
