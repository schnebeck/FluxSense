# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_budget.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Token estimator, markdown compaction and table
                de-duplication, graceful degradation, and retry after a
                server-side exceed_context_size_error (digit-heavy two-page
                phone bill).
------------------------------------------------------------------------------
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from fluxsense.config import LlmSettings
from fluxsense.extract import build_text_layer
from fluxsense.extract.base import PageTable, PageText, TextLayer
from fluxsense.llm import budget
from fluxsense.llm.base import LLMRequest
from fluxsense.llm.images import ImagePart
from fluxsense.llm.openai_client import OpenAICompatClient
from fluxsense.samples import make_phone_bill
from fluxsense.stages.extract import Extractor
from fluxsense.catalog import default_catalog
from tests.conftest import FakeBackend


def image(px: int) -> ImagePart:
    """A fake A4 image part."""
    return ImagePart(label="PAGE 1", data_uri="data:image/png;base64,AA", width=int(px * 0.707), height=px)


def test_estimator_counts_digits_and_images() -> None:
    """Digits are single tokens; images cost (w/32)*(h/32)."""
    assert budget.text_tokens("0123456789" * 100) >= 1000
    assert budget.text_tokens("abcdefgh" * 100) < 300
    assert budget.image_tokens(1280, 905) == 40 * 29 + budget.IMAGE_OVERHEAD
    request = LLMRequest(prompt="x" * 3200, label="t", model="m", images=[image(1280)], system="y" * 320)
    assert 1000 + 100 + 1100 < budget.estimate_request(request) < 1000 + 100 + 1300


def test_markdown_compaction_and_table_stripping() -> None:
    """Docling padding is removed; tables can be replaced by a placeholder."""
    md = "Titel\n|   Datum      |   Betrag   |\n|--------------|------------|\n|   01.09.     |   1,00     |\nEnde"
    compacted = budget.compact_markdown(md)
    assert "| Datum | Betrag |" in compacted and "|---|" in compacted and len(compacted) < len(md)
    stripped = budget.strip_markdown_tables(md)
    assert stripped.splitlines() == ["Titel", "[table: see PRE-PARSED TABLES]", "Ende"]


def test_fit_degrades_images_then_count_then_text() -> None:
    """Smaller images first, then fewer images, then truncated text."""
    rendered: List[int] = []

    def rerender(px: int) -> List[ImagePart]:
        rendered.append(px)
        return [image(px) for _ in range(6)]

    request = LLMRequest(prompt="1" * 30000, label="t", model="m", images=[image(1280) for _ in range(6)], max_tokens=4000)
    notes = budget.fit(request, 32768, rerender=rerender, tail="END")
    assert rendered == [1024, 768]
    assert budget.estimate_request(request) <= budget.available(request, 32768)
    assert notes and request.prompt.endswith("END")


def test_client_retries_after_context_overflow() -> None:
    """exceed_context_size_error -> reduced payload -> success (never an ERROR document)."""
    seen: List[Dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        payload = json.loads(req.content)
        seen.append(payload)
        content = payload["messages"][-1]["content"]
        images = sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url") if isinstance(content, list) else 0
        if images > 1:
            return httpx.Response(400, json={"error": {"code": 400, "type": "exceed_context_size_error",
                                                      "message": "request (45000 tokens) exceeds the available context size"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}], "usage": {}})

    client = OpenAICompatClient(LlmSettings(base_url="http://t/v1", api_key_command="", transport_retries=0),
                                transport=httpx.MockTransport(handler))
    request = LLMRequest(prompt="text " * 100, label="t", model="m", images=[image(800) for _ in range(4)])
    assert client.generate_json(request) == {"ok": True}
    assert request.overflow and len(seen) >= 2 and client.calls[0].ok


def test_phone_bill_prompt_fits_slot(settings: Any, tmp_path: Path) -> None:
    """The digit-heavy two-page bill is sent within the 32k slot context."""
    truth = make_phone_bill(tmp_path / "bill.pdf")
    layer = build_text_layer(truth.path, settings, "native")
    for page in layer.pages:
        page.sources["docling"] = page.sources["native"].replace("\n", "   |   ") * 2
    requests: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Optional[Any]:
        requests.append(request)
        return {"meta_header": {}, "bodies": {"finance_body": {}}}

    Extractor(FakeBackend(handler), settings, default_catalog()).run(truth.path, layer, [1, 2], ["INVOICE", "UTILITY_BILL"], None, None)
    assert requests
    for request in requests:
        assert budget.estimate_request(request) <= budget.available(request, settings.llm.context_tokens)
