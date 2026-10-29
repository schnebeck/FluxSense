# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_ovis.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    OvisOCR3 'document_structured' engine against mocked,
                synthetic responses (verbatim prompt, request parameters,
                robust parsing of pretty-printed/truncated/repeated output,
                text/layout/word boxes/tables, DIN 5008 window on scans).
------------------------------------------------------------------------------
"""

import json
from typing import Any, Dict, List

from fluxsense.extract import build_text_layer
from fluxsense.extract.ovis import parse_structured, structured_prompt
from fluxsense.layout.din5008 import analyze
from fluxsense.llm.base import LLMRequest
from tests.conftest import FakeBackend

LETTER_ITEMS: List[Dict[str, Any]] = [
    {"bbox": [562, 50, 865, 75], "category": "header", "content": "MUSTERBANK"},
    {"bbox": [98, 150, 420, 160], "category": "text", "content": "Musterbank eG · Postfach 1 · 00201 Berlin"},
    {"bbox": [98, 175, 300, 240], "category": "text", "content": "Frau\nErika Beispiel\nLindenweg 3\n00115 Berlin"},
    {"bbox": [364, 109, 568, 131], "category": "text", "content": "ÜBERWIESEN 12. MAI 2031"},
    {"bbox": [98, 330, 900, 345], "category": "text", "content": "Sehr geehrte Frau Beispiel,"},
    {"bbox": [98, 360, 900, 420], "category": "table",
     "content": "<table><tr><th>Datum</th><th>Betrag</th></tr><tr><td>01.03.2026</td><td>12,50</td></tr></table>"},
    {"bbox": [450, 960, 550, 975], "category": "number", "content": "1"},
]


def test_parse_pretty_truncated_and_repeated() -> None:
    """Pretty-printed JSON, a cut tail and repetition loops are handled."""
    pretty = json.dumps(LETTER_ITEMS, indent=4, ensure_ascii=False)
    assert len(parse_structured(pretty, 1)) == len(LETTER_ITEMS)
    cut = pretty[: pretty.index("ÜBERWIESEN") + 5]
    assert len(parse_structured(cut, 1)) >= 3
    loop = LETTER_ITEMS[:2] + [{"bbox": [1, 2, 3, 4], "category": "text", "content": "0000"}] * 30
    assert len(parse_structured(loop, 1)) == 4
    assert parse_structured("not json at all", 1) == []


def test_engine_request_and_layer(settings: Any, samples: Dict[str, Any]) -> None:
    """One verbatim request per page (2880 px, 16384 tokens); results become source, items, words, tables, stream."""
    seen: List[LLMRequest] = []

    def handler(request: LLMRequest) -> Any:
        seen.append(request)
        return LETTER_ITEMS

    layer = build_text_layer(samples["letter"].path, settings, "ovisocr3", backend=FakeBackend(handler))
    request = seen[0]
    assert request.prompt == structured_prompt() and request.verbatim and request.max_tokens == 16384
    assert request.model == "ovisocr3" and max(request.images[0].width, request.images[0].height) == 2880
    page = layer.pages[0]
    assert "Erika Beispiel" in page.sources["ovis"] and "| 01.03.2026 | 12,50 |" in page.sources["ovis"]
    assert page.tables and page.tables[0].rows[1] == ["01.03.2026", "12,50"]
    assert "stream" in page.sources and not page.sources["stream"].rstrip().endswith("\n1")
    assert "Erika Beispiel" in analyze(page).address_window  # native word boxes win on native PDFs


def test_ovis_word_boxes_give_the_din_window_on_scans() -> None:
    """Without a native layer, Ovis elements (address block = one element) yield the DIN 5008 regions."""
    from fluxsense.extract.base import PageText
    from fluxsense.extract.ovis import apply_items
    page = PageText(page=1, width=595.0, height=842.0)
    apply_items(page, parse_structured(LETTER_ITEMS, 1))
    hints = analyze(page)
    assert hints.address_window == ["Frau", "Erika Beispiel", "Lindenweg 3", "00115 Berlin"]
    assert hints.return_address == "Musterbank eG · Postfach 1 · 00201 Berlin" and hints.letterhead == ["MUSTERBANK"]


def test_client_sends_verbatim_prompt_without_json_hints() -> None:
    """Verbatim requests carry no minify instruction and no response_format."""
    import httpx
    from fluxsense.config import LlmSettings
    from fluxsense.llm.openai_client import OpenAICompatClient
    seen: List[Dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": []})
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "[]"}, "finish_reason": "stop"}], "usage": {}})

    client = OpenAICompatClient(LlmSettings(base_url="http://t/v1", api_key_command=""), transport=httpx.MockTransport(handler))
    client.generate_json(LLMRequest(prompt="PROMPT", label="ocr", model="ovisocr3", verbatim=True, max_tokens=16384))
    payload = seen[0]
    assert "response_format" not in payload and payload["messages"][-1]["content"] == "PROMPT"
    assert payload["temperature"] == 0.0 and payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_ovis_only_on_weak_pages(settings: Any, samples: Dict[str, Any]) -> None:
    """extract.ovis_pages = 'weak': a digital page with a good native text layer gets no OvisOCR3 call."""
    seen: List[LLMRequest] = []
    settings.extract.ovis_pages = "weak"
    layer = build_text_layer(samples["letter"].path, settings, "ovisocr3",
                             backend=FakeBackend(lambda r: seen.append(r) or LETTER_ITEMS))
    assert seen == [] and "native" in layer.pages[0].sources and "stream" in layer.pages[0].sources
