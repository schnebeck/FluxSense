# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_pointer.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Numbered native text lines; values are taken
                from the line the model points to (flag off by default).
------------------------------------------------------------------------------
"""

from typing import Any, Dict, Optional

from fluxsense.extract import build_text_layer
from fluxsense.grounding.pointer import numbered_text, value_from_line
from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from tests.conftest import FakeBackend


def test_value_from_line_by_kind() -> None:
    """Kind-specific parsing picks the printed characters closest to the model's reading."""
    assert value_from_line("identifier", "RE-2026-0851", "Rechnungsnummer: RE-2026-0815") == "RE-2026-0815"
    assert value_from_line("amount", 375.0, "Rechnungsbetrag 357,00 EUR") == 357.0
    assert value_from_line("date", "2026-09-41", "Rechnungsdatum: 14.09.2026") == "2026-09-14"
    assert value_from_line("iban", "DE44990123450532031000", "auf IBAN DE44 9901 2345 0532 0130 00, BIC") == "DE44990123450532013000"
    assert value_from_line("amount", 999.0, "Rechnungsbetrag 357,00 EUR") is None


def test_pointer_mode_takes_characters_from_the_text_layer(settings: Any, samples: Dict[str, Any]) -> None:
    """With the flag, misread digits are replaced by the referenced native line; without it, nothing changes."""
    layer = build_text_layer(samples["invoice_wrong_vat"].path, settings, "native")
    _, mapping = numbered_text(layer, [1], 100000)
    number_line = next(k for k, v in mapping.items() if "RE-2026-0815" in v and "Rechnungsnummer" in v)
    seen = []

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "UNKNOWN", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            seen.append(request)
            return {"meta_header": {}, "bodies": {"finance_body": {"invoice_number": {"v": "RE-2026-0851", "l": number_line}}}}
        return None

    settings.pipeline.pointer_extraction = True
    settings.grounding.max_escalations = 0
    doc = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["invoice_wrong_vat"].path).logical_documents[0]
    assert f"[{number_line}]" in seen[0].prompt and "LINE POINTERS" in seen[0].system
    assert doc.fields["bodies"]["finance_body"]["invoice_number"] == "RE-2026-0815"
    assert doc.pointer_stats == {"pointed": 1, "resolved": 1, "changed": 1}
    settings.pipeline.pointer_extraction = False
    seen.clear()
    doc = Analyzer(settings, backend=FakeBackend(handler)).analyze(samples["invoice_wrong_vat"].path).logical_documents[0]
    assert "[p1:L" not in seen[0].prompt and doc.pointer_stats is None
