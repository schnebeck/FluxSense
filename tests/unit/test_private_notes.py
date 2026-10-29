# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_private_notes.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Field contract §2.7 on generated death notices and greeting
                cards: a LETTER / OTHER answer is re-typed NOTE, and a NOTE
                (also as a secondary type tag) never keeps a sender or
                recipient.
------------------------------------------------------------------------------
"""

import random
from pathlib import Path
from typing import Any, List, Optional

import pytest

from fluxsense.extract.base import TextLayer
from fluxsense.llm.base import LLMRequest
from fluxsense.pipeline import Analyzer
from fluxsense.stages.classify import Entity, route_private_notes
from fluxsense.testdata import build_letter, build_private_note
from fluxsense.testdata.layout import render
from tests.conftest import FakeBackend


@pytest.mark.parametrize("card", [True, False])
def test_card_and_notice_vocabulary_routes_to_note(card: bool) -> None:
    """A short LETTER / OTHER without letterhead printing notice or card vocabulary becomes NOTE; a letter stays."""
    note = build_private_note(random.Random(f"note-{card}"), card=card)
    layer = TextLayer(pages=[note.pages[0].page_text(1)], engine="t")
    assert route_private_notes([Entity(type_tags=["LETTER"], pages=[1])], layer)[0].type_tags == ["NOTE"]
    letter = build_letter(random.Random("letter-1"))
    layer = TextLayer(pages=[letter.pages[0].page_text(1)], engine="t")
    entity = Entity(type_tags=["LETTER"], pages=[1])
    entity.letterhead = True
    assert route_private_notes([entity], layer)[0].type_tags == ["LETTER"]


@pytest.mark.parametrize("tags", [["LETTER"], ["LETTER", "NOTE"]])
def test_private_note_has_no_parties(settings: Any, tmp_path: Path, tags: List[str]) -> None:
    """The model names a sender and a recipient on a greeting card: both are removed (§2.7)."""
    note = build_private_note(random.Random("card-2"), card=True)
    pdf = render(note.pages, tmp_path / "card.pdf")
    words = note.text().split()

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": tags, "page_indices": [1], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Stage 2"):
            return {"meta_header": {"sender": {"name": {"v": f"{words[-3]} Familie", "p": 1}},
                                    "recipient": {"name": {"v": words[5].strip(","), "p": 1}}}, "bodies": {}}
        return None

    doc = Analyzer(settings, backend=FakeBackend(handler)).analyze(pdf).logical_documents[0]
    assert "NOTE" in [doc.type] + doc.type_tags
    assert not doc.fields["meta_header"].get("sender") and not doc.fields["meta_header"].get("recipient")
