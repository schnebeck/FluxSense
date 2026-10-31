# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_stamps_certificates.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Field contract 1.11 on synthetic texts: every type gets at
                least the stamp audit (seals and certification notes
                included; statements and payslips excepted, switchable),
                and an Urkunde is a CERTIFICATE.
------------------------------------------------------------------------------
"""

from typing import Any, List

import pytest

from fluxsense.catalog.builtin import default_catalog
from fluxsense.catalog.registry import AUDIT_FULL, AUDIT_NONE, AUDIT_STAMP
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.prompts import stage1
from fluxsense.stages.classify import Entity, route_certificates
from fluxsense.stages.extract import Extractor, audit_to_kpaperflux


def _layer(text: str) -> TextLayer:
    """One-page layer with the given native text."""
    return TextLayer(pages=[PageText(page=1, sources={"native": text})], engine="t")


def test_stamp_audit_floor_for_every_type_but_statements_and_payslips() -> None:
    """Without a floor the catalog modes apply; with it every type that allows a folded audit gets stamps."""
    cat = default_catalog()
    assert cat.audit_mode(["OTHER"]) == AUDIT_NONE and cat.audit_mode(["CERTIFICATE"]) == AUDIT_NONE
    for name in ("OTHER", "CERTIFICATE", "LETTER", "NOTE", "MEDICAL_DOCUMENT", "INVOICE"):
        assert cat.audit_mode([name], AUDIT_STAMP) == AUDIT_STAMP, name
    for name in ("BANK_STATEMENT", "CREDIT_CARD_STATEMENT", "PAYSLIP"):
        assert cat.audit_mode([name], AUDIT_STAMP) == AUDIT_NONE, name
    assert cat.audit_mode(["APPLICATION"], AUDIT_STAMP) == AUDIT_FULL


def test_extraction_schema_asks_for_stamps_and_seals(settings: Any) -> None:
    """The OTHER schema carries the audit block with SEAL / CERTIFICATION; the switch turns the floor off."""
    cat = default_catalog()
    build = Extractor(None, settings, cat).build([cat.get("OTHER")], False, "", audit=True)  # type: ignore[arg-type]
    stamp_types = build.schema["properties"]["audit"]["properties"]["stamps"]["items"]["properties"]["type"]["enum"]
    assert {"SEAL", "CERTIFICATION", "RECEIVED"} <= set(stamp_types) and "Dienstsiegel" in build.system
    statement = Extractor(None, settings, cat).build([cat.get("BANK_STATEMENT")], False, "", audit=True)  # type: ignore[arg-type]
    assert "audit" not in statement.schema["properties"]
    settings.pipeline.stamp_audit_all_types = False
    build = Extractor(None, settings, cat).build([cat.get("OTHER")], False, "", audit=True)  # type: ignore[arg-type]
    assert "audit" not in build.schema["properties"]
    seal = {"type_matches": True, "stamps": [{"type": "SEAL", "location": "BOTTOM", "page": 1, "text": "Standesamt Musterstadt"}]}
    assert audit_to_kpaperflux(seal, AUDIT_STAMP)["layer_stamps"][0]["type"] == "SEAL"


@pytest.mark.parametrize("title", ["Geburtsurkunde", "Heiratsurkunde", "URKUNDE", "Meisterurkunde",
                                   "Urkunde über die Ernennung"])
@pytest.mark.parametrize("tags", [["OTHER"], ["LETTER"], ["NOTE"]])
def test_urkunde_title_is_a_certificate(title: str, tags: List[str]) -> None:
    """A short Urkunde title line among the first lines re-types the answer CERTIFICATE."""
    layer = _layer(f"Standesamt Musterstadt\n{title}\nNr. 0815/2026\nErika Beispiel\ngeboren am 03.04.1941")
    assert route_certificates([Entity(type_tags=tags, pages=[1])], layer)[0].type_tags == ["CERTIFICATE"]


def test_urkunde_in_running_text_and_notarial_deeds_stay() -> None:
    """A letter enclosing an Urkunde, a notarial Vollmacht and a deed of sale keep their type."""
    cases = (("LETTER", "Erika Beispiel\nMusterweg 7\nSehr geehrte Damen und Herren,\nanbei erhalten Sie die Geburtsurkunde."),
             ("OTHER", "Urkunde\nVorsorgevollmacht\nVor mir, dem Notar ..."),
             ("OTHER", "Urkundenrolle Nr. 12/2026\nKaufvertrag\nVerhandelt zu Musterstadt"))
    for tag, text in cases:
        assert route_certificates([Entity(type_tags=[tag], pages=[1])], _layer(text))[0].type_tags == [tag], text
    assert "An Urkunde" in stage1.PROMPT_STAGE_1_1_SYSTEM and "Urkunde" in default_catalog().get("CERTIFICATE").description
