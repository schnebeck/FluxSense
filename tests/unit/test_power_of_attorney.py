# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_power_of_attorney.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Field contract §14 (powers of attorney) on synthetic texts:
                the type and its body schema, re-typing by title (a form
                with an authorisation box stays APPLICATION), and the
                principal / agent signature roles.
------------------------------------------------------------------------------
"""

from typing import List

import pytest

from fluxsense.catalog import contract
from fluxsense.catalog.builtin import default_catalog
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.grounding.evidence import Unwrapped, build_plan, unwrap, wire_schema
from fluxsense.models.semantic import BODY_MODELS, PowerOfAttorneyBody
from fluxsense.prompts import stage1
from fluxsense.stages import signers
from fluxsense.stages.classify import Entity, route_powers_of_attorney

POA_TEXT = """Vollmacht und Schweigepflichtentbindung

Hiermit bevollmächtige ich, Erika Beispiel, geb. 03.04.1941, Musterweg 7, 12345 Musterstadt,
meine Tochter Petra Probe, geb. 12.09.1968, Beispielallee 2, 54321 Probehausen,
mich gegenüber der Muster Pflegeversicherung AG in allen Angelegenheiten der Pflegeversicherung zu vertreten.
Die Vollmacht gilt bis auf schriftlichen Widerruf. Sie bleibt auch über meinen Tod hinaus wirksam.

Musterstadt, den 14.05.2026
Unterschrift Vollmachtgeberin                Unterschrift Bevollmächtigte"""

FORM_TEXT = """Antrag auf Leistungen der Pflegeversicherung
Versicherungsnummer A123456789
Name, Vorname: Beispiel, Erika
Vollmacht
[ ] Ich bevollmächtige folgende Person, mich in diesem Verfahren zu vertreten:"""


def _layer(text: str) -> TextLayer:
    """One-page layer with the given native text."""
    return TextLayer(pages=[PageText(page=1, sources={"native": text})], engine="t")


def test_type_body_and_contract_rows() -> None:
    """The type exists with its body, follows the §2.4 / form rows, and no longer hides under APPLICATION."""
    cat = default_catalog()
    spec = cat.get("POWER_OF_ATTORNEY")
    assert spec.name == "POWER_OF_ATTORNEY" and spec.body_key == "power_of_attorney_body"
    assert BODY_MODELS["power_of_attorney_body"] is PowerOfAttorneyBody
    assert contract.TYPE_CONTRACT["POWER_OF_ATTORNEY"] == contract.TypeContract("form", "form", "2.4")
    assert "handwritten allowed" in spec.doc_number_rule and "Vollmacht" not in cat.get("APPLICATION").description.split(";")[0]
    assert '"POWER_OF_ATTORNEY", never "APPLICATION" or "LEGAL_CORRESPONDENCE"' in stage1.PROMPT_STAGE_1_1_SYSTEM


def test_body_schema_kinds() -> None:
    """Codes are capped and ungrounded, the scope summary is prose, names / dates carry evidence."""
    plan = build_plan(PowerOfAttorneyBody)
    schema = wire_schema(plan)["properties"]
    assert schema["poa_type"]["maxLength"] == 80 and schema["scope_summary"]["maxLength"] >= 1000
    assert schema["valid_from"]["properties"]["v"]["maxLength"] == 24  # a date, wrapped with evidence
    agent = schema["agents"]["items"]["properties"]
    assert agent["birth_date"]["maxLength"] == 24 and agent["role"]["maxLength"] == 80  # list rows: plain values
    out = Unwrapped(None)
    wire = {"poa_type": "LASTING", "scope_summary": "Die Tochter vertritt die Mutter gegenüber der Pflegekasse.",
            "principal": {"name": "Erika Beispiel"}, "agents": [{"name": "Petra Probe", "relation": "Tochter", "role": "AGENT"}],
            "valid_beyond_death": True}
    clean = unwrap(plan, wire, "bodies.power_of_attorney_body", out)
    assert clean["agents"][0]["relation"] == "Tochter" and clean["valid_beyond_death"] is True
    kinds = {e.path.split(".")[-1]: e.kind for e in out.evidence}
    assert kinds.get("name") == "text" and "poa_type" not in kinds and "scope_summary" not in kinds and "role" not in kinds
    PowerOfAttorneyBody.model_validate(clean)


@pytest.mark.parametrize("tags", [["LEGAL_CORRESPONDENCE"], ["APPLICATION"], ["LETTER"]])
def test_title_retypes_to_power_of_attorney(tags: List[str]) -> None:
    """A document titled 'Vollmacht ...' is a power of attorney, whatever the classifier said."""
    assert route_powers_of_attorney([Entity(type_tags=tags, pages=[1])], _layer(POA_TEXT))[0].type_tags == ["POWER_OF_ATTORNEY"]
    for title in ("Vorsorgevollmacht", "Generalvollmacht", "Notarielle Vorsorgevollmacht", "VOLLMACHT"):
        layer = _layer(f"{title}\n\nHiermit bevollmächtige ich ...")
        assert route_powers_of_attorney([Entity(type_tags=tags, pages=[1])], layer)[0].type_tags == ["POWER_OF_ATTORNEY"]


def test_form_with_authorisation_box_and_letters_mentioning_it_stay() -> None:
    """An application with a 'Vollmacht' box, a letter that encloses one, and an invoice keep their type."""
    assert route_powers_of_attorney([Entity(type_tags=["APPLICATION"], pages=[1])], _layer(FORM_TEXT))[0].type_tags == ["APPLICATION"]
    letter = "Rechtsanwälte Muster & Partner\nIhr Zeichen 12/26\nSehr geehrte Damen und Herren,\nanbei die Vollmacht unseres Mandanten."
    assert route_powers_of_attorney([Entity(type_tags=["LEGAL_CORRESPONDENCE"], pages=[1])],
                                    _layer(letter))[0].type_tags == ["LEGAL_CORRESPONDENCE"]
    assert route_powers_of_attorney([Entity(type_tags=["INVOICE"], pages=[1])], _layer(POA_TEXT))[0].type_tags == ["INVOICE"]


def test_principal_and_agent_signature_roles() -> None:
    """'Vollmachtgeberin' signs as the principal (sender), 'Bevollmächtigte' as the only agent - never as the form filler."""
    assert signers.role_of("Unterschrift Vollmachtgeberin") == ("principal", False)
    assert signers.role_of("Unterschrift Bevollmächtigte") == ("agent", None)
    blocks = []
    for label in ("Unterschrift Vollmachtgeberin", "Unterschrift Bevollmächtigte"):
        role, issuing = signers.role_of(label)
        blocks.append(signers.SignatureBlock(page=1, kind="line", label=label, label_box=(0.0, 700.0, 200.0, 710.0),
                                             region=(0.0, 650.0, 200.0, 698.0), status="signed", role=role, issuing=issuing))
    fields = {"meta_header": {"sender": {"name": "Erika Beispiel"}, "recipient": {"company": "Muster Pflegeversicherung AG"}},
              "bodies": {"power_of_attorney_body": {"agents": [{"name": "Petra Probe"}]}}}
    out = {s.label: s for s in signers.attribute(blocks, fields, "2.4")}
    assert out["Unterschrift Vollmachtgeberin"].signer_name == "Erika Beispiel"
    assert out["Unterschrift Bevollmächtigte"].signer_name == "Petra Probe"
    assert out["Unterschrift Bevollmächtigte"].role == "agent"
    fields["bodies"]["power_of_attorney_body"]["agents"].append({"name": "Max Muster"})
    out = {s.label: s for s in signers.attribute(blocks, fields, "2.4")}
    assert out["Unterschrift Bevollmächtigte"].signer_name is None  # two agents: no printed evidence which one signed


def test_principal_and_agent_names_are_never_spell_corrected() -> None:
    """Names and places of the persons of a power of attorney are protected like party names."""
    from types import SimpleNamespace

    from fluxsense.text import _names
    doc = SimpleNamespace(fields={"bodies": {"power_of_attorney_body": {
        "principal": {"name": "Erika Beispiel", "city": "Musterstadt"}, "agents": [{"name": "Petra Probe"}],
        "addressees": ["Muster Pflegeversicherung AG"]}}}, signatures=[], form_fields=[])
    names = _names(SimpleNamespace(logical_documents=[doc]))
    assert {"Erika Beispiel", "Musterstadt", "Petra Probe", "Muster Pflegeversicherung AG"} <= set(names)
