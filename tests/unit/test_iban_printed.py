# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_iban_printed.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    An IBAN printed with a wrong checksum (generated fictional
                IBAN with one changed account digit): kept as printed when
                the native text prints it exactly, flagged with an ERROR,
                the bank-code status and one-digit hints that pass mod-97
                and keep a known bank code (never applied), never
                auto-accepted and not removed by the bank-identifier
                hygiene. An IBAN that is not printed so is still removed.
------------------------------------------------------------------------------
"""

import random
from typing import Any

from fluxsense.catalog import default_catalog
from fluxsense.config import Settings
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.grounding.acceptance import assess
from fluxsense.grounding.harness import CHECK_IBAN_PRINTED_INVALID, GroundingHarness, iban_digit_hints
from fluxsense.grounding.snap import iban_valid
from fluxsense.models.result import FieldEvidence
from fluxsense.stages.repair import sanitize_bank_identifiers
from fluxsense.testdata import ids
from tests.conftest import FakeBackend


def _typo(rng: random.Random) -> tuple:
    valid = ids.fictional_iban(rng)
    digit = str((int(valid[15]) + 3) % 10)
    wrong = valid[:15] + digit + valid[16:]
    assert not iban_valid(wrong)
    return valid, wrong


def _run(text: str, iban: str) -> tuple:
    layer = TextLayer(pages=[PageText(page=1, sources={"native": text})], engine="t")
    fields = {"meta_header": {"sender": {"company": "Beispiel Immobilien KG"}},
              "bodies": {"finance_body": {"payment_accounts": [{"iban": iban}]}}}
    evidence = [FieldEvidence(path="bodies.finance_body.payment_accounts.0.iban", kind="iban", value=iban, page=1, critical=True)]
    settings = Settings()
    settings.grounding.max_escalations = 0
    outcome = GroundingHarness(FakeBackend(lambda r: None), settings).run(None, layer, [1], fields, evidence, [],
                                                                         [default_catalog().get("INVOICE")])
    return layer, fields, evidence, outcome


def test_printed_invalid_iban_is_kept_flagged_with_hints(monkeypatch: Any) -> None:
    """The document prints an IBAN with a one-digit typo: kept, ERROR with bank-code status and the corrected IBAN
    among the hints (never applied), not auto-accepted, not removed by the hygiene."""
    valid, wrong = _typo(random.Random("iban-typo"))
    monkeypatch.setattr("fluxsense.grounding.harness.known_german_blz", lambda iban: iban[4:12] == valid[4:12])
    text = f"Bitte überweisen Sie die Miete auf das Konto IBAN {ids.group_iban(wrong)} BIC XMPLDEFFXXX"
    layer, fields, evidence, outcome = _run(text, wrong)
    assert fields["bodies"]["finance_body"]["payment_accounts"][0]["iban"] == wrong
    check = next(c for c in outcome.checks if c.check_id == CHECK_IBAN_PRINTED_INVALID)
    assert check.severity.value == "ERROR" and "bank code known" in check.message and valid in check.message
    hints = iban_digit_hints(wrong)
    assert valid in hints and len(hints) <= 2 and all(iban_valid(h) and h[4:12] == valid[4:12] for h in hints)
    assert sanitize_bank_identifiers(fields, {wrong}) == [] and fields["bodies"]["finance_body"]["payment_accounts"][0]["iban"] == wrong
    assess(fields, evidence, layer, [1], "INVOICE", outcome.checks)
    assert not evidence[0].auto_accepted and any("as printed fails the checksum" in r for r in evidence[0].review_reasons)


def test_unprinted_invalid_iban_is_removed() -> None:
    """The text does not print the invalid IBAN (a misreading): it is removed as before."""
    valid, wrong = _typo(random.Random("iban-misread"))
    _, fields, evidence, outcome = _run(f"IBAN {ids.group_iban(valid)}", wrong)
    assert not any(c.check_id == CHECK_IBAN_PRINTED_INVALID for c in outcome.checks)
    assert fields["bodies"]["finance_body"]["payment_accounts"][0].get("iban") != wrong
