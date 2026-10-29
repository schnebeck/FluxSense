# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_snap.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Snap misread identifiers/amounts/dates to the
                native text layer; IBAN prefix/masking/checksum handling and
                payee vs. direct-debit roles (synthetic telecom-bill text).
------------------------------------------------------------------------------
"""

from pathlib import Path
from typing import Any, Dict, List

from fluxsense.catalog import default_catalog
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.grounding.harness import snap_ibans, snap_ungrounded
from fluxsense.grounding.snap import damerau, iban_candidates, iban_valid, normalize_iban, snap_value
from fluxsense.models.result import FieldEvidence, GroundingStatus

PAYEE = "DE44990123450532013000"
DEBTOR = "DE02120300000000202051"
BILL = (f"Fernmelde Muster AG · USt-IdNr. DE 344487914 · Rechnungsdatum 05.10.2026\n"
        f"Rechnungsbetrag 62,30 EUR\n"
        f"Den Betrag buchen wir am 20.10.2026 von dem uns genannten Konto ab.\n"
        f"Bankverbindung: IBAN {DEBTOR[:4]} {DEBTOR[4:8]} {DEBTOR[8:12]} {DEBTOR[12:16]} {DEBTOR[16:20]} {DEBTOR[20:]}\n"
        f"Fernmelde Muster AG, IBAN {PAYEE}, BIC XMPLDEFFXXX, Beispielbank\n")


def layer() -> TextLayer:
    """One native page with the bill text."""
    return TextLayer(pages=[PageText(page=1, sources={"native": BILL})], engine="t")


def test_distance_and_iban_helpers() -> None:
    """Transposition counts as 1; prefix stripping; masked/short IBANs are invalid."""
    assert damerau("344487914", "344879114") == 2 and damerau("ab", "ba") == 1
    assert normalize_iban("IBAN DE44 9901 2345 0532 0130 00") == PAYEE
    assert iban_valid(PAYEE) and not iban_valid("DE4499012345053201XXXX") and not iban_valid("DE4499012345053201300000")


def test_snap_values_to_text() -> None:
    """VAT ID digit swap, amount transposition and date typo snap to the printed value; ambiguity does not."""
    assert snap_value("identifier", "DE 344879114", BILL) == "DE 344487914"
    assert snap_value("amount", 62.03, BILL) == 62.30
    assert snap_value("date", "2026-10-50", BILL) is None
    assert snap_value("date", "2026-10-02", BILL) in ("2026-10-05", None)
    assert snap_value("identifier", "ZZZZZZZZZ", BILL) is None


def test_iban_roles_and_repair() -> None:
    """Payee vs. debtor roles; a garbled payee IBAN is snapped; a debit account is moved to recipient.iban."""
    candidates = {c.iban: c.role for c in iban_candidates(layer(), [1])}
    assert candidates == {DEBTOR: "debtor", PAYEE: "payee"}
    fields: Dict[str, Any] = {"meta_header": {}, "bodies": {"finance_body": {"payment_accounts": [{"iban": "IBANDE4499012345053201300"}]}}}
    evidence = [FieldEvidence(path="bodies.finance_body.payment_accounts.0.iban", kind="iban", value="IBANDE4499012345053201300", page=1)]
    records = snap_ibans(layer(), [1], fields, evidence, [default_catalog().get("INVOICE")])
    assert fields["bodies"]["finance_body"]["payment_accounts"][0]["iban"] == PAYEE and records and evidence[0].action == "snapped"
    fields = {"meta_header": {}, "bodies": {"finance_body": {"payment_accounts": [{"iban": DEBTOR}]}}}
    evidence = [FieldEvidence(path="bodies.finance_body.payment_accounts.0.iban", kind="iban", value=DEBTOR, page=1)]
    snap_ibans(layer(), [1], fields, evidence, [default_catalog().get("INVOICE")])
    assert fields["meta_header"]["recipient"]["iban"] == DEBTOR
    assert fields["bodies"]["finance_body"]["payment_accounts"][0]["iban"] == PAYEE


def test_masked_iban_without_candidate_is_removed() -> None:
    """A masked IBAN is never output (left for the zoom, then removed by the harness)."""
    from fluxsense.config import Settings
    from fluxsense.grounding.harness import GroundingHarness
    from tests.conftest import FakeBackend
    text = TextLayer(pages=[PageText(page=1, sources={"native": "Konto DE44 9901 2345 0532 01XX XX " * 3})], engine="t")
    fields = {"meta_header": {"sender": {"iban": "DE4499012345053201XXXX"}}, "bodies": {}}
    evidence = [FieldEvidence(path="meta_header.sender.iban", kind="iban", value="DE4499012345053201XXXX", page=1)]
    settings = Settings()
    settings.grounding.max_escalations = 0
    GroundingHarness(FakeBackend(lambda r: None), settings).run(None, text, [1], fields, evidence, [], [default_catalog().get("INVOICE")])
    assert "iban" not in fields["meta_header"]["sender"] and evidence[0].action == "nulled"


def test_snap_ungrounded_records_original() -> None:
    """An ungrounded VAT ID is snapped and recorded with its original value."""
    fields = {"meta_header": {"sender": {"vat_id": "DE 344879114"}}}
    evidence = [FieldEvidence(path="meta_header.sender.vat_id", kind="identifier", value="DE 344879114", page=1,
                              status=GroundingStatus.UNGROUNDED)]
    records = snap_ungrounded(layer(), [1], fields, evidence)
    assert fields["meta_header"]["sender"]["vat_id"] == "DE 344487914"
    assert evidence[0].original_value == "DE 344879114" and evidence[0].status == GroundingStatus.GROUNDED
    assert records[0].applied and "DE 344879114" in records[0].note


def test_iban_checksum_repair_candidates(tmp_path: Any, monkeypatch: Any) -> None:
    """A single visual confusion (0 read as 6) is undone via mod-97; ranking prefers known IBANs and real BLZ."""
    from fluxsense.grounding.blz import known_german_blz, load_blz
    from fluxsense.grounding.snap import choose_iban_repair, iban_repair_candidates, rank_iban_candidates
    broken = PAYEE[:12] + "6" + PAYEE[13:]          # 0 misread as 6 in the account number
    assert not iban_valid(broken)
    candidates = iban_repair_candidates(broken)
    assert PAYEE in candidates and all(iban_valid(c) for c in candidates)
    ranked = rank_iban_candidates(candidates, [PAYEE])
    assert ranked[0] == (3, PAYEE)
    assert choose_iban_repair(broken, [PAYEE])[0] == PAYEE
    import fluxsense.grounding.blz as blz
    monkeypatch.setattr(blz, "USER_BLZ_FILE", tmp_path / "missing.txt")
    load_blz.cache_clear()
    assert load_blz() == frozenset() and known_german_blz(PAYEE) is None  # no list: check skipped
    monkeypatch.setattr(blz, "USER_BLZ_FILE", tmp_path / "blz.txt")
    (tmp_path / "blz.txt").write_text("99012345\n12030000\n", encoding="utf-8")
    load_blz.cache_clear()
    assert "99012345" in load_blz() and known_german_blz(PAYEE) is True and known_german_blz("AT611904300234573201") is None
    load_blz.cache_clear()


def test_iban_repair_confirmed_by_binary_zoom(monkeypatch: Any) -> None:
    """The repaired IBAN is only used if the 400 dpi binary question picks it."""
    from PIL import Image
    from fluxsense.config import Settings
    from fluxsense.grounding.harness import GroundingHarness
    from tests.conftest import FakeBackend
    monkeypatch.setattr("fluxsense.vision.zoom.render_crop", lambda *a, **k: (Image.new("RGB", (10, 10)), 400))
    broken = PAYEE[:12] + "6" + PAYEE[13:]
    scan = TextLayer(pages=[PageText(page=1, sources={"ocr": f"IBAN {broken}"})], engine="t")

    def run(choice: str) -> Dict[str, Any]:
        fields = {"meta_header": {"sender": {"iban": broken}}}
        evidence = [FieldEvidence(path="meta_header.sender.iban", kind="iban", value=broken, page=1)]

        def handler(request: Any) -> Any:
            if request.label.startswith("Zoom choice"):
                assert PAYEE in request.prompt and broken in request.prompt
                return {"choice": choice}
            return None

        settings = Settings()
        settings.grounding.max_escalations = 0
        settings.identity.private.iban = [PAYEE]
        GroundingHarness(FakeBackend(handler), settings).run(Path("x.pdf"), scan, [1], fields, evidence, [],
                                                              [default_catalog().get("INVOICE")])
        return fields["meta_header"]["sender"]

    assert run("A") == {"iban": PAYEE}
    assert run("B") == {}
