# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_formats.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Format registry: shapes, checksums and repairs of IBAN, BIC,
                EAN / GTIN, ISBN-10 / -13, VAT ID, German tax number and
                Steuer-ID, dates, times, postcodes and card numbers; forced
                repairs by position class, free IBAN repairs as before the
                move; formats supplied per call; the old call sites use the
                registry. Values are fictional (generator ranges) or public
                check-digit examples.
------------------------------------------------------------------------------
"""

import random

from fluxsense.formats import REGISTRY, FormatSpec, format_for_field, get_format, valid_tax_id
from fluxsense.formats.builtin import (BIC, CARD, DATE, EAN, IBAN, IBAN_LENGTHS, ISBN10, ISBN13, POSTCODE_AT, POSTCODE_CH,
                                       POSTCODE_DE, STEUER_ID, TAX_NUMBER, TIME, VAT_ID)
from fluxsense.formats.checksums import gs1_valid, iso7064_mod11_10_digit, mod97_check_digits
from fluxsense.formats.iban import choose_iban_repair, iban_repair_candidates, iban_valid, printed_ibans
from fluxsense.testdata import ids

PAYEE = "DE44990123450532013000"


def _gtin(body: str) -> str:
    return next(body + d for d in "0123456789" if gs1_valid(body + d))


def test_iban_per_country_structure_and_checksum() -> None:
    """Country length and BBAN structure, mod 97, masking; the free repairs of the old snap module are unchanged."""
    rng = random.Random("formats-iban")
    for _ in range(20):
        iban = ids.fictional_iban(rng)
        assert iban_valid(iban) and IBAN.valid(ids.group_iban(iban)) and IBAN.valid(f"IBAN {iban}")
    assert IBAN_LENGTHS["DE"] == 22 and IBAN_LENGTHS["NL"] == 18 and IBAN_LENGTHS["CH"] == 21
    assert not iban_valid(PAYEE[:-1]) and not iban_valid("DE4499012345053201XXXX")
    bban = "AAAA" + "1234567890"
    nl = f"NL{mod97_check_digits('NL', bban)}{bban}"
    assert iban_valid(nl) and not IBAN.matches(nl.replace("AAAA", "AA12"))  # NL: four letters, then digits
    broken = PAYEE[:12] + "6" + PAYEE[13:]  # 0 read as 6
    assert not iban_valid(broken) and PAYEE in iban_repair_candidates(broken)
    assert choose_iban_repair(broken, [PAYEE]) == (PAYEE, 3)
    assert printed_ibans("Konto: DE44 9901 2345 0532 0130 00") == [PAYEE]


def test_forced_repairs_by_position_class() -> None:
    """A letter at a digit position is replaced by its confusable digit; the repair must pass the checksum; a value
    without a class violation has no forced repair."""
    misread = "DE44 99O1 2345 0532 O130 00"
    assert IBAN.forced_positions(misread) and IBAN.forced_repairs(misread) == [PAYEE]
    assert IBAN.forced_repairs(PAYEE[:12] + "6" + PAYEE[13:]) == []  # digit / digit: only a re-read can decide
    assert IBAN.find(f"Bitte auf {misread} überweisen.") == [(10, 10 + len(misread), misread)]
    vat = ids.fictional_vat_id(random.Random("vat"))
    assert VAT_ID.valid(vat) and VAT_ID.forced_repairs(vat[:6] + "O" + vat[7:]) == ([vat] if vat[6] == "0" else [])


def test_bic_ean_isbn_card() -> None:
    assert BIC.valid("XMPLDEFFXXX") and BIC.valid("XMPLDEFF") and not BIC.valid("XMPLDEFFXX") and not BIC.valid("COBA-DE")
    for body in ("4000123", "40001234567", "400012345678", "1400012345678"):
        code = _gtin(body)
        assert EAN.valid(code) and not EAN.valid(code[:-1] + str((int(code[-1]) + 1) % 10))
    assert ISBN10.valid("080442957X") and ISBN10.valid("0306406152") and not ISBN10.valid("0306406153")
    assert ISBN13.valid("978-0-306-40615-7") and not ISBN13.valid("9790306406157") and not ISBN13.valid("9770306406157")
    assert CARD.valid("4111 1111 1111 1111")
    full = ids.fictional_card_number(random.Random("card"))
    assert CARD.matches(full) and not CARD.valid(full)  # generated card numbers fail Luhn on purpose


def test_vat_tax_number_and_steuer_id() -> None:
    """DE VAT ID with ISO 7064 MOD 11,10; ATU and other countries by shape; the old valid_tax_id semantics (shape)."""
    rng = random.Random("formats-vat")
    for _ in range(10):
        vat = ids.fictional_vat_id(rng)
        assert VAT_ID.valid(vat) and valid_tax_id(vat, vat=True)
        assert not VAT_ID.valid(vat[:-1] + str((int(vat[-1]) + 1) % 10))
    assert VAT_ID.valid("ATU12345678") and VAT_ID.valid("FR12345678901") and not VAT_ID.matches("DE1234")
    assert iso7064_mod11_10_digit("13669597") == 6
    assert valid_tax_id("123/456/78901") and valid_tax_id(ids.fictional_tax_number(rng)) and not valid_tax_id("99012345")
    assert TAX_NUMBER.valid("12/345/6789") and not TAX_NUMBER.valid("12/34/6789")
    head = "1234567891"  # one digit twice, the others once
    steuer_id = head + str(iso7064_mod11_10_digit(head))
    assert STEUER_ID.valid(steuer_id)
    assert not STEUER_ID.valid("1234567890" + str(iso7064_mod11_10_digit("1234567890")))  # no repeated digit
    assert not STEUER_ID.valid("1112345678" + str(iso7064_mod11_10_digit("1112345678")))  # three in a row
    assert format_for_field("meta_header.sender.tax_id", steuer_id) is STEUER_ID
    assert format_for_field("meta_header.sender.tax_id", "123/456/78901") is TAX_NUMBER


def test_dates_times_postcodes() -> None:
    assert DATE.valid("07.08.2024") and DATE.valid("7.8.24") and DATE.valid("2024-08-07") and not DATE.valid("31.02.2024")
    assert TIME.valid("23:59") and not TIME.valid("24:10")
    assert POSTCODE_DE.valid("01067") and not POSTCODE_DE.valid("00999") and not POSTCODE_DE.valid("1234")
    assert POSTCODE_AT.valid("1010") and POSTCODE_CH.valid("8001") and not POSTCODE_CH.valid("9999")
    assert format_for_field("meta_header.sender.zip_code", "8001", "CH") is POSTCODE_CH
    assert format_for_field("meta_header.sender.zip_code", "01067") is POSTCODE_DE


def test_formats_supplied_per_call() -> None:
    """An application's own format (here a customer-number pattern) uses the same spec type; it is looked up before
    the built-in ones and never stored in the registry."""
    customer = FormatSpec.custom("customer_number", ["\\K\\D999999"], description="customer number of one sender")
    assert customer.valid("KD-123456") and not customer.valid("KD-12345")
    assert customer.forced_repairs("KD-12345G") == ["KD123456"]
    assert get_format("customer_number", [customer]) is customer and "customer_number" not in REGISTRY
    assert get_format("iban") is IBAN


def test_testdata_uses_the_registry_checksums() -> None:
    """The generator's check digits come from the registry (no second implementation)."""
    assert ids.vat_check_digit("13669597") == 6 and ids.luhn_valid("4111 1111 1111 1111")
    assert ids.iban_check_digits("DE", PAYEE[4:]) == PAYEE[2:4]
