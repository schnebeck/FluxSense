# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/formats/builtin.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Built-in format specs: IBAN (per-country BBAN structure and
                length, mod 97), BIC (8 / 11), EAN / GTIN 8 / 12 / 13 / 14
                (mod 10), ISBN-10 (mod 11, 'X') and ISBN-13, VAT ID (per
                country; DE with ISO 7064 MOD 11,10), German tax number and
                tax identification number (Steuer-ID, ISO 7064 and the digit
                rule), dates, times, postcodes (DE 5, AT / CH 4 digits) and
                card numbers (Luhn).
------------------------------------------------------------------------------
"""

import re
from datetime import date
from itertools import product
from typing import Dict, Tuple

from fluxsense.formats.checksums import (gs1_valid, isbn10_valid, iso7064_mod11_10_valid, luhn_valid, mod97_valid,
                                         steuer_id_valid)
from fluxsense.formats.spec import FormatSpec

# BBAN structure per country (SWIFT IBAN registry): '9' digits, 'A' letters, 'X' letters or digits.
IBAN_BBAN: Dict[str, str] = {
    "DE": "9" * 18, "AT": "9" * 16, "CH": "9" * 5 + "X" * 12, "LI": "9" * 5 + "X" * 12, "NL": "AAAA" + "9" * 10,
    "BE": "9" * 12, "LU": "999" + "X" * 13, "GB": "AAAA" + "9" * 14, "IE": "AAAA" + "9" * 14,
    "FR": "9" * 10 + "X" * 11 + "99", "IT": "A" + "9" * 10 + "X" * 12, "ES": "9" * 20, "PL": "9" * 24, "DK": "9" * 14,
    "SE": "9" * 20, "NO": "9" * 11, "FI": "9" * 14, "PT": "9" * 21, "CZ": "9" * 20,
}
IBAN_LENGTHS: Dict[str, int] = {country: 4 + len(bban) for country, bban in IBAN_BBAN.items()}


def _literal(text: str) -> str:
    return "".join(f"\\{c}" for c in text)


def normalize_iban(value: object) -> str:
    """Strips an 'IBAN' prefix, spaces and punctuation; uppercase."""
    from fluxsense.grounding.normalize import compact
    text = re.sub(r"^\s*IBAN\s*[:.]?\s*", "", str(value or ""), flags=re.IGNORECASE)
    return compact(text)


def _iban_other_country(value: str) -> bool:
    """Shape of an IBAN of a country without a BBAN pattern: country letters, check digits, 11-30 characters."""
    return value[:2] not in IBAN_BBAN and re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", value) is not None


IBAN = FormatSpec(name="iban", patterns=tuple(_literal(c) + "99" + bban for c, bban in IBAN_BBAN.items()),
                  checksum=mod97_valid, normaliser=normalize_iban, fallback=_iban_other_country,
                  description="IBAN (ISO 13616): country, check digits, BBAN of the country's length; mod 97")
BIC = FormatSpec(name="bic", patterns=("AAAAAAXX", "AAAAAAXXXXX"), description="BIC (ISO 9362): 8 or 11 characters")
EAN = FormatSpec(name="ean", patterns=("9" * 8, "9" * 12, "9" * 13, "9" * 14), checksum=gs1_valid,
                 description="EAN / GTIN 8, 12 (UPC-A), 13, 14; GS1 mod 10")
ISBN10 = FormatSpec(name="isbn10", patterns=("9" * 9 + "C",), checksum=isbn10_valid, description="ISBN-10; mod 11, check 'X' = 10")
ISBN13 = FormatSpec(name="isbn13", patterns=(_literal("978") + "9" * 10, _literal("979") + "9" * 10), checksum=gs1_valid,
                    description="ISBN-13 (978 / 979); GS1 mod 10")


def _strip_spaces(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def _vat_checksum(value: str) -> bool:
    """German VAT IDs carry an ISO 7064 MOD 11,10 check digit; other countries are checked by shape only."""
    return iso7064_mod11_10_valid(value[2:]) if re.fullmatch(r"DE\d{9}", value) else True


VAT_ID = FormatSpec(name="vat_id", patterns=(_literal("DE") + "9" * 9, _literal("ATU") + "9" * 8), checksum=_vat_checksum,
                    normaliser=_strip_spaces,
                    fallback=lambda v: v[:2] not in ("DE", "AT") and re.fullmatch(r"[A-Z]{2}[0-9A-Z+*.]{8,12}", v) is not None,
                    description="VAT ID: DE + 9 digits (ISO 7064 MOD 11,10), ATU + 8 digits, other countries by shape")
TAX_NUMBER = FormatSpec(name="tax_number", patterns=tuple("9" * a + "/" + "9" * b + "/" + "9" * c
                                                          for a, b, c in product((2, 3), (3, 4), (4, 5)))
                        + ("9" * 11, "9" * 13), normaliser=_strip_spaces,
                        description="German tax number (Steuernummer): 2-3/3-4/4-5 digits, or 11 / 13 digits")
STEUER_ID = FormatSpec(name="steuer_id", patterns=("9" * 11,), checksum=steuer_id_valid,
                       description="German tax identification number: 11 digits, ISO 7064 MOD 11,10 and digit rule")


def _calendar(value: str) -> bool:
    """A real calendar date (day.month.year with 2- or 4-digit year, ISO, or day/month/year)."""
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    parts = (iso.group(3), iso.group(2), iso.group(1)) if iso else tuple(re.split(r"[./]", value))
    if len(parts) != 3:
        return False
    day, month, year = (int(p) for p in parts)
    year = year + (2000 if year < 70 else 1900) if len(parts[2]) == 2 else year
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


DATE = FormatSpec(name="date", patterns=tuple(f"{'9' * d}.{'9' * m}.{'9' * y}" for d, m, y in product((1, 2), (1, 2), (2, 4)))
                  + ("9999-99-99",) + tuple(f"{'9' * d}/{'9' * m}/{'9' * y}" for d, m, y in product((1, 2), (1, 2), (2, 4))),
                  checksum=_calendar, normaliser=_strip_spaces, description="Dates d.m.y / d/m/y (2- or 4-digit year), ISO")


def _clock(value: str) -> bool:
    parts = [int(p) for p in value.split(":")]
    return parts[0] < 24 and all(p < 60 for p in parts[1:])


TIME = FormatSpec(name="time", patterns=("9:99", "99:99", "99:99:99"), checksum=_clock, normaliser=_strip_spaces,
                  description="Times h:mm, hh:mm, hh:mm:ss")
POSTCODE_DE = FormatSpec(name="postcode_de", patterns=("99999",), checksum=lambda v: v >= "01001", normaliser=_strip_spaces,
                         description="German postcode: 5 digits from 01001")
POSTCODE_AT = FormatSpec(name="postcode_at", patterns=("9999",), checksum=lambda v: v >= "1000", normaliser=_strip_spaces,
                         description="Austrian postcode: 4 digits from 1000")
POSTCODE_CH = FormatSpec(name="postcode_ch", patterns=("9999",), checksum=lambda v: "1000" <= v <= "9699",
                         normaliser=_strip_spaces, description="Swiss postcode: 4 digits 1000-9699")
CARD = FormatSpec(name="card", patterns=tuple("9" * n for n in range(13, 20)), checksum=luhn_valid,
                  description="Payment card number: 13-19 digits, Luhn")

BUILTIN: Tuple[FormatSpec, ...] = (IBAN, BIC, EAN, ISBN10, ISBN13, VAT_ID, TAX_NUMBER, STEUER_ID, DATE, TIME, POSTCODE_DE,
                                   POSTCODE_AT, POSTCODE_CH, CARD)
