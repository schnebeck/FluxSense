# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/formats/checksums.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Check-digit algorithms of the format registry: ISO 13616
                mod 97 (IBAN), GS1 mod 10 (EAN / GTIN, ISBN-13), mod 11 with
                'X' (ISBN-10), Luhn (card numbers), ISO 7064 MOD 11,10
                (German VAT ID and tax identification number) and the
                digit-frequency rule of the German Steuer-ID.
------------------------------------------------------------------------------
"""

import re


def mod97_valid(value: str) -> bool:
    """ISO 13616: the IBAN rearranged (BBAN + country + check digits) as a number is 1 modulo 97."""
    try:
        return int("".join(str(int(c, 36)) for c in value[4:] + value[:4])) % 97 == 1
    except ValueError:
        return False


def mod97_check_digits(country: str, bban: str) -> str:
    """ISO 13616 check digits for a country code and BBAN."""
    digits = "".join(str(int(c, 36)) for c in bban + country + "00")
    return f"{98 - int(digits) % 97:02d}"


def gs1_valid(value: str) -> bool:
    """GS1 mod 10 (EAN-8/13, UPC-A, GTIN-14, ISBN-13): weights 3 and 1 from the right, the last digit is the check."""
    if not value.isdigit() or len(value) < 2:
        return False
    body, check = value[:-1], int(value[-1])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == check


def isbn10_valid(value: str) -> bool:
    """ISBN-10: sum of digit x weight (10..1) is 0 modulo 11; the check character may be 'X' (= 10)."""
    if len(value) != 10 or not value[:9].isdigit() or not (value[9].isdigit() or value[9] in "Xx"):
        return False
    digits = [int(c) for c in value[:9]] + [10 if value[9] in "Xx" else int(value[9])]
    return sum(d * w for d, w in zip(digits, range(10, 0, -1))) % 11 == 0


def luhn_valid(number: str) -> bool:
    """Luhn (mod 10) check of a card number (non-digits ignored)."""
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(reversed(digits)))
    return bool(digits) and total % 10 == 0


def iso7064_mod11_10_digit(body: str) -> int:
    """ISO 7064 MOD 11,10 check digit of a digit string."""
    product = 10
    for char in body:
        total = (int(char) + product) % 10 or 10
        product = (2 * total) % 11
    check = 11 - product
    return 0 if check == 10 else check


def iso7064_mod11_10_valid(value: str) -> bool:
    """True if the last digit is the ISO 7064 MOD 11,10 check digit of the others."""
    return value.isdigit() and len(value) >= 2 and iso7064_mod11_10_digit(value[:-1]) == int(value[-1])


def steuer_id_valid(value: str) -> bool:
    """
    German tax identification number (11 digits): no leading zero; in the
    first ten digits exactly one digit occurs twice or three times (three
    times never in a row) and all others at most once; ISO 7064 MOD 11,10
    check digit.
    """
    if len(value) != 11 or not value.isdigit() or value[0] == "0":
        return False
    head = value[:10]
    counts = [head.count(d) for d in set(head)]
    if sorted(c for c in counts if c > 1) not in ([2], [3]) or re.search(r"(\d)\1\1", head):
        return False
    return iso7064_mod11_10_valid(value)
