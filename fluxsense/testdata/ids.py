# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/ids.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Fictional identifiers in valid formats (phone numbers only
                from the Bundesnetzagentur drama-number blocks, card numbers
                with the unused prefix 9999 that fail Luhn): German IBANs that
                pass mod-97 in the bank-code range 99xxxxxx, VAT IDs with
                a correct ISO 7064 check digit in the DE9... range (both
                not observed in practice; values are fictional), tax numbers with the 999/ prefix,
                and document / customer numbers.
------------------------------------------------------------------------------
"""

import random
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Tuple

# Bank codes in the 99xxxxxx range: not observed in practice; all generated values are fictional.
FICTIONAL_BLZ_PREFIX = "99"
# VAT IDs in the DE9... range: not observed in practice; all generated values are fictional.
FICTIONAL_VAT_PREFIX = "9"


def iban_check_digits(country: str, bban: str) -> str:
    """ISO 13616 check digits for a country code and BBAN."""
    digits = "".join(str(int(c, 36)) for c in bban + country + "00")
    return f"{98 - int(digits) % 97:02d}"


def fictional_iban(rng: random.Random) -> str:
    """A German IBAN (compact) that passes mod-97, bank code from the fictional 99xxxxxx range."""
    bban = FICTIONAL_BLZ_PREFIX + "".join(str(rng.randint(0, 9)) for _ in range(6)) + "".join(
        str(rng.randint(0, 9)) for _ in range(10))
    return f"DE{iban_check_digits('DE', bban)}{bban}"


def group_iban(iban: str) -> str:
    """Prints an IBAN in groups of four ('DE12 9912 ...')."""
    return " ".join(iban[i:i + 4] for i in range(0, len(iban), 4))


def fictional_bic(rng: random.Random) -> str:
    """A BIC-shaped code with the reserved-looking bank part 'XMPL'."""
    return f"XMPLDE{rng.choice('ABCDEFGH')}{rng.randint(1, 9)}XXX"


def vat_check_digit(body: str) -> int:
    """ISO 7064 MOD 11,10 check digit of the 8-digit body of a German VAT ID."""
    product = 10
    for char in body:
        total = (int(char) + product) % 10 or 10
        product = (2 * total) % 11
    check = 11 - product
    return 0 if check == 10 else check


def fictional_vat_id(rng: random.Random) -> str:
    """A German VAT ID ('DE9xxxxxxxx') with a valid check digit, from the fictional DE9 range."""
    body = FICTIONAL_VAT_PREFIX + "".join(str(rng.randint(0, 9)) for _ in range(7))
    return f"DE{body}{vat_check_digit(body)}"


# Bundesnetzagentur "Drama Numbers": local numbers permanently not assigned to any subscriber, reserved for films,
# books and other media (bundesnetzagentur.de, Nummerierung / DramaNumbers): (area code, block prefix), suffix 000-999.
DRAMA_NUMBER_BLOCKS: Tuple[Tuple[str, str], ...] = (("030", "23125"), ("069", "90009"), ("040", "66969"), ("0221", "4710"),
                                                    ("089", "99998"))
# No card network issues numbers starting with 9999; generated card numbers also fail the Luhn check on purpose.
FICTIONAL_CARD_PREFIX = "9999"


def fictional_phone(rng: random.Random) -> str:
    """A phone number from the reserved drama-number blocks ('030 23125042')."""
    area, block = rng.choice(DRAMA_NUMBER_BLOCKS)
    return f"{area} {block}{rng.randint(0, 999):03d}"


def is_drama_number(text: str) -> bool:
    """True if a printed phone number lies in a reserved drama-number block."""
    digits = re.sub(r"\D", "", text)
    return any(digits.startswith(area + block) and len(digits) == len(area + block) + 3 for area, block in DRAMA_NUMBER_BLOCKS)


def luhn_valid(number: str) -> bool:
    """Luhn (mod 10) check of a card number (non-digits ignored)."""
    digits = [int(c) for c in re.sub(r"\D", "", number)]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(reversed(digits)))
    return bool(digits) and total % 10 == 0


def fictional_card_number(rng: random.Random) -> str:
    """A 16-digit card number with the unused prefix 9999 that fails the Luhn check ('9999 1234 5678 9012')."""
    body = FICTIONAL_CARD_PREFIX + "".join(str(rng.randint(0, 9)) for _ in range(11))
    last = next(d for d in "0123456789" if not luhn_valid(body + d))
    number = body + last
    return " ".join(number[i:i + 4] for i in range(0, 16, 4))


def masked_card_number(number: str) -> str:
    """'9999 xxxx xxxx 9012'."""
    return f"{number[:4]} xxxx xxxx {number[-4:]}"


def fictional_tax_number(rng: random.Random) -> str:
    """A tax number in the 3/3/5 format with the unused office prefix 999."""
    return f"999/{rng.randint(100, 999)}/{rng.randint(10000, 99999)}"


def document_number(rng: random.Random, year: int, style: int) -> str:
    """Invoice / reference numbers in a few common shapes."""
    if style == 0:
        return f"RE-{year}-{rng.randint(1, 99999):05d}"
    if style == 1:
        return f"{rng.randint(100000, 999999)}-{year % 100:02d}"
    if style == 2:
        return f"INV{year % 100:02d}{rng.randint(100000, 999999)}"
    return f"{year}{rng.randint(1000, 9999)}"


def customer_number(rng: random.Random) -> str:
    """A customer number."""
    return f"KD-{rng.randint(100000, 999999)}"


def random_date(rng: random.Random, start: date, days: int) -> date:
    """A date within [start, start + days)."""
    return start + timedelta(days=rng.randrange(days))


def money(rng: random.Random, low_cents: int, high_cents: int) -> Decimal:
    """An amount between two cent values."""
    return (Decimal(rng.randint(low_cents, high_cents)) / 100).quantize(Decimal("0.01"))


def fmt_de(value: Decimal) -> str:
    """German amount format ('1.234,56', '-12,30')."""
    text = f"{abs(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"-{text}" if value < 0 else text


def fmt_en(value: Decimal) -> str:
    """English amount format ('1,234.56')."""
    return f"{value:,.2f}"


def fmt_date_de(value: date) -> str:
    """'DD.MM.YYYY'."""
    return value.strftime("%d.%m.%Y")
