# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/parties.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Fictional persons and organisations of the test-document
                generator, drawn deterministically from the built-in
                vocabulary and identifier generators.
------------------------------------------------------------------------------
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from fluxsense.testdata import ids, vocab


@dataclass
class Address:
    """Street line and town."""

    street: str
    zip_code: str
    city: str

    def lines(self) -> List[str]:
        """Street and ZIP/city lines."""
        return [self.street, f"{self.zip_code} {self.city}"]

    def one_line(self) -> str:
        """'Street · ZIP City'."""
        return f"{self.street} · {self.zip_code} {self.city}"


@dataclass
class Person:
    """A fictional private person."""

    first: str
    last: str
    female: bool
    address: Address
    iban: str

    @property
    def name(self) -> str:
        """First and last name (the ground-truth form)."""
        return f"{self.first} {self.last}"

    @property
    def salutation_line(self) -> str:
        """'Frau' / 'Herrn' line of the address window."""
        return vocab.SALUTATIONS["f" if self.female else "m"][0]

    @property
    def greeting(self) -> str:
        """'Sehr geehrte Frau Muster,' greeting."""
        return f"{vocab.SALUTATIONS['f' if self.female else 'm'][1]} {self.last},"


@dataclass
class Organisation:
    """A fictional company, bank, shop or authority."""

    name: str
    address: Address
    iban: str
    bic: str
    vat_id: str
    tax_number: str
    bank: str


def town(rng: random.Random) -> str:
    """A placeholder town name ('Probehausen')."""
    return rng.choice(vocab.TOWN_STEMS) + rng.choice(vocab.TOWN_SUFFIXES)


FICTIONAL_ZIP_PREFIX = "00"  # German postcodes start at 01001: 00100-00999 never collide with a real one


def address(rng: random.Random) -> Address:
    """A fictional address (postcode from the 00xxx range)."""
    return Address(f"{rng.choice(vocab.STREETS)} {rng.randint(1, 120)}", f"{FICTIONAL_ZIP_PREFIX}{rng.randint(100, 999)}", town(rng))


def person(rng: random.Random) -> Person:
    """A fictional person with address and account."""
    first = rng.choice(vocab.FIRST_NAMES)
    return Person(first, rng.choice(vocab.SURNAMES), first in vocab.FEMALE, address(rng), ids.fictional_iban(rng))


def organisation(rng: random.Random, names: Optional[Sequence[str]] = None, legal_form: bool = True) -> Organisation:
    """A fictional organisation; names: base names to choose from (default: company stems)."""
    base = rng.choice(list(names or vocab.COMPANY_STEMS))
    name = f"{base} {rng.choice(vocab.LEGAL_FORMS)}" if legal_form and not base.endswith(("AG", "eG", "GmbH", "a.G.")) else base
    return Organisation(name, address(rng), ids.fictional_iban(rng), ids.fictional_bic(rng), ids.fictional_vat_id(rng),
                        ids.fictional_tax_number(rng), rng.choice(vocab.BANKS))
