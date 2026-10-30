# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/formats/__init__.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Format registry: one spec per fixed format (shape per
                position, checksum, normalisation, OCR confusions), looked
                up by name or by the field a value belongs to. Formats an
                application supplies for one call (FormatSpec.custom) are
                used the same way; the registry stores nothing of them.
------------------------------------------------------------------------------
"""

import re
from typing import Dict, Iterable, Optional

from fluxsense.formats.builtin import (BIC, BUILTIN, CARD, IBAN, POSTCODE_AT, POSTCODE_CH, POSTCODE_DE, STEUER_ID, TAX_NUMBER,
                                       VAT_ID)
from fluxsense.formats.spec import FormatSpec

REGISTRY: Dict[str, FormatSpec] = {spec.name: spec for spec in BUILTIN}

__all__ = ["FormatSpec", "REGISTRY", "format_for_field", "get_format", "valid_tax_id"]


def get_format(name: str, extra: Iterable[FormatSpec] = ()) -> Optional[FormatSpec]:
    """A spec by name: the call's own formats first, then the built-in ones."""
    for spec in extra:
        if spec.name == name:
            return spec
    return REGISTRY.get(name)


def format_for_field(path: str, value: object = None, country: Optional[str] = None) -> Optional[FormatSpec]:
    """
    The format of a field by its path ('...iban', '...bic', '...vat_id',
    '...tax_id', '...zip_code', '...card_number'); a tax_id of 11 digits
    without separators is a Steuer-ID, other ones a tax number; a postcode
    follows the party's country (DE by default).
    """
    name = path.rsplit(".", 1)[-1]
    if name in ("iban", "counterparty_iban", "payout_iban"):
        return IBAN
    if name == "bic":
        return BIC
    if name == "vat_id":
        return VAT_ID
    if name == "tax_id":
        text = re.sub(r"\s+", "", str(value or ""))
        if text[:2].isalpha():
            return VAT_ID
        return STEUER_ID if re.fullmatch(r"\d{11}", text) and STEUER_ID.valid(text) else TAX_NUMBER
    if name == "zip_code":
        return {"AT": POSTCODE_AT, "CH": POSTCODE_CH}.get((country or "DE").upper()[:2], POSTCODE_DE)
    if name in ("card_number", "card"):
        return CARD
    return None


def valid_tax_id(value: object, vat: bool = False) -> bool:
    """§7: a VAT ID ('DE123456789') or a German tax number ('123/456/78901', 11 or 13 digits); a BLZ or account number is
    neither (shape check)."""
    text = re.sub(r"\s+", "", str(value or "")).upper()
    if not text:
        return False
    if vat or text[:2].isalpha():
        return VAT_ID.matches(text)
    return TAX_NUMBER.matches(text)
