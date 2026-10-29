# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_identity.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tenant context: a person-name match alone never decides
                BUSINESS when that person is also the private identity.
                Synthetic profiles.
------------------------------------------------------------------------------
"""

from fluxsense.models.identity import IdentityProfile
from fluxsense.stages.identity import business_only, direction_from_parties

PRIVATE = IdentityProfile(name="Max Mustermann", aliases=["M. Mustermann"], address_keywords=["00345", "Musterstadt"])
BUSINESS = IdentityProfile(name="Max Mustermann", company_name="Mustermann Technik GmbH", aliases=["Max Mustermann"],
                           vat_id="DE123456789", address_keywords=["54321", "Beispielhausen"])


def test_person_only_match_is_private() -> None:
    """An invoice to the person is PRIVATE although the business profile lists the person as an alias."""
    direction, context = direction_from_parties({"company": "Autohaus Beispiel GmbH"}, {"name": "Max Mustermann"},
                                                PRIVATE, BUSINESS)
    assert (direction, context) == ("INBOUND", "PRIVATE")
    trimmed = business_only(BUSINESS, PRIVATE)
    assert trimmed.name is None and trimmed.aliases == [] and trimmed.company_name == "Mustermann Technik GmbH"


def test_company_or_vat_match_is_business() -> None:
    """The company name or VAT ID of the business profile decides BUSINESS."""
    assert direction_from_parties({"company": "Lieferant AG"}, {"company": "Mustermann Technik GmbH"}, PRIVATE, BUSINESS) == \
        ("INBOUND", "BUSINESS")
    assert direction_from_parties({"company": "Mustermann Technik GmbH", "vat_id": "DE123456789"}, {"name": "Erika Beispiel"},
                                  PRIVATE, BUSINESS) == ("OUTBOUND", "BUSINESS")
