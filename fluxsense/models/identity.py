# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/models/identity.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    IdentityProfile (ported from KPaperFlux): the user's names,
                aliases, address keywords, VAT ID and IBANs. Used to decide
                direction (INBOUND/OUTBOUND) and tenant context.
------------------------------------------------------------------------------
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class IdentityProfile(BaseModel):
    """Structured identity profile of a user or business entity."""

    name: Optional[str] = None
    aliases: List[str] = Field(default_factory=list)
    company_name: Optional[str] = None
    company_aliases: List[str] = Field(default_factory=list)
    address_keywords: List[str] = Field(default_factory=list)
    vat_id: Optional[str] = None
    iban: List[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        """Returns True if the profile carries no identifying data."""
        return not (self.name or self.company_name or self.aliases or self.company_aliases
                    or self.address_keywords or self.vat_id or self.iban)

    def summary(self) -> str:
        """Compact one-line description used in prompts."""
        if self.is_empty():
            return "Not Configured"
        return f"{self.name} ({self.company_name}) | Keywords: {', '.join(self.address_keywords)}"
