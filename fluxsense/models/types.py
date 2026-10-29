# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/models/types.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Document type and classification enumerations (KPaperFlux
                DocType plus CREDIT_CARD_STATEMENT, LETTER, SICK_NOTE,
                TECHNICAL_DOC which the KPaperFlux classifier already used).
------------------------------------------------------------------------------
"""

from enum import Enum
from typing import List


class DocType(str, Enum):
    """Recognized document types."""

    QUOTE = "QUOTE"
    ORDER = "ORDER"
    ORDER_CONFIRMATION = "ORDER_CONFIRMATION"
    DELIVERY_NOTE = "DELIVERY_NOTE"
    INVOICE = "INVOICE"
    CREDIT_NOTE = "CREDIT_NOTE"
    RECEIPT = "RECEIPT"
    DUNNING = "DUNNING"
    BANK_STATEMENT = "BANK_STATEMENT"
    CREDIT_CARD_STATEMENT = "CREDIT_CARD_STATEMENT"
    TAX_ASSESSMENT = "TAX_ASSESSMENT"
    EXPENSE_REPORT = "EXPENSE_REPORT"
    UTILITY_BILL = "UTILITY_BILL"
    CONTRACT = "CONTRACT"
    INSURANCE_POLICY = "INSURANCE_POLICY"
    PAYSLIP = "PAYSLIP"
    SICK_NOTE = "SICK_NOTE"
    LEGAL_CORRESPONDENCE = "LEGAL_CORRESPONDENCE"
    OFFICIAL_LETTER = "OFFICIAL_LETTER"
    LETTER = "LETTER"
    CERTIFICATE = "CERTIFICATE"
    MEDICAL_DOCUMENT = "MEDICAL_DOCUMENT"
    VEHICLE_REGISTRATION = "VEHICLE_REGISTRATION"
    TECHNICAL_DOC = "TECHNICAL_DOC"
    APPLICATION = "APPLICATION"
    NOTE = "NOTE"
    OTHER = "OTHER"


class Direction(str, Enum):
    """Flow direction of a document relative to the user."""

    INBOUND = "INBOUND"
    OUTBOUND = "OUTBOUND"
    INTERNAL = "INTERNAL"
    UNKNOWN = "UNKNOWN"


class TenantContext(str, Enum):
    """Which identity (private or business) a document belongs to."""

    PRIVATE = "PRIVATE"
    BUSINESS = "BUSINESS"
    UNKNOWN = "UNKNOWN"


# Pre-flight coarse types (Stage 1.0).
PREFLIGHT_TYPES: List[str] = ["MANUAL", "DATASHEET", "BOOK", "CATALOG", "INVOICE", "LETTER", "OTHER"]

# Types for which the pre-flight selects the SANDWICH scan strategy.
SANDWICH_TYPES: frozenset = frozenset({"MANUAL", "DATASHEET", "BOOK", "CATALOG"})

# Tags that describe direction/context rather than a document type.
NON_TYPE_TAGS: frozenset = frozenset(
    {"INBOUND", "OUTBOUND", "INTERNAL", "CTX_PRIVATE", "CTX_BUSINESS", "UNKNOWN"}
)


def all_doc_types() -> List[str]:
    """Returns all DocType values as strings."""
    return [t.value for t in DocType]
