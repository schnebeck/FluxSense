# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/models/semantic.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Semantic extraction schema, ported field-compatible from
                KPaperFlux core/models/semantic.py (EN 16931 / ZUGFeRD 2.2
                aligned). KPaperFlux workflow/UI state is intentionally not
                part of the analysis output. Extensions (statement_body,
                payslip_body, correspondence_body, LineItem.source_page) are
                additive: KPaperFlux keeps unknown bodies as plain dicts.
------------------------------------------------------------------------------
"""

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from fluxsense.log import get_logger

logger = get_logger("models.semantic")


class AddressInfo(BaseModel):
    """Standardized address and contact block."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    company: Optional[str] = None
    name: Optional[str] = None
    street: Optional[str] = None
    house_number: Optional[str] = None
    zip_code: Optional[str] = Field(None, alias="zip")
    city: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    iban: Optional[str] = None
    bic: Optional[str] = None
    bank_name: Optional[str] = None
    tax_id: Optional[str] = None
    vat_id: Optional[str] = None

    @field_validator("iban", "bic", mode="before")
    @classmethod
    def normalize_banking(cls, v: Any) -> Any:
        """Removes whitespace and an 'IBAN'/'BIC' label from banking identifiers and uppercases them."""
        if isinstance(v, str):
            return re.sub(r"^(IBAN|BIC|SWIFT)[:.]?", "", "".join(v.split()).upper())
        return v


class DocumentReference(BaseModel):
    """External ID or reference found in the document (e.g. Project ID, Order ID)."""

    model_config = ConfigDict(populate_by_name=True)
    ref_type: str = Field(..., description="e.g. CUSTOMER_ID, ORDER_NUMBER, PROJECT_ID")
    ref_value: str = Field(..., description="The actual ID value")


class MetaHeader(BaseModel):
    """Document header with sender, recipient and metadata."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    sender: Optional[AddressInfo] = Field(default_factory=AddressInfo)
    recipient: Optional[AddressInfo] = Field(default_factory=AddressInfo)
    doc_date: Optional[str] = None
    doc_number: Optional[str] = Field(None, description=(
        "The sender's reference of THIS document: Rechnungsnummer, Unser Zeichen, Aktenzeichen, Vorgangs-/Schaden-/"
        "Vertragsnummer. NOT print/form codes in footers or margins (e.g. XQFRM001234, DV01.234.56)"))
    language: str = "de"
    references: List[DocumentReference] = Field(default_factory=list, description="IDs for semantic linking")


class LineItem(BaseModel):
    """Single line item (BT-126 to BT-155)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    pos: Optional[str] = Field(None, alias="BT-126")
    description: Optional[str] = Field(None, alias="BT-153")
    quantity: Optional[Decimal] = Field(None, alias="BT-129")
    unit: Optional[str] = Field("C62", alias="BT-130")
    unit_price: Optional[Decimal] = Field(None, alias="BT-146")
    total_price: Optional[Decimal] = Field(None, alias="BT-131")
    tax_rate: Optional[Decimal] = Field(None, alias="BT-152")
    article_number: Optional[str] = Field(None, alias="BT-155")
    source_page: Optional[int] = Field(None, description="1-based PDF page the row was printed on (FluxSense extension)")

    @field_validator("pos", mode="before")
    @classmethod
    def coerce_pos(cls, v: Any) -> Any:
        """Accepts numeric positions from the model and stores them as strings."""
        if isinstance(v, (int, float)):
            return str(int(v)) if float(v).is_integer() else str(v)
        return v


class TaxBreakdownRow(BaseModel):
    """Breakdown per tax rate (BT-116 to BT-121)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    tax_basis_amount: Decimal = Field(..., alias="BT-116")
    tax_amount: Decimal = Field(..., alias="BT-117")
    tax_rate: Decimal = Field(..., alias="BT-119")
    tax_category: str = Field("S", alias="BT-118")


class MonetarySummation(BaseModel):
    """Final totals (BT-106 to BT-115)."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    line_total_amount: Optional[Decimal] = Field(None, alias="BT-106")
    charge_total_amount: Optional[Decimal] = Field(None, alias="BT-107")
    allowance_total_amount: Optional[Decimal] = Field(None, alias="BT-108")
    tax_basis_total_amount: Optional[Decimal] = Field(None, alias="BT-109")
    tax_total_amount: Optional[Decimal] = Field(None, alias="BT-110")
    grand_total_amount: Optional[Decimal] = Field(None, alias="BT-112")
    due_payable_amount: Optional[Decimal] = Field(None, alias="BT-115")


class FinanceBody(BaseModel):
    """Enforce strict ZUGFeRD 2.2 / EN 16931 alignment."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    invoice_number: Optional[str] = Field(None, alias="BT-1")
    invoice_date: Optional[str] = Field(None, alias="BT-2")
    due_date: Optional[str] = Field(None, alias="BT-9")
    currency: str = Field("EUR", alias="BT-5")
    payment_reference: Optional[str] = Field(None, alias="BT-83")
    order_number: Optional[str] = Field(None, alias="BT-13")
    order_date: Optional[str] = Field(None, alias="BT-14")
    service_date: Optional[str] = Field(None, alias="BT-7")
    customer_id: Optional[str] = Field(None, alias="BT-46")
    buyer_reference: Optional[str] = Field(None, alias="BT-10")
    project_reference: Optional[str] = Field(None, alias="BT-11")
    accounting_reference: Optional[str] = Field(None, alias="BT-19")
    payment_terms: Optional[str] = Field(None, alias="BT-20")
    payment_accounts: List[AddressInfo] = Field(default_factory=list)
    line_items: List[LineItem] = Field(default_factory=list, alias="IncludedSupplyChainTradeLineItem")
    monetary_summation: Optional[MonetarySummation] = Field(
        default_factory=MonetarySummation, alias="SpecifiedTradeSettlementMonetarySummation"
    )
    tax_breakdown: List[TaxBreakdownRow] = Field(default_factory=list, alias="ApplicableTradeTax")


class NoticePeriod(BaseModel):
    """Strictly structured notice period for legal evaluation."""

    model_config = ConfigDict(populate_by_name=True)
    value: Optional[int] = Field(None, description="Numerical value (e.g. 3, 14)")
    unit: Optional[str] = Field(None, description="MUST be: DAYS, WEEKS, MONTHS, YEARS")
    anchor_type: Optional[str] = Field(None, description="MUST be: START_OF, END_OF, ANY_TIME")
    anchor_scope: Optional[str] = Field(None, description="MUST be: WEEK, MONTH, QUARTER, HALF_YEAR, YEAR")
    original_text: Optional[str] = Field(None, description="Raw extracted text for verification")


class LegalBody(BaseModel):
    """Enhanced data model for contracts and legal documents."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")
    document_title: Optional[str] = None
    contract_id: Optional[str] = None
    issuer: Optional[str] = None
    beneficiary: Optional[str] = None
    subject_reference: Optional[str] = None
    statements: List[str] = Field(default_factory=list)
    compliance_standards: List[str] = Field(default_factory=list)
    effective_date: Optional[str] = None
    termination_date: Optional[str] = None
    valid_until: Optional[str] = None
    notice_period: Optional[NoticePeriod] = Field(default_factory=NoticePeriod)
    renewal_clause: Optional[str] = None
    contract_type: Optional[str] = None
    parties: List[str] = Field(default_factory=list)


class SubscriptionInfo(BaseModel):
    """Details for recurring payments and subscriptions."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    is_recurring: bool = Field(False, description="True if the document is part of a subscription")
    frequency: Optional[str] = Field(None, description="MUST be: DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY, ONCE")
    service_period_start: Optional[str] = Field(None, description="Start of service period (YYYY-MM-DD)")
    service_period_end: Optional[str] = Field(None, description="End of service period (YYYY-MM-DD)")
    next_billing_date: Optional[str] = Field(None, description="Predicted next billing date (YYYY-MM-DD)")


# --- FluxSense extensions ----------------------------------------------------


class Transaction(BaseModel):
    """One booking line of a bank or credit card statement (FluxSense extension)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    booking_date: Optional[str] = Field(None, description="Booking date (YYYY-MM-DD)")
    value_date: Optional[str] = Field(None, description="Value date / Wertstellung (YYYY-MM-DD)")
    description: Optional[str] = Field(None, description="Booking text / Verwendungszweck (all lines joined)")
    counterparty: Optional[str] = Field(None, description="Payer or payee name")
    counterparty_iban: Optional[str] = None
    reference: Optional[str] = Field(None, description="Mandate, end-to-end or creditor reference")
    transaction_type: Optional[str] = Field(None, description="e.g. SEPA transfer, direct debit, card payment")
    amount: Optional[Decimal] = Field(None, description="Signed amount: negative = debit/charge, positive = credit/payment")
    balance: Optional[Decimal] = Field(None, description="Running balance after this booking, only if printed")
    currency: Optional[str] = None
    source_page: Optional[int] = Field(None, description="1-based PDF page the row was printed on")


class StatementBody(BaseModel):
    """Bank account or credit card statement (FluxSense extension, body key 'statement_body')."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    statement_type: Optional[str] = Field(None, description="BANK_ACCOUNT | CREDIT_CARD | SAVINGS | DEPOT")
    account_holder: Optional[str] = None
    bank_name: Optional[str] = None
    iban: Optional[str] = None
    bic: Optional[str] = None
    account_number: Optional[str] = None
    card_number_masked: Optional[str] = None
    statement_number: Optional[str] = None
    statement_date: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None
    currency: str = "EUR"
    opening_balance: Optional[Decimal] = Field(None, description="Opening / previous balance, signed as printed")
    closing_balance: Optional[Decimal] = Field(None, description="Closing / new balance, signed as printed")
    total_debits: Optional[Decimal] = Field(None, description="Printed sum of all debits (positive magnitude)")
    total_credits: Optional[Decimal] = Field(None, description="Printed sum of all credits (positive magnitude)")
    debit_count: Optional[int] = Field(None, description="Printed number of debit bookings")
    credit_count: Optional[int] = Field(None, description="Printed number of credit bookings")
    credit_limit: Optional[Decimal] = None
    minimum_payment: Optional[Decimal] = None
    payment_due_date: Optional[str] = None
    transactions: List[Transaction] = Field(default_factory=list)

    @field_validator("iban", "bic", mode="before")
    @classmethod
    def normalize_banking(cls, v: Any) -> Any:
        """Removes whitespace from banking identifiers and uppercases them."""
        if isinstance(v, str):
            return "".join(v.split()).upper()
        return v


class PayslipItem(BaseModel):
    """One wage component or deduction of a payslip (FluxSense extension)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    description: Optional[str] = None
    category: Optional[str] = Field(None, description="EARNING | TAX | SOCIAL_SECURITY | DEDUCTION | OTHER")
    amount: Optional[Decimal] = Field(None, description="Signed: earnings positive, deductions negative")
    source_page: Optional[int] = None


class PayslipBody(BaseModel):
    """Payslip / Gehaltsabrechnung (FluxSense extension, body key 'payslip_body')."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    employee_name: Optional[str] = None
    employee_id: Optional[str] = None
    employer: Optional[str] = None
    pay_period_start: Optional[str] = None
    pay_period_end: Optional[str] = None
    payment_date: Optional[str] = None
    currency: str = "EUR"
    tax_class: Optional[str] = None
    gross_pay: Optional[Decimal] = None
    income_tax: Optional[Decimal] = None
    solidarity_surcharge: Optional[Decimal] = None
    church_tax: Optional[Decimal] = None
    social_security_total: Optional[Decimal] = None
    net_pay: Optional[Decimal] = Field(None, description="Statutory net pay (Nettobezug)")
    payout_amount: Optional[Decimal] = Field(None, description="Amount transferred (Auszahlungsbetrag)")
    payout_iban: Optional[str] = None
    items: List[PayslipItem] = Field(default_factory=list)


class CorrespondenceBody(BaseModel):
    """General letter content (FluxSense extension, body key 'correspondence_body')."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")
    subject: Optional[str] = None
    summary: Optional[str] = Field(None, description="Two to four sentence summary in the document language")
    requested_action: Optional[str] = Field(None, description="What the recipient is asked to do, if anything")
    deadline: Optional[str] = Field(None, description="Deadline for the requested action (YYYY-MM-DD)")
    amount_mentioned: Optional[Decimal] = None
    reference_numbers: List[str] = Field(default_factory=list)


# Body key -> model. The catalog registers additional (custom) bodies here.
BODY_MODELS: Dict[str, Type[BaseModel]] = {
    "finance_body": FinanceBody,
    "legal_body": LegalBody,
    "subscription_info": SubscriptionInfo,
    "statement_body": StatementBody,
    "payslip_body": PayslipBody,
    "correspondence_body": CorrespondenceBody,
}


class VisualAuditResult(BaseModel):
    """Structured result of Stage 1.5 Forensic Audit."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", validate_assignment=True)
    audit_summary: Optional[Dict[str, Any]] = Field(default_factory=dict)
    layer_stamps: List[Dict[str, Any]] = Field(default_factory=list)
    signatures: Optional[Dict[str, Any]] = None
    integrity: Optional[Dict[str, Any]] = Field(default_factory=dict)
    arbiter_decision: Optional[Dict[str, Any]] = Field(default_factory=dict)
    meta_mode: Optional[str] = None


class SemanticExtraction(BaseModel):
    """The central extraction model (KPaperFlux compatible)."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore", validate_assignment=True)
    meta_header: Optional[MetaHeader] = Field(default_factory=MetaHeader)
    bodies: Dict[str, Any] = Field(default_factory=dict)
    repaired_text: Optional[str] = None
    type_tags: List[str] = Field(default_factory=list)
    direction: Optional[str] = "INBOUND"
    tenant_context: Optional[str] = "PRIVATE"
    ai_confidence: float = Field(1.0, description="Confidence score for the overall extraction (0.0 - 1.0)")
    extraction_source: Optional[str] = Field(None, description="Data source: 'AI', 'ZUGFERD_NATIVE', or None")
    visual_audit: Optional[VisualAuditResult] = None

    @field_validator("bodies", mode="after")
    @classmethod
    def parse_bodies(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Hydrates known bodies into their models; invalid bodies stay plain dicts."""
        parsed: Dict[str, Any] = {}
        for key, value in v.items():
            model = BODY_MODELS.get(key)
            if model is not None and isinstance(value, dict):
                try:
                    parsed[key] = model.model_validate(value)
                except ValidationError as e:
                    logger.debug(f"Body '{key}' kept as raw dict: {e.error_count()} validation errors")
                    parsed[key] = value
            else:
                parsed[key] = value
        return parsed

    @property
    def sender_summary(self) -> Optional[str]:
        """Company or name of the sender."""
        if self.meta_header and self.meta_header.sender:
            return self.meta_header.sender.company or self.meta_header.sender.name
        return None

    @property
    def recipient_summary(self) -> Optional[str]:
        """Company or name of the recipient."""
        if self.meta_header and self.meta_header.recipient:
            return self.meta_header.recipient.company or self.meta_header.recipient.name
        return None
