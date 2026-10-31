# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/prompts/guide.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    ZUGFERD_GUIDE ported from KPaperFlux (EN 16931 / ZUGFeRD 2.2
                terminology, legal directives, semantic linking, subscription
                detection), split per body so only relevant parts are sent.
------------------------------------------------------------------------------
"""

from typing import Iterable

GUIDE_FINANCE = """
### EXTRACTION DIRECTIVES (EN 16931 / ZUGFeRD 2.2, Cross Industry Invoice semantics)
Use the following standardized terminology for extraction (finance_body):
- **SellerTradeParty** -> meta_header.sender; **BuyerTradeParty** -> meta_header.recipient (billing address, not delivery address).
- **BT-1 (Invoice ID):** -> "invoice_number"
- **BT-2 (Issue Date):** -> "invoice_date" (ISO 8601: YYYY-MM-DD)
- **BT-9 (Due Date):** -> "due_date" (ISO 8601: YYYY-MM-DD)
- **BT-5 (Currency):** -> "currency" (ISO 4217, default: "EUR")
- **BT-13 (Order ID):** -> "order_number"
- **BT-7 (Service Date):** -> "service_date" (Leistungsdatum, often separate from invoice date)
- **BT-46 (Customer ID):** -> "customer_id" (Internal number assigned to recipient)
- **BT-10 (Buyer Reference):** -> "buyer_reference"
- **BT-11 (Project ID):** -> "project_reference"
- **BT-19 (Cost Center):** -> "accounting_reference" (Buyer accounting reference)
- **BT-20 (Payment Terms):** -> "payment_terms" (Cash discounts, valid for 10 days, etc.)
- **Line Items (IncludedSupplyChainTradeLineItem):**
  - BT-126 (ID) -> "pos"
  - BT-153 (Description) -> "description"
  - BT-129 (Quantity) -> "quantity"
  - BT-146 (Price) -> "unit_price"
  - BT-131 (Net Amount) -> "total_price"
- **Tax breakdown (ApplicableTradeTax), one row per printed tax rate:** BT-116 basis -> "tax_basis_amount", BT-117 -> "tax_amount", BT-119 -> "tax_rate".
- **Monetary Summation (SpecifiedTradeSettlementMonetarySummation):**
  - BT-106 (Line Total) -> "line_total_amount"
  - BT-109 (Tax Basis) -> "tax_basis_total_amount"
  - BT-110 (Tax Total) -> "tax_total_amount"
  - BT-112 (Grand Total) -> "grand_total_amount"
  - BT-115 (Payable) -> "due_payable_amount"
- **Payment means (SpecifiedTradeSettlementPaymentMeans):** IBAN/BIC/bank of the payee -> "payment_accounts".
- **BT-731 (Service Start Date):** -> "service_period_start"; **BT-732 (Service End Date):** -> "service_period_end" (subscription_info).

### SUBSCRIPTION & RECURRING DETECTION
If the document is an "INVOICE" or "UTILITY_BILL", look for recurring patterns:
- **Keywords:** "Subscription", "Membership", "Contract Period", "monthly", "yearly", "monatlich", "jährlich", "Grundgebühr".
- **Service Period:** Map identified billing periods to `service_period_start` and `service_period_end`.
- **Frequency:** Determine the cycle (DAILY, WEEKLY, MONTHLY, QUARTERLY, YEARLY, ONCE).
- **Trigger:** If any period or recurring keyword is found, set `is_recurring` to true in `subscription_info`.
"""

GUIDE_LEGAL = """
### LEGAL & CONTRACTUAL DIRECTIVES
If extracting into "legal_body":
- **document_title:** The exact title (e.g., "Rental Agreement", "Insurance Policy").
- **contract_id:** Any unique reference or policy number.
- **issuer:** The issuing party (e.g., Landlord, Insurer).
- **beneficiary:** The protected or served party (e.g., You).
- **effective_date:** The start date of the contract (YYYY-MM-DD).
- **termination_date:** The date the contract ends IF it has already been cancelled (YYYY-MM-DD).
- **valid_until:** The fixed expiration or warranty end date (YYYY-MM-DD).
- **notice_period:** value, unit (DAYS, WEEKS, MONTHS, YEARS), anchor_type (START_OF, END_OF, ANY_TIME), anchor_scope (WEEK, MONTH, QUARTER, HALF_YEAR, YEAR), original_text (raw text, e.g. '6 weeks to quarter end').
- **minimum_term:** a minimum term or a (mutual) waiver of ordinary termination for a duration ('verzichten
  wechselseitig auf ihr Recht zur Kündigung für die Dauer von 3 Jahren', 'Mindestlaufzeit 24 Monate'): value, unit,
  mutual_waiver, original_text. This is NEVER the notice_period; a clause like '... erstmalig nach Ablauf mit
  gesetzlicher Frist' leaves notice_period empty unless a concrete period of notice is printed.
- **renewal_clause:** Text describing auto-renewal (e.g., '12 months').
- **contract_type:** RENTAL, INSURANCE, EMPLOYMENT, etc.
- **parties:** List of str.
- **payment_terms:** the recurring payment: amount = the printed TOTAL (e.g. 'monatlich insgesamt zu zahlende Miete'),
  currency, frequency (MONTHLY, QUARTERLY, HALF_YEARLY, YEARLY, ONCE), due_rule as printed, components = every printed
  component with label and amount (they add up to the total), payment_account (name = account holder, iban, bic,
  bank_name) and payment_reference (Verwendungszweck).
- **deposit:** the security deposit (Kaution) amount.
- **contract_object:** the object of the contract (rented flat: street, house_number, zip_code, city, location_detail
  such as 'EG links'; vehicle or meter: identifier).
- **Certificates / confirmations:** certified_persons (who the confirmation is about), certified_event (MOVE_IN,
  MOVE_OUT, EMPLOYMENT, MEMBERSHIP, COMPLETION, BIRTH, MARRIAGE, DEATH, APPOINTMENT, AWARD, OTHER), event_date / event_end_date, subject_address; the issuer is the
  party that confirms (sender). Handwritten entries on a printed form count.
"""

GUIDE_STATEMENT = """
### STATEMENT DIRECTIVES (statement_body)
- One row in `transactions` per booking, all pages, printed order. Multi-line booking texts belong to ONE row (join the lines).
- 'Alter Kontostand' / previous balance -> opening_balance; 'Neuer Kontostand' / new balance -> closing_balance. They are NOT transactions.
- 'Übertrag' rows at page breaks are NOT transactions.
- Dates without year (e.g. '01.09.') take the year from the statement period.
- period_start / period_end ONLY from a printed period ('Zeitraum', 'vom ... bis ...', 'Abrechnungszeitraum'). If no period
  is printed: period_end = the printed statement/settlement date (Abrechnungsdatum, Auszugsdatum, 'Datum'), NOT the last
  booking date; period_start = the day after the previous statement date if that is printed, else omit it.
- Balances keep the printed sign: '512,30 -', '512,30 S' or 'Soll' are negative. Use ONE sign convention for opening balance,
  closing balance and all transactions.
"""

GUIDE_LINKING = """
### UNIVERSAL SEMANTIC LINKING (MetaHeader)
For ALL document types, extract ALL found IDs into the "references" list:
- **CUSTOMER_ID:** Your reference number at the sender.
- **ORDER_NUMBER:** Reference to a specific purchase order (BT-13).
- **PROJECT_REFERENCE:** Reference to a project or site (BT-11).
- **INVOICE_ID:** If found as a reference in a non-invoice document.
- **OTHER:** Use for any other significant technical identifiers.
"""

_BY_BODY = {"finance_body": GUIDE_FINANCE, "legal_body": GUIDE_LEGAL, "statement_body": GUIDE_STATEMENT}


def guide_for(body_keys: Iterable[str]) -> str:
    """
    Returns the guide sections relevant for the given bodies.

    Args:
        body_keys: Body keys of the extraction.

    Returns:
        Guide text.
    """
    parts = [_BY_BODY[k] for k in body_keys if k in _BY_BODY]
    parts.append(GUIDE_LINKING)
    return "\n".join(parts)
