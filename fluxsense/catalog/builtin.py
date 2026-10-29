# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/catalog/builtin.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Built-in type definitions. Body mapping follows KPaperFlux's
                TYPE_TO_BODY (stage2.py) with FluxSense extensions:
                statements -> statement_body, PAYSLIP -> payslip_body,
                LETTER/other types -> correspondence_body.
------------------------------------------------------------------------------
"""

from dataclasses import replace
from typing import Optional, Tuple

from fluxsense.catalog.contract import (STRICT_NUMBER_GROUPS, contract_for, date_labels, label_regex, number_labels,
                                       type_prompt_rules)
from fluxsense.catalog.registry import AUDIT_FULL, AUDIT_NONE, AUDIT_STAMP, Catalog, TableSpec, TypeSpec

LINE_ITEMS = TableSpec(
    name="line_items", body_key="finance_body", field_name="line_items",
    columns=("pos", "description", "quantity", "unit", "unit_price", "total_price", "tax_rate", "source_page"),
    amount_fields=("unit_price", "total_price"),
)

TRANSACTIONS = TableSpec(
    name="transactions", body_key="statement_body", field_name="transactions",
    columns=("booking_date", "value_date", "description", "counterparty", "amount", "balance", "source_page"),
    amount_fields=("amount", "balance"),
    date_fields=("booking_date", "value_date"),
)

PAYSLIP_ITEMS = TableSpec(
    name="payslip_items", body_key="payslip_body", field_name="items",
    columns=("description", "category", "amount", "source_page"),
    amount_fields=("amount",),
)

FIN = "bodies.finance_body"
FIN_CRITICAL: Tuple[str, ...] = (
    f"{FIN}.invoice_number", f"{FIN}.invoice_date", f"{FIN}.due_date",
    f"{FIN}.monetary_summation.*", f"{FIN}.tax_breakdown.*.*", f"{FIN}.payment_accounts.*.iban",
    "meta_header.doc_date", "meta_header.doc_number", "meta_header.sender.iban",
)
FIN_RULES: Tuple[str, ...] = ("invoice.arithmetic", "invoice.line_items", "invoice.tax_rates", "currency",
                              "iban", "dates", "completeness", "identity")
INVOICE_QUESTIONS: Tuple[str, ...] = (
    "Who is the seller (SellerTradeParty) and who is the buyer (BuyerTradeParty)? Name, address, VAT ID / tax number.",
    "What is the invoice number (BT-1), issue date (BT-2), due date (BT-9), service/delivery date (BT-7) or service period?",
    "Which order number (BT-13), customer number (BT-46) or other references are printed?",
    "List EVERY billed line item of the whole document (all pages, in order) with quantity, unit price, net line total and tax rate. "
    "An itemised call list / Einzelverbindungsnachweis / usage report is NOT line items: only the billed positions "
    "(e.g. Grundgebühr, Verbindungen, Gutschrift) are.",
    "What are the printed totals: line total (BT-106), tax basis (BT-109), tax total (BT-110), grand total (BT-112), amount due (BT-115)?",
    "What is the tax breakdown per tax rate (basis, rate, tax amount) as printed?",
    "Which payment means are given (IBAN, BIC, bank), which payment terms and payment reference?",
    "Is this a recurring charge (subscription, monthly fee)? Which service period?",
)

# Statement-specific: the statement number field is the doc_number; a due date needs a printed debit date (§6).
STATEMENT_DOC_RULES = dict(
    doc_number_from="bodies.statement_body.statement_number",
    due_date_labels=(r"wird\s+am", r"belastet\s+am", r"f[äa]llig\s+am", r"zahlbar\s+bis", r"abgebucht\s+am", r"lastschrift\s+am"),
)


def with_contract(spec: TypeSpec) -> TypeSpec:
    """
    Derives the doc_number / doc_date rules of a type from the field contract
    (fluxsense/catalog/contract.py, mirroring docs/field_contract.md): the
    prompt rule text, and the removal label lists for the groups where an
    unlabelled value is removed (statements, forms, enclosures).
    """
    rows = contract_for(spec.name)
    numbers = number_labels(spec.name) if rows.number_group in STRICT_NUMBER_GROUPS else ()
    dates = date_labels(spec.name) if rows.date_group == "enclosure" else ()
    return replace(spec, doc_number_rule=type_prompt_rules(spec.name),
                   doc_number_labels=tuple(label_regex(label, compound_head=True) for label in numbers),
                   doc_date_labels=tuple(label_regex(label) for label in dates))


STATEMENT_QUESTIONS: Tuple[str, ...] = (
    "Which account is this (holder, bank, IBAN, BIC, account or masked card number)?",
    "Statement number, statement date and covered period (from - to)?",
    "Opening (previous) balance and closing (new) balance, signed as printed (debit/overdraft/S = negative)?",
    "List EVERY booking of the whole statement in printed order (all pages): booking date, value date, text, counterparty, signed amount, running balance if printed.",
    "Printed totals of debits and credits and their counts, if present?",
)


def _finance(name: str, description: str, rules: Tuple[str, ...] = FIN_RULES, audit: str = AUDIT_STAMP,
             zugferd: bool = False, required: Tuple[str, ...] = ()) -> TypeSpec:
    """Builds a finance_body type."""
    return TypeSpec(name=name, description=description, body_key="finance_body", extra_bodies=("subscription_info",),
                    questions=INVOICE_QUESTIONS, tables=(LINE_ITEMS,), rules=rules, critical_fields=FIN_CRITICAL,
                    required_fields=required, audit_mode=audit, zugferd=zugferd)


def _legal(name: str, description: str, audit: str = AUDIT_FULL, extra: Tuple[str, ...] = (),
           kpf_type: Optional[str] = None) -> TypeSpec:
    """Builds a legal_body type."""
    return TypeSpec(
        name=name, description=description, body_key="legal_body", extra_bodies=extra, kpf_type=kpf_type,
        questions=("What is the exact document title and the contract/policy/reference number? The sender's reference "
                   "('Unser Zeichen', 'Aktenzeichen', 'Vorgangs-/Schadennummer') is meta_header.doc_number; print or form "
                   "codes in footers or margins are never document numbers.",
                   "Who is the issuer and who is the beneficiary? Which parties are named?",
                   "Effective date, valid-until date, termination date (only if already cancelled)?",
                   "Notice period (value, unit, anchor) and renewal clause, quoted as printed?",
                   "Key statements or obligations (short, as printed)?"),
        rules=("dates", "completeness", "identity"),
        critical_fields=("bodies.legal_body.contract_id", "bodies.legal_body.effective_date",
                         "bodies.legal_body.valid_until", "meta_header.doc_date"),
        required_fields=("bodies.legal_body.document_title",),
        audit_mode=audit,
    )


def _correspondence(name: str, description: str, audit: str = AUDIT_NONE, kpf_type: Optional[str] = None) -> TypeSpec:
    """Builds a correspondence_body type."""
    return TypeSpec(
        name=name, description=description, body_key="correspondence_body", kpf_type=kpf_type,
        questions=("What is the subject?", "Summarize the content in two to four sentences.",
                   "Is the recipient asked to do something, and until when (deadline)?",
                   "Which reference numbers (file number, customer number, case number) are printed? The sender's reference "
                   "('Unser Zeichen', 'Aktenzeichen', 'Vorgangs-/Schadennummer') is meta_header.doc_number; print or form codes "
                   "in footers or margins (e.g. 'XQFRM001234', 'DV01.234.56') are never document numbers.",
                   "Is an amount of money mentioned?"),
        rules=("dates", "identity"),
        critical_fields=("meta_header.doc_date", "bodies.correspondence_body.deadline",
                         "bodies.correspondence_body.amount_mentioned"),
        audit_mode=audit,
    )


def build_catalog() -> Catalog:
    """
    Creates the catalog with all built-in types.

    Returns:
        The populated catalog.
    """
    cat = Catalog()
    inv_required = ("bodies.finance_body.invoice_number", "bodies.finance_body.invoice_date",
                    "bodies.finance_body.monetary_summation.grand_total_amount")
    cat.register(_finance("INVOICE", "Rechnung / invoice asking for payment of goods or services (EN 16931). Telecom, energy and water bills: tag INVOICE and UTILITY_BILL", zugferd=True, required=inv_required))
    cat.register(_finance("CREDIT_NOTE", "Gutschrift / Rechnungskorrektur / credit note", zugferd=True, required=inv_required))
    cat.register(_finance("RECEIPT", "Quittung, Kassenbon, Zahlungsbeleg, Spendenquittung", zugferd=True,
                          required=("bodies.finance_body.monetary_summation.grand_total_amount",)))
    cat.register(_finance("UTILITY_BILL", "Strom, Gas, Wasser, Telefon, Internet, Mobilfunk, Rundfunkbeitrag bill or Abschlag - always together with INVOICE", zugferd=True, required=inv_required))
    cat.register(_finance("DUNNING", "Mahnung, Zahlungserinnerung, Inkasso",
                          required=("bodies.finance_body.monetary_summation.due_payable_amount",)))
    for name, desc in (("ORDER", "Bestellung, Auftrag (sent by the buyer)"), ("ORDER_CONFIRMATION", "Auftragsbestätigung, Bestellbestätigung"),
                       ("QUOTE", "Angebot, Kostenvoranschlag"), ("EXPENSE_REPORT", "Reisekostenabrechnung, Spesenabrechnung"),
                       ("TAX_ASSESSMENT", "Steuerbescheid, Grundsteuer-/Gewerbesteuerbescheid, Vorauszahlungsbescheid")):
        cat.register(_finance(name, desc))
    cat.register(TypeSpec(
        name="DELIVERY_NOTE", description="Delivery note / packing list (Lieferschein), no prices",
        body_key="finance_body", questions=(
            "Delivery note number, date, order number and customer number?",
            "List EVERY delivered item of the whole document with quantity, unit and article number."),
        tables=(LINE_ITEMS,), rules=("dates", "completeness", "identity"),
        critical_fields=("meta_header.doc_number", "meta_header.doc_date", f"{FIN}.order_number"),
        audit_mode=AUDIT_FULL))
    statement_rules = ("statement.balance", "statement.running_balance", "statement.period", "statement.totals",
                       "iban", "dates", "completeness")
    st_critical = ("bodies.statement_body.opening_balance", "bodies.statement_body.closing_balance",
                   "bodies.statement_body.iban", "bodies.statement_body.period_start", "bodies.statement_body.period_end",
                   "bodies.statement_body.total_debits", "bodies.statement_body.total_credits",
                   "bodies.statement_body.debit_count", "bodies.statement_body.credit_count")
    cat.register(TypeSpec(**STATEMENT_DOC_RULES,
        name="BANK_STATEMENT", description="Kontoauszug, Depotauszug, Rechnungsabschluss of a bank account", body_key="statement_body",
        questions=STATEMENT_QUESTIONS, tables=(TRANSACTIONS,), rules=statement_rules, critical_fields=st_critical,
        required_fields=("bodies.statement_body.opening_balance", "bodies.statement_body.closing_balance")))
    cat.register(TypeSpec(**STATEMENT_DOC_RULES,
        name="CREDIT_CARD_STATEMENT", kpf_type="BANK_STATEMENT", description="Kreditkartenabrechnung, Umsatzaufstellung of a credit card", body_key="statement_body",
        questions=STATEMENT_QUESTIONS + ("Credit limit, minimum payment and payment due date?",),
        tables=(TRANSACTIONS,), rules=("statement.credit_card", "statement.period", "statement.totals", "iban", "dates", "completeness"),
        critical_fields=st_critical + ("bodies.statement_body.payment_due_date",),
        required_fields=("bodies.statement_body.closing_balance",)))
    cat.register(TypeSpec(
        name="TRANSFER_CONFIRMATION", kpf_type="BANK_STATEMENT", body_key="statement_body",
        description="Bank confirmation of ONE transaction: Überweisungsbeleg, Auftrags-/Ausführungsbestätigung of a transfer, "
                    "Auslandsüberweisung, Einzahlungsbeleg, Lastschriftbeleg (issued by the bank, not an invoice or receipt)",
        questions=("Which bank issued the confirmation (sender) and who is the account holder / ordering customer "
                   "(Auftraggeber, Kontoinhaber = recipient)? The payee is a counterparty, never the recipient.",
                   "Ordering account (IBAN, BIC) of the account holder = statement_body.iban / bic?",
                   "The ONE transaction as a single row: execution / booking date, payee (counterparty) and payee IBAN, "
                   "purpose (Verwendungszweck), signed amount (a transfer out is negative).",
                   "The transfer amount is no invoice total: leave totals and tax empty."),
        tables=(TRANSACTIONS,), rules=("iban", "dates", "completeness"),
        critical_fields=("bodies.statement_body.iban", "bodies.statement_body.transactions.*.amount", "meta_header.doc_date"),
        audit_mode=AUDIT_STAMP))
    cat.register(TypeSpec(
        name="PAYSLIP", description="Gehaltsabrechnung, Entgeltabrechnung, Lohnabrechnung, Bezügemitteilung", body_key="payslip_body",
        questions=("Employee, employee id, employer, pay period and payment date?",
                   "Gross pay, income tax, solidarity surcharge, church tax, total social security, statutory net pay, payout amount?",
                   "List every wage component and deduction with signed amount (earnings positive, deductions negative)."),
        tables=(PAYSLIP_ITEMS,), rules=("payslip.net", "iban", "dates", "completeness"),
        critical_fields=("bodies.payslip_body.gross_pay", "bodies.payslip_body.net_pay", "bodies.payslip_body.payout_amount",
                         "bodies.payslip_body.payout_iban"),
        required_fields=("bodies.payslip_body.gross_pay", "bodies.payslip_body.net_pay")))
    for name, desc in (("CONTRACT", "Vertrag, Vereinbarung, Vertragsbestätigung with parties and terms"), ("INSURANCE_POLICY", "Versicherungsschein, Police, Nachtrag, Beitragsrechnung of an insurer"),
                       ("LEGAL_CORRESPONDENCE", "Letter from or to a court, lawyer (Anwalt), arbitration board (Schlichtungsstelle), ombudsman "
                        "or consumer advice centre about proceedings or a complaint (Schlichtungsverfahren, Beschwerde, "
                        "Klage, Aktenzeichen of the proceeding) - also a bank's or company's statement addressed to the ombudsman")):
        cat.register(_legal(name, desc))
    cat.register(_legal("CERTIFICATE", "Zertifikat, Bescheinigung, Nachweis, Zeugnis, Konformitätserklärung; test/inspection reports "
                        "(Prüfbericht, Testing report, Messprotokoll, Safety checks) with their check lists", audit=AUDIT_NONE))
    cat.register(_legal("OFFICIAL_LETTER", "Behördenschreiben, Bescheid (not Steuerbescheid), formal letter of a bank, insurer or company to the customer "
                        "about a case (not addressed to a court/ombudsman)",
                        extra=("correspondence_body",)))
    cat.register(_correspondence("LETTER", "Private or personal letter with an addressee, family/friends correspondence (greeting cards and notices are NOTE)",
                                 kpf_type="OTHER"))
    kpf = {"SICK_NOTE": "MEDICAL_DOCUMENT", "TECHNICAL_DOC": "OTHER"}
    for name, desc, audit in (("SICK_NOTE", "Arbeitsunfähigkeitsbescheinigung (AU), Krankmeldung", AUDIT_FULL),
                              ("MEDICAL_DOCUMENT", "Arztbrief, Befund, Rezept, Heil- und Kostenplan", AUDIT_NONE),
                              ("VEHICLE_REGISTRATION", "Zulassungsbescheinigung Teil I/II, Fahrzeugschein, Fahrzeugbrief", AUDIT_NONE),
                              ("TECHNICAL_DOC", "Bedienungsanleitung, Datenblatt, technical description (test reports are CERTIFICATE)", AUDIT_NONE),
                              ("APPLICATION", "Antrag, Formular, Fragebogen, Kündigung, Widerruf, Vollmacht, Erklärung filled in / signed by the user (also a company form the user filled in and sends back)", AUDIT_FULL),
                              ("NOTE", "Notiz, Aktenvermerk, Memo, handschriftliche Notiz, Traueranzeige, Glückwunsch-/Grußkarte, Einladungskarte (private notices and cards: no parties)", AUDIT_NONE),
                              ("OTHER", "Enclosures without letterhead (screenshots, app logs, protocols, printed web pages, transaction "
                               "lists, photos), advertising, leaflets, anything else", AUDIT_NONE)):
        cat.register(_correspondence(name, desc, audit, kpf.get(name)))
    for name in cat.names():
        cat.register(with_contract(cat.get(name)))
    return cat


_DEFAULT: Catalog = build_catalog()


def default_catalog() -> Catalog:
    """Returns the process-wide default catalog."""
    return _DEFAULT
