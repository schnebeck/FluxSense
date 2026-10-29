# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_validators.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for the deterministic plausibility rules (invoices, VAT, IBAN, dates, completeness, statements, credit cards, payslips, identity).
------------------------------------------------------------------------------
"""

import copy
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from fluxsense.catalog import default_catalog
from fluxsense.config import PlausibilitySettings
from fluxsense.models.identity import IdentityProfile
from fluxsense.models.result import CheckStatus
from fluxsense.validate import run_plausibility
from fluxsense.validate.core import CheckContext
from fluxsense.validate.general import validate_bic, validate_iban

GOOD_INVOICE: Dict[str, Any] = {
    "meta_header": {"sender": {"company": "Muster GmbH", "vat_id": "DE123456789", "country": "DE"},
                    "recipient": {"name": "Erika Beispiel"}, "doc_date": "2026-09-14"},
    "bodies": {"finance_body": {
        "invoice_number": "R-1", "invoice_date": "2026-09-14", "due_date": "2026-09-28", "currency": "EUR",
        "line_items": [{"quantity": 4, "unit_price": 25, "total_price": 100, "tax_rate": 19},
                       {"quantity": 1, "unit_price": 50, "total_price": 50, "tax_rate": 7}],
        "monetary_summation": {"line_total_amount": 150, "tax_basis_total_amount": 150, "tax_total_amount": "22.50",
                               "grand_total_amount": "172.50"},
        "tax_breakdown": [{"tax_basis_amount": 100, "tax_amount": 19, "tax_rate": 19},
                          {"tax_basis_amount": 50, "tax_amount": "3.50", "tax_rate": 7}],
        "payment_accounts": [{"iban": "DE44990123450532013000", "bic": "XMPLDEFFXXX"}]}},
}


def run(fields: Dict[str, Any], type_name: str = "INVOICE", **kwargs: Any) -> Dict[str, List[Any]]:
    """Runs the type's rules and groups the checks by id."""
    spec = default_catalog().get(type_name)
    ctx = CheckContext(fields=fields, spec=spec, settings=PlausibilitySettings(), today=date(2026, 10, 1), **kwargs)
    report = run_plausibility(ctx, [spec])
    grouped: Dict[str, List[Any]] = {}
    for check in report.checks:
        grouped.setdefault(check.check_id, []).append(check)
    grouped["_report"] = [report]
    return grouped


def failed(grouped: Dict[str, List[Any]]) -> List[str]:
    """Failed check ids."""
    return sorted({c.check_id for k, v in grouped.items() if k != "_report" for c in v if c.status == CheckStatus.FAIL})


def test_good_invoice_passes() -> None:
    """A consistent two-rate invoice passes every rule."""
    grouped = run(copy.deepcopy(GOOD_INVOICE))
    assert failed(grouped) == []
    assert grouped["_report"][0].status == "PASS" and grouped["_report"][0].score == 1.0


def test_wrong_vat_sum_detected() -> None:
    """Printed VAT that does not match the rate and the gross total is reported."""
    fields = copy.deepcopy(GOOD_INVOICE)
    fin = fields["bodies"]["finance_body"]
    fin["monetary_summation"]["tax_total_amount"] = "24.50"
    fin["tax_breakdown"][0]["tax_amount"] = 21
    codes = failed(run(fields))
    assert "invoice.net_plus_tax" in codes and "invoice.vat_row" in codes
    assert run(fields)["_report"][0].status == "FAIL"


def test_line_items_checks() -> None:
    """qty x price mismatch and sum of lines vs. net are reported with rows."""
    fields = copy.deepcopy(GOOD_INVOICE)
    fin = fields["bodies"]["finance_body"]
    fin["line_items"][1]["total_price"] = 60
    fin["line_items"][1]["source_page"] = 2
    grouped = run(fields)
    assert grouped["invoice.qty_x_price"][0].rows == [1] and grouped["invoice.qty_x_price"][0].pages == [2]
    assert "invoice.line_sum" in failed(grouped) and "invoice.line_sum_per_rate" in failed(grouped)


def test_net_from_line_total_when_basis_missing() -> None:
    """net + tax = gross also works when only the line total is printed."""
    fields = copy.deepcopy(GOOD_INVOICE)
    del fields["bodies"]["finance_body"]["monetary_summation"]["tax_basis_total_amount"]
    fields["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] = 175
    assert "invoice.net_plus_tax" in failed(run(fields))


@pytest.mark.parametrize("rate,invoice_date,ok", [("19", "2026-01-01", True), ("16", "2026-01-01", False),
                                                   ("16", "2020-08-01", True), ("19.5", "2026-01-01", False)])
def test_german_vat_rates(rate: str, invoice_date: str, ok: bool) -> None:
    """German rates incl. the 2020 reduction."""
    fields = {"meta_header": {"sender": {"country": "DE"}}, "bodies": {"finance_body": {
        "invoice_date": invoice_date, "tax_breakdown": [{"tax_basis_amount": 100, "tax_amount": Decimal(rate), "tax_rate": rate}]}}}
    assert ("invoice.tax_rate_de" in failed(run(fields))) is not ok


def test_foreign_seller_skips_german_rates() -> None:
    """Rates are not checked for sellers outside Germany."""
    fields = {"meta_header": {"sender": {"country": "FR"}}, "bodies": {"finance_body": {
        "tax_breakdown": [{"tax_basis_amount": 100, "tax_amount": 20, "tax_rate": 20}]}}}
    assert run(fields)["invoice.tax_rate_de"][0].status == CheckStatus.SKIP


def test_iban_and_bic() -> None:
    """ISO 13616 checksum, country length and BIC format."""
    assert validate_iban("DE44 9901 2345 0532 0130 00")
    assert not validate_iban("DE44 9901 2345 0532 0130 01")
    assert not validate_iban("DE44990123450532013")
    assert validate_bic("XMPLDEFFXXX") and validate_bic("XMPLDEFF") and not validate_bic("COBA-DE")
    fields = copy.deepcopy(GOOD_INVOICE)
    fields["bodies"]["finance_body"]["payment_accounts"][0]["iban"] = "DE44990123450532013001"
    assert "iban.checksum" in failed(run(fields))


def test_dates() -> None:
    """Due before issue date, future dates and invalid formats."""
    fields = copy.deepcopy(GOOD_INVOICE)
    fin = fields["bodies"]["finance_body"]
    fin["due_date"] = "2026-09-01"
    fin["invoice_date"] = "2026-09-14"
    fields["meta_header"]["doc_date"] = "2027-05-01"
    fin["order_date"] = "31.02.2026"
    codes = failed(run(fields))
    assert {"dates.order", "dates.not_future", "dates.format"} <= set(codes)


def test_completeness_and_vat_id_requirement() -> None:
    """Required fields per type and §14 UStG VAT-ID check above 250 EUR."""
    fields = copy.deepcopy(GOOD_INVOICE)
    del fields["bodies"]["finance_body"]["invoice_number"]
    del fields["meta_header"]["sender"]["vat_id"]
    fields["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"] = "172.50"
    assert "completeness.required" in failed(run(fields))
    fin = fields["bodies"]["finance_body"]
    fin["monetary_summation"].update({"tax_basis_total_amount": 300, "line_total_amount": 300, "grand_total_amount": 357,
                                      "tax_total_amount": 57})
    assert "completeness.vat_id" in failed(run(fields))


def statement(transactions: List[Dict[str, Any]], opening: str, closing: str, **extra: Any) -> Dict[str, Any]:
    """Statement fields."""
    body = {"opening_balance": opening, "closing_balance": closing, "period_start": "2026-09-01", "period_end": "2026-09-30",
            "transactions": transactions}
    body.update(extra)
    return {"meta_header": {}, "bodies": {"statement_body": body}}


ROWS = [{"booking_date": "2026-09-02", "amount": "-10.00", "balance": "90.00", "source_page": 1},
        {"booking_date": "2026-09-05", "amount": "250.00", "balance": "340.00", "source_page": 1},
        {"booking_date": "2026-09-20", "amount": "-40.50", "balance": "299.50", "source_page": 2}]


def test_statement_balance_and_running_balance_pass() -> None:
    """opening + sum = closing, running balance, period and printed totals."""
    fields = statement(copy.deepcopy(ROWS), "100.00", "299.50", total_debits="50.50", total_credits="250.00",
                       debit_count=2, credit_count=1)
    assert failed(run(fields, "BANK_STATEMENT")) == []


def test_statement_sign_error_diagnosed() -> None:
    """A flipped sign is diagnosed with row and page; running balance breaks at that row."""
    rows = copy.deepcopy(ROWS)
    rows[2]["amount"] = "40.50"
    grouped = run(statement(rows, "100.00", "299.50"), "BANK_STATEMENT")
    assert "sign error in row 2" in grouped["statement.balance"][0].message
    assert grouped["statement.running_balance"][0].rows == [2] and grouped["statement.running_balance"][0].pages == [2]


def test_statement_missing_row_and_dates_outside_period() -> None:
    """Missing booking breaks the balance; a date outside the period is flagged."""
    rows = copy.deepcopy(ROWS[:2])
    rows[1]["booking_date"] = "2026-10-03"
    grouped = run(statement(rows, "100.00", "299.50", debit_count=2), "BANK_STATEMENT")
    codes = failed(grouped)
    assert {"statement.balance", "statement.dates_in_period", "statement.debit_count"} <= set(codes)


def test_credit_card_balance_both_conventions() -> None:
    """previous + charges - payments = new (debt positive) and signed balances both pass."""
    rows = [{"amount": "-120.00"}, {"amount": "-30.00"}, {"amount": "100.00"}]
    assert failed(run(statement(rows, "200.00", "250.00"), "CREDIT_CARD_STATEMENT")) == []
    assert failed(run(statement(rows, "-200.00", "-250.00"), "CREDIT_CARD_STATEMENT")) == []
    assert "statement.credit_card_balance" in failed(run(statement(rows, "200.00", "260.00"), "CREDIT_CARD_STATEMENT"))


def test_payslip_net() -> None:
    """gross - taxes - social security = net."""
    body = {"gross_pay": "4000.00", "income_tax": "600.00", "solidarity_surcharge": "0", "church_tax": "40.00",
            "social_security_total": "820.00", "net_pay": "2540.00", "payout_amount": "2540.00"}
    fields = {"meta_header": {}, "bodies": {"payslip_body": body}}
    assert failed(run(fields, "PAYSLIP")) == []
    body["net_pay"] = "2640.00"
    assert "payslip.gross_minus_deductions" in failed(run(fields, "PAYSLIP"))


def test_identity_direction_and_own_iban() -> None:
    """INBOUND while the user is the sender, and the user's IBAN as payee, are suspicious."""
    me = IdentityProfile(name="Erika Beispiel", address_keywords=["00115", "Lindenweg"], iban=["DE44990123450532013000"])
    fields = copy.deepcopy(GOOD_INVOICE)
    fields["meta_header"]["sender"] = {"name": "Erika Beispiel", "zip_code": "00115"}
    fields["meta_header"]["recipient"] = {"company": "Fremd AG", "zip_code": "00345"}
    codes = failed(run(fields, direction="INBOUND", private_id=me))
    assert "identity.direction" in codes and "identity.own_iban_as_payee" in codes


def test_currency_code() -> None:
    """Unknown currency codes are warnings."""
    fields = copy.deepcopy(GOOD_INVOICE)
    fields["bodies"]["finance_body"]["currency"] = "EURO"
    assert "currency.iso4217" in failed(run(fields))


def test_invalid_line_rates_are_one_finding() -> None:
    """Many invalid per-line rates produce one aggregated finding with the rows."""
    fields = {"meta_header": {"sender": {"country": "DE"}}, "bodies": {"finance_body": {
        "line_items": [{"tax_rate": "0.5"}, {"tax_rate": "5"}, {"tax_rate": "19"}]}}}
    checks = run(fields)["invoice.tax_rate_de"]
    assert len(checks) == 1 and checks[0].rows == [0, 1]


def test_iban_checked_for_every_type() -> None:
    """IBAN and date rules apply to all types (e.g. a letter with bank details)."""
    fields = {"meta_header": {"sender": {"iban": "DE19990123450000000000"}}, "bodies": {}}
    assert "iban.checksum" in failed(run(fields, "LETTER"))


def test_strict_identity_match() -> None:
    """Party decisions need IBAN/VAT ID, the full name, or ZIP + name token; loose prefixes do not count."""
    from fluxsense.stages.identity import strong_identity_match
    me = IdentityProfile(name="Erika Beispiel", address_keywords=["00115", "Lindenweg"], iban=["DE44990123450532013000"])
    assert strong_identity_match("Erika Beispiel Lindenweg 3 00115 Berlin", me)
    assert strong_identity_match("Frau E. Beispiel, 00115 Berlin", me)
    assert strong_identity_match("IBAN DE44 9901 2345 0532 0130 00", me)
    assert not strong_identity_match("Muster Elektro GmbH Industriestraße 7 00173 Stuttgart", me)
    assert not strong_identity_match("Erik Beisp GmbH 00345 Musterstadt", me)


def test_party_harmonization() -> None:
    """Statements: bank = sender; INBOUND with the user as sender swaps; 'Musterbank Musterbank' deduplicated."""
    from fluxsense.stages.identity import harmonize_parties
    me = IdentityProfile(name="Erika Beispiel", address_keywords=["00115"])
    meta = {"sender": {"name": "Erika Beispiel", "zip_code": "00115"}, "recipient": {"company": "Musterbank", "name": "Musterbank"}}
    notes, _ = harmonize_parties(meta, "INBOUND", ["CREDIT_CARD_STATEMENT"], me, None)
    assert meta["sender"] == {"company": "Musterbank"} and meta["recipient"]["name"] == "Erika Beispiel" and len(notes) == 2
    meta = {"sender": {"name": "Erika Beispiel", "zip_code": "00115"}, "recipient": {"company": "Stadtwerke Musterstadt"}}
    notes, direction = harmonize_parties(meta, "INBOUND", ["LETTER"], me, None)
    assert meta["sender"]["name"] == "Erika Beispiel" and direction == "OUTBOUND"  # layout roles trusted, direction fixed
    meta = {"sender": {"company": "Musterbank eG"}, "recipient": {"name": "Erika Beispiel"}}
    assert harmonize_parties(meta, "INBOUND", ["BANK_STATEMENT"], None, None) == ([], "INBOUND")
    meta = {"sender": {"name": "Erika Beispiel"}, "recipient": {"company": "Musterbank eG"}}
    harmonize_parties(meta, "INBOUND", ["BANK_STATEMENT"], None, None)
    assert meta["sender"]["company"] == "Musterbank eG"


def test_credit_card_mixed_signs_fail() -> None:
    """Opening +512.30 with closing -48.17 cannot pass via the debt convention."""
    rows = [{"amount": "-730.02"}]
    assert "statement.sign_convention" in failed(run(statement(rows, "512.30", "-48.17"), "CREDIT_CARD_STATEMENT"))
    assert failed(run(statement(rows, "-512.30", "-1242.32"), "CREDIT_CARD_STATEMENT")) == []
    assert "statement.credit_card_balance" in failed(run(statement(rows, "500.00", "48.17"), "CREDIT_CARD_STATEMENT"))


def test_forms_by_any_person_and_role_labels() -> None:
    """A form filled by a relative (not the user): filler = sender; role labels are no parties."""
    from fluxsense.stages.identity import apply_form_semantics, is_role_label, remove_role_labels
    me = IdentityProfile(name="Erika Beispiel", address_keywords=["00115"])
    meta = {"sender": {"company": "Musterkasse für Beamte"}, "recipient": {"name": "Max Muster", "zip_code": "00345"}}
    swapped, _ = apply_form_semantics(meta, me, None)
    assert swapped and meta["sender"]["name"] == "Max Muster" and meta["recipient"]["company"].startswith("Musterkasse")
    assert is_role_label("Patient/ Patientin") and is_role_label("Privat Versicherte/ Selbstzahler")
    assert not is_role_label("Erika Muster") and not is_role_label("Musterbank eG")
    meta = {"recipient": {"name": "Patient/ Patientin", "city": "Musterstadt"}}
    assert remove_role_labels(meta) and meta["recipient"] == {"city": "Musterstadt"}
