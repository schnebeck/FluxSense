# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/validate/finance.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Invoice plausibility: net + tax = gross, tax breakdown sums
                and per-row VAT, sum of line items vs. line total / net (per
                tax rate), quantity x unit price = line total, German VAT
                rates (incl. the 2020 reduction), currency codes.
------------------------------------------------------------------------------
"""

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Dict, List, Optional, Set

from fluxsense.grounding.normalize import parse_date
from fluxsense.models.result import CheckResult, Severity
from fluxsense.validate.core import CheckContext, dec, failed, money, outcome, passed, rule, skipped

FIN = "bodies.finance_body"
DE_RATES: Set[Decimal] = {Decimal("0"), Decimal("7"), Decimal("19")}
DE_RATES_2020: Set[Decimal] = {Decimal("0"), Decimal("5"), Decimal("16")}
ISO_CURRENCIES: Set[str] = {
    "EUR", "USD", "GBP", "CHF", "JPY", "CNY", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RON", "BGN", "HRK", "TRY",
    "CAD", "AUD", "NZD", "INR", "BRL", "MXN", "ZAR", "RUB", "UAH", "ISK", "HKD", "SGD", "KRW", "THB", "ILS", "AED",
}


def _totals(ctx: CheckContext) -> Dict[str, Optional[Decimal]]:
    """Monetary summation values as Decimals."""
    ms = ctx.body("finance_body").get("monetary_summation") or {}
    return {k: dec(ms.get(k)) for k in ("line_total_amount", "charge_total_amount", "allowance_total_amount",
                                         "tax_basis_total_amount", "tax_total_amount", "grand_total_amount", "due_payable_amount")}


def _net(t: Dict[str, Optional[Decimal]], rows: List[Dict]) -> tuple:
    """Net amount: printed tax basis, else line total +/- charges/allowances, else sum of breakdown bases."""
    if t["tax_basis_total_amount"] is not None:
        return t["tax_basis_total_amount"], "tax basis"
    if t["line_total_amount"] is not None:
        return t["line_total_amount"] + (t["charge_total_amount"] or 0) - (t["allowance_total_amount"] or 0), "line total"
    bases = [dec(r.get("tax_basis_amount")) for r in rows]
    if bases and all(b is not None for b in bases):
        return sum(bases, Decimal(0)), "sum of tax bases"
    return None, ""


@rule("invoice.arithmetic")
def check_invoice_arithmetic(ctx: CheckContext) -> List[CheckResult]:
    """Net + tax = gross, breakdown sums, per-row VAT, due <= gross."""
    name = "arithmetic"
    out: List[CheckResult] = []
    t = _totals(ctx)
    tol = ctx.tolerance
    rows = ctx.body("finance_body").get("tax_breakdown") or []
    net, net_source = _net(t, rows)
    tax, gross = t["tax_total_amount"], t["grand_total_amount"]
    if tax is None and rows and all(dec(r.get("tax_amount")) is not None for r in rows):
        tax = sum((dec(r.get("tax_amount")) for r in rows), Decimal(0))
    if net is not None and tax is not None and gross is not None:
        expected = net + tax
        if abs(expected - gross) > tol:
            out.append(failed("invoice.net_plus_tax", name, Severity.ERROR,
                              f"Net {money(net)} ({net_source}) + tax {money(tax)} = {money(expected)}, but grand total is {money(gross)}",
                              field=f"{FIN}.monetary_summation.grand_total_amount", expected=money(expected), actual=money(gross)))
        else:
            out.append(passed("invoice.net_plus_tax", name, f"{money(net)} + {money(tax)} = {money(gross)}"))
    else:
        out.append(skipped("invoice.net_plus_tax", name, "net, tax or gross not printed"))

    if rows:
        basis_sum = sum((dec(r.get("tax_basis_amount")) or Decimal(0) for r in rows), Decimal(0))
        tax_sum = sum((dec(r.get("tax_amount")) or Decimal(0) for r in rows), Decimal(0))
        if net is not None:
            out.append(outcome(
                abs(basis_sum - net) <= tol, "invoice.breakdown_basis_sum", name, Severity.ERROR,
                f"Sum of tax bases {money(basis_sum)} vs net {money(net)}",
                field=f"{FIN}.tax_breakdown", expected=money(net), actual=money(basis_sum)))
        if tax is not None:
            out.append(outcome(
                abs(tax_sum - tax) <= tol, "invoice.breakdown_tax_sum", name, Severity.ERROR,
                f"Sum of tax amounts {money(tax_sum)} vs tax total {money(tax)}",
                field=f"{FIN}.tax_breakdown", expected=money(tax), actual=money(tax_sum)))
        bad_rows = []
        for i, r in enumerate(rows):
            basis, rate, amount = dec(r.get("tax_basis_amount")), dec(r.get("tax_rate")), dec(r.get("tax_amount"))
            if basis is None or rate is None or amount is None:
                continue
            expected = (basis * rate / Decimal(100)).quantize(Decimal("0.01"))
            if abs(expected - amount) > tol:
                bad_rows.append(i)
                out.append(failed("invoice.vat_row", name, Severity.ERROR,
                                  f"{money(rate)} % of {money(basis)} = {money(expected)}, but printed tax is {money(amount)}",
                                  field=f"{FIN}.tax_breakdown.{i}.tax_amount", expected=money(expected), actual=money(amount),
                                  rows=[i], table="tax_breakdown"))
        if not bad_rows:
            out.append(passed("invoice.vat_row", name, "tax amount = basis x rate for every rate"))
    elif net is not None and tax is not None and net != 0:
        implied = (tax / net * Decimal(100)).quantize(Decimal("0.1"))
        out.append(passed("invoice.implied_rate", name, f"implied VAT rate {implied} %"))

    due = t["due_payable_amount"]
    if due is not None and gross is not None and due > gross + tol:
        out.append(failed("invoice.due_le_gross", name, Severity.WARNING,
                          f"Amount due {money(due)} exceeds grand total {money(gross)}",
                          field=f"{FIN}.monetary_summation.due_payable_amount", expected=f"<= {money(gross)}", actual=money(due)))
    return out


@rule("invoice.line_items")
def check_line_items(ctx: CheckContext) -> List[CheckResult]:
    """Quantity x price = line total; sum of lines = line total / net (per rate)."""
    name = "line_items"
    out: List[CheckResult] = []
    items = ctx.body("finance_body").get("line_items") or []
    if not items:
        return [skipped("invoice.line_sum", name, "no line items")]
    tol = ctx.tolerance
    bad: List[int] = []
    for i, item in enumerate(items):
        qty, price, total = dec(item.get("quantity")), dec(item.get("unit_price")), dec(item.get("total_price"))
        if qty is None or price is None or total is None:
            continue
        expected = (qty * price).quantize(Decimal("0.01"))
        if abs(expected - total) > max(tol, abs(total) * Decimal("0.005")):
            bad.append(i)
    pages = sorted({int(items[i]["source_page"]) for i in bad if isinstance(items[i].get("source_page"), int)})
    if bad:
        out.append(failed("invoice.qty_x_price", name, Severity.ERROR,
                          f"quantity x unit price != line total in {len(bad)} rows", rows=bad, pages=pages,
                          table="line_items", field=f"{FIN}.line_items"))
    else:
        out.append(passed("invoice.qty_x_price", name, "quantity x unit price = line total for all rows"))

    totals = [dec(i.get("total_price")) for i in items]
    if any(v is None for v in totals):
        out.append(skipped("invoice.line_sum", name, "some line totals missing"))
        return out
    line_sum = sum(totals, Decimal(0))
    t = _totals(ctx)
    target_name, target = "line_total_amount", t["line_total_amount"]
    if target is None and t["tax_basis_total_amount"] is not None:
        target_name = "tax_basis_total_amount"
        target = t["tax_basis_total_amount"] - (t["charge_total_amount"] or 0) + (t["allowance_total_amount"] or 0)
    if target is None:
        out.append(skipped("invoice.line_sum", name, "no printed line total or net"))
    elif abs(line_sum - target) > tol:
        out.append(failed("invoice.line_sum", name, Severity.ERROR,
                          f"Sum of {len(items)} line totals {money(line_sum)} != {target_name} {money(target)} "
                          f"(difference {money(line_sum - target)})", field=f"{FIN}.line_items",
                          expected=money(target), actual=money(line_sum), table="line_items",
                          pages=sorted({int(i["source_page"]) for i in items if isinstance(i.get("source_page"), int)})))
    else:
        out.append(passed("invoice.line_sum", name, f"Sum of {len(items)} line totals = {money(target)}"))

    breakdown = ctx.body("finance_body").get("tax_breakdown") or []
    if len(breakdown) > 1 and all(dec(i.get("tax_rate")) is not None for i in items):
        per_rate: Dict[Decimal, Decimal] = defaultdict(Decimal)
        for item in items:
            per_rate[dec(item.get("tax_rate")).normalize()] += dec(item.get("total_price"))
        for row in breakdown:
            rate, basis = dec(row.get("tax_rate")), dec(row.get("tax_basis_amount"))
            if rate is None or basis is None:
                continue
            got = per_rate.get(rate.normalize(), Decimal(0))
            if abs(got - basis) > tol:
                out.append(failed("invoice.line_sum_per_rate", name, Severity.ERROR,
                                  f"Lines at {money(rate)} % sum to {money(got)}, tax basis printed {money(basis)}",
                                  expected=money(basis), actual=money(got), table="line_items"))
    return out


def _is_german(ctx: CheckContext) -> bool:
    """Decides whether German VAT rates apply (seller in DE)."""
    sender = ctx.get("meta_header.sender") or {}
    country = str(sender.get("country") or "").strip().upper()
    vat_id = str(sender.get("vat_id") or "").strip().upper()
    if country:
        return country in ("DE", "DEU", "GERMANY", "DEUTSCHLAND")
    if vat_id:
        return vat_id.startswith("DE")
    return ctx.settings.country.upper() == "DE"


def _valid_rates(invoice_date: Optional[date]) -> Set[Decimal]:
    """German VAT rates valid at a date."""
    if invoice_date and date(2020, 7, 1) <= invoice_date <= date(2020, 12, 31):
        return DE_RATES_2020 | DE_RATES
    return DE_RATES


@rule("invoice.tax_rates")
def check_tax_rates(ctx: CheckContext) -> List[CheckResult]:
    """Tax rates must be legal German rates (0, 7, 19; 5/16 in H2 2020) with tolerance."""
    name = "tax_rates"
    if not _is_german(ctx):
        return [skipped("invoice.tax_rate_de", name, "seller not in Germany")]
    body = ctx.body("finance_body")
    valid = _valid_rates(parse_date(body.get("invoice_date") or ctx.get("meta_header.doc_date")))
    rates = [(f"{FIN}.tax_breakdown.{i}.tax_rate", dec(r.get("tax_rate"))) for i, r in enumerate(body.get("tax_breakdown") or [])]
    rates += [(f"{FIN}.line_items.{i}.tax_rate", dec(r.get("tax_rate"))) for i, r in enumerate(body.get("line_items") or [])]
    rates = [(p, r) for p, r in rates if r is not None]
    t = _totals(ctx)
    if not rates and t["tax_basis_total_amount"] and t["tax_total_amount"] is not None and t["tax_basis_total_amount"] != 0:
        implied = t["tax_total_amount"] / t["tax_basis_total_amount"] * Decimal(100)
        rates = [(f"{FIN}.monetary_summation.tax_total_amount", implied)]
    if not rates:
        return [skipped("invoice.tax_rate_de", name, "no tax rate information")]
    bad = [(path, rate) for path, rate in rates if abs(min(valid, key=lambda v: abs(v - rate)) - rate) > Decimal("0.1")]
    if not bad:
        return [passed("invoice.tax_rate_de", name, "all VAT rates valid")]
    item_rows = [int(path.split(".")[3]) for path, _ in bad if ".line_items." in path]
    distinct = sorted({f"{rate:.2f}" for _, rate in bad})
    return [failed("invoice.tax_rate_de", name, Severity.ERROR,
                   f"{len(bad)} VAT rates {distinct} % are not German rates {sorted(int(v) for v in valid)}",
                   field=bad[0][0], actual=", ".join(distinct), rows=item_rows, table="line_items" if item_rows else None)]


@rule("currency")
def check_currency(ctx: CheckContext) -> List[CheckResult]:
    """Currency must be an ISO 4217 code."""
    currencies = [(f"bodies.{k}.currency", v.get("currency")) for k, v in (ctx.fields.get("bodies") or {}).items()
                  if isinstance(v, dict) and v.get("currency")]
    out: List[CheckResult] = []
    for path, cur in currencies:
        code = str(cur).strip().upper()
        if code not in ISO_CURRENCIES:
            out.append(failed("currency.iso4217", "currency", Severity.WARNING, f"Unknown currency code '{cur}'", field=path, actual=str(cur)))
    return out or [passed("currency.iso4217", "currency", "currency codes valid")]
