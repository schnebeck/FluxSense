# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/validate/statement.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Statement and payslip plausibility on the merged tables:
                opening + sum(transactions) = closing (to the cent) with
                sign-error / duplicate-row diagnostics, running balance per
                row, booking dates inside the period, printed debit/credit
                totals and counts, credit card balance (both sign
                conventions), payslip gross - deductions = net.
------------------------------------------------------------------------------
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional

from fluxsense.grounding.normalize import parse_date
from fluxsense.models.result import CheckResult, Severity
from fluxsense.validate.core import CheckContext, dec, failed, money, outcome, passed, rule, skipped

ST = "bodies.statement_body"
CENT = Decimal("0.01")


def _transactions(ctx: CheckContext) -> List[Dict[str, Any]]:
    """Transactions of the statement body."""
    return [t for t in ctx.body("statement_body").get("transactions") or [] if isinstance(t, dict)]


def _pages(rows: List[Dict[str, Any]], indices: List[int]) -> List[int]:
    """Source pages of the given rows."""
    return sorted({int(rows[i]["source_page"]) for i in indices if isinstance(rows[i].get("source_page"), int)})


def _diagnose(diff: Decimal, rows: List[Dict[str, Any]]) -> str:
    """
    Explains a balance difference (closing - computed) by a single row:
    diff = -2 x amount -> sign error, diff = -amount -> duplicate / wrongly included row.
    """
    for i, row in enumerate(rows):
        amount = dec(row.get("amount"))
        if amount is None or amount == 0:
            continue
        if abs(diff + 2 * amount) < CENT:
            return f" Likely sign error in row {i} (amount {money(amount)}, page {row.get('source_page')})."
        if abs(diff + amount) < CENT:
            return f" Row {i} (amount {money(amount)}, page {row.get('source_page')}) may be a duplicate or wrongly included."
    return " A row may be missing or misread."


@rule("statement.balance")
def check_balance(ctx: CheckContext) -> List[CheckResult]:
    """opening + sum(amounts) = closing."""
    body = ctx.body("statement_body")
    rows = _transactions(ctx)
    opening, closing = dec(body.get("opening_balance")), dec(body.get("closing_balance"))
    amounts = [dec(r.get("amount")) for r in rows]
    if opening is None or closing is None or not rows:
        return [skipped("statement.balance", "statement", "opening/closing balance or transactions missing")]
    if any(a is None for a in amounts):
        missing = [i for i, a in enumerate(amounts) if a is None]
        return [failed("statement.balance", "statement", Severity.ERROR, f"{len(missing)} transactions without amount",
                       rows=missing, pages=_pages(rows, missing), table="transactions")]
    total = sum(amounts, Decimal(0))
    expected = opening + total
    diff = closing - expected
    message = (f"opening {money(opening)} + sum of {len(rows)} transactions {money(total)} = {money(expected)}, "
               f"closing balance {money(closing)}")
    if abs(diff) >= CENT:
        message += f" (difference {money(diff)})." + _diagnose(diff, rows)
    return [outcome(abs(diff) < CENT, "statement.balance", "statement", Severity.ERROR, message, field=f"{ST}.closing_balance",
                    expected=money(expected), actual=money(closing), table="transactions",
                    pages=sorted({int(r["source_page"]) for r in rows if isinstance(r.get("source_page"), int)}))]


@rule("statement.running_balance")
def check_running_balance(ctx: CheckContext) -> List[CheckResult]:
    """previous balance + amount = printed balance, row by row."""
    rows = _transactions(ctx)
    previous = dec(ctx.body("statement_body").get("opening_balance"))
    with_balance = [i for i, r in enumerate(rows) if dec(r.get("balance")) is not None]
    if not with_balance:
        return [skipped("statement.running_balance", "statement", "no running balance printed")]
    bad: List[int] = []
    for i, row in enumerate(rows):
        amount, balance = dec(row.get("amount")), dec(row.get("balance"))
        if balance is None:
            if previous is not None and amount is not None:
                previous += amount
            continue
        if previous is not None and amount is not None and abs(previous + amount - balance) >= CENT:
            bad.append(i)
        previous = balance
    message = f"running balance consistent in {len(with_balance) - len(bad)}/{len(with_balance)} rows"
    return [outcome(not bad, "statement.running_balance", "statement", Severity.ERROR, message, rows=bad,
                    pages=_pages(rows, bad), table="transactions")]


@rule("statement.period")
def check_period(ctx: CheckContext) -> List[CheckResult]:
    """Booking dates inside the statement period."""
    body = ctx.body("statement_body")
    start, end = parse_date(body.get("period_start")), parse_date(body.get("period_end"))
    rows = _transactions(ctx)
    if start is None or end is None or not rows:
        return [skipped("statement.period", "statement", "period or transactions missing")]
    out: List[CheckResult] = [outcome(start <= end, "statement.period_order", "statement", Severity.WARNING,
                                      f"period {start} - {end}", field=f"{ST}.period_start")]
    outside = [i for i, r in enumerate(rows) if (d := parse_date(r.get("booking_date"))) is not None and not start <= d <= end]
    out.append(outcome(not outside, "statement.dates_in_period", "statement", Severity.WARNING,
                       f"{len(outside)} booking dates outside {start} - {end}", rows=outside, pages=_pages(rows, outside),
                       table="transactions"))
    return out


@rule("statement.totals")
def check_totals(ctx: CheckContext) -> List[CheckResult]:
    """Printed debit/credit sums and counts."""
    body = ctx.body("statement_body")
    rows = _transactions(ctx)
    amounts = [a for a in (dec(r.get("amount")) for r in rows) if a is not None]
    debits = [a for a in amounts if a < 0]
    credits = [a for a in amounts if a > 0]
    out: List[CheckResult] = []
    for key, values, label in (("total_debits", debits, "debits"), ("total_credits", credits, "credits")):
        printed = dec(body.get(key))
        if printed is not None:
            got = abs(sum(values, Decimal(0)))
            out.append(outcome(abs(abs(printed) - got) < CENT, f"statement.{key}", "statement", Severity.ERROR,
                               f"sum of {len(values)} {label} {money(got)} vs printed {money(abs(printed))}",
                               field=f"{ST}.{key}", expected=money(abs(printed)), actual=money(got), table="transactions"))
    for key, values, label in (("debit_count", debits, "debits"), ("credit_count", credits, "credits")):
        printed_count = body.get(key)
        if isinstance(printed_count, int):
            out.append(outcome(printed_count == len(values), f"statement.{key}", "statement", Severity.ERROR,
                               f"{len(values)} {label} extracted, {printed_count} printed", field=f"{ST}.{key}",
                               expected=str(printed_count), actual=str(len(values)), table="transactions"))
    return out or [skipped("statement.totals", "statement", "no printed totals")]


@rule("statement.credit_card")
def check_credit_card(ctx: CheckContext) -> List[CheckResult]:
    """previous balance + charges - payments = new balance (either sign convention)."""
    body = ctx.body("statement_body")
    rows = _transactions(ctx)
    opening, closing = dec(body.get("opening_balance")), dec(body.get("closing_balance"))
    amounts = [dec(r.get("amount")) for r in rows]
    if opening is None or closing is None or not rows or any(a is None for a in amounts):
        return [skipped("statement.credit_card_balance", "statement", "balances or amounts missing")]
    total = sum(amounts, Decimal(0))
    as_account = opening + total
    as_debt = opening - total
    # The debt convention (balances = amount owed, positive) is only valid if both balances are non-negative;
    # mixed signs must satisfy the signed-account formula.
    debt_ok = opening >= 0 and closing >= 0 and abs(as_debt - closing) < CENT
    ok = abs(as_account - closing) < CENT or debt_ok
    charges = abs(sum((a for a in amounts if a < 0), Decimal(0)))
    payments = sum((a for a in amounts if a > 0), Decimal(0))
    message = (f"previous {money(opening)} + charges {money(charges)} - payments {money(payments)} = {money(as_debt)} "
               f"(or {money(as_account)} if balances are signed), new balance {money(closing)}")
    if not ok:
        message += "." + _diagnose(closing - as_account, rows)
    checks = [outcome(ok, "statement.credit_card_balance", "statement", Severity.ERROR, message, field=f"{ST}.closing_balance",
                      expected=money(as_debt), actual=money(closing), table="transactions")]
    if opening * closing < 0:
        checks.append(failed("statement.sign_convention", "statement", Severity.WARNING,
                             f"opening {money(opening)} and closing {money(closing)} have opposite signs - "
                             f"credit card balances should use one sign convention",
                             field=f"{ST}.opening_balance"))
    return checks


@rule("payslip.net")
def check_payslip(ctx: CheckContext) -> List[CheckResult]:
    """gross - (income tax + soli + church tax + social security) = net; payout <= net."""
    body = ctx.body("payslip_body")
    gross, net = dec(body.get("gross_pay")), dec(body.get("net_pay"))
    income_tax, social = dec(body.get("income_tax")), dec(body.get("social_security_total"))
    out: List[CheckResult] = []
    if gross is not None and net is not None and income_tax is not None and social is not None:
        deductions = income_tax + social + (dec(body.get("solidarity_surcharge")) or 0) + (dec(body.get("church_tax")) or 0)
        expected = gross - abs(deductions)
        out.append(outcome(abs(expected - net) <= ctx.tolerance, "payslip.gross_minus_deductions", "payslip", Severity.ERROR,
                           f"gross {money(gross)} - deductions {money(abs(deductions))} = {money(expected)}, net {money(net)}",
                           field="bodies.payslip_body.net_pay", expected=money(expected), actual=money(net)))
    payout: Optional[Decimal] = dec(body.get("payout_amount"))
    if payout is not None and gross is not None:
        out.append(outcome(payout <= gross, "payslip.payout_le_gross", "payslip", Severity.WARNING,
                           f"payout {money(payout)} vs gross {money(gross)}", field="bodies.payslip_body.payout_amount"))
    return out or [skipped("payslip.gross_minus_deductions", "payslip", "pay components missing")]
