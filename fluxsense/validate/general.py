# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/validate/general.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Type-independent plausibility: IBAN checksums (ISO 13616) and
                BIC format/country, date sanity (format, ranges, order, not in
                the future), required-field completeness per type (incl. the
                German VAT-ID requirement), identity/direction consistency.
------------------------------------------------------------------------------
"""

import re
from datetime import timedelta
from decimal import Decimal
from typing import Any, Iterator, List, Tuple

from fluxsense.grounding.normalize import compact, parse_date
from fluxsense.models.result import CheckResult, Severity
from fluxsense.stages.identity import party_text, strong_identity_match
from fluxsense.validate.core import CheckContext, dec, failed, outcome, passed, rule, skipped

BIC_RE = re.compile(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$")
IBAN_LENGTHS = {"DE": 22, "AT": 20, "CH": 21, "NL": 18, "BE": 16, "FR": 27, "IT": 27, "ES": 24, "LU": 20, "GB": 22,
                "PL": 28, "DK": 18, "SE": 24, "NO": 15, "FI": 18, "IE": 22, "PT": 25, "CZ": 24, "LI": 21}
DATE_ORDER: Tuple[Tuple[str, str], ...] = (
    ("bodies.finance_body.invoice_date", "bodies.finance_body.due_date"),
    ("bodies.finance_body.order_date", "bodies.finance_body.invoice_date"),
    ("bodies.subscription_info.service_period_start", "bodies.subscription_info.service_period_end"),
    ("bodies.statement_body.period_start", "bodies.statement_body.period_end"),
    ("bodies.payslip_body.pay_period_start", "bodies.payslip_body.pay_period_end"),
    ("bodies.legal_body.effective_date", "bodies.legal_body.valid_until"),
)
NOT_FUTURE: Tuple[str, ...] = ("meta_header.doc_date", "bodies.finance_body.invoice_date", "bodies.statement_body.statement_date",
                               "bodies.finance_body.order_date")


def validate_iban(iban: str) -> bool:
    """
    ISO 13616 mod-97 check (ported from KPaperFlux core/utils/validation.py).

    Args:
        iban: IBAN (any spacing).

    Returns:
        True if the checksum and the country length are valid.
    """
    value = compact(iban)
    if not 15 <= len(value) <= 34 or not value[:2].isalpha() or not value[2:4].isdigit():
        return False
    expected = IBAN_LENGTHS.get(value[:2])
    if expected and len(value) != expected:
        return False
    numeric = "".join(str(int(c, 36)) for c in value[4:] + value[:4])
    return int(numeric) % 97 == 1


def validate_bic(bic: str) -> bool:
    """BIC format check (8 or 11 characters)."""
    return bool(BIC_RE.match(compact(bic)))


def _walk(node: Any, path: str = "") -> Iterator[Tuple[str, Any]]:
    """Yields (path, value) for all scalars in a nested structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk(value, f"{path}.{i}")
    else:
        yield path, node


@rule("iban")
def check_ibans(ctx: CheckContext) -> List[CheckResult]:
    """Checksums of all IBANs, BIC format and BIC/IBAN country consistency."""
    out: List[CheckResult] = []
    ibans = [(p, v) for p, v in _walk(ctx.fields) if p.split(".")[-1] in ("iban", "counterparty_iban", "payout_iban") and v]
    for path, value in ibans:
        out.append(outcome(validate_iban(str(value)), "iban.checksum", "iban", Severity.ERROR,
                           f"IBAN {value} {'valid' if validate_iban(str(value)) else 'has an invalid checksum or length'}",
                           field=path, actual=str(value)))
    for path, value in _walk(ctx.fields):
        if path.split(".")[-1] != "bic" or not value:
            continue
        if not validate_bic(str(value)):
            out.append(failed("iban.bic_format", "iban", Severity.WARNING, f"BIC {value} has an invalid format", field=path))
            continue
        sibling = ctx.get(path.rsplit(".", 1)[0] + ".iban")
        if sibling and compact(str(sibling))[:2] != compact(str(value))[4:6]:
            out.append(failed("iban.bic_country", "iban", Severity.INFO,
                              f"BIC country {compact(str(value))[4:6]} differs from IBAN country {compact(str(sibling))[:2]}", field=path))
    return out or [skipped("iban.checksum", "iban", "no IBAN extracted")]


@rule("dates")
def check_dates(ctx: CheckContext) -> List[CheckResult]:
    """Date format, plausible range, order and not-in-future."""
    out: List[CheckResult] = []
    tolerance = timedelta(days=ctx.settings.future_days_tolerance)
    for path, value in _walk(ctx.fields):
        name = path.split(".")[-1]
        if not value or not ("date" in name or name in ("period_start", "period_end", "valid_until", "deadline")):
            continue
        if ".transactions." in path:
            continue
        parsed = parse_date(value)
        if parsed is None:
            out.append(failed("dates.format", "dates", Severity.WARNING, f"'{value}' is not a valid date", field=path))
        elif parsed.year < ctx.settings.min_year or parsed.year > ctx.today.year + 30:
            out.append(failed("dates.range", "dates", Severity.WARNING, f"{parsed} is outside the plausible range", field=path))
    for path in NOT_FUTURE:
        parsed = parse_date(ctx.get(path))
        if parsed is not None and parsed > ctx.today + tolerance:
            out.append(failed("dates.not_future", "dates", Severity.WARNING, f"{parsed} lies in the future", field=path,
                              expected=f"<= {ctx.today}", actual=str(parsed)))
    for first, second in DATE_ORDER:
        a, b = parse_date(ctx.get(first)), parse_date(ctx.get(second))
        if a is not None and b is not None:
            out.append(outcome(a <= b, "dates.order", "dates", Severity.WARNING, f"{first.split('.')[-1]} {a} <= {second.split('.')[-1]} {b}",
                               field=second, expected=f">= {a}", actual=str(b)))
    meta, inv = parse_date(ctx.get("meta_header.doc_date")), parse_date(ctx.get("bodies.finance_body.invoice_date"))
    if meta and inv and meta != inv:
        out.append(failed("dates.doc_vs_invoice", "dates", Severity.INFO, f"document date {meta} differs from invoice date {inv}",
                          field="meta_header.doc_date"))
    return out or [passed("dates.format", "dates", "all dates valid")]


@rule("completeness")
def check_completeness(ctx: CheckContext) -> List[CheckResult]:
    """Required fields per type and the German VAT-ID requirement for invoices > 250 EUR."""
    out: List[CheckResult] = []
    for path in ctx.spec.required_fields:
        value = ctx.get(path)
        out.append(outcome(value not in (None, "", []), "completeness.required", "completeness", Severity.WARNING,
                           f"{path.split('.')[-1]} {'present' if value not in (None, '', []) else 'missing'}", field=path))
    if ctx.spec.body_key == "finance_body" and ctx.spec.name in ("INVOICE", "CREDIT_NOTE", "UTILITY_BILL"):
        sender = ctx.get("meta_header.sender") or {}
        if not (sender.get("name") or sender.get("company")):
            out.append(failed("completeness.seller", "completeness", Severity.WARNING, "seller name missing",
                              field="meta_header.sender"))
        gross = dec(ctx.get("bodies.finance_body.monetary_summation.grand_total_amount"))
        if gross is not None and gross > Decimal(250) and not (sender.get("vat_id") or sender.get("tax_id")):
            out.append(failed("completeness.vat_id", "completeness", Severity.WARNING,
                              "invoice above 250 EUR without seller VAT ID or tax number (§14 UStG)",
                              field="meta_header.sender.vat_id"))
    return out or [skipped("completeness.required", "completeness", "no required fields defined")]


@rule("identity")
def check_identity(ctx: CheckContext) -> List[CheckResult]:
    """Direction must agree with where the user's identity appears; own IBAN as payee is suspicious."""
    profiles = [p for p in (ctx.private_id, ctx.business_id) if p and not p.is_empty()]
    if not profiles:
        return [skipped("identity.direction", "identity", "no identity profile configured")]
    sender = party_text(ctx.get("meta_header.sender"))
    recipient = party_text(ctx.get("meta_header.recipient"))
    user_is_sender = any(strong_identity_match(sender, p) for p in profiles) if sender else False
    user_is_recipient = any(strong_identity_match(recipient, p) for p in profiles) if recipient else False
    out: List[CheckResult] = []
    if ctx.direction == "INBOUND" and user_is_sender and not user_is_recipient:
        out.append(failed("identity.direction", "identity", Severity.WARNING,
                          "direction INBOUND, but the user's identity is the sender", field="direction"))
    elif ctx.direction == "OUTBOUND" and user_is_recipient and not user_is_sender:
        out.append(failed("identity.direction", "identity", Severity.WARNING,
                          "direction OUTBOUND, but the user's identity is the recipient", field="direction"))
    else:
        out.append(passed("identity.direction", "identity", f"direction {ctx.direction} consistent"))
    own = {compact(i) for p in profiles for i in p.iban if i}
    payee = [a.get("iban") for a in ctx.get("bodies.finance_body.payment_accounts") or [] if isinstance(a, dict)]
    payee.append(ctx.get("meta_header.sender.iban"))
    if ctx.direction == "INBOUND" and any(i and compact(str(i)) in own for i in payee):
        out.append(failed("identity.own_iban_as_payee", "identity", Severity.WARNING,
                          "incoming invoice asks for payment to one of the user's own IBANs", field="bodies.finance_body.payment_accounts"))
    return out
