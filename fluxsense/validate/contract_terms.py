# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/validate/contract_terms.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Contract payment terms and terms of duration (field contract
                §10): the printed components of a recurring payment must add
                up to the printed total (deterministic check, like the
                invoice arithmetic), and a minimum term / mutual waiver of
                termination the model put into notice_period is moved to
                minimum_term (a waiver is never the period of notice).
------------------------------------------------------------------------------
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set

from fluxsense.models.result import CheckResult, CheckStatus, Severity

LEGAL = "bodies.legal_body"
PAYMENT = f"{LEGAL}.payment_terms"
CHECK_PAYMENT_SUM = "contract.payment_sum"
# Wording of a minimum term / waiver of ordinary termination (not a period of notice).
WAIVER_RE = re.compile(r"(verzicht\w*|kündigungsverzicht|ausschluss\s+der\s+(?:ordentlichen\s+)?kündigung|"
                       r"kündigung\s+(?:ist\s+)?ausgeschlossen|mindestlaufzeit|mindestvertragslaufzeit|feste\s+laufzeit|"
                       r"erstmal(?:ig|s)\s+(?:nach|zum)|minimum\s+term|waive)", re.IGNORECASE)
MUTUAL_RE = re.compile(r"(wechselseitig|beiderseitig|gegenseitig|beide\s+(?:vertrags)?parteien|die\s+vertragsparteien|mutual)",
                       re.IGNORECASE)
_DURATION_RE = re.compile(r"(\d{1,3})\s*(jahr\w*|monat\w*|years?|months?)", re.IGNORECASE)


def _dec(value: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(value)) if value not in (None, "") else None
    except InvalidOperation:
        return None


def _legal(fields: Dict[str, Any]) -> Dict[str, Any]:
    body = (fields.get("bodies") or {}).get("legal_body")
    return body if isinstance(body, dict) else {}


def payment_components_sum(fields: Dict[str, Any]) -> Optional[bool]:
    """True / False if two or more printed components can be compared with the total; None if not checkable."""
    payment = _legal(fields).get("payment_terms") or {}
    total = _dec(payment.get("amount"))
    amounts = [_dec(c.get("amount")) for c in payment.get("components") or [] if isinstance(c, dict)]
    if total is None or len(amounts) < 2 or any(a is None for a in amounts):
        return None
    return abs(sum(amounts, Decimal(0)) - total) <= Decimal("0.011")  # type: ignore[arg-type]


def payment_checked_paths(fields: Dict[str, Any]) -> Set[str]:
    """Paths confirmed by the component sum (the total and every component amount)."""
    if payment_components_sum(fields) is not True:
        return set()
    count = len(_legal(fields)["payment_terms"]["components"])
    return {f"{PAYMENT}.amount"} | {f"{PAYMENT}.components.{i}.amount" for i in range(count)}


def check_contract_payment(fields: Dict[str, Any]) -> List[CheckResult]:
    """'contract.payment_sum': the printed components of the recurring payment add up to the printed total."""
    result = payment_components_sum(fields)
    if result is None:
        return []
    payment = _legal(fields)["payment_terms"]
    amounts = [_dec(c.get("amount")) for c in payment["components"]]
    total = sum(amounts, Decimal(0))  # type: ignore[arg-type]
    if result:
        return [CheckResult(check_id=CHECK_PAYMENT_SUM, rule="contract", status=CheckStatus.PASS, severity=Severity.WARNING,
                            field=f"{PAYMENT}.amount", message=f"components add up to the total {payment['amount']}")]
    return [CheckResult(check_id=CHECK_PAYMENT_SUM, rule="contract", status=CheckStatus.FAIL, severity=Severity.WARNING,
                        field=f"{PAYMENT}.amount", expected=str(total), actual=str(payment.get("amount")),
                        message=f"payment components add up to {total}, the printed total is {payment.get('amount')}")]


def fix_minimum_term(fields: Dict[str, Any]) -> List[str]:
    """
    A minimum term / waiver of termination extracted as notice_period is moved
    to minimum_term (value and unit from the clause; mutual_waiver if both
    parties waive); notice_period is cleared.

    Returns:
        Notes about the move.
    """
    legal = _legal(fields)
    notice = legal.get("notice_period")
    if not isinstance(notice, dict):
        return []
    text = str(notice.get("original_text") or "")
    if not text or not WAIVER_RE.search(text):
        return []
    term: Dict[str, Any] = dict(legal.get("minimum_term") or {})
    duration = _DURATION_RE.search(text)
    if duration:
        term.setdefault("value", int(duration.group(1)))
        term.setdefault("unit", "YEARS" if duration.group(2).lower().startswith(("jahr", "year")) else "MONTHS")
    elif notice.get("value") and notice.get("unit"):
        term.setdefault("value", notice["value"])
        term.setdefault("unit", notice["unit"])
    term.setdefault("mutual_waiver", bool(MUTUAL_RE.search(text)))
    term.setdefault("original_text", text)
    legal["minimum_term"] = term
    legal["notice_period"] = {}
    return [f"notice_period {text[:60]!r} is a minimum term / termination waiver: moved to minimum_term (§10)"]
