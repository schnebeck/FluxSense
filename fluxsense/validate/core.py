# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/validate/core.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Plausibility framework: check context, rule registry, check
                helpers and the scored PlausibilityReport.
------------------------------------------------------------------------------
"""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional

from fluxsense.catalog.registry import TypeSpec
from fluxsense.config import PlausibilitySettings
from fluxsense.grounding.normalize import to_decimal
from fluxsense.models.identity import IdentityProfile
from fluxsense.models.result import CheckResult, CheckStatus, PlausibilityReport, Severity

SEVERITY_WEIGHT: Dict[Severity, float] = {Severity.ERROR: 0.25, Severity.WARNING: 0.08, Severity.INFO: 0.02}


@dataclass
class CheckContext:
    """Inputs of all plausibility rules for one logical document."""

    fields: Dict[str, Any]
    spec: TypeSpec
    settings: PlausibilitySettings
    direction: str = "UNKNOWN"
    tenant_context: str = "UNKNOWN"
    private_id: Optional[IdentityProfile] = None
    business_id: Optional[IdentityProfile] = None
    today: date = field(default_factory=date.today)

    def get(self, path: str) -> Any:
        """Dotted-path lookup in the fields (list indices allowed)."""
        node: Any = self.fields
        for part in path.split("."):
            if isinstance(node, dict):
                node = node.get(part)
            elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
                node = node[int(part)]
            else:
                return None
        return node

    def body(self, key: str) -> Dict[str, Any]:
        """Returns a body dict (empty if absent)."""
        value = (self.fields.get("bodies") or {}).get(key)
        return value if isinstance(value, dict) else {}

    @property
    def tolerance(self) -> Decimal:
        """Absolute amount tolerance."""
        return Decimal(str(self.settings.amount_tolerance))


def dec(value: Any) -> Optional[Decimal]:
    """Safe Decimal conversion."""
    try:
        return to_decimal(value)
    except (InvalidOperation, ValueError):
        return None


def money(value: Optional[Decimal]) -> str:
    """Formats a Decimal for messages."""
    return "None" if value is None else f"{value:.2f}"


Rule = Callable[[CheckContext], List[CheckResult]]
RULES: Dict[str, Rule] = {}


def rule(rule_id: str) -> Callable[[Rule], Rule]:
    """Decorator registering a rule under an id."""
    def decorator(func: Rule) -> Rule:
        RULES[rule_id] = func
        return func
    return decorator


def passed(check_id: str, rule_name: str, message: str = "", **kwargs: Any) -> CheckResult:
    """Builds a PASS result."""
    return CheckResult(check_id=check_id, rule=rule_name, status=CheckStatus.PASS, message=message, **kwargs)


def failed(check_id: str, rule_name: str, severity: Severity, message: str, **kwargs: Any) -> CheckResult:
    """Builds a FAIL result."""
    return CheckResult(check_id=check_id, rule=rule_name, status=CheckStatus.FAIL, severity=severity, message=message, **kwargs)


def outcome(ok: bool, check_id: str, rule_name: str, severity: Severity, message: str, **kwargs: Any) -> CheckResult:
    """PASS if ok else FAIL with the given severity."""
    if ok:
        return passed(check_id, rule_name, message, **kwargs)
    return failed(check_id, rule_name, severity, message, **kwargs)


def skipped(check_id: str, rule_name: str, message: str) -> CheckResult:
    """Builds a SKIP result."""
    return CheckResult(check_id=check_id, rule=rule_name, status=CheckStatus.SKIP, message=message)


def score_report(checks: List[CheckResult]) -> PlausibilityReport:
    """
    Scores a list of checks.

    Args:
        checks: All executed checks.

    Returns:
        The PlausibilityReport (score 1.0 minus severity weights; status
        FAIL on any error, WARN on any warning).
    """
    penalty = sum(SEVERITY_WEIGHT[c.severity] for c in checks if c.status == CheckStatus.FAIL)
    severities = {c.severity for c in checks if c.status == CheckStatus.FAIL}
    status = "FAIL" if Severity.ERROR in severities else "WARN" if Severity.WARNING in severities else "PASS"
    return PlausibilityReport(status=status, score=round(max(0.0, 1.0 - penalty), 3), checks=checks)
