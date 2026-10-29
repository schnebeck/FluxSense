# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/validate/__init__.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Deterministic plausibility entry point: applies the rules
                listed in the TypeSpec(s) and returns a scored report.
------------------------------------------------------------------------------
"""

from typing import List

from fluxsense.catalog.registry import TypeSpec
from fluxsense.log import get_logger
from fluxsense.models.result import CheckResult, PlausibilityReport
from fluxsense.validate import finance, general, statement  # noqa: F401  (registers rules)
from fluxsense.validate.core import RULES, CheckContext, score_report

logger = get_logger("validate")

# Rules applied to every document type (an IBAN or a date can appear anywhere).
UNIVERSAL_RULES = ("iban", "dates")


def run_plausibility(ctx: CheckContext, specs: List[TypeSpec]) -> PlausibilityReport:
    """
    Runs all rules of the given type specs once.

    Args:
        ctx: Check context (ctx.spec = primary spec).
        specs: All specs of the logical document.

    Returns:
        The scored report.
    """
    rule_ids: List[str] = list(UNIVERSAL_RULES)
    for spec in specs:
        rule_ids += [r for r in spec.rules if r not in rule_ids]
    checks: List[CheckResult] = []
    for rule_id in rule_ids:
        func = RULES.get(rule_id)
        if func is None:
            logger.warning(f"Unknown plausibility rule '{rule_id}'")
            continue
        checks.extend(func(ctx))
    return score_report(checks)
