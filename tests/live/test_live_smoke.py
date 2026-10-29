# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/live/test_live_smoke.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Live smoke tests against a real backend (llama-swap + docling-serve).
                Excluded by default; run with: pytest -m live
------------------------------------------------------------------------------
"""

from typing import Any, Dict

import pytest

from fluxsense.config import load_settings
from fluxsense.pipeline import Analyzer

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def analyzer() -> Analyzer:
    """Analyzer with the user's real settings."""
    return Analyzer(load_settings())


def test_faulty_invoice(analyzer: Analyzer, samples: Dict[str, Any]) -> None:
    """The wrong VAT is reported and printed values are kept."""
    doc = analyzer.analyze(samples["invoice_wrong_vat"].path).logical_documents[0]
    assert doc.type == "INVOICE"
    assert {"invoice.net_plus_tax", "invoice.vat_row"} <= set(doc.plausibility.summary_codes())
    assert doc.fields["bodies"]["finance_body"]["monetary_summation"]["grand_total_amount"].startswith("357")


def test_bank_statement_three_pages(analyzer: Analyzer, samples: Dict[str, Any]) -> None:
    """The 3-page statement yields one table with all rows and a passing balance check."""
    truth = samples["bank_statement_3p"].data
    result = analyzer.analyze(samples["bank_statement_3p"].path)
    doc = result.logical_documents[0]
    assert doc.type == "BANK_STATEMENT" and len(result.logical_documents) == 1
    assert len(doc.tables[0].rows) == len(truth["entries"])
    checks = {c.check_id: c.status.value for c in doc.plausibility.checks}
    assert checks["statement.balance"] == "PASS"
