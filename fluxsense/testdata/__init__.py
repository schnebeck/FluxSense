# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/__init__.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Synthetic test-document generator: fictional data only,
                deterministic per seed, PDFs (digital and scan variants)
                with exact ground truth. Test data is generated, never
                derived from real documents.
------------------------------------------------------------------------------
"""

from fluxsense.testdata.documents import (InvoiceStyle, LogicalDoc, build_contract, build_enclosure, build_form,
                                          build_invoice, build_letter, build_private_note, build_receipt, build_stack,
                                          build_statement, build_telecom_invoice, build_transfer)
from fluxsense.testdata.generate import (KINDS, GeneratedFile, denylist_hits, generate_file, generate_set, rng_for,
                                         smoke_set, vocabulary)

__all__ = ["InvoiceStyle", "LogicalDoc", "build_contract", "build_enclosure", "build_form", "build_invoice",
           "build_letter", "build_private_note", "build_receipt", "build_stack", "build_statement",
           "build_telecom_invoice", "build_transfer", "KINDS", "GeneratedFile", "denylist_hits", "generate_file",
           "generate_set", "rng_for", "smoke_set", "vocabulary"]
