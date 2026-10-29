# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/pdflock.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Process-wide lock for PyMuPDF. PyMuPDF is not thread-safe
                (e.g. page.find_tables() keeps its text page in a module
                global), so every access - text, tables, renders, embedded
                files - runs under this lock; concurrent analyses (bench
                prepare pool, REST uploads, parallel OCR renders) stay safe.
------------------------------------------------------------------------------
"""

import threading

PYMUPDF_LOCK = threading.RLock()
