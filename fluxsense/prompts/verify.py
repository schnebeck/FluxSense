# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/prompts/verify.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Prompts of the grounding harness and reviews: zoomed region
                verification (closed question per field), page-tile
                localisation, text-only second opinion, vision review.
------------------------------------------------------------------------------
"""

PROMPT_ZOOM_VERIFY = """
You are a meticulous proof-reader. You see high-resolution crops of page {page} of a document.
For each question, look ONLY at the crop with the given id and answer with what is ACTUALLY PRINTED there.
Do not compute, do not guess, do not use world knowledge. If the requested item is not visible in the crop, set visible=false.

QUESTIONS:
{questions}

Return ONLY JSON: {{"answers": [{{"id": "F1", "visible": true, "printed": "text exactly as printed or null", "matches": true}}]}}
'matches' is true only if the printed value equals the proposed value (ignoring formatting such as 1.234,56 vs 1234.56 or 01.03.2024 vs 2024-03-01).
"""

PROMPT_LOCATE = """
The page image is divided into a 3x3 grid of tiles: TL, TC, TR (top row), ML, MC, MR (middle row), BL, BC, BR (bottom row).
For each item, name the tile where it is printed, or NONE if it is not on this page.
ITEMS:
{items}
Return ONLY JSON: {{"tiles": [{{"id": "F1", "tile": "TR"}}]}}
"""

PROMPT_TEXT_SECOND_OPINION = """
You are a careful auditor. Below is the text of a document and values a model extracted from it.
For each value decide whether the TEXT supports it (copied, not computed or guessed).
Return ONLY JSON: {{"answers": [{{"id": "F1", "supported": true, "printed": "value as printed or null"}}]}}

VALUES:
{questions}

DOCUMENT TEXT:
{document_text}
"""

PROMPT_REVIEW_VISION = """
You see the page images of a document and a list of values extracted from it.
For each value answer whether the images show exactly this value for this field.
Return ONLY JSON: {{"verdicts": [{{"field": "...", "verdict": "MATCH | MISMATCH | NOT_VISIBLE", "observed": "what is printed or null"}}]}}

EXTRACTED VALUES:
{values}
"""

PROMPT_REVIEW_TEXT = """
You are a senior accountant reviewing an automated document extraction for plausibility.
Document type: {doc_type}
Deterministic findings already detected:
{findings}

Extracted data (JSON):
{extraction}

Document text:
{document_text}

Judge whether the extraction is plausible and complete. Report only concrete issues (wrong values, missing values, inconsistent sums, suspicious content such as a changed IBAN).
Return ONLY JSON: {{"verdict": "PLAUSIBLE | SUSPICIOUS | IMPLAUSIBLE", "issues": [{{"field": "...", "severity": "INFO | WARNING | ERROR", "explanation": "..."}}]}}
"""
