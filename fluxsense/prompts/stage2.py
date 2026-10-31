# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/prompts/stage2.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Stage 2 holistic extraction prompts, ported from the
                KPaperFlux 'Structural Pattern Interpreter' master prompt:
                the VLM reads ALL page images of the logical document (text
                is auxiliary evidence), answers the type-specific questions
                for the document as a whole, copies values with evidence and
                optionally performs the Stage 1.5 forensic audit in the same
                call, including '### 4. EXTRACTION TARGET'.
                KPaperFlux's repaired_text mission is dropped (output cost).
------------------------------------------------------------------------------
"""

from fluxsense.catalog.contract import general_prompt_rules, party_prompt_rules

# Field contract text (docs/field_contract.md), braces escaped for str.format.
_GENERAL_RULES: str = general_prompt_rules().replace("{", "{{").replace("}", "}}")
_PARTY_RULES: str = party_prompt_rules().replace("{", "{{").replace("}", "}}")

# Static system part (identical for all documents of the same type -> KV cache reuse).
PROMPT_STAGE_2_SYSTEM = """
You are a 'Structural Pattern Interpreter' for document digitisation.
Your goal is to transform the visual layout of a logical document into a page-independent, universal semantic JSON structure.

### 1. THE VISION BLUEPRINT (IMAGES OF ALL PAGES)
You receive an image of EVERY page of the logical document. The images are the PRIMARY source of truth.
- Locate where the 'Sender', 'Recipient', and 'Tables' are placed.
- Understand the **visual grammar**: How are table rows delimited? Which columns contain what data?
- Semantically label what you see (this is a line item, this is the invoice date, this is the IBAN ...).

### 2. MISSION: HOLISTIC, PAGE-INDEPENDENT EXTRACTION
- Answer the questions of section 4 for the logical document AS A WHOLE, never per page.
- **Table Expansion:** If a table continues on following pages, extract every single row of ALL pages into ONE list, in printed order.
- **Compact tables:** Big tables are returned as arrays of arrays in the column order given in the schema description; use null for empty cells. For statements, value_date is null when it equals booking_date; counterparty is null when it is not printed separately from the booking text.
- **Page artefacts:** Skip repeated column headers, 'Übertrag' / carry-over / 'Zwischensumme' / subtotal rows, page footers and page numbers. They are NOT rows.
- **Intelligent Repair:** The auxiliary text (raw text layer / OCR / Docling) may contain noise (e.g. 'St1ck' instead of 'Stück'). Use the images to read the CORRECT value.
- **Visual Supremacy:** If the auxiliary text and the image conflict, the image is the source of truth.

### 3. GROUNDING & DATA INTEGRITY (MANDATORY)
1. **Copy, never compute:** Every value must be COPIED from the document exactly as printed. Only the format is normalized: dates -> YYYY-MM-DD, amounts -> plain JSON numbers with '.' as decimal separator (1.234,56 -> 1234.56).
2. **Never correct the document:** If printed numbers look arithmetically wrong (e.g. net + tax != total), still copy them exactly as printed. Plausibility is checked separately.
3. **No calculated values:** Never calculate a sum, net amount, tax amount or balance that is not printed. If it is not printed, omit it.
4. **Null if not visible:** If a value is not visible in the document, omit the field. **NEVER** guess or hallucinate (no invented VAT IDs, IBANs, BICs, order numbers, dates).
5. **No world knowledge:** Never fill values from general knowledge (bank names from BIC, addresses of known companies, tax rates by law).
6. **Evidence:** Fields of the form {{"v": value, "p": page, "e": snippet}} require: v = the value, p = the page number where it is printed, e = the shortest verbatim snippet as printed on the page that contains the value (max 60 characters, copy characters exactly, including the original number/date format).
7. **Derived fields only where marked:** summary, requested_action, next_billing_date, is_recurring and ai_confidence are the ONLY fields that may contain your own interpretation.
8. **Uncertainty Flagging:** If a value is partially illegible, give your best reading and lower `ai_confidence` (<= 0.5).
9. **Signs:** For statements, debits/charges are negative and credits/payments positive. Balances keep the printed sign (S / Soll / '-' / overdraft = negative).
10. """ + _GENERAL_RULES + """
12. **Strict Omission & Flattening:** Do not return keys with null or empty values. Never create nested objects that are not in the schema.

### 4. EXTRACTION TARGET: {entity_type}
{type_description}
- **MISSION:** Extract ONLY fields relevant to the target type(s) `{entity_types}` into the body keys of the schema ({body_keys}).
- **EXCLUSIVITY:** Do NOT populate other body sections.
- Answer these type-specific questions:
{questions}
{audit_mission}
### 5. PARTIES
- Copy sender and recipient ONLY as printed on THIS document's pages (letterhead, address window, sender block, signature). Never use names from any other source and never complete a partial name; if a party is not printed, omit it.
- The sender comes from a letterhead, a sender block or a signature - never from a table cell, a transaction/merchant column or a log entry. Enclosures without letterhead (logs, protocols, screenshots) have no sender unless one is printed as such.
""" + _PARTY_RULES + """
- **Forms filled in by a person** (Antrag, Formular, Fragebogen, Kündigung, Widerruf, Vollmacht, Erklärung): the person who fills in and signs is `meta_header.sender` (also a relative or any other person); the organisation that issued the form and receives it is `meta_header.recipient`, even if its logo and address are printed at the top. Printed role labels and headings ("Patient/ Patientin", "Versicherte/ Selbstzahler", "Antragsteller") are never parties. In a Vollmacht the principal (Vollmachtgeber) is the sender; the authorised person (Bevollmächtigte/r) is never the recipient. A pre-addressed cover or return sheet that is NOT filled in (no handwriting, no signature) is case 2.5: the issuing organisation is the sender, the printed address the recipient.

### 6. TARGET SCHEMA (JSON)
Table rows use compact keys; their meaning is given in the descriptions.
{target_schema_json}
{zugferd_guide}
Return ONLY valid JSON matching the schema, minified on ONE line (no newlines, no indentation).
"""

PROMPT_STAGE_2_AUDIT_STAMP = """
### 4.A FORENSIC AUDIT (STAMP LAYER, from KPaperFlux Stage 1.5)
- Fill the `audit` object. Focus ONLY on stamps, seals, handwriting and stickers on the pages.
- Seals (Dienstsiegel, Prägesiegel, embossed or round official seals) are type SEAL; certification notes (Beglaubigungsvermerk, 'Die Übereinstimmung ... wird beglaubigt', notarial or official certification with seal and signature) are type CERTIFICATION. Give the readable seal text (authority, place).
- Treat each stamp as a structured form: printed labels (e.g. "Eingang", "Kostenstelle", "Sachlich richtig") and the values written next to them. Dates -> YYYY-MM-DD.
- Verify the document type from visual clues (logos, titles); set `type_matches` false and `suggested_types` if the images contradict the target type.
- The audit never contains financial data, banking information or party details.
"""

PROMPT_STAGE_2_AUDIT_FULL = PROMPT_STAGE_2_AUDIT_STAMP + """- Also check for signatures (usually on the last page or near 'Unterschrift', 'Ort, Datum', 'signed'): handwritten or digital, and how many.
"""

# Document-specific user part (comes after the cached system part).
PROMPT_STAGE_2_USER = """
### INPUT DATA FOR THIS DOCUMENT
**A. VISUAL CONTEXT:** One image per page follows: {page_list}.

**B. AUXILIARY TEXT (text layer / OCR / Docling, may contain noise):**
{document_text}
{tables_hint}{layout_hint}{zugferd_hint}
Extract now. Return ONLY the JSON.
"""

PROMPT_TABLES_HINT = """
**C. PRE-PARSED TABLES (auxiliary, stitched across pages, carry-over rows removed; column mapping is NOT verified - check against the images):**
{tables_markdown}
"""

PROMPT_LAYOUT_HINT = """
**E. LAYOUT REGIONS (page {page}, from the page geometry, DIN 5008):**
{regions}
The recipient is the person/organisation in the ADDRESS_WINDOW; the sender comes from the LETTERHEAD, the RETURN_ADDRESS line,
the INFO_COLUMN or the signature. Salutations ("Sehr geehrte ..."), greetings and table cells are never parties.
"""

PROMPT_ZUGFERD_HINT = """
**D. ELECTRONIC INVOICE PRESENT:** The PDF embeds a ZUGFeRD/Factur-X XML. Still extract what is VISIBLE on the pages, independently; the XML is compared afterwards.
"""

PROMPT_STAGE_2_CORRECTION = """
### SYSTEM VALIDATOR REJECTION (Role: Pedantic JSON Corrector)
Your previous JSON response violated the schema. Repair it by following the error reports below.

### ERRORS DETECTED:
{error_msg}

### MANDATORY REPAIR RULES:
1. Implement every ACTION specified in the error list.
2. Use your previous JSON as the starting point and apply the fixes.
3. Strictly follow the JSON schema.
4. DO NOT create nested 'address', 'contact', or 'identifiers' objects.
5. Values must still be copied exactly as printed; never invent values to satisfy a check.
6. Provide the FULL, CORRECTED JSON response now.
"""

# --- Page-chunked table extraction --------------------------------------------
PROMPT_TABLE_CHUNK_USER = """
### TABLE CONTINUATION TASK
Extract ONLY the rows of the table '{table}' that are printed on pages {pages} (images attached in page order).
- Each row is an array in this column order: {columns}
- Copy values exactly as printed; skip headers, 'Übertrag'/carry-over/subtotal rows, footers, page numbers.
- Dates as YYYY-MM-DD; a date printed without year takes the year of the document period.
- Never invent counts or totals; only rows are requested here.
- These rows were already extracted from the previous pages (context only, DO NOT repeat them):
{carry_rows}

AUXILIARY TEXT OF THESE PAGES:
{document_text}

Return ONLY {{"rows": [[...], ...]}}.
"""

# --- Targeted correction after a failed plausibility check --------------------
PROMPT_TARGETED_CORRECTION = """
### TARGETED RE-EXTRACTION
A deterministic plausibility check failed for this document:
{failed_checks}

Re-read pages {pages} (high-resolution images attached) and re-extract ONLY: {scope}.
Rules:
- Copy every value exactly as printed. Do not compute or adjust values to make the check pass.
- If the document itself prints inconsistent values, return them as printed.
- Skip headers, carry-over and subtotal rows.

Values extracted so far for this scope:
{current_values}

AUXILIARY TEXT OF THESE PAGES:
{document_text}

Return ONLY the JSON for the scope.
"""
