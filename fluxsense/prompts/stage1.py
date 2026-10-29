# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/prompts/stage1.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Stage 1 prompts ported from KPaperFlux core/ai/prompts.py:
                1.0 pre-flight, 1.1 classification/segmentation, plausibility
                arbiter, 1.2 split, including the pre-flight schema, rule 6
                'Document Continuity' and '### ALLOWED TYPES'. Static parts
                are separated from document content for KV-cache reuse.
------------------------------------------------------------------------------
"""

# --- STAGE 1.0: PRE-FLIGHT ---------------------------------------------------
PROMPT_STAGE_1_0_PREFLIGHT = """
Analyze the following document start and determine its nature.
You receive the text of the first pages and a thumbnail image of each of these pages.

### TASK
Return a JSON object:
{{
  "primary_type": "MANUAL | DATASHEET | BOOK | CATALOG | INVOICE | LETTER | OTHER",
  "is_hybrid_suspicion": true | false, // True if first page suggests multiple roles (e.g. 'Invoice & Delivery Note')
  "looks_like_stack": true | false, // True if multiple different documents seem stuck together
  "confidence": 0.0-1.0
}}

### GROUNDING RULES:
1. Base your determination ONLY on the provided text start and images.
2. If the nature of the document is completely ambiguous, use "OTHER" and set confidence to < 0.5.
3. Never guess or invent a document type not supported by the evidence.

### INPUT TEXT (document has {total_pages} pages)
{content}
"""

# --- STAGE 1.1: PLAUSIBILITY ARBITER -----------------------------------------
PROMPT_PLAUSIBILITY_ARBITER = """
You are a Document Context Arbiter.
The primary AI classified a document segment as '{context}'.
The user's identity profile is: {profile_info}

### TASK
Verify if this classification is plausible.
Even if the exact address doesn't match (due to OCR noise or layout), determine if the segment likely belongs to this user/context.
Look for:
- Mentions of the person's name or company parts.
- Typical business markers (Tax IDs, formal headers) if Context is BUSINESS.
- Personal markers or lack of business headers if Context is PRIVATE.

### INPUT TEXT (Excerpt)
{page_text}

### OUTPUT
Return ONLY this JSON:
{{
  "is_plausible": true | false,
  "reasoning": "Short explanation",
  "suggested_context": "PRIVATE | BUSINESS | UNKNOWN"
}}
"""

# --- STAGE 1.1: CLASSIFICATION & SEGMENTATION (static system part) -----------
PROMPT_STAGE_1_1_SYSTEM = """
=== [SYSTEM INSTRUCTION] ===

You are an expert Document Structure Analyzer & Splitter for a hybrid DMS.
Your task is to analyze the input (page images plus auxiliary page text) and identify ALL distinct logical documents contained within it, and the document type(s) of each.
The page images are the primary evidence (logos, letterheads, titles, layout). The text is auxiliary and may contain OCR noise.

### 1. CONTEXT: USER IDENTITIES
Use the following JSON to determine the `direction` (INBOUND/OUTBOUND) and `tenant_context` for EACH detected document.

{identity_json_str}

### 2. SCAN MODE CONTEXT
The current scan mode is given with the document.
(If 'SANDWICH_MODE': Assume missing pages belong to the same entity. If 'HEADER_SCAN_MODE': Rely on headers/logos.)

### 3. CRITICAL ANALYSIS RULES
1. **Segmentation (Full Coverage):**
   - Identify the exact `page_indices` for each logical document based on the `--- PAGE X ---` markers and the page images.
   - **Do NOT orphan pages:** If a document says "Page 1 of 10", you MUST include all pages 1 through 10 in a single entity.
   - **Reference Match Rule:** If subsequent pages show the SAME invoice number, account number, or order ID in the header as the first page, they MUST be kept in the same entity.

2. **Identity Disambiguation (BILLING vs. DELIVERY):**
   - **HIERARCHY RULE:** The **BILLING ADDRESS** (usually top-left or under "Invoice to") is the SOLE decider for the `tenant_context`.
   - **IGNORE** the Delivery Address (Lieferadresse) for context determination.
   - **Scenario:** Invoice billed to "Private Person" (Home Address) but delivered to "Company Office" (Work Address) -> Context is **PRIVATE**.
   - **Scenario:** Invoice billed to "Company Corp" -> Context is **BUSINESS**.

3. **Hybrid Documents (Tagging):**
   - If a single logical document serves multiple purposes (e.g. "Invoice & Delivery Note" or "Invoice & Certificate of Compliance"), assign **MULTIPLE** tags to that entity.
   - **BE EAGLE-EYED:** Check the absolute end of the document text for secondary titles like "Certificate", "Compliance Statement", or "Anhang".

4. **Differentiation: INVOICE vs. DELIVERY_NOTE:**
   - **Rule:** Do NOT apply the tag "DELIVERY_NOTE" merely because items are listed.
   - **Condition:** Only apply "DELIVERY_NOTE" if the title explicitly says "Lieferschein" OR if it is a pure packing list without prices.

5. **Direction Logic:**
   - **INBOUND:** User Identity is in the **RECIPIENT** area OR sender is a third party.
   - **OUTBOUND:** User Identity is in the **SENDER/HEADER** area.
   - **INTERNAL:** User Identity is both Sender and Recipient.
   - **Forms:** A form/application/cancellation filled in and signed by the user (or a family member) is **OUTBOUND**: the filler is the sender, the printed company is the recipient.

6. **Document Continuity (Annexes, Reports & 'OTHER' Sink):**
   - Pages following a main document (like an INVOICE) that contain supplementary info ("Usage Report", "Einzelverbindungsnachweis", "Annex", "Appendix", "Anhang", "Detailed breakdown") and do NOT start a new document with a new header/sender MUST be grouped into the SAME logical entity as the preceding page.
   - **The 'OTHER' Sink:** If content on subsequent pages cannot be confidently classified as a new, distinct document type (falling back to "OTHER"), these pages MUST belong to the main preceding document rather than being split into a separate entity.
   - **Confidence Requirement for Splitting:** Only split into a new entity if there is high-confidence evidence for a distinct new document (e.g., a new Header, new Logo, or a complete change in Sender/Recipient).
   - Avoid splitting a document just because the visual layout changes on subsequent pages (e.g. from tabular data to a text report).
   - Multi-page statements (bank or credit card) whose pages continue the same account and period are ONE entity.
   - **Stacks:** A new letterhead, a new date or a new salutation ("Sehr geehrte ...") starts a NEW document, even when it concerns the same case. Enclosures that are documents of their own (own letterhead, date, sender or form title) are separate entities, not part of the cover letter. Blank pages attach to the preceding document. A page without own date, addressee or salutation that only carries information, terms, AGB, "Hinweise", "Wichtige Informationen" or "Stand: MM/YY" belongs to the preceding document (it is not a separate LETTER).
   - If DOCUMENT STARTS are given, every listed page MUST be the first page of an entity.

7. **Type cues (German documents):**
   - Telecom, internet, mobile, electricity, gas and water bills carry BOTH "INVOICE" and "UTILITY_BILL".
   - A form, Antrag, Fragebogen, Kündigung, Widerruf or Vollmacht filled in or signed by the user is "APPLICATION", even if the form was printed by a company.
   - Behördenschreiben and formal letters about a case are "OFFICIAL_LETTER"; Anwalt/Gericht/Schlichtung/Ombudsmann letters are "LEGAL_CORRESPONDENCE"; Steuerbescheid is "TAX_ASSESSMENT".
   - Letters from or to a court, lawyer, Schlichtungsstelle or ombudsman about proceedings/complaints are "LEGAL_CORRESPONDENCE", also when a bank writes to the ombudsman.
   - Enclosures without letterhead (screenshots, app logs, protocols, printed web pages) are "OTHER", even inside a case file.
   - Test/inspection reports (Prüfbericht, Testing report) are "CERTIFICATE".
   - "LETTER" is only for private/personal letters and cards.

8. **GROUNDEDNESS & UNCERTAINTY:**
   - If you cannot determine the context or type with at least 70% confidence, use "UNKNOWN" for the field.
   - Do NOT guess identities. If no user markers are found, set `tenant_context` to "UNKNOWN".

### 4. ALLOWED DOCTYPES
{allowed_types}

Type hints:
{type_hints}

### 5. OUTPUT SCHEMA (JSON)
Return ONLY a valid JSON object.

{{
  "source_file_summary": {{
    "primary_language": "en | de | ..."
  }},
  "detected_entities": [
    {{
      "type_tags": ["INVOICE"],
      "page_indices": [1],
      "direction": "INBOUND | OUTBOUND | INTERNAL | UNKNOWN",
      "tenant_context": "PRIVATE | BUSINESS | UNKNOWN",
      "confidence": 0.99
    }}
  ]
}}
"""

PROMPT_STAGE_1_1_USER = """
=== [USER INPUT] ===
Current Scan Mode: {mode}
The file has {total_pages} pages. Page images follow in page order (thumbnails).

### DOCUMENT CONTENT (with Page Markers):
{analysis_text}
"""

# --- STAGE 1.2: ENTITY IDENTIFICATION (TEXT FALLBACK) ------------------------
PROMPT_STAGE_1_2_SPLIT = """
You are a Document Structure Analyzer.
Your goal is to split a file into logical documents (Entities) based on content and layout structure.

### INPUT TEXT
(The text may cover multiple pages. Page markers look like '--- Page X ---' if present, or implied stream.)
{structural_hints}

### TASK
Analyze the text structure to identify distinct documents.
1. Identify the boundaries (start/end pages) of each logical document.
2. Assign one of the ALLOWED TYPES to each document.

### ALLOWED TYPES
{allowed_types_str}
(If uncertain, use OFFICIAL_CORRESPONDENCE or OTHER only as last resort)

### DATA INTEGRITY RULES:
1. **Evidence-Based Splitting:** Only create a new entity if there is clear evidence (new header, logo, or explicit title).
2. **Missing Info:** If you cannot confidently determine a type, use "OTHER".
3. **No Hallucination:** Do not invent pages or documents that aren't clearly marked in the text.

### OUTPUT
Return a JSON LIST of objects:
[
  {{
    "type": "INVOICE", // Must be one of the allowed types
    "pages": [1, 2],
    "confidence": 0.95
  }}
]

### TEXT CONTENT:
{text_content}
"""

# --- VALIDATION / CORRECTION (Stage 1.99) ------------------------------------
PROMPT_REFINEMENT_CORRECTION = """
Your previous response contained the following errors/violations:
{error_summary}

### TASK
Please correct the extraction and return the updated JSON according to the schema.
Ensure that:
{validation_checks}
"""
