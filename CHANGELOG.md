# Changelog

## Unreleased

- Field contract 1.8, §13: text levels per page (`raw`, `clean` -
  dehyphenated, soft hyphens removed, running text reflowed to paragraphs,
  addresses / tables / lists line-based - and `corrected` with the
  corrections list `{page, offset, bbox, before, after, reason,
  evidence_kind}`); digital text is never corrected.
- Format registry `fluxsense.formats`: IBAN, BIC, EAN / GTIN, ISBN-10 / -13,
  VAT ID, German tax number and Steuer-ID, dates, times, postcodes, card
  numbers - shape per position, checksum, normalisation, OCR confusion map.
  The IBAN, BIC and tax-ID checks of grounding, plausibility and the
  contract now use it; applications can pass their own formats per call
  (`AnalyzeOptions.formats`).
- Evidence-based correction after the plausibility checks: unique
  OCR-explained dictionary words (hunspell via `hunspell` / `enchant-2`,
  skipped when not installed; `text.spell_check`), forced repairs of fixed
  formats by position class + checksum, and native-resolution crop re-reads
  without the old value for digit confusions and arithmetic mismatches
  (`text.format_reread`); confirmed printed errors are flagged as document
  errors. Review hooks `confirm_text` / `confirm_field`.
- `render_text_layer()`: a searchable copy of a scanned PDF with an
  invisible text layer from FluxSense's text and geometry (corrected words,
  soft-hyphen pairs with ActualText, form entries, nothing in signature
  regions; page images untouched).
- Field contract 1.7, §12: every scanned page has a handwriting level
  (`none | signature | entries | full`) from the Stage 1.0 page signatures
  or a probe call (`pipeline.handwriting_probe`), output per document
  (`handwriting`). A page with only a signature reads just its date /
  signature lines and flags no printed value; a printed form with
  handwritten entries is read form-first (the vision model locates the
  label / entry pairs, the lexicon only gives kind and target); a freely
  handwritten page is read twice from native-resolution line bands and
  only agreeing lines are accepted (`handread` source). The field reader
  reports `typed | handwritten | empty`: typed entries that the page text
  agrees with may be auto-accepted again. Native AcroForm fields of
  digital PDFs are read from their widgets without vision. The readings
  are output as `form_fields[]`.
- Letter deadlines: `bis zum` / `bis` count as a deadline label only on a
  line that asks the recipient to act; an offer's validity date is no
  longer auto-accepted as a deadline.
- Signatures as structured output (`signatures[]`: signer, role, place,
  date, page, bbox, status `signed | empty | unclear`, evidence): the
  status comes from the pen ink in the signature space, the signer only
  from printed content (i. A. / i. V. name, printed name, line label ->
  party, §2.4 form filler; else null). OCR text of signature scribbles is
  removed from scanned pages before the extraction (`pipeline.signatures`),
  also when the OCR joins a handwritten date and the signature into one
  line (only the date stays; its share of the line is masked for the ink).
  Long labels starting with `Unterschrift` are signature labels; form field
  labels naming a role (`Name des Wohnungsgebers`) are not.
- An IBAN printed with a wrong checksum is kept as printed and flagged
  (bank-code status, one-digit correction hints) instead of being removed;
  a one-time hint names `fluxsense blz-update` when no bank-code list is
  installed.
- Field contract 1.6, §12: form fields of scanned form pages are read by
  their printed label - one native-resolution crop per field, a typed
  question without the full-page value; dates with their candidate
  readings; the readings feed the extraction (FORM FIELDS) and grounding
  and map to the schema where the label is unambiguous
  (`pipeline.fieldread`).
- Defensive detection of runaway repetition and early stops in OvisOCR3
  answers, with a re-read of the page as overlapping tiles.
- Field contract 1.5: contract payment terms (total, components with a
  sum check, frequency, due rule, payee account, reference), deposit and
  contract object; minimum term / termination waiver separate from the
  notice period; names read only from signatures are no parties (§2.9);
  landlord / employer confirmations are certificates with the certified
  persons, event, date and address; form print numbers are no doc_number.
- Field contract 1.4, §9 date consistency: the document date is checked
  against the document's other dates (contract start, due / order /
  service dates, statement period, reply deadline) in type-aware windows;
  conflicts are warnings on both fields and block auto-acceptance. An
  ambiguous year (two-digit, zoom / vision reading) is re-read only when
  exactly one reading is consistent, and stays flagged; a zoom reading
  that contradicts the other dates is no visual support.

## 0.1.0 - initial release

- Holistic, type-driven extraction of logical documents from PDF stacks
  (segmentation, classification, one structured extraction per document
  with cross-page tables) on any OpenAI-compatible vision-language server.
- Field contract (`docs/field_contract.md`) defining every core field.
- Grounding against the text layer, zoom re-verification, plausibility
  checks (EN 16931 arithmetic, IBAN mod-97, statement balances), targeted
  corrections and per-field provenance with auto-acceptance.
- ZUGFeRD / Factur-X XML as authoritative source when embedded.
- Text engines: native PDF text, ocrmypdf, local Docling, docling-serve,
  OvisOCR3 (optional).
- CLI (`analyze`, `bench`, `reassess`, `serve`, `doctor`, `catalog`,
  `config`, `samples`, `gen-testdocs`, `score-testdocs`, `blz-update`),
  REST API and Python library.
- Generator of fictional test documents with exact ground truth (digital
  and scan variants) and a public smoke benchmark.
- Optional energy management of a suspended AI server (wake hook or a
  WOL-Server device request).
