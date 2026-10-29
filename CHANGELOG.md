# Changelog

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
