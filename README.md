# FluxSense

FluxSense turns PDFs into structured, page-independent data with a **local
vision-language model**. It is a command-line tool, a Python library and a
REST service.

For every logical document in a PDF (a scanned stack can hold several) it
returns:

- the document type and the fields that matter for it (parties, dates,
  numbers, amounts, IBANs, statement periods and balances, ...);
- tables merged across pages (invoice line items, bank transactions);
- deterministic plausibility checks (EN 16931 arithmetic, IBAN mod-97,
  statement balances, dates);
- for every field: where it was printed, how it was determined and whether
  it can be **accepted without review**.

FluxSense runs against any OpenAI-compatible server, for example
`llama-server` (llama.cpp) or `llama-swap`. No cloud service is needed.

Licence: GPL-3.0-or-later.

## The idea: precision first, then coverage

Language models read documents well, but they also guess. FluxSense never
trusts a value just because the model returned it. Every filled field gets a
**provenance**:

| Provenance | Meaning |
|---|---|
| `label` | printed next to a label that the field contract lists for this document type (e.g. `Rechnungsnummer`, `Kontoauszug Nr.`) |
| `geometry` | printed in the layout region of its role (DIN 5008 address window, letterhead, closing) |
| `check` | confirmed by a deterministic check (net + tax = gross, line sum, balance, mod-97, embedded ZUGFeRD XML) |
| `model` | none of the above |

A field is **auto-accepted** only with provenance `label`, `geometry` or
`check`, when it is grounded in the text layer (or confirmed on a zoomed
crop of the page), and when no failing check touches it. Everything else is
flagged for review with a reason. The meaning of every field per document
type is defined in the [field contract](docs/field_contract.md).

On the author's mixed benchmark of German business documents (invoices,
bank and credit-card statements, letters, forms, certificates), the values
that FluxSense auto-accepted were correct in 99 % or more of the cases,
while it auto-accepted about 85 % of all values. The benchmark itself is
private and not part of the repository.

## Quick start

### 1. An AI server

FluxSense needs one always-on, OpenAI-compatible server with a
**vision-language model**, for example `llama-server` from llama.cpp:

```bash
llama-server -m Qwen3.6-35B-A3B-Q4_K_M.gguf --mmproj mmproj-Qwen3.6-35B-A3B-F16.gguf \
    --host 0.0.0.0 --port 8080 -c 131072 -np 4 --jinja
```

`-np 4` gives four parallel slots of 32k context each (FluxSense uses up to
`llm.max_concurrency` = 4 parallel requests). With several models,
`llama-swap` serves them under one URL; a minimal `config.yaml`:

```yaml
models:
  qwen3.6-35b-a3b:                  # vision-language model (required)
    cmd: llama-server --port ${PORT} -m /models/Qwen3.6-35B-A3B-Q4_K_M.gguf
         --mmproj /models/mmproj-Qwen3.6-35B-A3B-F16.gguf -c 131072 -np 4 --jinja
  ovisocr3:                         # optional OCR/layout engine
    cmd: llama-server --port ${PORT} -m /models/OvisOCR3-Q8_0.gguf
         --mmproj /models/mmproj-OvisOCR3-F16.gguf -c 32768 -np 2
```

Any vision-language model that understands JSON-schema constrained output
(`response_format: json_schema`) works; the model names are configured in
`[models]`.

### 2. Install

```bash
python3.12 -m venv venv
./venv/bin/pip install -e .            # add .[docling] for the local Docling engine
```

Optional tools: `ocrmypdf` + `tesseract` for scans without OvisOCR3.

### 3. Configure

`~/.config/fluxsense/config.toml` (or `$FLUXSENSE_CONFIG`):

```toml
[llm]
base_url = "http://ai-server.local:8080/v1"   # default: http://localhost:8080/v1
# the API key, if your server needs one, comes from $FLUXSENSE_LLM_API_KEY
# or from a command that prints it:
# api_key_command = "pass show ai-server/api-key"

[models]
primary = "qwen3.6-35b-a3b"           # the ids your server lists under /v1/models
vision = "qwen3.6-35b-a3b"
ovis_ocr = "ovisocr3"                 # only for --ocr-engine ovisocr3

[identity.private]                    # optional: your own name and address
name = "Max Mustermann"
address_keywords = ["00345", "Musterstadt"]
```

Every setting can also be given as environment variable
`FLUXSENSE_<SECTION>__<KEY>`, e.g. `FLUXSENSE_LLM__BASE_URL`. Secrets are
never written to the configuration.

Check the setup:

```bash
fluxsense doctor
```

It reports the endpoint, every configured model the server does not serve,
the text engines and the optional tools.

## Usage

### Command line

```bash
fluxsense analyze invoice.pdf                 # short report with findings
fluxsense analyze invoice.pdf --json          # the full result as JSON
fluxsense analyze scans/ --out results/ --ocr-engine ovisocr3
fluxsense bench pdfs/ --out bench-out         # resumable batch run with summary
fluxsense reassess bench-out --out bench-re   # re-score saved results offline (no model calls)
fluxsense serve --port 8765                   # REST API, OpenAPI docs at /docs
fluxsense catalog -v                          # document types, questions, rules
fluxsense gen-testdocs --out testdocs --seed 7 --count 24   # fictional test documents + ground truth
fluxsense blz-update                          # optional German bank-code list
```

`--ocr-engine` selects the auxiliary text layer: `auto` (native text, plus
OCR for scans), `native`, `ocrmypdf`, `ovisocr3`, `docling`, `docling-vlm`
or `docling-serve`. The vision model always reads the page images; the text
layer supports it and is what values are grounded in.

### Smoke benchmark with generated documents

FluxSense ships a generator of fictional test documents: invoices (one page,
long multi-page distributor invoices with carry-over rows, a bilingual
variant, ZUGFeRD / Factur-X XML, telecom invoices followed by an itemised
usage list), bank and credit-card statements, DIN 5008 letters, filled-in
forms, receipts, bank transfer confirmations, private notes and cards, and
multi-document stacks, as digital PDFs or image-only scans (skew, noise,
stamps, handwriting-like marks). Every file comes with an exact ground truth
in `ground_truth.json`, so anyone can measure FluxSense against their own
model server:

```bash
fluxsense gen-testdocs --smoke --out smoke         # 28 files, seed 1
fluxsense bench smoke --out smoke-run
fluxsense score-testdocs --truth smoke/ground_truth.json --results smoke-run
```

`--seed` and `--count` produce other, equally deterministic sets; all names,
addresses and numbers are invented (IBANs and VAT IDs are valid in format
but use ranges not observed in practice; postcodes are 00100-00999, below
the lowest German postcode 01001).

### REST API

- `POST /v1/analyze`, multipart `file`; optional query parameters
  `ocr_engine`, `review_vision`, `review_text`, `grounding`,
  `include_page_text`. Returns an `AnalysisResult`.
- Errors carry a machine-readable `error_code`: 503 `BACKEND_UNREACHABLE`,
  502 `BACKEND_AUTH` (the AI server rejected the request: missing or invalid
  API key), 504 `BACKEND_TIMEOUT`, 422 `INPUT_INVALID`.
- `GET /v1/health`, `GET /v1/models`, `GET /v1/catalog`.

### Library

```python
from pathlib import Path
from fluxsense.config import load_settings
from fluxsense.pipeline import Analyzer, AnalyzeOptions

result = Analyzer(load_settings()).analyze(Path("invoice.pdf"), AnalyzeOptions(include_page_text=True))
for doc in result.logical_documents:
    print(doc.type, doc.pages, doc.plausibility.status)
    for ev in doc.evidence:
        if ev.auto_accepted is False:
            print("review:", ev.path, ev.value, ev.review_reasons)
```

## How it works

1. **Text layer.** Native PDF text and word boxes always; for scans and
   weak pages OCR (OvisOCR3, ocrmypdf or Docling).
2. **Segmentation and classification.** Page signatures (letterhead, date,
   page numbering, salutation, closing) and the vision model split a stack
   into logical documents and assign types from the catalog.
3. **Holistic extraction.** One structured call per logical document with
   the page images, the auxiliary text and the type's questions. Long item
   tables are extracted page by page in parallel and stitched. An embedded
   ZUGFeRD / Factur-X XML is authoritative.
4. **Post-checks** that implement the field contract (label priority,
   "never" lists, party roles, statement periods) and the DIN 5008 layout.
5. **Grounding and zoom.** Every value is matched against the text layer;
   uncertain critical values are re-read on a high-resolution crop.
6. **Plausibility and one targeted correction** when a sum or balance
   check fails.
7. **Provenance and acceptance** per field, summarised in a trust report.

## Output

`AnalysisResult` (`fluxsense/models/result.py`):

- `source`, `text_layer` (engine, sources; optional per-page text),
  `classification`, `zugferd`, `status`, `error_code`;
- `logical_documents[]` with `type`, `type_tags`, `pages`, `direction`,
  `tenant_context`, `fields` (`meta_header`, `bodies` such as
  `finance_body`, `statement_body`, `correspondence_body`), `tables`,
  `plausibility` (all checks), `evidence` (per field: value, page,
  snippet, grounding status, provenance, `auto_accepted`,
  `review_reasons`), `trust`, `corrections`;
- `timings` and `ai_calls` (model, tokens, seconds).

The field layout follows the KPaperFlux `SemanticExtraction` model with
EN 16931 business terms; FluxSense adds statement, payslip and
correspondence bodies.

## Models

| Role (`[models]`) | Used for | Needed |
|---|---|---|
| `primary`, `vision` | classification, extraction, zoom verification | yes (a vision-language model) |
| `ovis_ocr` | the `ovisocr3` text engine | optional |
| `review_text` | an optional text-only second opinion | optional |
| `docling_vlm` | the `docling-vlm` engine | optional |

The defaults are the model names of the author's llama-swap setup; set
them to the ids your server reports.

## Development

```bash
./venv/bin/pip install -e .[dev]
./venv/bin/pytest            # unit tests, no network needed
./venv/bin/pytest -m live    # optional live tests against your server
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Third-party material is listed in
[THIRD_PARTY.md](THIRD_PARTY.md).

## Optional: energy management of a suspended AI server

Most setups run the AI server permanently and need nothing of this. For a
server that sleeps when idle:

- **Wake hook:** `hooks.wake_command` runs a command (e.g. a wake-on-LAN
  script) when the endpoint does not answer; FluxSense then polls until it
  does (`hooks.wake_timeout_s`).
- **WOL-Server device requests:** with `hooks.wol_url`, `hooks.wol_device`
  and `hooks.wol_token_command`, FluxSense requests the device on the first
  model call (waking it if needed), sends heartbeats while it works and
  releases the request after `hooks.wol_release_after_idle_s` without
  analyses. It never asks for sleep; the server's own idle logic decides.
  Without `wol_url` none of this code runs.

```toml
[hooks]
wol_url = "https://wol.example.lan"
wol_device = "ai-server"
wol_token_command = "pass show wol/api-token"
wol_verify_tls = "/etc/ssl/certs/wol-ca.pem"   # or false for a self-signed test setup
```
