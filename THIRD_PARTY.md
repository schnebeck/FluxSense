# Third-party material

FluxSense itself is licensed under GPL-3.0-or-later (see `LICENSE`). The
repository contains, or uses at run time, the following third-party
material:

| Item | Where | Origin | Licence |
|---|---|---|---|
| OvisOCR3 `document_structured` prompt | `fluxsense/data/ovisocr3_document_structured.txt` | verbatim copy of `prompts/document_structured.txt` from the OvisOCR3 model repository (ATH-MaaS, <https://huggingface.co/ATH-MaaS/OvisOCR3>); the model card states the licence | Apache-2.0 (`LICENSES/Apache-2.0.txt`) |
| German bank-code list (Bankleitzahlendatei) | not bundled; `fluxsense blz-update` downloads it into `~/.local/share/fluxsense/blz.txt` | Deutsche Bundesbank, <https://www.bundesbank.de> | subject to the Bundesbank's terms of use; FluxSense does not redistribute it and works without it (the bank-code check is then skipped) |

Models (vision-language model, optional OvisOCR3, optional text reviewer)
are not part of FluxSense; each is used under its own licence through the
OpenAI-compatible server you run.
