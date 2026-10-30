# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/text/__init__.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Text levels per page (raw / clean / corrected with the
                corrections list), evidence-based corrections and the
                searchable text layer. text_pages() builds the page outputs
                of an analysis; names of its parties and signers are never
                spell-corrected.
------------------------------------------------------------------------------
"""

from typing import Any, List, Optional, Sequence

from fluxsense.config import Settings
from fluxsense.extract.base import TextLayer
from fluxsense.models.result import PageTextOut
from fluxsense.text.correct import confirm_field, confirm_text, protected_words, repeated_words
from fluxsense.text.levels import clean_text, page_levels
from fluxsense.text.render import render_text_layer

__all__ = ["clean_text", "confirm_field", "confirm_text", "page_levels", "render_text_layer", "text_pages"]


def _names(result: Any) -> List[str]:
    """Party, person and signer names of all documents of a result (never spell-corrected)."""
    names: List[str] = []
    for doc in getattr(result, "logical_documents", []) or []:
        meta = (doc.fields or {}).get("meta_header") or {}
        for role in ("sender", "recipient"):
            party = meta.get(role) or {}
            if isinstance(party, dict):
                names += [str(party.get(k)) for k in ("name", "company", "street", "city", "contact") if party.get(k)]
        legal = ((doc.fields or {}).get("bodies") or {}).get("legal_body") or {}
        names += [str(n) for n in (legal.get("parties") or []) + (legal.get("certified_persons") or [])]
        names += [s.signer_name for s in doc.signatures if s.signer_name]
        names += [f.entry for f in doc.form_fields if f.kind in ("name", "persons", "address") and f.entry]
    return names


def text_pages(layer: TextLayer, settings: Settings, result: Optional[Any] = None, formats: Sequence[Any] = ()) -> List[PageTextOut]:
    """
    The text levels of every page.

    Args:
        layer: Text layer after the analysis (pending corrections of the field checks are on its pages).
        settings: Settings (spell check and its languages).
        result: The AnalysisResult whose names are protected from the word correction.
        formats: Formats supplied by the application for this call.

    Returns:
        One PageTextOut per page.
    """
    lookup = None
    if settings.text.spell_check and any(p.scanned() for p in layer.pages):
        from fluxsense.text.dictionary import Dictionary
        dictionary = Dictionary.system(settings.text.spell_languages)
        lookup = dictionary.lookup if dictionary.available else None
    protected = protected_words(_names(result)) if result is not None else set()
    if lookup is not None:
        protected |= repeated_words(page.best_text() for page in layer.pages)
    return [page_levels(page, lookup, protected, formats) for page in layer.pages]
