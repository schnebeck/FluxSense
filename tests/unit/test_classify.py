# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_classify.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for Stage 1: adaptive pre-flight routing, header/footer reduction, coverage repair, classification retries and the prompt restorations.
------------------------------------------------------------------------------
"""

from typing import Any, Dict, List, Optional

from fluxsense.catalog import default_catalog
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.llm.base import LLMRequest
from fluxsense.prompts import stage1, stage2
from fluxsense.stages.classify import (FULL_READ, HEADER_SCAN, SANDWICH, Classifier, Entity, extract_headers_footers,
                                       repair_coverage, route)
from fluxsense.stages.signatures import PageSignature, document_starts
from tests.conftest import FakeBackend


def test_routing() -> None:
    """SANDWICH for manuals, HEADER_SCAN for long or stacked files, else FULL_READ."""
    assert route(30, {"primary_type": "MANUAL"}, 11) == (SANDWICH, [1, 2, 3, 30])
    assert route(2, {"primary_type": "DATASHEET"}, 11) == (SANDWICH, [1, 2])
    assert route(12, {"primary_type": "INVOICE"}, 11)[0] == HEADER_SCAN
    assert route(4, {"looks_like_stack": True}, 11)[0] == HEADER_SCAN
    assert route(3, {}, 11) == (FULL_READ, [1, 2, 3])


def test_header_footer_reduction() -> None:
    """Long pages are reduced to header and footer areas."""
    page = "\n".join(f"line {i}" for i in range(40))
    reduced = extract_headers_footers([page, "short"])
    assert reduced[0].startswith("[HEADER AREA]") and "line 39" in reduced[0] and "line 20" not in reduced[0]
    assert reduced[1] == "short"


def test_coverage_repair() -> None:
    """Orphans join the preceding entity, duplicates and invalid pages are dropped."""
    entities = [Entity(["INVOICE"], [1, 2]), Entity(["DELIVERY_NOTE"], [2, 4, 9])]
    fixed = repair_coverage(entities, 5)
    assert [e.pages for e in fixed] == [[1, 2, 3], [4, 5]]
    assert repair_coverage([], 2)[0].pages == [1, 2]


def layer(pages: int) -> TextLayer:
    """A text layer with simple page texts."""
    return TextLayer(pages=[PageText(page=i, sources={"native": f"Rechnung Seite {i}"}) for i in range(1, pages + 1)], engine="t")


def test_classifier_retries_on_orphans_and_skips_preflight(settings: Any, samples: Dict[str, Any]) -> None:
    """Missing pages trigger a correction turn; short files skip the pre-flight call."""
    answers = [{"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": [1], "direction": "INBOUND",
                                       "tenant_context": "UNKNOWN", "confidence": 0.9}]},
               {"detected_entities": [{"type_tags": ["INVOICE", "BOGUS"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                       "tenant_context": "UNKNOWN", "confidence": 0.9}]}]

    def handler(request: LLMRequest) -> Optional[Any]:
        return answers.pop(0) if request.label.startswith("Stage 1.1") else None

    settings.pipeline.signature_min_pages = 99
    backend = FakeBackend(handler)
    result = Classifier(backend, settings, default_catalog()).run(samples["bank_statement_3p"].path, layer(3), None, None)
    assert backend.labels() == ["Stage 1.1 (Classification)", "Stage 1.1 Correction 1"]
    assert "SEGMENTATION_ERROR" in backend.requests[1].followups[1]["content"]
    assert result.entities[0].pages == [1, 2, 3] and result.entities[0].type_tags == ["INVOICE"]
    assert len(backend.requests[0].images) == 3 and backend.requests[0].system


def test_classifier_stack_split_with_preflight(settings: Any, samples: Dict[str, Any]) -> None:
    """Long files get a pre-flight; a stack yields several logical documents."""
    settings.pipeline.preflight_min_pages = 3
    settings.pipeline.signature_min_pages = 99

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.0"):
            return {"primary_type": "INVOICE", "is_hybrid_suspicion": False, "looks_like_stack": True, "confidence": 0.8}
        return {"detected_entities": [
            {"type_tags": ["INVOICE"], "page_indices": [1, 2], "direction": "INBOUND", "tenant_context": "PRIVATE", "confidence": 0.9},
            {"type_tags": ["BANK_STATEMENT"], "page_indices": [3], "direction": "INBOUND", "tenant_context": "PRIVATE", "confidence": 0.9}]}

    backend = FakeBackend(handler)
    result = Classifier(backend, settings, default_catalog()).run(samples["bank_statement_3p"].path, layer(3), None, None)
    assert result.strategy == HEADER_SCAN
    assert [(e.type_tags, e.pages) for e in result.entities] == [(["INVOICE"], [1, 2]), (["BANK_STATEMENT"], [3])]
    assert "Current Scan Mode: HEADER_SCAN_MODE" in backend.requests[1].prompt


def sig(page: int, **kwargs: Any) -> PageSignature:
    """A page signature."""
    return PageSignature(page=page, **kwargs)


def test_document_starts_from_signatures() -> None:
    """Cover letter, bank letter, enclosure form, blank page, multi-page statement."""
    signatures = [
        sig(1, letterhead=True, sender="Ombudsstelle Musterverband", date="2026-03-01", salutation=True),
        sig(2, closing=True),
        sig(3, letterhead=True, sender="Musterbank eG", date="2026-02-10", salutation=True),
        sig(4, form=True, title="Antrag auf Schlichtung", date="2026-02-20"),
        sig(5, blank=True),
        sig(6, letterhead=True, sender="Musterbank eG", page_number="Seite 1 von 2", title="Kontoauszug"),
        sig(7, letterhead=True, sender="Musterbank eG", page_number="Seite 2 von 2"),
        sig(8, letterhead=True, sender="Musterbank eG", date="2026-01-05"),
        sig(9, letterhead=True, sender="Musterbank eG", date="2026-01-07"),
    ]
    assert document_starts(signatures) == [1, 3, 4, 6, 9]


def test_no_split_without_positive_evidence() -> None:
    """Statement + 'Wichtige Informationen' page and a report numbered 1/1 then 2, 3 stay one document."""
    statement = [sig(1, letterhead=True, sender="Musterbank", date="2031-03-31", page_number="1/3"),
                 sig(2, letterhead=True, sender="Musterbank", date="2031-03-31", page_number="2/3"),
                 sig(3, letterhead=True, sender="Musterbank", date="2031-03-31", page_number="3/3"),
                 sig(4, title="Wichtige Informationen", boilerplate=True)]
    assert document_starts(statement) == [1]
    report = [sig(1, letterhead=True, date="2031-04-09", page_number="1/1", form=True, title="INSPECTION PROTOCOL"),
              sig(2, page_number="2", title="Measurement Results"), sig(3, page_number="3")]
    assert document_starts(report) == [1]
    two = [sig(1, letterhead=True, title="Vertrag", date="2031-06-01", addressee=True),
           sig(2, form=True, title="Antrag", date="2031-06-12", addressee=True)]
    assert document_starts(two) == [1, 2]


def test_boilerplate_entity_attached_to_statement() -> None:
    """The model's separate 'LETTER' for an information page is merged back."""
    from fluxsense.stages.classify import attach_boilerplate
    sigs = [sig(1, letterhead=True, date="2031-03-31", page_number="1/2"), sig(2, page_number="2/2"),
            sig(3, title="Wichtige Informationen", boilerplate=True)]
    merged = attach_boilerplate([Entity(["CREDIT_CARD_STATEMENT"], [1, 2]), Entity(["LETTER"], [3])], sigs)
    assert [(e.type_tags, e.pages) for e in merged] == [(["CREDIT_CARD_STATEMENT"], [1, 2, 3])]


def test_merged_stack_is_split_at_signature_starts(settings: Any, samples: Dict[str, Any]) -> None:
    """The model merges a 3-page stack into one document; signature starts split it."""
    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.0 signatures"):
            return {"pages": [{"page": 1, "blank": False, "letterhead": True, "sender": "Ombudsmann", "salutation": True, "closing": True, "form": False},
                              {"page": 2, "blank": False, "letterhead": True, "sender": "Musterbank eG", "salutation": True, "closing": False, "form": False},
                              {"page": 3, "blank": True, "letterhead": False, "salutation": False, "closing": False, "form": False}]}
        if request.label.startswith("Stage 1.1"):
            assert "each MUST start a new entity" in request.prompt and "): [1, 2]" in request.prompt
            return {"detected_entities": [{"type_tags": ["LEGAL_CORRESPONDENCE"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        return None

    result = Classifier(FakeBackend(handler), settings, default_catalog()).run(samples["bank_statement_3p"].path, layer(3), None, None)
    assert [e.pages for e in result.entities] == [[1], [2, 3]]
    assert result.starts == [1, 2]


def test_restored_prompt_text() -> None:
    """Prompt sections inherited from the KPaperFlux prompts are present."""
    assert '"primary_type"' in stage1.PROMPT_STAGE_1_0_PREFLIGHT and '"looks_like_stack"' in stage1.PROMPT_STAGE_1_0_PREFLIGHT
    assert "Document Continuity (Annexes, Reports & 'OTHER' Sink)" in stage1.PROMPT_STAGE_1_1_SYSTEM
    assert "### ALLOWED TYPES" in stage1.PROMPT_STAGE_1_2_SPLIT
    assert "### 4. EXTRACTION TARGET: {entity_type}" in stage2.PROMPT_STAGE_2_SYSTEM
    stage1.PROMPT_STAGE_1_0_PREFLIGHT.format(content="x", total_pages=1)


def test_identity_doubt_does_not_reclassify(settings: Any, samples: Dict[str, Any]) -> None:
    """A context mismatch is settled by the arbiter; without confirmation the context becomes UNKNOWN."""
    from fluxsense.models.identity import IdentityProfile
    settings.pipeline.signature_min_pages = 99

    def handler(request: LLMRequest) -> Optional[Any]:
        if request.label.startswith("Stage 1.1"):
            return {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": [1, 2, 3], "direction": "INBOUND",
                                           "tenant_context": "PRIVATE", "confidence": 0.9}]}
        if request.label.startswith("Arbiter"):
            return {"is_plausible": False, "suggested_context": "UNKNOWN"}
        return None

    backend = FakeBackend(handler)
    me = IdentityProfile(name="Zacharias Quux", address_keywords=["99999"])
    pages = TextLayer(pages=[PageText(page=i, sources={"native": "Muster GmbH Rechnung " * 5}) for i in (1, 2, 3)], engine="t")
    result = Classifier(backend, settings, default_catalog()).run(samples["bank_statement_3p"].path, pages, me, None)
    assert [l for l in backend.labels() if l.startswith("Stage 1.1")] == ["Stage 1.1 (Classification)"]
    assert result.entities[0].tenant_context == "UNKNOWN"


def test_coverage_repair_handles_zero_based_and_invalid_pages() -> None:
    """0-based answers are shifted; unusable page numbers keep the model's types."""
    assert [e.pages for e in repair_coverage([Entity(["INVOICE"], [0]), Entity(["LETTER"], [1])], 2)] == [[1], [2]]
    fixed = repair_coverage([Entity(["INVOICE"], [7])], 1)
    assert fixed[0].type_tags == ["INVOICE"] and fixed[0].pages == [1]



def test_zero_based_answer_is_not_retried(settings: Any, samples: Dict[str, Any]) -> None:
    """A 0-based classification answer is normalized instead of triggering correction turns."""
    settings.pipeline.signature_min_pages = 99
    backend = FakeBackend(lambda r: {"detected_entities": [{"type_tags": ["INVOICE"], "page_indices": [0], "direction": "INBOUND",
                                                            "tenant_context": "UNKNOWN", "confidence": 0.9}]})
    result = Classifier(backend, settings, default_catalog()).run(samples["invoice_wrong_vat"].path, layer(1), None, None)
    assert backend.labels() == ["Stage 1.1 (Classification)"] and result.entities[0].type_tags == ["INVOICE"]


def test_segmentation_refinements() -> None:
    """Repeated address block on a bill's detail page is no start; letter->form and new titled letterhead are."""
    bill = [sig(1, letterhead=True, title="Stromrechnung für März 2031", addressee=True),
            sig(2, letterhead=True, title="Ihre detaillierte Stromrechnung für", addressee=True)]
    assert document_starts(bill) == [1]
    letter_form = [sig(1, letterhead=True, salutation=True, closing=True, addressee=True),
                   sig(2, form=True, title="Erklärung zur Hausratversicherung"),
                   sig(3, form=True, title="Erklärung zur Hausratversicherung")]
    assert document_starts(letter_form) == [1, 2]
    two_letters = [sig(1, letterhead=True, salutation=True, title="Bestätigung der Kündigung", addressee=True),
                   sig(2, letterhead=True, title="Hinweise zum Tarifwechsel")]
    assert document_starts(two_letters) == [1, 2]


def test_enclosures_are_not_attached_but_boilerplate_is() -> None:
    """An app-log enclosure without letterhead stays separate; an information page is attached."""
    from fluxsense.stages.classify import attach_boilerplate
    sigs = [sig(1, letterhead=True, salutation=True), sig(2, title="App-Protokoll"), sig(3, boilerplate=True, title="Hinweise")]
    merged = attach_boilerplate([Entity(["LETTER"], [1]), Entity(["OTHER"], [2]), Entity(["LETTER"], [3])], sigs)
    assert [e.pages for e in merged] == [[1], [2, 3]]


def test_missing_signatures_are_requested_again(settings: Any, samples: Dict[str, Any]) -> None:
    """If the signature answer skips pages, the missing pages are asked again."""
    from fluxsense.stages.signatures import collect_signatures
    calls = []

    def handler(request: LLMRequest) -> Optional[Any]:
        calls.append(request.label)
        if "retry" in request.label:
            return {"pages": [{"page": p, "blank": False, "letterhead": False, "salutation": False, "closing": False, "form": False}
                              for p in (2, 3)]}
        return {"pages": [{"page": 1, "blank": False, "letterhead": True, "salutation": True, "closing": False, "form": False}]}

    result = collect_signatures(FakeBackend(handler), settings, samples["bank_statement_3p"].path, layer(3))
    assert [s.page for s in result] == [1, 2, 3] and any("retry" in c for c in calls)


def test_enclosures_without_letterhead_are_other() -> None:
    """A 'letter' without letterhead/salutation/address block is an enclosure -> OTHER (type regression fix)."""
    from fluxsense.stages.classify import retype_enclosures
    sigs = [sig(1, letterhead=True, salutation=True), sig(2, title="App-Protokoll"), sig(3, letterhead=True, addressee=True)]
    entities = retype_enclosures([Entity(["LEGAL_CORRESPONDENCE"], [1]), Entity(["OFFICIAL_LETTER"], [2]),
                                  Entity(["INVOICE"], [3])], sigs)
    assert [e.type_tags for e in entities] == [["LEGAL_CORRESPONDENCE"], ["OTHER"], ["INVOICE"]]


def test_type_cues_for_proceedings_reports_and_private_letters() -> None:
    """General cues: ombudsman/court proceedings, test reports, private letters (type regression fix)."""
    catalog = default_catalog()
    assert "ombudsman" in catalog.get("LEGAL_CORRESPONDENCE").description
    assert "Prüfbericht" in catalog.get("CERTIFICATE").description
    assert "screenshots" in catalog.get("OTHER").description
    assert catalog.get("LETTER").output_type == "OTHER"
    assert "LEGAL_CORRESPONDENCE" in stage1.PROMPT_STAGE_1_1_SYSTEM and "Enclosures without letterhead" in stage1.PROMPT_STAGE_1_1_SYSTEM


def test_layout_headers_and_closing_segmentation() -> None:
    """Report sections with titles but no own header are one document; the same running layout header is continuation;
    the page after a letter's closing starts an enclosure; layout page numbers fill the signature."""
    from fluxsense.extract.base import PageText, TextLayer
    from fluxsense.layout.items import FOOTER, HEADER, TEXT, LayoutItem
    from fluxsense.stages.signatures import apply_layout
    report = [sig(1, letterhead=True, form=True, title="INSPECTION PROTOCOL", addressee=True),
              sig(2, form=True, title="Measurement Results Unit 7"), sig(3), sig(4, form=True, title="Check list for emergency lighting"),
              sig(5, form=True, title="Check list for fire doors")]
    assert document_starts(report) == [1]
    letters = [sig(1, letterhead=True, salutation=True, addressee=True), sig(2, letterhead=True, closing=True),
               sig(3), sig(4, blank=True), sig(5, title="Letzter Umsatz")]
    assert document_starts(letters) == [1, 3]
    pages = [PageText(page=n, items=[LayoutItem(n, (50, 20, 950, 60), HEADER, "Beispiel Prüfdienst GmbH · Prüfbericht 42"),
                                     LayoutItem(n, (50, 100, 950, 200), TEXT, "Inhalt"),
                                     LayoutItem(n, (50, 950, 950, 980), FOOTER, f"Seite {n} von 2")]) for n in (1, 2)]
    same = [sig(1, letterhead=True, date="2031-01-01", title="Prüfbericht"),
            sig(2, letterhead=True, title="Messwerte")]
    apply_layout(same, TextLayer(pages=pages, engine="t"))
    assert same[0].header and same[0].header == same[1].header and same[1].page_number == "Seite 2 von 2"
    assert document_starts(same) == [1]
    same[1].page_number = None
    assert document_starts(same) == [1]
    same[1].header = "anderer kopf"
    assert document_starts(same) == [1, 2]


def test_continuity_signals_need_positive_start_evidence() -> None:
    """A long manual / offer with chapter title pages, a table of contents, a running header and page numbers stays one
    document; a new letterhead with a new date or a numbering restart still splits (synthetic)."""
    from fluxsense.extract.base import PageText, TextLayer
    from fluxsense.layout.items import HEADER, PAGE_NUMBER, TEXT, LayoutItem
    from fluxsense.stages.signatures import apply_layout
    offer = [sig(1, letterhead=True, salutation=True, closing=True, title="Angebot", addressee=True),
             sig(2, letterhead=True, title="Angebot"), sig(3), sig(4), sig(5)]
    pages = [PageText(page=n, items=[LayoutItem(n, (900, 950, 950, 980), PAGE_NUMBER, str(n))] if n >= 3 else [])
             for n in range(1, 6)]
    apply_layout(offer, TextLayer(pages=pages, engine="t"))
    assert offer[2].layout_page == 3 and document_starts(offer) == [1]
    manual = [sig(1, letterhead=True, title="Bedienungsanleitung"), sig(2, title="Inhaltsverzeichnis"),
              sig(3, letterhead=True, title="Kapitel 1 Sicherheit"), sig(4, letterhead=True, title="Kapitel 2 Montage"),
              sig(5, letterhead=True, title="Kapitel 3 Wartung")]
    texts = ["Bedienungsanleitung", "Inhaltsverzeichnis\n1 Sicherheit ..... 3\n2 Montage ..... 4\n3 Wartung ..... 5", "", "", ""]
    pages = [PageText(page=n, sources={"native": texts[n - 1]},
                      items=[LayoutItem(n, (50, 20, 950, 60), HEADER, "Beispielgerät X1 · Handbuch"),
                             LayoutItem(n, (50, 100, 950, 200), TEXT, "Inhalt")]) for n in range(1, 6)]
    apply_layout(manual, TextLayer(pages=pages, engine="t"))
    assert manual[1].toc and document_starts(manual) == [1]
    leaflet = [sig(1, letterhead=True, title="ZWEI TARIFE EIN PREIS", header="kundenmagazin 02 31 tarif zuhause tarif mobil"),
               sig(2, letterhead=True, title="NEUES TELEFON", header="tarif mobil")]
    assert document_starts(leaflet) == [1]
    stack = [sig(1, letterhead=True, date="2031-01-01", page_number="1/2", header="musterbank ag"),
             sig(2, page_number="2/2", header="musterbank ag"),
             sig(3, letterhead=True, date="2031-02-01", header="musterbank ag"), sig(4, header="musterbank ag")]
    assert document_starts(stack) == [1, 3]


def test_running_reference_bridges_a_cover_letter() -> None:
    """A cover letter naming the offer number that runs in the footers of the following offer pages is one document
    with them; the layout-only fields stay out of the classification prompt (synthetic)."""
    from fluxsense.extract.base import PageText, TextLayer
    from fluxsense.layout.items import FOOTER, PAGE_NUMBER, LayoutItem
    from fluxsense.stages.signatures import apply_layout, signatures_prompt
    offer = [sig(1, letterhead=True, salutation=True, closing=True, title="Angebot", addressee=True),
             sig(2, letterhead=True, title="Angebot"), sig(3), sig(4)]
    pages = [PageText(page=1, sources={"native": "Angebotsnummer: QX-31-0815\nSehr geehrte Damen und Herren,"})]
    pages += [PageText(page=n, sources={"native": "Leistungsbeschreibung"},
                       items=[LayoutItem(n, (900, 950, 950, 980), PAGE_NUMBER, str(n - 1)),
                              LayoutItem(n, (50, 950, 400, 980), FOOTER, "QX-31-0815")]) for n in (2, 3, 4)]
    apply_layout(offer, TextLayer(pages=pages, engine="t"))
    assert offer[0].mentions == ("QX-31-0815",) and document_starts(offer) == [1]
    prompt = signatures_prompt(offer, [1])
    assert "mentions" not in prompt and "layout_page" not in prompt
