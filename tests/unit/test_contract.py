# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_contract.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    The field contract data (fluxsense/catalog/contract.py)
                matches docs/field_contract.md row by row (party table,
                doc_date / doc_number label lists in priority order, 'never'
                lists), the catalog and prompts are generated from it, and
                the contract post-checks behave as specified. Synthetic
                values only.
------------------------------------------------------------------------------
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from fluxsense.catalog import contract
from fluxsense.catalog import default_catalog
from fluxsense.catalog.contract import adjacent_label, find_labelled, label_regex, labelled_dates, never_date_label

DOC = Path(__file__).resolve().parents[2] / "docs" / "field_contract.md"


def _section(title: str) -> str:
    """Markdown text of one '## ' section."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 3)
    return text[start:end if end > 0 else None]


def _table(section: str) -> List[List[str]]:
    """Body rows of the first markdown table of a section (cells stripped)."""
    rows = [line for line in section.splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[2:]]


def _ticks(cell: str) -> Tuple[str, ...]:
    """Backticked labels of a cell in order."""
    return tuple(re.findall(r"`([^`]+)`", cell))


def test_contract_version_matches_document() -> None:
    """The data mirrors the document version."""
    assert f"Version {contract.CONTRACT_VERSION} " in DOC.read_text(encoding="utf-8")


def test_party_table_matches_document() -> None:
    """§2: every row (case, sender, recipient) is in the data verbatim, in order."""
    rows = _table(_section("2. Parties"))
    expected: Dict[str, Tuple[str, str, str]] = {}
    for case, sender, recipient in rows:
        number, _, name = case.partition(" ")
        expected[number] = (name, sender, recipient)
    data = {k: (r.row, r.sender, r.recipient) for k, r in contract.PARTY_ROWS.items()}
    assert data == expected
    section = _section("2. Parties")
    for label in contract.PARTY_NEVER[1:] + contract.TITLE_WORDS:
        assert f"`{label}`" in section
    assert "`Sehr geehrte ...`" in section


def _check_rows(title: str, rows: Dict[str, contract.FieldRow]) -> None:
    """Rows of a §3/§4 table equal the data (first cell, meaning, backticked labels in priority order)."""
    table = _table(_section(title))
    by_row = {cells[0]: cells for cells in table}
    assert set(by_row) == {r.row for r in rows.values()}
    for row in rows.values():
        cells = by_row[row.row]
        assert cells[1] == row.meaning, row.row
        assert _ticks(cells[2]) == row.labels, row.row


def test_doc_date_labels_match_document() -> None:
    """§3 label lists and the 'never' list."""
    _check_rows("3. doc_date", contract.DOC_DATE_ROWS)
    never = _section("3. doc_date").split("Never a doc_date:")[1]
    assert _ticks(never) == contract.DOC_DATE_NEVER


def test_doc_number_labels_match_document() -> None:
    """§4 label lists and the 'never' examples."""
    _check_rows("4. doc_number", contract.DOC_NUMBER_ROWS)
    never = _section("4. doc_number").split("Never a doc_number:")[1]
    assert _ticks(never) == contract.DOC_NUMBER_NEVER


def test_every_contract_type_exists_in_the_catalog() -> None:
    """Type mapping covers only catalog types; the catalog carries the generated rules."""
    cat = default_catalog()
    for name, rows in contract.TYPE_CONTRACT.items():
        assert cat.has(name), name
        spec = cat.get(name)
        if rows.date_group or rows.number_group or rows.party_case:
            assert spec.doc_number_rule == contract.type_prompt_rules(name)
        for label in contract.number_labels(name):
            if rows.number_group in contract.STRICT_NUMBER_GROUPS:
                assert label_regex(label, compound_head=True) in spec.doc_number_labels
    assert "'Auszug', 'Kontoauszug Nr.', 'Abrechnung Nr.'" in cat.get("BANK_STATEMENT").doc_number_rule
    assert "handwritten allowed" in cat.get("APPLICATION").doc_number_rule
    assert cat.get("INVOICE").doc_number_labels == ()  # flagged by acceptance, not removed


def test_prompt_is_generated_from_the_contract() -> None:
    """The Stage 2 system prompt carries the party table and the 'never' lists."""
    from fluxsense.prompts.stage2 import PROMPT_STAGE_2_SYSTEM
    for row in contract.PARTY_ROWS.values():
        assert row.sender in PROMPT_STAGE_2_SYSTEM and row.recipient in PROMPT_STAGE_2_SYSTEM
    for label in contract.DOC_DATE_NEVER:
        assert f"'{label}'" in PROMPT_STAGE_2_SYSTEM
    PROMPT_STAGE_2_SYSTEM.format(entity_type="X", type_description="", entity_types="", body_keys="", questions="",
                                 audit_mission="", target_schema_json="{}", zugferd_guide="")


def test_label_regex_is_tolerant_but_word_bounded() -> None:
    """Dots, hyphens and number suffixes vary; a label never matches inside a longer word."""
    rech = re.compile(label_regex("Rechnungs-Nr."), re.IGNORECASE)
    for printed in ("Rechnungs-Nr.: 1", "Rechnungsnummer 1", "Rechnung Nr. 1", "RECHNUNGS NR 1"):
        assert rech.search(printed), printed
    datum = re.compile(label_regex("Datum"), re.IGNORECASE)
    assert not datum.search("Rechnungsdatum 01.02.2021") and datum.search("Datum: 01.02.2021")


def test_labelled_values_priority_and_adjacency() -> None:
    """Same line or directly below a label standing alone; earlier labels first; never labels excluded."""
    lines = ["Lieferdatum: 01.02.2021", "Datum fertig 02.02.2021", "Datum: 05.02.2021", "Rechnungsdatum", "04.02.2021"]
    assert [str(d) for d in labelled_dates("INVOICE", lines)] == ["2021-02-04", "2021-02-05"]
    assert never_date_label("2021-02-01", lines) == "Lieferdatum" and never_date_label("2021-02-05", lines) is None
    assert never_date_label("2028-09-14", [], "Auftr.-Dat. : 14.09.28") == "Auftragsdatum"
    refs = ["Ihr Zeichen: 77-1", "Kundennummer 123456", "Unser Zeichen  AB/4711 vom 03.02.2021"]
    hits = find_labelled("identifier", contract.number_labels("OFFICIAL_LETTER"), refs)
    assert [(h.label, h.value) for h in hits] == [("Unser Zeichen", "AB/4711"), ("Kundennummer", "123456")]
    assert adjacent_label("identifier", "123456", contract.number_labels("OFFICIAL_LETTER"), refs).label == "Kundennummer"
    assert adjacent_label("identifier", "77-1", contract.number_labels("OFFICIAL_LETTER"), refs) is None
    form = ["Musterstadt, 12.06.2022", "Ort, Datum", "Unterschrift"]
    assert [str(d) for d in labelled_dates("APPLICATION", form)] == ["2022-06-12"]


def test_contract_date_post_checks() -> None:
    """A labelled date beats an unlabelled one; a 'never' date without labelled alternative is removed;
    form dates are exempt from the stamp rule."""
    from fluxsense.stages.repair import fix_document_dates
    fields = {"meta_header": {"doc_date": "2021-02-01"}}
    fix_document_dates(fields, [], "Vertrag\nausgestellt am 10.02.2021\nBeginn 01.02.2021", None, "CONTRACT")
    assert fields["meta_header"]["doc_date"] == "2021-02-10"
    fields = {"meta_header": {"doc_date": "2021-02-01"}}
    notes = fix_document_dates(fields, [], "Leistungsdatum: 01.02.2021", None, "OFFICIAL_LETTER")
    assert "doc_date" not in fields["meta_header"] and "removed" in notes[0]
    fields = {"meta_header": {"doc_date": "2021-03-03"}}
    audit = {"layer_stamps": [{"raw_content": "Eingang 03.03.2021", "form_fields": []}]}
    fix_document_dates(fields, [], "Antrag\nOrt, Datum: 03.03.2021", audit, "APPLICATION")
    assert fields["meta_header"]["doc_date"] == "2021-03-03"
    fields = {"meta_header": {"doc_date": "2021-03-03"}}
    fix_document_dates(fields, [], "Musterstadt, den 28.02.2021\nEingang 03.03.2021", audit, "OFFICIAL_LETTER")
    assert fields["meta_header"]["doc_date"] == "2021-02-28"
    fields = {"meta_header": {"doc_date": "2021-04-01"}}
    fix_document_dates(fields, [], "Musterstadt\n01.04.2021\nTermin am 12.04.2021", None, "LETTER")
    assert fields["meta_header"]["doc_date"] == "2021-04-01"


def test_contract_number_post_checks() -> None:
    """Earlier labels beat later ones; statement customer numbers, table cells and headings are never doc_numbers."""
    from fluxsense.stages.repair import contract_never_number, fix_document_number
    letter = "\n".join(["Kundennummer 555123", "Unser Zeichen VG-2021-0042", "Sehr geehrte Frau Muster"])
    fields = {"meta_header": {"doc_number": "555123"}}
    fix_document_number(fields, [], letter, False, "OFFICIAL_LETTER")
    assert fields["meta_header"]["doc_number"] == "VG-2021-0042"
    fields = {"meta_header": {"doc_number": "VG-2021-0042"}}
    assert fix_document_number(fields, [], letter, False, "OFFICIAL_LETTER") == []
    statement = "Kontoauszug\nKundennummer 98765432\nAuszug 7"
    assert contract_never_number({}, "98765432", "BANK_STATEMENT", statement) is not None
    tx = {"bodies": {"statement_body": {"transactions": [{"description": "Kartenzahlung REF 77TX99812"}]}}}
    assert contract_never_number(tx, "77TX99812", "OTHER", "Kartenzahlung REF 77TX99812") == "is a table cell / transaction ID"
    assert contract_never_number({}, "RK7", "INVOICE", "RECHNUNG RK7\nPosition 1") == "is part of a heading"
    assert contract_never_number({}, "880412-29", "INVOICE", "RECHNUNG RK7\nRech-Nr. : 880412-29") is None


def test_unfilled_form_is_a_cover_sheet() -> None:
    """§2.5: a form with audit but neither signature nor handwriting is a pre-addressed cover sheet."""
    from fluxsense.stages.identity import unfilled_form
    assert unfilled_form({"signatures": {"has_signature": False}, "audit_summary": {"has_handwriting": False}})
    assert not unfilled_form({"signatures": {"has_signature": True}, "audit_summary": {"has_handwriting": False}})
    assert not unfilled_form({"signatures": {"has_signature": False}, "audit_summary": {"has_handwriting": True}})
    assert not unfilled_form(None) and not unfilled_form({"audit_summary": {}})


def test_never_a_party_and_form_filler() -> None:
    """Info-column labels and postal service marks are no parties; a form's filler is read from its name field (§2.4)."""
    from fluxsense.catalog.contract import form_filler, not_a_party
    from fluxsense.stages.identity import remove_role_labels
    assert not_a_party("Ihr Ansprechpartner") and not_a_party("PREMIUMADRESS") and not not_a_party("Erika Muster")
    meta = {"recipient": {"name": "Ihr Ansprechpartner", "company": "Beispiel Versicherung AG"}, "sender": {"company": "PREMIUMADRESS"}}
    remove_role_labels(meta)
    assert meta == {"recipient": {"company": "Beispiel Versicherung AG"}, "sender": {}}
    lines = ["Kündigungsformular", "Vorname und Name des Ansprechpartners / Domaininhabers: Erika Muster", "Auftragsnummer 123456"]
    assert form_filler(lines) == ("Vorname und Name", "Erika Muster")
    assert form_filler(["Name, Vorname", "Probe Hans"]) == ("Name, Vorname", "Probe Hans")
    assert form_filler(["Name: ________"]) is None
    hits = find_labelled("identifier", contract.number_labels("APPLICATION"), lines)
    assert [(h.label, h.value) for h in hits] == [("Auftragsnummer", "123456")]


def test_dates_and_periods_are_no_numbers_and_stamp_due_dates() -> None:
    """A doc_number that is a date or period is removed; a due date equal to a stamp date is replaced or removed."""
    from fluxsense.stages.repair import clean_document_numbers, fix_document_dates, is_date_or_period
    for value in ("23. Juli 1998", "3. Quartal 2024", "27 MAI 09", "01.02.2024", "Q3/2024", "KW 12 2024"):
        assert is_date_or_period(value), value
    for value in ("880412-29", "RE-2024-05", "4711", "2024-05"):
        assert not is_date_or_period(value), value
    fields = {"meta_header": {"doc_number": "3. Quartal 2024"}}
    clean_document_numbers(fields)
    assert "doc_number" not in fields["meta_header"]
    from tests.unit.test_repair import garage_invoice
    doc = garage_invoice()  # generated: paid stamp line and a printed 'Zahlbar bis' date
    stamp = doc.facts["stamp_date"]
    unpaid = "\n".join(line for line in doc.text().splitlines() if not line.startswith("Zahlbar bis"))
    fields = {"meta_header": {}, "bodies": {"finance_body": {"due_date": stamp}}}
    fix_document_dates(fields, [], unpaid, None, "INVOICE")
    assert "due_date" not in fields["bodies"]["finance_body"]
    fields = {"meta_header": {}, "bodies": {"finance_body": {"due_date": stamp}}}
    fix_document_dates(fields, [], doc.text(), None, "INVOICE")
    assert fields["bodies"]["finance_body"]["due_date"] == doc.truth["due_date"]


def test_payee_accounts() -> None:
    """§7: payee accounts in reading order (first = primary); a credit note keeps only the payout account."""
    from fluxsense.models.result import FieldEvidence
    from fluxsense.stages.repair import order_payee_accounts
    first, second = "DE44990123450532013000", "DE02120300000000202051"
    text = f"Bankverbindung 1: {second}\nBankverbindung 2: {first}"
    fields = {"bodies": {"finance_body": {"payment_accounts": [{"iban": first}, {"iban": second}]}}}
    evidence = [FieldEvidence(path="bodies.finance_body.payment_accounts.0.iban", kind="iban", value=first)]
    order_payee_accounts(fields, evidence, text, False)
    assert [a["iban"] for a in fields["bodies"]["finance_body"]["payment_accounts"]] == [second, first]
    assert evidence[0].path == "bodies.finance_body.payment_accounts.1.iban"
    credit = f"Den Betrag erstatten wir auf Ihr Konto\nIBAN {first}\nUnsere Bankverbindung: {second}"
    fields = {"bodies": {"finance_body": {"payment_accounts": [{"iban": second}, {"iban": first}]}}}
    order_payee_accounts(fields, [], credit, True)
    assert fields["bodies"]["finance_body"]["payment_accounts"] == [{"iban": first}]
    fields = {"bodies": {"finance_body": {"payment_accounts": [{"iban": second}]}}}
    order_payee_accounts(fields, [], f"Unsere Bankverbindung: {second}", True)
    assert "payment_accounts" not in fields["bodies"]["finance_body"]


def test_private_notes_have_no_parties() -> None:
    """§2.7: notes, cards and memos map to a party row with empty sender and recipient."""
    assert contract.contract_for("NOTE").party_case == "2.7"
    assert contract.PARTY_ROWS["2.7"].sender == "empty" and contract.PARTY_ROWS["2.7"].recipient == "empty"


def test_table_labels_date_lines_and_form_filler() -> None:
    """Labels inside pipe tables (next cell / same column below), slash and month-name dates, date and signature lines,
    and the form filler outside the subject block."""
    from fluxsense.catalog.contract import date_line_dates, form_filler, signature_line
    rows = ["| Rechnung Nr. | 71K42 | Datum | 17/11/2031 |", "| Seite | Abrechnungsdatum |", "| 1/2 | 10. Januar 2031 |",
            "Kartenkonto", "| Rechnungsdatum 27 MAI 31 | Neuer Saldo |"]
    assert {h.value for h in find_labelled("date", contract.date_labels("INVOICE"), rows)} == {"2031-11-17", "2031-05-27"}
    assert [h.value for h in find_labelled("date", contract.date_labels("BANK_STATEMENT"), rows)] == ["2031-01-10"]
    assert [h.value for h in find_labelled("identifier", ("Rechnungsnummer",), rows)] == ["71K42"]
    assert date_line_dates(["Musterstadt, den 21. September 2031", "00345 Musterstadt, 09.07.2031", "14.10.2031 18:33",
                            "Text 01.01.2031 mit mehr"]) == ["2031-09-21", "2031-07-09", "2031-10-14"]
    assert signature_line("Musterstadt 28.11.2031 Erika Muster") == ("2031-11-28", "Erika Muster")
    form = ["Name, Vorname des Mitglieds", "Probe Hans", "Antragsteller", "Name: Erika Muster"]
    assert form_filler(form) == ("Name", "Erika Muster")
    assert form_filler(["| Firma | Vorname und Name des Ansprechpartners |", "|  | Erika Muster |"]) == ("Vorname und Name", "Erika Muster")


def test_post_checks_keep_correct_values() -> None:
    """A printed period keeps period_start; an unlabelled form number is kept (flagged later); a log-entry ID is no number;
    a form's signature date is no deadline; the address window never yields 'Ihr Ansprechpartner' or a street."""
    from fluxsense.layout.din5008 import recipient_from_window
    from fluxsense.stages.repair import contract_never_number, drop_form_deadline, enforce_doc_semantics, fix_statement_semantics
    from fluxsense.models.result import FieldEvidence
    body = {"statement_date": "2031-08-30", "period_start": "2031-08-01", "period_end": "2031-08-30"}
    evidence = [FieldEvidence(path="bodies.statement_body.period_start", kind="date", value="2031-08-01", evidence="01.08.2031")]
    fix_statement_semantics(body, evidence, "## Kontoauszug vom 01.08.2031 bis 30.08.2031")
    assert body["period_start"] == "2031-08-01"
    fix_statement_semantics(body, evidence, "Kontoauszug")
    assert "period_start" not in body
    fields = {"meta_header": {"doc_number": "42/0815"}}
    enforce_doc_semantics(fields, [], default_catalog().get("APPLICATION"), "Mitglieds-Nr.: 42/0815", True)
    assert fields["meta_header"]["doc_number"] == "42/0815"
    assert contract_never_number({}, "55501234170420310b", "OTHER", "17.04.2031 09:41 Zahlung ID 55501234170420310b") is not None
    fields = {"meta_header": {"doc_date": "2031-07-05"}, "bodies": {"correspondence_body": {"deadline": "2031-07-05"}}}
    assert drop_form_deadline(fields, "APPLICATION") and "deadline" not in fields["bodies"]["correspondence_body"]
    assert recipient_from_window(["Ihr Ansprechpartner", "Erika Beispiel", "Lindenweg 5", "00345 Musterstadt"])["name"] == "Erika Beispiel"
    assert recipient_from_window(["Lindenallee 13", "00345 Musterstadt"]) is None


def test_form_letter_and_report_number_labels() -> None:
    """Form member numbers and 'Unser(e) Zeichen'; enclosure 'Unser Zeichen'; authority case numbers only for
    courts / ombudsman bodies / authorities; 'No.'/'Nr.' only in a report's header area; 'Rechnungsdatum' only on
    credit-card statements."""
    from fluxsense.catalog.contract import date_labels, number_labels, own_number_hits
    form = number_labels("APPLICATION")
    assert form.index("Vertragsnummer") < form.index("Mitglieds-Nr.") < form.index("Unser Zeichen") < form.index("Auftragsnummer")
    assert [h.value for h in own_number_hits("APPLICATION", ["Mitglieds-Nr.: 42/0815"])] == ["42/0815"]
    assert [h.value for h in own_number_hits("APPLICATION", ["Unsere Zeichen: QZ07"])] == ["QZ07"]
    assert [h.value for h in own_number_hits("OTHER", ["17.04.2031 Protokoll", "Unser Zeichen: 47-3920"])] == ["47-3920"]
    ombudsman = ["Ombudsstelle des Musterverbands", "Musterstraße 1", "Betreff: Schlichtungsverfahren 29-Q7741"]
    assert [h.label for h in own_number_hits("LETTER", ombudsman)][:1] == ["Verfahren"]
    company = ["Musterwerk GmbH", "Musterstraße 1", "Ihr Schlichtungsverfahren 29-Q7741"]
    assert own_number_hits("LETTER", company) == []
    court = ["Amtsgericht Musterstadt", "Az. 12 C 345/31"]
    assert [h.value for h in own_number_hits("OFFICIAL_LETTER", court)] == ["12 C 345/31"]
    report = ["Beispiel Prüfdienst GmbH", "Inspection report No.: 2031-4-9_CD34"] + ["Text"] * 20 + ["Nr. 4711"]
    assert {h.value for h in own_number_hits("CERTIFICATE", report)} == {"2031-4-9_CD34"}
    assert "Rechnungsdatum" in date_labels("CREDIT_CARD_STATEMENT") and "Rechnungsdatum" not in date_labels("BANK_STATEMENT")


def test_table_column_header_is_no_document_label() -> None:
    """A 'Datum' header of a multi-row table (instalment plan, bookings) labels a column, not the document date;
    the first-page date line of an invoice is kept."""
    from fluxsense.stages.repair import fix_document_dates
    plan = ["| Datum | Abschlagsbetrag in EUR |", "| 01.07.2031 | 455,00 |", "| 01.08.2031 | 455,00 |"]
    assert find_labelled("date", contract.date_labels("INVOICE"), plan) == []
    single = ["| Rechnungsdatum | Kundennummer |", "| 23.05.2031 | 4711 |"]
    assert [h.value for h in find_labelled("date", contract.date_labels("INVOICE"), single)] == ["2031-05-23"]
    fields = {"meta_header": {"doc_date": "2031-05-23"}}
    fix_document_dates(fields, [], "\n".join(["Beispiel Energie GmbH", "23.05.2031", "Abschlagsplan"] + plan), None, "INVOICE")
    assert fields["meta_header"]["doc_date"] == "2031-05-23"


def test_bank_transfer_confirmation() -> None:
    """§2.8: routing by vocabulary, parties (bank = sender, ordering customer = recipient, payee = counterparty), no invoice
    totals; §7: BLZ / account numbers are no tax ID or IBAN (synthetic)."""
    from fluxsense.catalog.contract import is_bank_transfer_confirmation
    from fluxsense.formats import valid_tax_id
    from fluxsense.extract.base import PageText, TextLayer
    from fluxsense.stages.classify import Entity, route_transfer_confirmations
    from fluxsense.stages.repair import fix_transfer_confirmation, sanitize_bank_identifiers
    import random
    from fluxsense.testdata import build_transfer
    doc = build_transfer(random.Random("transfer-1"))  # generated confirmation (seed recorded)
    text, customer, payee = doc.text(), doc.facts["customer"], doc.facts["payee"]
    assert is_bank_transfer_confirmation(text) and not is_bank_transfer_confirmation("Auftragsbestätigung Ihrer Bestellung")
    layer = TextLayer(pages=[PageText(page=1, sources={"native": text})], engine="t")
    entities = route_transfer_confirmations([Entity(type_tags=["RECEIPT"], pages=[1])], layer)
    assert entities[0].type_tags == ["TRANSFER_CONFIRMATION"]
    assert default_catalog().get("TRANSFER_CONFIRMATION").output_type == "BANK_STATEMENT"
    assert contract.contract_for("TRANSFER_CONFIRMATION").party_case == "2.8"
    fields = {"meta_header": {"sender": {"name": customer}, "recipient": {"company": payee}},
              "bodies": {"finance_body": {"monetary_summation": {"grand_total_amount": doc.facts["amount"]}}}}
    notes = fix_transfer_confirmation(fields, text)
    assert fields["meta_header"] == {"recipient": {"name": customer}} and "monetary_summation" not in fields["bodies"]["finance_body"]
    assert len(notes) == 2
    bank, iban = doc.truth["sender"], doc.truth["iban"]
    assert not valid_tax_id(doc.facts["bank_code"]) and valid_tax_id("123/456/78901") and valid_tax_id("DE123456789", vat=True)
    fields = {"meta_header": {"sender": {"company": bank, "tax_id": doc.facts["bank_code"], "iban": doc.facts["account"]}},
              "bodies": {"statement_body": {"iban": iban}}}
    sanitize_bank_identifiers(fields)
    assert fields["meta_header"]["sender"] == {"company": bank}
    assert fields["bodies"]["statement_body"]["iban"] == iban


def test_info_column_header_and_form_question_dates() -> None:
    """'Unsere Zeichen, Nachricht' with the value below beats a later insurance number; a form's 'Datum' that answers a
    question is no doc_date, the date next to the signature is (synthetic)."""
    from fluxsense.catalog.contract import form_signature_dates, own_number_hits
    from fluxsense.stages.repair import fix_document_number
    letter = ["Ihre Zeichen, Nachricht", "Unsere Zeichen, Nachricht", "XQ4711", "Hausratversicherung Nr. 31415926535"]
    assert [h.value for h in own_number_hits("OFFICIAL_LETTER", letter)][:1] == ["XQ4711"]
    fields = {"meta_header": {"doc_number": "XQ4711"}}
    assert fix_document_number(fields, [], "\n".join(letter), False, "OFFICIAL_LETTER") == [] and \
        fields["meta_header"]["doc_number"] == "XQ4711"
    form = ["3. Wann ist der Schaden entstanden?", "Datum: 23.6.2031", "Uhrzeit: 17:30", "Musterstadt 28.06.2031",
            "Unterschrift des Versicherungsnehmers"]
    assert form_signature_dates(form) == ["2031-06-28"]
    assert form_signature_dates(form[:3]) == []
