# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_acceptance.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Provenance (label / geometry / check / model) and
                auto-acceptance of filled fields (field contract §0.5),
                semantic review rules and the trust-report counts.
                Synthetic values only.
------------------------------------------------------------------------------
"""

from typing import Any, Dict, List, Optional

from fluxsense.extract.base import PageText, TextLayer, Word
from fluxsense.grounding.acceptance import assess, checked_paths
from fluxsense.grounding.harness import trust_report
from fluxsense.models.result import CheckResult, CheckStatus, FieldEvidence, GroundingStatus

G = GroundingStatus.GROUNDED


def _line(x: float, y: float, text: str, size: float = 10.0) -> List[Word]:
    """Words of one line at x/y (points)."""
    words, cursor = [], x
    for token in text.split():
        width = len(token) * size * 0.5
        words.append(Word(cursor, y, cursor + width, y + size, token))
        cursor += width + size * 0.3
    return words


INVOICE_LINES = [
    "RECHNUNG", "Rechnungsnummer: RE-2031-0815", "Rechnungsdatum: 03.05.2031", "Auftragsdatum: 28.04.2031",
    "Nettobetrag 100,00", "MwSt 19,00", "Gesamtbetrag 119,00", "IBAN DE44 9901 2345 0532 0130 00", "Beispielwerk GmbH",
]


def _layer(lines: List[str], words: Optional[List[Word]] = None) -> TextLayer:
    """One-page layer with a native text source."""
    return TextLayer(pages=[PageText(page=1, words=words or [], sources={"native": "\n".join(lines)})], engine="t")


def _ev(path: str, kind: str, value: Any, status: GroundingStatus = G, snippet: Optional[str] = None) -> FieldEvidence:
    """Grounded evidence record."""
    return FieldEvidence(path=path, kind=kind, value=value, status=status, evidence=snippet)


def _invoice() -> Dict[str, Any]:
    """Fields of the synthetic invoice."""
    return {"meta_header": {"doc_number": "RE-2031-0815", "doc_date": "2031-05-03", "sender": {"company": "Beispielwerk GmbH"}},
            "bodies": {"finance_body": {"invoice_number": "RE-2031-0815",
                                        "monetary_summation": {"tax_basis_total_amount": "100.00", "tax_total_amount": "19.00",
                                                               "grand_total_amount": "119.00"},
                                        "payment_accounts": [{"iban": "DE44990123450532013000"}]}}}


def test_provenance_label_check_model() -> None:
    """Labelled number/date -> label; net + tax = gross and valid IBAN -> check; unplaced party -> model (flagged)."""
    fields = _invoice()
    evidence = [_ev("meta_header.doc_number", "identifier", "RE-2031-0815"), _ev("meta_header.doc_date", "date", "2031-05-03"),
                _ev("meta_header.sender.company", "text", "Beispielwerk GmbH")]
    counts = assess(fields, evidence, _layer(INVOICE_LINES), [1], "INVOICE", [])
    by_path = {e.path: e for e in evidence}
    assert by_path["meta_header.doc_number"].provenance == "label" and by_path["meta_header.doc_number"].auto_accepted
    assert by_path["meta_header.doc_date"].provenance == "label"
    assert by_path["bodies.finance_body.monetary_summation.grand_total_amount"].provenance == "check"
    assert by_path["bodies.finance_body.payment_accounts.0.iban"].provenance == "check"
    assert by_path["bodies.finance_body.payment_accounts.0.iban"].action == "synthesized"
    sender = by_path["meta_header.sender.company"]
    assert sender.provenance == "model" and sender.auto_accepted is False and "model reading only" in sender.review_reasons[0]
    assert counts["flagged"] == 1 and counts["auto_accepted"] == len(evidence) - 1


def test_never_label_date_and_ungrounded_values_are_flagged() -> None:
    """An order date is no doc_date label (model); an ungrounded labelled value and a failing check force a review."""
    fields = _invoice()
    fields["meta_header"]["doc_date"] = "2031-04-28"
    evidence = [_ev("meta_header.doc_date", "date", "2031-04-28"),
                _ev("meta_header.doc_number", "identifier", "RE-2031-0815", GroundingStatus.WEAKLY_GROUNDED)]
    checks = [CheckResult(check_id="invoice.arithmetic", rule="arithmetic", status=CheckStatus.FAIL,
                          field="bodies.finance_body.monetary_summation.grand_total_amount")]
    assess(fields, evidence, _layer(INVOICE_LINES), [1], "INVOICE", checks)
    by_path = {e.path: e for e in evidence}
    assert by_path["meta_header.doc_date"].provenance == "model" and not by_path["meta_header.doc_date"].auto_accepted
    assert any("not grounded" in r for r in by_path["meta_header.doc_number"].review_reasons)
    gross = by_path["bodies.finance_body.monetary_summation.grand_total_amount"]
    assert gross.provenance == "check" and gross.auto_accepted is False and "failing check" in gross.review_reasons[0]


def test_geometry_and_semantic_party_rules() -> None:
    """Address-window recipient -> geometry; sender == recipient and salutation parties are flagged."""
    words = (_line(70, 40, "Musterhaus AG", 16) + _line(62, 128, "Musterhaus AG · Postfach 1 · 00201 Berlin", 6)
             + _line(62, 158, "Erika Beispiel") + _line(62, 171, "Lindenweg 5") + _line(62, 184, "00345 Musterstadt")
             + _line(62, 300, "Sehr geehrter Herr Hans Probe,"))
    lines = ["Musterhaus AG", "Erika Beispiel", "Lindenweg 5", "00345 Musterstadt", "Sehr geehrter Herr Hans Probe,"]
    layer = _layer(lines, words)
    fields = {"meta_header": {"sender": {"company": "Musterhaus AG"}, "recipient": {"name": "Erika Beispiel"}}}
    evidence = [_ev("meta_header.sender.company", "text", "Musterhaus AG"), _ev("meta_header.recipient.name", "text", "Erika Beispiel")]
    assess(fields, evidence, layer, [1], "OFFICIAL_LETTER", [])
    assert [(e.provenance, e.auto_accepted) for e in evidence] == [("geometry", True), ("geometry", True)]
    fields = {"meta_header": {"sender": {"name": "Hans Probe"}, "recipient": {"name": "Erika Beispiel"}}}
    evidence = [_ev("meta_header.sender.name", "text", "Hans Probe")]
    assess(fields, evidence, layer, [1], "OFFICIAL_LETTER", [])
    assert "party is a salutation or greeting" in evidence[0].review_reasons
    fields = {"meta_header": {"sender": {"company": "Musterhaus AG"}, "recipient": {"company": "Musterhaus AG"}}}
    evidence = [_ev("meta_header.sender.company", "text", "Musterhaus AG"), _ev("meta_header.recipient.company", "text", "Musterhaus AG")]
    assess(fields, evidence, layer, [1], "OFFICIAL_LETTER", [])
    sender, recipient = evidence[:2]  # the letterhead supports the sender; the recipient is the suspect
    assert "sender == recipient" in recipient.review_reasons and not recipient.auto_accepted
    assert sender.provenance == "geometry" and sender.auto_accepted


def test_doc_date_after_stamp_and_unlabelled_number() -> None:
    """A doc_date later than a stamp date and a doc_number without a listed label are flagged."""
    lines = ["Musterstadt, den 10.06.2031", "Vorgang AB-77-123"]
    fields = {"meta_header": {"doc_date": "2031-06-10", "doc_number": "AB-77-123"}}
    evidence = [_ev("meta_header.doc_date", "date", "2031-06-10"), _ev("meta_header.doc_number", "identifier", "AB-77-123")]
    audit = {"layer_stamps": [{"raw_content": "Eingang 05.06.2031", "form_fields": []}]}
    assess(fields, evidence, _layer(lines), [1], "OFFICIAL_LETTER", [], audit)
    date_ev, number_ev = evidence
    assert date_ev.provenance == "label" and "doc_date after the stamp date 2031-06-05" in date_ev.review_reasons
    assert any("not printed next to a listed label" in r for r in number_ev.review_reasons)


def test_statement_balance_check_and_removed_values() -> None:
    """opening + bookings = closing -> check; removed/stale records carry no acceptance."""
    fields = {"bodies": {"statement_body": {"opening_balance": "100.00", "closing_balance": "70.00",
                                            "transactions": [{"amount": "-50.00"}, {"amount": "20.00"}]}}}
    assert {"bodies.statement_body.opening_balance", "bodies.statement_body.closing_balance"} <= checked_paths(fields)
    lines = ["alter Kontostand 100,00", "neuer Kontostand 70,00"]
    stale = _ev("meta_header.doc_number", "identifier", "999")
    removed = _ev("bodies.statement_body.iban", "iban", "DE00", GroundingStatus.REMOVED)
    evidence = [stale, removed]
    assess(fields, evidence, _layer(lines), [1], "BANK_STATEMENT", [])
    assert stale.auto_accepted is None and removed.auto_accepted is None
    closing = next(e for e in evidence if e.path.endswith("closing_balance"))
    assert closing.provenance == "check" and closing.auto_accepted


def test_zugferd_values_are_check_and_trust_counts() -> None:
    """A value equal to the embedded XML is 'check'; the trust report counts accepted / flagged / provenance."""
    fields = {"meta_header": {"doc_number": "X-1"}}
    evidence = [_ev("meta_header.doc_number", "identifier", "X-1")]
    assess(fields, evidence, _layer(["Beleg X-1"]), [1], "INVOICE", [], None, {"meta_header": {"doc_number": "X-1"}})
    assert evidence[0].provenance == "check" and evidence[0].auto_accepted
    report = trust_report(evidence + [_ev("meta_header.doc_date", "date", "2031-01-01")], [], [], 1.0, 0, 0)
    assert report.auto_accepted == 1 and report.flagged_for_review == 0 and report.provenance == {"check": 1}


def test_form_filler_label_and_service_marks() -> None:
    """A form sender next to a name-field label is 'label'; info-column labels as parties and stamp due dates are flagged."""
    lines = ["Kündigung", "Vorname und Name des Domaininhabers: Erika Muster", "Ihr Ansprechpartner", "ÜBERWIESEN 04.04.2031"]
    fields = {"meta_header": {"sender": {"name": "Erika Muster"}, "recipient": {"name": "Ihr Ansprechpartner"}},
              "bodies": {"finance_body": {"due_date": "2031-04-04"}}}
    evidence = [_ev("meta_header.sender.name", "text", "Erika Muster"), _ev("meta_header.recipient.name", "text", "Ihr Ansprechpartner"),
                _ev("bodies.finance_body.due_date", "date", "2031-04-04")]
    audit = {"layer_stamps": [{"raw_content": "ÜBERWIESEN 04.04.2031", "form_fields": []}]}
    assess(fields, evidence, _layer(lines), [1], "APPLICATION", [], audit)
    sender, recipient, due = evidence
    assert sender.provenance == "label" and sender.auto_accepted
    assert "party is an info-column label / postal service mark" in recipient.review_reasons
    assert "due date is a stamp date" in due.review_reasons


def test_statement_provenance() -> None:
    """Statement date from a table header column or the printed period end; balances after 'Saldo' in a table row or
    on the line below 'Kontosaldo per ...'; period_start = day after the printed previous statement date (§8)."""
    lines = ["Kontoauszug", "| Seite | Kartennummer | Abrechnungsdatum |", "| 1/2 | 4111 xxxx 1111 | 10. Januar 2031 |",
             "## Kontoauszug vom 01.12.2030 bis 10.01.2031", "| 28.12. | ALTER KONTOSTAND VOM 10.12.30 | 512,30 - |",
             "| Saldo EUR | -1.384,15 |", "Kontosaldo per 10.01.2031", "EUR + 18.905,40"]
    fields = {"meta_header": {"doc_date": "2031-01-10"},
              "bodies": {"statement_body": {"statement_date": "2031-01-10", "period_end": "2031-01-10", "period_start": "2030-12-11",
                                            "closing_balance": -1384.15, "opening_balance": 18905.40}}}
    evidence = [_ev("meta_header.doc_date", "date", "2031-01-10"), _ev("bodies.statement_body.period_end", "date", "2031-01-10"),
                _ev("bodies.statement_body.period_start", "date", "2030-12-11", GroundingStatus.UNGROUNDED),
                _ev("bodies.statement_body.closing_balance", "amount", -1384.15),
                _ev("bodies.statement_body.opening_balance", "amount", 18905.40)]
    assess(fields, evidence, _layer(lines), [1], "CREDIT_CARD_STATEMENT", [])
    by_path = {e.path: e for e in evidence}
    for path in ("meta_header.doc_date", "bodies.statement_body.period_end", "bodies.statement_body.closing_balance"):
        assert by_path[path].auto_accepted, (path, by_path[path].review_reasons)
    start = by_path["bodies.statement_body.period_start"]
    assert start.provenance == "check" and start.auto_accepted
    fields["bodies"]["statement_body"]["period_start"] = "2030-12-10"
    evidence = [_ev("bodies.statement_body.period_start", "date", "2030-12-10")]
    assess(fields, evidence, _layer(lines), [1], "CREDIT_CARD_STATEMENT", [])
    assert evidence[0].provenance == "model"  # the previous statement date itself is no period start


def test_dates_and_amounts_in_tables_and_date_lines() -> None:
    """'| Datum | 17/11/2031 |', a place/date line on page 1 and 'Gesamtsumme' for an amount due equal to the total."""
    lines = ["Beispiel Versand GmbH", "| Rechnung Nr. | 71K42 | Datum | 17/11/2031 |", "| Gesamtsumme (inkl. USt) EUR 314,57 |"]
    fields = {"meta_header": {"doc_date": "2031-11-17"},
              "bodies": {"finance_body": {"monetary_summation": {"grand_total_amount": 314.57, "due_payable_amount": 314.57}}}}
    evidence = [_ev("meta_header.doc_date", "date", "2031-11-17"),
                _ev("bodies.finance_body.monetary_summation.due_payable_amount", "amount", 314.57)]
    assess(fields, evidence, _layer(lines), [1], "INVOICE", [])
    assert [e.auto_accepted for e in evidence[:2]] == [True, True]
    letter = ["Musterstadt, den 21. September 2031", "Sehr geehrte Damen und Herren,"]
    evidence = [_ev("meta_header.doc_date", "date", "2031-09-21")]
    assess({"meta_header": {"doc_date": "2031-09-21"}}, evidence, _layer(letter), [1], "CERTIFICATE", [])
    assert evidence[0].provenance == "label"


def test_text_layout_parties() -> None:
    """Without word boxes: recipient from an address block under 'Frau', sender from the legal footer / closing block."""
    lines = ["Beispielbank AG", "Frau", "Erika Beispiel", "Lindenweg 5", "00345 Musterstadt", "Sehr geehrte Frau Beispiel,",
             "Text", "Mit freundlichen Grüßen", "Hans Probe", "Musterkasse AG - Vorstand: A. B. - Sitz Musterstadt HRB 1"]
    fields = {"meta_header": {"recipient": {"name": "Erika Beispiel"}, "sender": {"company": "Musterkasse AG", "name": "Hans Probe"}}}
    evidence = [_ev("meta_header.recipient.name", "text", "Erika Beispiel"), _ev("meta_header.sender.company", "text", "Musterkasse AG"),
                _ev("meta_header.sender.name", "text", "Hans Probe")]
    assess(fields, evidence, _layer(lines), [1], "OFFICIAL_LETTER", [])
    assert [(e.provenance, e.auto_accepted) for e in evidence] == [("geometry", True)] * 3


def test_never_party_rules_and_checks() -> None:
    """Titles, street-only values, department lines and form subjects are flagged; tax breakdown rows check out;
    a form's signature date is no due date; non-core fields are informational."""
    lines = ["PRÜFBERICHT", "Lindenallee 13", "Niederlassung Kundenservice Nord", "Name, Vorname des Mitglieds", "Probe, Hans",
             "Musterstadt 28.11.2031 Erika Muster"]
    fields = {"meta_header": {"sender": {"company": "PRÜFBERICHT"}, "recipient": {"name": "Lindenallee 13"},
                              "doc_date": "2031-11-28"},
              "bodies": {"correspondence_body": {"deadline": "2031-11-28"},
                         "finance_body": {"tax_breakdown": [{"tax_basis_amount": 100.0, "tax_amount": 19.0, "tax_rate": 19.0}]}}}
    evidence = [_ev("meta_header.sender.company", "text", "PRÜFBERICHT"), _ev("meta_header.recipient.name", "text", "Lindenallee 13"),
                _ev("bodies.correspondence_body.deadline", "date", "2031-11-28"),
                _ev("bodies.finance_body.tax_breakdown.0.tax_basis_amount", "amount", 100.0)]
    counts = assess(fields, evidence, _layer(lines), [1], "APPLICATION", [])
    reasons = {e.path: e.review_reasons for e in evidence}
    assert "party is a document heading / title" in reasons["meta_header.sender.company"]
    assert "party is a street / P.O. box / city, no name" in reasons["meta_header.recipient.name"]
    assert "a form's signature date is never a due date" in reasons["bodies.correspondence_body.deadline"]
    assert next(e for e in evidence if "tax_breakdown" in e.path).provenance == "check"
    fields = {"meta_header": {"sender": {"name": "Probe, Hans"}}}
    evidence = [_ev("meta_header.sender.name", "text", "Probe, Hans")]
    assess(fields, evidence, _layer(lines), [1], "APPLICATION", [])
    assert any("form subject" in r for r in evidence[0].review_reasons)
    fields = {"meta_header": {"sender": {"name": "Niederlassung Kundenservice Nord"}}}
    evidence = [_ev("meta_header.sender.name", "text", "Niederlassung Kundenservice Nord")]
    assess(fields, evidence, _layer(lines), [1], "OFFICIAL_LETTER", [])
    assert any("department" in r for r in evidence[0].review_reasons)
    fields = {"meta_header": {"sender": {"name": "Erika Muster", "phone": "0123 4567"}}}
    evidence = [_ev("meta_header.sender.phone", "text", "0123 4567")]
    counts = assess(fields, evidence, _layer(lines + ["Tel. 0123 4567"]), [1], "APPLICATION", [])
    assert counts["informational_flagged"] + counts["informational_auto_accepted"] == 1


def test_party_gate_and_geometry_policy() -> None:
    """Every party value passes the plausibility gate; a geometry/model conflict is a review reason; enclosures get no
    address-window geometry."""
    from fluxsense.layout.din5008 import GEOMETRY_CONFLICT
    words = _line(62, 158, "Erika Beispiel") + _line(62, 171, "Lindenweg 5") + _line(62, 184, "00345 Musterstadt")
    lines = ["Erika Beispiel", "Lindenweg 5", "00345 Musterstadt", "Meier", "Ihre Kundennummer"]
    layer = _layer(lines, words)
    for value, reason in (("Meier", "single name token"), ("Ihre Kundennummer", "form vocabulary"),
                          ("Q 12 7AB4 902C D3 E117 AF05", "ID / code string"), ("Inspection sheet - 2031-04-17", "date"),
                          ("c/o Hans Probe", "c/o line")):
        fields = {"meta_header": {"recipient": {"name": value}}}
        evidence = [_ev("meta_header.recipient.name", "text", value)]
        assess(fields, evidence, layer, [1], "OFFICIAL_LETTER", [])
        assert any(reason in r for r in evidence[0].review_reasons), (value, evidence[0].review_reasons)
    fields = {"meta_header": {"recipient": {"name": "Erika Beispiel"}}}
    evidence = [_ev("meta_header.recipient.name", "text", "Erika Beispiel")]
    conflict = CheckResult(check_id=GEOMETRY_CONFLICT, rule="layout", status=CheckStatus.FAIL, field="meta_header.recipient.name")
    assess(fields, evidence, layer, [1], "OFFICIAL_LETTER", [conflict])
    assert evidence[0].provenance == "geometry" and evidence[0].review_reasons == ["geometry/model conflict"]
    evidence = [_ev("meta_header.recipient.name", "text", "Erika Beispiel")]
    assess(fields, evidence, layer, [1], "OTHER", [])
    assert evidence[0].provenance == "model" and not evidence[0].auto_accepted


def test_letter_date_label_on_first_page_only() -> None:
    """A 'Datum' in a booking table on a later page is no letter date label."""
    pages = [PageText(page=1, sources={"native": "Musterbank AG\nSehr geehrte Frau Beispiel,"}),
             PageText(page=2, sources={"native": "| Nr. | Datum | Betrag |\n| 1. | 06.02.2031 | 725,00 |"})]
    fields = {"meta_header": {"doc_date": "2031-02-06"}}
    evidence = [_ev("meta_header.doc_date", "date", "2031-02-06")]
    assess(fields, evidence, TextLayer(pages=pages, engine="t"), [1, 2], "LEGAL_CORRESPONDENCE", [])
    assert evidence[0].provenance == "model"


def test_amount_checks_must_hold() -> None:
    """A ticket printing only the gross price and 'enthaltene MwSt 19 %: 2,42': a net equal to the gross is no check and
    is flagged; consistent net + tax = gross stays a check; a breakdown row with basis x rate != tax is flagged."""
    from fluxsense.grounding.acceptance import checked_paths, inconsistent_amounts
    lines = ["Fahrkarte Musterstadt - Beispielhausen", "Summe 15,20 EUR", "enthaltene MwSt 19 %: 2,42", "Netto 15,20"]
    ms = "bodies.finance_body.monetary_summation"
    fields = {"bodies": {"finance_body": {
        "monetary_summation": {"tax_basis_total_amount": 15.20, "line_total_amount": 15.20, "tax_total_amount": 2.42,
                               "grand_total_amount": 15.20},
        "line_items": [{"total_price": 15.20}],
        "tax_breakdown": [{"tax_basis_amount": 15.20, "tax_amount": 2.42, "tax_rate": 19}]}}}
    assert f"{ms}.tax_basis_total_amount" not in checked_paths(fields)
    assert {f"{ms}.tax_basis_total_amount", f"{ms}.line_total_amount", "bodies.finance_body.tax_breakdown.0.tax_basis_amount"} \
        <= inconsistent_amounts(fields)
    evidence = [_ev(f"{ms}.tax_basis_total_amount", "amount", 15.20), _ev(f"{ms}.grand_total_amount", "amount", 15.20)]
    assess(fields, evidence, _layer(lines), [1], "RECEIPT", [])
    net, gross = evidence[:2]
    assert not net.auto_accepted and "amount arithmetic inconsistent" in net.review_reasons and gross.auto_accepted
    ok = {"bodies": {"finance_body": {"monetary_summation": {"tax_basis_total_amount": 12.78, "tax_total_amount": 2.42,
                                                             "grand_total_amount": 15.20}}}}
    assert f"{ms}.tax_basis_total_amount" in checked_paths(ok) and not inconsistent_amounts(ok)


def test_transfer_confirmation_acceptance() -> None:
    """§2.8 review reasons: ordering customer as sender, payee as recipient, invoice totals; BLZ as tax ID."""
    lines = ["Beispielbank AG", "Überweisungsbeleg", "Auftraggeber: Erika Muster", "Empfänger: Musterwerk GmbH",
             "Betrag: 1.250,00 EUR", "BLZ 99012345"]
    fields = {"meta_header": {"sender": {"name": "Erika Muster", "tax_id": "99012345"},
                              "recipient": {"company": "Musterwerk GmbH"}},
              "bodies": {"finance_body": {"monetary_summation": {"grand_total_amount": 1250.0}}}}
    evidence = [_ev("meta_header.sender.name", "text", "Erika Muster"), _ev("meta_header.recipient.company", "text", "Musterwerk GmbH"),
                _ev("bodies.finance_body.monetary_summation.grand_total_amount", "amount", 1250.0),
                _ev("meta_header.sender.tax_id", "identifier", "99012345")]
    assess(fields, evidence, _layer(lines), [1], "TRANSFER_CONFIRMATION", [])
    reasons = [e.review_reasons for e in evidence[:4]]
    assert "the ordering customer is never the sender (§2.8)" in reasons[0]
    assert "the payee is a counterparty, not the recipient (§2.8)" in reasons[1]
    assert "a transfer amount is no invoice total (§2.8)" in reasons[2]
    assert any("not a valid tax number" in r for r in reasons[3])
    fields = {"meta_header": {"recipient": {"name": "Erika Muster"}}}
    evidence = [_ev("meta_header.recipient.name", "text", "Erika Muster")]
    assess(fields, evidence, _layer(lines), [1], "TRANSFER_CONFIRMATION", [])
    assert evidence[0].provenance == "label" and evidence[0].auto_accepted


def test_derived_period_values_are_grounded_in_their_text() -> None:
    """A period start normalised from a printed quarter is accepted although the ISO date is not printed verbatim."""
    from fluxsense.catalog.contract import derived_period_value
    lines = ["Rechnungsabschluss 3. Quartal 2031", "Kontosaldo per 30.09.2031"]
    assert derived_period_value("bodies.statement_body.period_start", "2031-07-01", lines)
    assert not derived_period_value("bodies.statement_body.period_start", "2031-07-02", lines)
    fields = {"bodies": {"statement_body": {"period_start": "2031-07-01"}}}
    evidence = [_ev("bodies.statement_body.period_start", "date", "2031-07-01", GroundingStatus.UNGROUNDED)]
    check = CheckResult(check_id="grounding.ungrounded", rule="grounding", status=CheckStatus.FAIL,
                        field="bodies.statement_body.period_start")
    assess(fields, evidence, _layer(lines), [1], "BANK_STATEMENT", [check])
    assert evidence[0].provenance == "label" and evidence[0].auto_accepted, evidence[0].review_reasons


def test_party_labels_form_issuer_and_same_party_rank() -> None:
    """Table party labels ('| Customer: | ... | Contractor: | ... |'), a form issuer in the header despite an address
    window, the first-page date line of a quote, and sender == recipient flagging only the weaker side (synthetic)."""
    lines = ["| Customer: | Beispiel Druck GmbH Lindenweg 5 00345 Musterstadt | Contractor: | Muster Prüftechnik GmbH |",
             "Ein Angebot von: Musterkasse AG, Ringstraße 1, 00345 Musterstadt"]
    fields = {"meta_header": {"recipient": {"company": "Beispiel Druck GmbH"}, "sender": {"company": "Muster Prüftechnik GmbH"}}}
    evidence = [_ev("meta_header.recipient.company", "text", "Beispiel Druck GmbH"),
                _ev("meta_header.sender.company", "text", "Muster Prüftechnik GmbH")]
    assess(fields, evidence, _layer(lines), [1], "CERTIFICATE", [])
    assert [e.provenance for e in evidence[:2]] == ["label", "label"]
    evidence = [_ev("meta_header.sender.company", "text", "Musterkasse AG")]
    assess({"meta_header": {"sender": {"company": "Musterkasse AG"}}}, evidence, _layer(lines), [1], "OTHER", [])
    assert evidence[0].provenance == "label"
    words = _line(62, 158, "Erika Beispiel") + _line(62, 171, "Lindenweg 5") + _line(62, 184, "00345 Musterstadt")
    form = ["Musterversicherung AG", "Ringstraße 1 / 00345 Musterstadt", "Schadenanzeige", "Erika Beispiel"]
    evidence = [_ev("meta_header.recipient.company", "text", "Musterversicherung AG")]
    assess({"meta_header": {"recipient": {"company": "Musterversicherung AG"}}}, evidence, _layer(form, words), [1],
           "APPLICATION", [])
    assert evidence[0].provenance == "geometry" and evidence[0].auto_accepted
    quote = ["Musterwerk GmbH", "30.06.2031", "Angebot"]
    evidence = [_ev("meta_header.doc_date", "date", "2031-06-30")]
    assess({"meta_header": {"doc_date": "2031-06-30"}}, evidence, _layer(quote), [1], "QUOTE", [])
    assert evidence[0].provenance == "label"


def test_deadline_label_needs_a_request_to_act() -> None:
    """'bis zum <Datum>' labels a letter's deadline only where the recipient is asked to act; an offer's validity
    ('gilt bei Buchung bis zum ...') is no deadline label."""
    path = "bodies.correspondence_body.deadline"
    offer = ["Ihr Vorteil", "Das Angebot gilt bei Buchung eines Tarifs bis zum 31.03.2031.", "Mit freundlichen Grüßen"]
    evidence = [_ev(path, "date", "2031-03-31")]
    assess({"bodies": {"correspondence_body": {"deadline": "2031-03-31"}}}, evidence, _layer(offer), [1], "OTHER", [])
    assert evidence[0].provenance != "label" and evidence[0].auto_accepted is False
    request = ["Bitte senden Sie uns die Unterlagen bis zum", "31.03.2031 zurück.", "Mit freundlichen Grüßen"]
    evidence = [_ev(path, "date", "2031-03-31")]
    assess({"bodies": {"correspondence_body": {"deadline": "2031-03-31"}}}, evidence, _layer(request), [1], "OTHER", [])
    assert evidence[0].provenance == "label"
    explicit = ["Ihre Stellungnahme erwarten wir spätestens am 31.03.2031."]
    evidence = [_ev(path, "date", "2031-03-31")]
    assess({"bodies": {"correspondence_body": {"deadline": "2031-03-31"}}}, evidence, _layer(explicit), [1], "OTHER", [])
    assert evidence[0].provenance == "label"
