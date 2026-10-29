# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_repair.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Transactions without amount are filled from the
                running balance or the matching text line; signed amount
                parsing of statement lines.
------------------------------------------------------------------------------
"""

from decimal import Decimal
from typing import Any

from fluxsense.extract.base import PageText, TextLayer
from fluxsense.stages.repair import fill_row_amounts, line_amounts

PAGE = ("09.09. 09.09. Stadtwerke Musterstadt Abschlag 85,00- 1.415,00\n"
        "10.09. 10.09. Gehalt Beispiel AG 2.450,00 3.865,00\n"
        "11.09. 11.09. REWE Markt 12,34 S")


def layer() -> TextLayer:
    """One-page statement text."""
    return TextLayer(pages=[PageText(page=9, sources={"native": PAGE})], engine="t")


def test_line_amounts_signs() -> None:
    """Trailing minus and S markers make amounts negative."""
    assert line_amounts("Abschlag 85,00- 1.415,00") == [Decimal("-85.00"), Decimal("1415.00")]
    assert line_amounts("REWE 12,34 S") == [Decimal("-12.34")]
    assert line_amounts("Gehalt -2.450,00") == [Decimal("-2450.00")]


def test_fill_from_balance_difference() -> None:
    """With running balances, the amount is balance minus previous balance."""
    rows = [{"booking_date": "2026-09-09", "description": "Stadtwerke", "amount": None, "balance": "1415.00", "source_page": 9},
            {"booking_date": "2026-09-10", "description": "Gehalt", "balance": "3865.00", "source_page": 9}]
    repair = fill_row_amounts(rows, "1500.00", layer())
    assert repair.filled == [0, 1] and repair.methods == {0: "balance_difference", 1: "balance_difference"}
    assert Decimal(rows[0]["amount"]) == Decimal("-85.00") and Decimal(rows[1]["amount"]) == Decimal("2450.00")


def test_fill_from_text_line_without_balance() -> None:
    """Without balances, the single non-balance amount of the matching line is used."""
    rows = [{"booking_date": "2026-09-11", "description": "REWE Markt", "source_page": 9},
            {"booking_date": "2026-09-30", "description": "Unbekannt", "source_page": 9}]
    repair = fill_row_amounts(rows, None, layer())
    assert repair.filled == [0] and Decimal(rows[0]["amount"]) == Decimal("-12.34")
    assert repair.unresolved == [1]


def test_dates_normalized_to_iso() -> None:
    """German, short and month-name dates become ISO; non-dates are untouched."""
    from fluxsense.stages.repair import normalize_dates
    fields = {"meta_header": {"doc_date": "07.11.2013", "doc_number": "07.11.2013-A"},
              "bodies": {"finance_body": {"due_date": "1. April 2026", "invoice_date": "2026-03-18"},
                         "statement_body": {"period_start": "01.09.26", "transactions": [{"booking_date": "05.09.2026"}]}}}
    changed = normalize_dates(fields)
    assert fields["meta_header"]["doc_date"] == "2013-11-07" and fields["meta_header"]["doc_number"] == "07.11.2013-A"
    assert fields["bodies"]["finance_body"]["due_date"] == "2026-04-01"
    assert fields["bodies"]["statement_body"]["period_start"] == "2026-09-01"
    assert fields["bodies"]["statement_body"]["transactions"][0]["booking_date"] == "2026-09-05"
    assert "bodies.finance_body.invoice_date" not in changed and len(changed) == 4


def test_form_codes_are_not_document_numbers() -> None:
    """A footer/margin print code without reference label is rejected; 'Unser Zeichen' is kept."""
    from fluxsense.extract.base import Word
    from fluxsense.models.semantic import MetaHeader
    from fluxsense.stages.repair import is_form_code
    page = PageText(page=1, width=595, height=842, words=[
        Word(60, 100, 120, 110, "Unser"), Word(122, 100, 170, 110, "Zeichen:"), Word(172, 100, 240, 110, "HV-2026-117"),
        Word(40, 815, 120, 825, "XQFRM001234"), Word(575, 400, 590, 470, "DV01.234.56")])
    assert is_form_code("XQFRM001234", "XQFRM001234", page)
    assert is_form_code("DV01.234.56", None, page)
    assert not is_form_code("HV-2026-117", "Unser Zeichen: HV-2026-117", page)
    assert not is_form_code("HV-2026-117", None, page)
    assert "Unser Zeichen" in MetaHeader.model_fields["doc_number"].description


def test_yearless_statement_dates_completed() -> None:
    """'01.09.' takes the year of the statement period; December rows of a January statement the previous year."""
    from fluxsense.stages.repair import complete_statement_years
    body = {"period_end": "2026-01-31", "transactions": [{"booking_date": "30.12.", "value_date": "02.01."}, {"booking_date": "2026-01-05"}]}
    assert complete_statement_years(body) == 2
    assert body["transactions"][0] == {"booking_date": "2025-12-30", "value_date": "2026-01-02"}


def test_statement_semantics_sign_and_period() -> None:
    """'512,30 -' is negative; without a printed period, period_end = statement date and period_start is dropped."""
    from fluxsense.models.result import FieldEvidence
    from fluxsense.stages.repair import fix_statement_semantics
    body = {"opening_balance": "512.30", "closing_balance": "-48.17", "statement_date": "2024-07-10",
            "period_start": "2024-06-11", "period_end": "2024-07-05"}
    evidence = [FieldEvidence(path="bodies.statement_body.opening_balance", kind="amount", value=512.30, evidence="512,30 -"),
                FieldEvidence(path="bodies.statement_body.period_end", kind="date", value="2024-07-05", evidence="05.07."),
                FieldEvidence(path="bodies.statement_body.period_start", kind="date", value="2024-06-11", evidence="11.06.")]
    notes = fix_statement_semantics(body, evidence)
    assert body["opening_balance"] == "-512.30" and body["period_end"] == "2024-07-10" and "period_start" not in body
    assert len(notes) == 3
    kept = {"period_start": "2024-06-01", "period_end": "2024-06-30", "statement_date": "2024-07-02"}
    fix_statement_semantics(kept, [FieldEvidence(path="bodies.statement_body.period_end", kind="date", value="2024-06-30",
                                                 evidence="Zeitraum 01.06.2024 bis 30.06.2024"),
                                   FieldEvidence(path="bodies.statement_body.period_start", kind="date", value="2024-06-01",
                                                 evidence="Zeitraum 01.06.2024 bis 30.06.2024")])
    assert kept["period_end"] == "2024-06-30" and kept["period_start"] == "2024-06-01"


def test_sender_from_table_cell_is_rejected() -> None:
    """A 'sender' that only appears in table cells (merchant, transaction id) is not a sender."""
    from fluxsense.extract.base import PageTable, Word
    from fluxsense.stages.repair import sender_only_in_table
    page = PageText(page=1, words=[Word(50, 300, 120, 310, "CARDPAY3D", cell=True), Word(130, 300, 160, 310, "PAYX24", cell=True),
                                   Word(50, 40, 140, 50, "Beispielbank"), Word(142, 40, 160, 50, "AG")],
                    tables=[PageTable(page=1, rows=[["a", "b"], ["c", "d"]], source="native", bbox=(40, 280, 400, 500))])
    layer = TextLayer(pages=[page], engine="t")
    assert sender_only_in_table({"company": "CARDPAY3D"}, layer, [1])
    assert sender_only_in_table({"company": "PAYX24"}, layer, [1])
    assert not sender_only_in_table({"company": "Beispielbank AG"}, layer, [1])
    assert not sender_only_in_table({"company": "Nowhere GmbH"}, layer, [1])


def garage_invoice() -> Any:
    """A generated workshop invoice (seed recorded): a spaced heading with a code, order / completion / service dates
    and a paid stamp line next to the 'Liefer=Re.Dat' invoice date."""
    import random
    from fluxsense.testdata import InvoiceStyle, build_invoice
    style = InvoiceStyle("Liefer=Re.Dat", "Rech-Nr.", ("Auftr.-Dat.", "Datum fertig", "Leistungsdat."), "R E C H N U N G  RK7")
    return build_invoice(random.Random("garage-invoice-1"), style=style, paid_stamp=True)


def test_stamp_date_is_not_the_invoice_date() -> None:
    """A stamp date or an order/service date is replaced by the labelled invoice date."""
    from fluxsense.models.result import FieldEvidence
    from fluxsense.stages.repair import fix_document_dates, labelled_invoice_dates, stamp_dates
    doc = garage_invoice()
    text, truth, order = doc.text(), doc.truth["doc_date"], doc.facts["other_dates"]["Auftr.-Dat."]
    assert str(labelled_invoice_dates(text)[0]) == truth
    assert [str(d) for d in stamp_dates(text, None)] == [doc.facts["stamp_date"]]
    order_line = next(line for line in text.splitlines() if line.startswith("Auftr.-Dat."))
    fields = {"meta_header": {"doc_date": doc.facts["stamp_date"]}, "bodies": {"finance_body": {"invoice_date": order}}}
    evidence = [FieldEvidence(path="bodies.finance_body.invoice_date", kind="date", value=order, evidence=order_line)]
    notes = fix_document_dates(fields, evidence, text, None)
    assert fields["meta_header"]["doc_date"] == truth and fields["bodies"]["finance_body"]["invoice_date"] == truth
    assert len(notes) == 2


def test_direct_debit_due_date() -> None:
    """The printed debit date is the due date of a direct-debit invoice."""
    from fluxsense.stages.repair import fix_document_dates
    fields = {"meta_header": {}, "bodies": {"finance_body": {"due_date": "2015-01-16"}}}
    fix_document_dates(fields, [], "Den Rechnungsbetrag buchen wir am 28.01.2015 von Ihrem Konto ab.", None)
    assert fields["bodies"]["finance_body"]["due_date"] == "2015-01-28"


def test_own_reference_beats_heading_and_case_numbers() -> None:
    """'Rech-Nr.' beats a heading code; 'Unser Zeichen' beats a third-party case number."""
    from fluxsense.stages.repair import fix_document_number, own_references
    doc = garage_invoice()
    number = doc.truth["doc_number"]
    fields = {"meta_header": {"doc_number": "RK7"}, "bodies": {"finance_body": {"invoice_number": "RK7"}}}
    notes = fix_document_number(fields, [], doc.text(), finance=True)
    assert fields["meta_header"]["doc_number"] == number and fields["bodies"]["finance_body"]["invoice_number"] == number
    assert len(notes) == 2
    letter = "\n".join(["Ombudsstelle Az. 31-S000123", "Ihr Zeichen: 4711-ABC", "Unser Zeichen 31-10001", "Sehr geehrte Damen und Herren"])
    fields = {"meta_header": {"doc_number": "31-S000123"}}
    fix_document_number(fields, [], letter, finance=False)
    assert fields["meta_header"]["doc_number"] == "31-10001"
    assert own_references("Versicherungsschein-Nr. 123 45 678901 A 123 45 vom") == [("Versicherungsschein-Nr.", "123 45 678901 A 123 45")]


def test_document_number_hygiene() -> None:
    """Page numbering, word-only values and printed labels are not document numbers."""
    from fluxsense.stages.repair import clean_document_numbers, drop_bank_statement_due_date
    fields = {"meta_header": {"doc_number": "1/2"},
              "bodies": {"finance_body": {"invoice_number": "Rechnungsnummer 73019245800417"},
                         "statement_body": {"statement_number": "Skizze", "payment_due_date": "2024-08-30"}}}
    notes = clean_document_numbers(fields)
    assert "doc_number" not in fields["meta_header"] and "statement_number" not in fields["bodies"]["statement_body"]
    assert fields["bodies"]["finance_body"]["invoice_number"] == "73019245800417" and len(notes) == 3
    assert drop_bank_statement_due_date(fields, ["BANK_STATEMENT"]) and "payment_due_date" not in fields["bodies"]["statement_body"]
    keep = {"bodies": {"statement_body": {"payment_due_date": "2024-08-30"}}}
    assert drop_bank_statement_due_date(keep, ["CREDIT_CARD_STATEMENT"]) == []


def test_doc_semantics_by_type() -> None:
    """Statements: printed statement number, never account/card numbers, no unprinted due date; forms: labelled
    numbers only, never a postcode; enclosures: no unlabelled doc date."""
    from fluxsense.catalog import default_catalog
    from fluxsense.models.result import FieldEvidence
    from fluxsense.stages.repair import enforce_doc_semantics
    cat = default_catalog()
    fields = {"meta_header": {"doc_number": "4111111111111111"},
              "bodies": {"statement_body": {"statement_number": "Auszug 3", "payment_due_date": "2024-08-30"}}}
    notes = enforce_doc_semantics(fields, [], cat.get("BANK_STATEMENT"), "Kontoauszug 3 vom 31.07.2024", True)
    assert fields["meta_header"]["doc_number"] == "3" and "payment_due_date" not in fields["bodies"]["statement_body"]
    fields = {"meta_header": {"doc_number": "4111 1111"}, "bodies": {"statement_body": {"card_number_masked": "4111 1111 xxxx 1111"}}}
    enforce_doc_semantics(fields, [], cat.get("CREDIT_CARD_STATEMENT"), "Karte 4111 1111 xxxx 1111", True)
    assert "doc_number" not in fields["meta_header"]
    fields = {"meta_header": {"doc_number": "00345"}}
    enforce_doc_semantics(fields, [], cat.get("APPLICATION"), "00345 Musterstadt", True)
    assert "doc_number" not in fields["meta_header"]
    fields = {"meta_header": {"doc_number": "LV-123456"}}
    enforce_doc_semantics(fields, [FieldEvidence(path="meta_header.doc_number", kind="identifier", value="LV-123456",
                                                 evidence="Versicherungsschein-Nr. LV-123456")], cat.get("APPLICATION"), "", True)
    assert fields["meta_header"]["doc_number"] == "LV-123456"
    fields = {"meta_header": {"doc_date": "2031-04-17", "doc_number": "55501234170420310b"}}
    notes = enforce_doc_semantics(fields, [], cat.get("OTHER"), "17.04.2031 09:41 Zahlung freigegeben ID 55501234170420310b", False)
    assert fields["meta_header"] == {} and len(notes) == 2
    fields = {"meta_header": {"doc_date": "2031-04-17"}}
    enforce_doc_semantics(fields, [], cat.get("OTHER"), "Protokoll vom 17.04.2031", False)
    assert fields["meta_header"]["doc_date"] == "2031-04-17"


def test_statement_period_fills() -> None:
    """§8: a printed quarter is the period; without a printed period period_end = statement date and period_start = the
    day after the printed previous statement date; interest periods do not count as the statement period (synthetic)."""
    from fluxsense.stages.repair import fill_statement_period
    body = {"statement_date": "2031-09-30"}
    fill_statement_period(body, "Rechnungsabschluss 3. Quartal 2031\nKontosaldo per 30.09.2031")
    assert body == {"statement_date": "2031-09-30", "period_start": "2031-07-01", "period_end": "2031-09-30"}
    body = {"statement_date": "2031-05-27"}
    fill_statement_period(body, "| 28 04 | ALTER KONTOSTAND VOM 27.04.31 | 512,30 - |")
    assert body["period_end"] == "2031-05-27" and body["period_start"] == "2031-04-28"
    body = {"statement_date": "2031-03-31"}
    fill_statement_period(body, "Sollzinssatz vom 01.03.2031 bis 16.03.2031\nSollzinssatz vom 17.03.2031 bis 31.03.2031\n"
                                "Kontoauszug vom 01.03.2031 bis 31.03.2031")
    assert body.get("period_end") is None or body["period_end"] == "2031-03-31"


def test_validity_date_is_no_due_date() -> None:
    """§6: 'Ihr Tarif ist gültig bis 23.11.2031' is a term end, never the invoice due date; 'zahlbar bis' is."""
    from fluxsense.stages.repair import fix_document_dates
    fields = {"meta_header": {}, "bodies": {"finance_body": {"due_date": "2031-11-23"}}}
    fix_document_dates(fields, [], "Rechnung\nIhr Tarif ist gültig bis 23.11.2031", None, "INVOICE")
    assert "due_date" not in fields["bodies"]["finance_body"]
    fields = {"meta_header": {}, "bodies": {"finance_body": {"due_date": "2031-03-08"}}}
    fix_document_dates(fields, [], "zahlbar bis 08.03.2031\ngültig bis 08.03.2031", None, "INVOICE")
    assert fields["bodies"]["finance_body"]["due_date"] == "2031-03-08"


def test_printed_net_fills_the_tax_basis() -> None:
    """§5: a printed net put only into BT-106 is copied into BT-109 when no allowances / charges exist and it is
    consistent with net + tax = gross; otherwise nothing is filled."""
    from fluxsense.stages.repair import fill_tax_basis
    fields = {"bodies": {"finance_body": {"monetary_summation": {"line_total_amount": "100.00", "tax_total_amount": "19.00",
                                                                 "grand_total_amount": "119.00"}}}}
    assert fill_tax_basis(fields) and fields["bodies"]["finance_body"]["monetary_summation"]["tax_basis_total_amount"] == "100.00"
    charged = {"bodies": {"finance_body": {"monetary_summation": {"line_total_amount": "100.00", "charge_total_amount": "5.00"}}}}
    assert fill_tax_basis(charged) == []
    gross = {"bodies": {"finance_body": {"monetary_summation": {"line_total_amount": "119.00", "tax_total_amount": "19.00",
                                                                "grand_total_amount": "119.00"}}}}
    assert fill_tax_basis(gross) == []


def test_stamp_line_from_the_text_layer() -> None:
    """A stamp line the visual audit missed ('ÜBERWIESEN 12. MAI 2031') is added; sentences never are."""
    from fluxsense.stages.repair import stamps_from_text
    audit = {"layer_stamps": [], "audit_summary": {}}
    assert stamps_from_text(audit, "Rechnung\nÜBERWIESEN 12. MAI 2031\nSie erhalten 5 % Rabatt") == 1
    assert audit["layer_stamps"][0]["type"] == "PAID" and audit["layer_stamps"][0]["form_fields"][0]["raw_value"] == "12. MAI 2031"
    assert stamps_from_text({"layer_stamps": []}, "Sie erhalten monatlich einen Auszug") == 0


def test_statements_keep_their_extraction_without_folded_audit() -> None:
    """Statements fold no visual audit into their extraction call (it shortened their answers); text-layer stamp lines
    still become an audit (see stamps_from_text)."""
    from fluxsense.catalog import default_catalog
    from fluxsense.catalog.registry import AUDIT_NONE
    assert default_catalog().get("CREDIT_CARD_STATEMENT").audit_mode == AUDIT_NONE
