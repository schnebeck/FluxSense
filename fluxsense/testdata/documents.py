# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/documents.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Document builders of the test-document generator. Each
                builder draws fictional content, lays it out on page models
                following the field contract (labels, DIN 5008 regions,
                carry-over rows, totals on the last item page) and records
                the ground truth from the same data, so it is exact by
                construction: invoices (one page, long distributor style,
                bilingual, optional CII XML), bank and credit-card
                statements, DIN 5008 letters, forms filled in by a person,
                receipts, bank transfer confirmations and enclosures.
------------------------------------------------------------------------------
"""

import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from fluxsense.testdata import ids, vocab
from fluxsense.testdata.layout import Page, Run, signature, wrap
from fluxsense.testdata.parties import Organisation, Person, organisation, person, town

LEFT, RIGHT = 62.0, 533.0
ROW = 13.0
FOOTER_Y = 800.0
EPOCH = date(2022, 1, 1)  # fixed (determinism) and in the past (no future-date findings)


@dataclass
class LogicalDoc:
    """One generated logical document with its pages and ground truth."""

    kind: str
    doc_types: List[str]
    pages: List[Page]
    truth: Dict[str, Any]
    language: str = "de"
    summary: str = ""
    xml: Optional[bytes] = None
    visual: Dict[str, bool] = field(default_factory=lambda: {"has_stamp": False, "has_signature": False,
                                                            "has_handwriting": False})
    facts: Dict[str, Any] = field(default_factory=dict)  # printed details outside the ground-truth format

    def text(self) -> str:
        """All page lines (no rendering needed)."""
        return "\n".join(line for page in self.pages for line in page.lines())


@dataclass
class InvoiceStyle:
    """Label choices of an invoice header (field contract §3/§4 label lists and 'never' labels)."""

    date_label: str = "Rechnungsdatum"
    number_label: str = "Rechnungsnummer"
    other_dates: Tuple[str, ...] = ("Lieferdatum",)
    heading: str = "Rechnung"


def _q(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _empty_truth() -> Dict[str, Any]:
    return {"sender": None, "recipient": None, "doc_date": None, "doc_number": None, "total_gross": None, "total_net": None,
            "tax_amount": None, "currency": None, "due_date": None, "iban": None,
            "statement": {"period_start": None, "period_end": None, "opening_balance": None, "closing_balance": None},
            "table": None}


def _letterhead(page: Page, org: Organisation) -> None:
    page.text(70, 48, org.name, 15, bold=True)
    page.text(70, 62, org.address.one_line(), 8)


def _window(page: Page, sender: Organisation, lines: Sequence[str]) -> None:
    """DIN 5008 form B: return-address line and the address window (45-90 mm from the top)."""
    page.text(LEFT, 128, f"{sender.name} · {sender.address.one_line()}", 6)
    y = 145.0
    for line in lines:
        page.text(LEFT, y, line, 10)
        y += 13.0


def _info(page: Page, pairs: Sequence[Tuple[str, str]], y: float = 140.0) -> None:
    """Info block right of the address window, 'Label: value' per line."""
    for label, value in pairs:
        page.text(360, y, f"{label}: {value}", 8)
        y += 12.0


def _footer(page: Page, org: Organisation, number: Optional[Tuple[int, int]] = None, bank: bool = False) -> None:
    """Legal footer; a bank (bank=True) prints its BIC and register entry, not an account of its own."""
    page.rule(LEFT, FOOTER_Y - 12, RIGHT)
    page.text(LEFT, FOOTER_Y, f"{org.name} · {org.address.one_line()} · USt-IdNr. {org.vat_id}", 6.5)
    if bank:
        page.text(LEFT, FOOTER_Y + 9, f"BIC {org.bic} · Sitz {org.address.city} · Registergericht {org.address.city} "
                                      f"HRB {int(org.address.zip_code) * 7 + 1000}", 6.5)
    else:
        page.text(LEFT, FOOTER_Y + 9, f"{org.bank} · IBAN {ids.group_iban(org.iban)} · BIC {org.bic}", 6.5)
    if number:
        page.text(RIGHT, FOOTER_Y + 9, f"Seite {number[0]} von {number[1]}", 6.5, align="right")


def _recipient_lines(recipient: Any) -> List[str]:
    if isinstance(recipient, Person):
        return [recipient.salutation_line, recipient.name] + recipient.address.lines()
    return [recipient.name] + recipient.address.lines()


def _split_rows(count: int, first: int, rest: int, last_reserve: int) -> List[int]:
    """Rows per page; rows move to a further page so that the last item page keeps room for the totals block."""
    sizes: List[int] = []
    left, capacity = count, first
    while left > 0:
        take = min(left, capacity)
        sizes.append(take)
        left -= take
        capacity = rest
    free = (first if len(sizes) == 1 else rest) - sizes[-1]
    shift = min(last_reserve - free, sizes[-1] - 1)
    if free < last_reserve and shift > 0:
        sizes[-1] -= shift
        sizes.append(shift)
    return sizes


# ---------------------------------------------------------------------------------------------------------------- invoice


def build_invoice(rng: random.Random, long: bool = False, bilingual: bool = False, zugferd: bool = False,
                  style: Optional[InvoiceStyle] = None, paid_stamp: bool = False,
                  items: Optional[int] = None) -> LogicalDoc:
    """
    An invoice: one page (2-8 items) or long distributor style (45-110 items
    across pages with carry-over rows, totals on the last item page, a terms
    page after it); bilingual DE/EN header variant; optional CII XML.

    Args:
        rng: Random source (deterministic per seed).
        long: Multi-page distributor invoice.
        bilingual: German/English labels and English descriptions.
        zugferd: Build a matching CII XML (doc.xml).
        style: Header labels (default: drawn from the contract label lists).
        paid_stamp: Print a 'BEZAHLT' stamp line with a later date (text layer).
        items: Fixed item count.

    Returns:
        LogicalDoc.
    """
    seller = organisation(rng)
    buyer: Any = person(rng) if rng.random() < 0.7 else organisation(rng)
    if style is None:
        style = InvoiceStyle(rng.choice(("Rechnungsdatum", "Re.-Dat", "Datum")),
                             rng.choice(("Rechnungsnummer", "Rech-Nr.", "Rechnungs-Nr.")),
                             rng.choice((("Lieferdatum",), ("Leistungsdatum",), ("Auftragsdatum", "Lieferdatum"))))
    inv_date = ids.random_date(rng, EPOCH, 720)
    number = ids.document_number(rng, inv_date.year, rng.randrange(4))
    customer = ids.customer_number(rng)
    due = inv_date + timedelta(days=rng.choice((14, 30)))
    paid_on = due - timedelta(days=rng.randint(1, 9))
    count = items or (rng.randint(45, 110) if long else rng.randint(2, 8))
    rows: List[Tuple[str, str, str, Decimal, Decimal, Decimal]] = []
    for pos in range(1, count + 1):
        de, en, (low, high) = rng.choice(vocab.ARTICLES)
        qty = Decimal(rng.randint(1, 8) if high > 2000 else rng.randint(1, 50))
        unit = ids.money(rng, low, high)
        rows.append((str(pos), f"{rng.randint(100, 999)}-{rng.randint(10000, 99999)}", en if bilingual else de, qty, unit,
                     _q(qty * unit)))
    net = sum((r[5] for r in rows), Decimal("0"))
    tax = _q(net * Decimal("0.19"))
    gross = net + tax

    def lbl(de: str, en: str) -> str:
        return f"{de} / {en}" if bilingual else de

    headers = [lbl("Pos.", "Item"), lbl("Artikel-Nr.", "Part No."), lbl("Beschreibung", "Description"), lbl("Menge", "Qty"),
               lbl("Einzelpreis", "Unit price"), lbl("Gesamt EUR", "Total EUR")]
    cols = [(LEFT, "left"), (95, "left"), (165, "left"), (390, "right"), (460, "right"), (RIGHT, "right")]
    sizes = _split_rows(count, 30, 46, 7)
    terms_pages = 1 if long else 0
    total_pages = len(sizes) + terms_pages
    pages: List[Page] = []
    other_dates = {label: inv_date - timedelta(days=rng.randint(1, 12)) for label in style.other_dates}
    index = 0
    carried = Decimal("0")
    for page_no, size in enumerate(sizes, 1):
        page = Page()
        if page_no == 1:
            _letterhead(page, seller)
            _window(page, seller, _recipient_lines(buyer))
            info = [(lbl(style.number_label, "Invoice No."), number), (lbl(style.date_label, "Invoice date"),
                                                                      ids.fmt_date_de(inv_date)),
                    (lbl("Kundennummer", "Customer No."), customer)]
            info += [(label, ids.fmt_date_de(value)) for label, value in other_dates.items()]
            _info(page, info)
            page.text(LEFT, 290, lbl(style.heading, "Invoice") if bilingual else style.heading, 13, bold=True)
            y = 318.0
        else:
            page.text(LEFT, 50, f"{seller.name} · {lbl('Rechnung', 'Invoice')} {number}", 8)
            y = 80.0
        for text, (x, align) in zip(headers, cols):
            page.text(x, y, text, 7.5, bold=True, align=align)
        page.rule(LEFT, y + 4, RIGHT)
        y += ROW + 2
        if page_no > 1:
            page.text(165, y, lbl("Übertrag von Seite", "Carried forward from page") + f" {page_no - 1}", 8)
            page.text(RIGHT, y, ids.fmt_de(carried), 8, align="right")
            y += ROW
        for pos, sku, desc, qty, unit, total in rows[index:index + size]:
            for text, (x, align) in zip((pos, sku, desc, str(qty), ids.fmt_de(unit), ids.fmt_de(total)), cols):
                page.text(x, y, text, 8, align=align)
            carried += total
            y += ROW
        index += size
        if page_no < len(sizes):
            page.rule(390, y - 8, RIGHT)
            page.text(165, y + 2, lbl("Übertrag auf Seite", "Carried forward to page") + f" {page_no + 1}", 8)
            page.text(RIGHT, y + 2, ids.fmt_de(carried), 8, align="right")
        else:
            page.rule(390, y - 8, RIGHT)
            for label, value, bold in ((lbl("Nettobetrag", "Net amount"), net, False),
                                       (lbl("zzgl. 19 % USt", "VAT 19 %"), tax, False),
                                       (lbl("Gesamtbetrag", "Total amount"), gross, True)):
                y += ROW
                page.text(390, y, label, 8.5, bold=bold, align="right")
                page.text(RIGHT, y, f"{ids.fmt_de(value)} EUR", 8.5, bold=bold, align="right")
            y += ROW * 2
            page.text(LEFT, y, f"Zahlbar bis {ids.fmt_date_de(due)} ohne Abzug auf das unten genannte Konto.", 8.5)
            if paid_stamp:
                page.text(360, 250, f"BEZAHLT {ids.fmt_date_de(paid_on)}", 11, bold=True)
        _footer(page, seller, (page_no, total_pages))
        pages.append(page)
    for extra in range(terms_pages):
        page = build_terms_page(seller.name)
        _footer(page, seller, (len(sizes) + extra + 1, total_pages))
        pages.append(page)
    truth = _empty_truth()
    truth.update({"sender": seller.name, "recipient": buyer.name, "doc_date": inv_date.isoformat(), "doc_number": number,
                  "total_gross": float(gross), "total_net": float(net), "tax_amount": float(tax), "currency": "EUR",
                  "due_date": due.isoformat(), "iban": seller.iban,
                  "table": {"kind": "line_items", "row_count": count, "sum_line_totals": float(net)}})
    doc = LogicalDoc("invoice", ["INVOICE"], pages, truth, summary=f"Invoice {number} from {seller.name}")
    doc.visual["has_stamp"] = paid_stamp
    doc.facts.update({"sender_vat_id": seller.vat_id, "sender_bic": seller.bic, "customer_number": customer,
                      "other_dates": {label: value.isoformat() for label, value in other_dates.items()}})
    if paid_stamp:
        doc.facts["stamp_date"] = paid_on.isoformat()
    if zugferd:
        from fluxsense.testdata.cii import build_cii
        doc.xml = build_cii(number, inv_date, due, seller, buyer, rows, net, tax, gross)
    return doc


def build_telecom_invoice(rng: random.Random, totals_page: int = 1, usage_pages: int = 2, rows_per_page: int = 78) -> LogicalDoc:
    """
    A telecom-style invoice: a short charge summary (the line items, without
    position numbers) with the totals block on page 1 or 2, followed by an
    itemised usage list (numbered call rows with times, durations and
    amounts, a 'Summe Verbindungen' subtotal) that is NOT the line-item table.

    Args:
        rng: Random source.
        totals_page: Page of the totals block (1 or 2).
        usage_pages: Pages of the itemised usage list.
        rows_per_page: Usage rows per page (a dense list makes the rows exceed one call's output budget).

    Returns:
        LogicalDoc.
    """
    seller = organisation(rng, ["Testfeld Mobilfunk", "Probe Telekommunikation", "Exempel Netz"])
    buyer = person(rng)
    inv_date = ids.random_date(rng, EPOCH, 720)
    number = ids.document_number(rng, inv_date.year, 2)
    customer = ids.customer_number(rng)
    debit = inv_date + timedelta(days=rng.randint(8, 14))
    per_page = rows_per_page
    calls: List[Tuple[str, str, str, str, Decimal]] = []
    for _ in range(usage_pages * per_page - rng.randint(0, 10)):
        day = inv_date - timedelta(days=rng.randint(5, 35))
        calls.append((day.strftime("%d.%m."), f"{rng.randint(7, 21):02d}:{rng.randint(0, 59):02d}",
                      ids.fictional_phone(rng), f"{rng.randint(0, 59):02d}:{rng.randint(1, 59):02d}",
                      ids.money(rng, 1, 290)))
    calls.sort(key=lambda c: (c[0][3:], c[0][:2], c[1]))
    usage = sum((c[4] for c in calls), Decimal("0"))
    items: List[Tuple[str, Decimal]] = [("Grundgebühr Tarif Probe M", ids.money(rng, 990, 3990)),
                                        ("Datenoption 10 GB", ids.money(rng, 500, 1500)),
                                        ("Verbindungen laut Einzelverbindungsnachweis", usage)]
    extra_items: List[Tuple[str, Decimal]] = [("Leistungen Dritter (Mehrwertdienste)", ids.money(rng, 100, 900)),
                                              ("Gutschrift Treuebonus", -ids.money(rng, 100, 500))] if totals_page == 2 else []
    net = sum((a for _, a in items + extra_items), Decimal("0"))
    tax = _q(net * Decimal("0.19"))
    gross = net + tax
    total_pages = max(totals_page, 1) + usage_pages
    pages: List[Page] = []

    def totals(page: Page, y: float) -> None:
        page.rule(390, y - 8, RIGHT)
        for label, value, bold in (("Summe netto", net, False), ("zzgl. 19 % USt", tax, False), ("Rechnungsbetrag", gross, True)):
            y += ROW
            page.text(390, y, label, 8.5, bold=bold, align="right")
            page.text(RIGHT, y, f"{ids.fmt_de(value)} EUR", 8.5, bold=bold, align="right")
        page.text(LEFT, y + 28, f"Den Rechnungsbetrag buchen wir am {ids.fmt_date_de(debit)} von Ihrem Konto ab.", 8.5)

    def charges(page: Page, y: float, rows: Sequence[Tuple[str, Decimal]]) -> float:
        page.text(LEFT, y, "Leistung", 7.5, bold=True)
        page.text(RIGHT, y, "Betrag netto EUR", 7.5, bold=True, align="right")
        page.rule(LEFT, y + 4, RIGHT)
        y += ROW + 2
        for label, amount in rows:
            page.text(LEFT, y, label, 8.5)
            page.text(RIGHT, y, ids.fmt_de(amount), 8.5, align="right")
            y += ROW
        return y

    first = Page()
    _letterhead(first, seller)
    _window(first, seller, _recipient_lines(buyer))
    _info(first, [("Rechnungsnummer", number), ("Rechnungsdatum", ids.fmt_date_de(inv_date)), ("Kundennummer", customer)])
    first.text(LEFT, 290, f"Ihre Rechnung für {(inv_date - timedelta(days=20)).strftime('%m/%Y')}", 13, bold=True)
    y = charges(first, 318, items)
    if totals_page == 1:
        totals(first, y)
    else:
        first.text(LEFT, y + 10, "Fortsetzung auf Seite 2", 8)
    _footer(first, seller, (1, total_pages))
    pages.append(first)
    if totals_page == 2:
        second = Page()
        second.text(LEFT, 50, f"{seller.name} · Rechnung {number}", 8)
        y = charges(second, 80, extra_items)
        totals(second, y)
        _footer(second, seller, (2, total_pages))
        pages.append(second)
    for index in range(usage_pages):
        page = Page()
        page.text(LEFT, 50, f"Einzelverbindungsnachweis zur Rechnung {number}", 10, bold=True)
        cols = [(LEFT, "left"), (95, "left"), (150, "left"), (200, "left"), (330, "left"), (RIGHT, "right")]
        for text, (x, align) in zip(("Nr.", "Datum", "Uhrzeit", "Zielrufnummer", "Dauer", "Betrag EUR"), cols):
            page.text(x, 80, text, 7.5, bold=True, align=align)
        page.rule(LEFT, 84, RIGHT)
        y = 95.0
        step = min(ROW, 680.0 / per_page)
        start = index * per_page
        for n, (day, clock, target, duration, amount) in enumerate(calls[start:start + per_page], start + 1):
            for text, (x, align) in zip((str(n), day, clock, target, duration, ids.fmt_de(amount)), cols):
                page.text(x, y, text, min(8.0, step * 0.8), align=align)
            y += step
        if index == usage_pages - 1:
            page.rule(330, y - step + 2, RIGHT)
            page.text(330, y + 4, "Summe Verbindungen", 8.5, bold=True)
            page.text(RIGHT, y + 4, ids.fmt_de(usage), 8.5, bold=True, align="right")
        _footer(page, seller, (len(pages) + 1, total_pages))
        pages.append(page)
    truth = _empty_truth()
    truth.update({"sender": seller.name, "recipient": buyer.name, "doc_date": inv_date.isoformat(), "doc_number": number,
                  "total_gross": float(gross), "total_net": float(net), "tax_amount": float(tax), "currency": "EUR",
                  "due_date": debit.isoformat(), "iban": seller.iban,
                  "table": {"kind": "line_items", "row_count": len(items) + len(extra_items), "sum_line_totals": float(net)}})
    doc = LogicalDoc("telecom_invoice", ["INVOICE"], pages, truth, summary=f"Telecom invoice {number} of {seller.name}")
    doc.facts.update({"sender_vat_id": seller.vat_id, "totals_page": totals_page, "usage_rows": len(calls),
                      "line_items": [(label, float(amount)) for label, amount in items + extra_items]})
    return doc


# ------------------------------------------------------------------------------------------------------------- statements


def build_statement(rng: random.Random, card: bool = False, info_page: Optional[bool] = None,
                    print_number: bool = True, counterparty_ibans: bool = False) -> LogicalDoc:
    """
    A bank statement (giro account, running balance) or a credit-card
    statement (charges, previous balance settled, debit date) with a
    multi-page transaction table and carry-over rows; balances add up.

    Args:
        rng: Random source.
        card: Credit-card statement instead of a bank statement.
        info_page: Append a 'Wichtige Informationen' page (default: random).
        print_number: Print a statement number; without it only the customer / account / card numbers are printed
            and the truth has no doc_number (§4).
        counterparty_ibans: Bank statement bookings print the counterparty's IBAN, and an interest sub-period
            ending on the statement date is printed below the bookings.

    Returns:
        LogicalDoc.
    """
    bank = organisation(rng, vocab.CARD_ISSUERS if card else vocab.BANKS, legal_form=False)
    holder = person(rng)
    month_start = date(rng.choice((2022, 2023, 2024)), rng.randint(1, 12), 1)
    period_end = (month_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
    number = rng.randint(1, 12) if not card else rng.randint(1, 99)
    if card:
        opening = -ids.money(rng, 5000, 150000)
    else:
        opening = ids.money(rng, -50000, 800000)
    count = rng.randint(8, 40) if card else rng.randint(18, 70)
    days = sorted(rng.randrange((period_end - month_start).days + 1) for _ in range(count))
    tx: List[Tuple[date, str, Decimal]] = []
    for i, offset in enumerate(days):
        when = month_start + timedelta(days=offset)
        if card and i == 0:
            tx.append((when, "Zahlung - vielen Dank", -opening))
            continue
        if card:
            tx.append((when, rng.choice(vocab.CARD_TEXTS), -ids.money(rng, 450, 25000)))
            continue
        text, sign = rng.choice(vocab.BOOKING_TEXTS)
        if counterparty_ibans and i % 3 == 0 and not text.startswith(("Kartenzahlung", "Bargeld")):
            text = f"{text} IBAN {ids.fictional_iban(rng)}"
        amount = ids.money(rng, 180000, 420000) if "Gehalt" in text else ids.money(rng, 500, 120000)
        tx.append((when, text, amount if sign > 0 else -amount))
    closing = opening + sum((t[2] for t in tx), Decimal("0"))
    title = "Kreditkartenabrechnung" if card else "Kontoauszug"
    card_full = ids.fictional_card_number(rng)
    card_no = ids.masked_card_number(card_full)
    debit_date = period_end + timedelta(days=rng.randint(8, 14))
    customer = f"{rng.randint(1000000, 9999999)}"
    account = ids.fictional_card_number(rng)
    sizes = _split_rows(count, 26, 44, 6)
    extra = rng.random() < 0.5 if info_page is None else info_page
    total_pages = len(sizes) + (1 if extra else 0)
    pages: List[Page] = []
    balance = opening
    index = 0
    rows_out: List[Tuple[str, str, str, float, float, int]] = []
    cols = [(LEFT, "left"), (115, "left"), (170, "left"), (RIGHT, "right")]
    for page_no, size in enumerate(sizes, 1):
        page = Page()
        if page_no == 1:
            _letterhead(page, bank)
            _window(page, bank, _recipient_lines(holder))
            info = [("Abrechnung Nr." if card else "Kontoauszug Nr.", str(number))] if print_number else []
            info += [("Abrechnungsdatum" if card else "Auszugsdatum", ids.fmt_date_de(period_end)),
                     ("Kundennummer", customer)]
            info += [("Kartennummer", card_no), ("Kontonummer", account)] if card else \
                [("IBAN", ids.group_iban(holder.iban)), ("BIC", bank.bic)]
            _info(page, info)
            page.text(LEFT, 280, f"{title} vom {ids.fmt_date_de(month_start)} bis {ids.fmt_date_de(period_end)}", 12, bold=True)
            y = 306.0
        else:
            page.text(LEFT, 50, f"{bank.name} · {title}" + (f" Nr. {number}" if print_number else ""), 8)
            y = 80.0
        header = ("Belegdatum", "Buchung", "Umsatz", "Betrag EUR") if card else ("Buchung", "Valuta", "Vorgang", "Betrag EUR")
        for text, (x, align) in zip(header, cols):
            page.text(x, y, text, 7.5, bold=True, align=align)
        page.rule(LEFT, y + 4, RIGHT)
        y += ROW + 2
        if page_no == 1:
            label = "Saldo der Vorabrechnung" if card else f"Kontostand am {ids.fmt_date_de(month_start - timedelta(days=1))}"
            page.text(170, y, label, 8, bold=True)
            page.text(RIGHT, y, ids.fmt_de(opening), 8, bold=True, align="right")
        else:
            page.text(170, y, f"Übertrag von Seite {page_no - 1}", 8)
            page.text(RIGHT, y, ids.fmt_de(balance), 8, align="right")
        y += ROW
        for when, text, amount in tx[index:index + size]:
            booked = when + timedelta(days=1 if card else 0)
            rows_out.append((when.isoformat(), booked.isoformat(), text, float(amount), float(balance + amount), page_no))
            values = (when.strftime("%d.%m."), booked.strftime("%d.%m."), text, ids.fmt_de(amount))
            for value, (x, align) in zip(values, cols):
                page.text(x, y, value, 8, align=align)
            balance += amount
            y += ROW
        index += size
        page.rule(LEFT, y - 8, RIGHT)
        if page_no < len(sizes):
            page.text(170, y + 2, f"Übertrag auf Seite {page_no + 1}", 8)
            page.text(RIGHT, y + 2, ids.fmt_de(balance), 8, align="right")
        else:
            label = "Neuer Saldo" if card else f"Neuer Kontostand am {ids.fmt_date_de(period_end)}"
            page.text(170, y + 2, label, 8.5, bold=True)
            page.text(RIGHT, y + 2, ids.fmt_de(closing), 8.5, bold=True, align="right")
            if card:
                page.text(LEFT, y + 30, f"Der Betrag von {ids.fmt_de(-closing)} EUR wird am {ids.fmt_date_de(debit_date)} "
                                        "von Ihrem Referenzkonto abgebucht.", 8.5)
            elif counterparty_ibans:
                page.text(LEFT, y + 30, f"Zinsabrechnung für den Zeitraum {ids.fmt_date_de(month_start + timedelta(days=14))} "
                                        f"bis {ids.fmt_date_de(period_end)}: 0,00 EUR", 8.5)
        _footer(page, bank, (page_no, total_pages), bank=True)
        pages.append(page)
    if extra:
        page = Page()
        page.text(LEFT, 60, "Wichtige Informationen", 12, bold=True)
        for i, line in enumerate(("Bitte prüfen Sie die Umsätze dieser Abrechnung.",
                                  "Einwendungen sind innerhalb von sechs Wochen nach Zugang zu erheben.",
                                  "Unterlassen rechtzeitiger Einwendungen gilt als Genehmigung.")):
            page.text(LEFT, 90 + i * 14, line, 9)
        _footer(page, bank, (total_pages, total_pages), bank=True)
        pages.append(page)
    credits = sum((t[2] for t in tx if t[2] > 0), Decimal("0"))
    debits = sum((t[2] for t in tx if t[2] < 0), Decimal("0"))
    truth = _empty_truth()
    truth.update({"sender": bank.name, "recipient": holder.name, "doc_date": period_end.isoformat(), "doc_number": str(number) if print_number else None,
                  "currency": "EUR", "due_date": debit_date.isoformat() if card else None,
                  "iban": None if card else holder.iban,
                  "statement": {"period_start": month_start.isoformat(), "period_end": period_end.isoformat(),
                                "opening_balance": float(opening), "closing_balance": float(closing)},
                  "table": {"kind": "transactions", "row_count": count, "sum_credits": float(credits),
                            "sum_debits": float(debits),
                            "first_row": {"date": tx[0][0].isoformat(), "amount": float(tx[0][2])},
                            "last_row": {"date": tx[-1][0].isoformat(), "amount": float(tx[-1][2])}}})
    types = ["BANK_STATEMENT", "CREDIT_CARD_STATEMENT"] if card else ["BANK_STATEMENT"]
    doc = LogicalDoc("card_statement" if card else "bank_statement", types, pages, truth, summary=f"{title} of {bank.name}")
    doc.facts.update({"customer_number": customer, "account_number": account if card else holder.iban[12:],
                      "card_number": card_no if card else None, "statement_number": str(number), "transactions": rows_out})
    return doc


# ---------------------------------------------------------------------------------------------------------------- letters


def build_letter(rng: random.Random, official: bool = False, deadline: Optional[bool] = None,
                 private: bool = False) -> LogicalDoc:
    """
    A DIN 5008 form-B letter: letterhead, return-address line, address
    window, info block ('Unser Zeichen', 'Datum'), subject, body, closing
    and a signature; optionally a reply deadline.

    Args:
        rng: Random source.
        official: Letter from a public authority (OFFICIAL_LETTER).
        private: Letter from a private person whose printed name matches the signature.
        deadline: Print a reply deadline (default: random).

    Returns:
        LogicalDoc.
    """
    sender = organisation(rng, [f"{rng.choice(vocab.AUTHORITIES)} {town(rng)}"] if official else
                          list(vocab.COMPANY_STEMS) + list(vocab.INSURERS), legal_form=not official)
    recipient = person(rng)
    when = ids.random_date(rng, EPOCH, 720)
    reference = f"{rng.choice('ABCDEFGHKLMNPRSTVZ')}{rng.choice('ABCDEFGHKLMNPRSTVZ')}-{rng.randint(1000, 9999)}/{when.year % 100:02d}"
    contact = person(rng)
    if private:
        sender.name = contact.name  # the printed sender signs the letter
    subject, body = rng.choice(vocab.LETTER_SUBJECTS)
    subject = subject.format(ref_date=ids.fmt_date_de(when - timedelta(days=rng.randint(5, 30))))
    with_deadline = rng.random() < 0.5 if deadline is None else deadline
    reply_by = when + timedelta(days=rng.randint(10, 28))
    page = Page()
    _letterhead(page, sender)
    _window(page, sender, _recipient_lines(recipient))
    _info(page, [("Unser Zeichen", reference), ("Ansprechpartner", f"{contact.salutation_line.replace('Herrn', 'Herr')} "
                                                                    f"{contact.last}"),
                 ("Datum", ids.fmt_date_de(when))])
    page.text(LEFT, 290, subject, 10.5, bold=True)
    y = 320.0
    paragraphs = [recipient.greeting, body]
    if with_deadline:
        paragraphs.append(f"Bitte senden Sie uns die Unterlagen bis zum {ids.fmt_date_de(reply_by)} zurück.")
    paragraphs.append("Für Rückfragen stehen wir Ihnen gern zur Verfügung.")
    for paragraph in paragraphs:
        for line in wrap(paragraph, RIGHT - LEFT, 10):
            page.text(LEFT, y, line, 10)
            y += 14
        y += 8
    page.text(LEFT, y + 6, "Mit freundlichen Grüßen", 10)
    page.text(LEFT, y + 20, sender.name, 10)
    signature(page, LEFT, y + 46, rng.randint(0, 999))
    page.text(LEFT, y + 66, contact.name, 9)
    if private:
        page.text(LEFT, y + 78, "(Unterschrift)", 7)
    _footer(page, sender)
    truth = _empty_truth()
    truth.update({"sender": sender.name, "recipient": recipient.name, "doc_date": when.isoformat(), "doc_number": reference,
                  "due_date": reply_by.isoformat() if with_deadline else None})
    doc = LogicalDoc("official_letter" if official else "letter", ["OFFICIAL_LETTER"] if official else ["OTHER", "LETTER"],
                     [page], truth, summary=f"Letter from {sender.name}: {subject}")
    doc.visual["has_signature"] = True
    doc.facts.update({"greeting": recipient.greeting, "recipient_zip": recipient.address.zip_code,
                      "recipient_street": recipient.address.street, "contact": contact.name})
    return doc


def build_terms_page(issuer: str) -> Page:
    """A boilerplate terms page (attaches to the preceding document)."""
    page = Page()
    page.text(LEFT, 60, vocab.TERMS_TITLE, 12, bold=True)
    y = 90.0
    for line in vocab.TERMS_LINES:
        for part in wrap(line, RIGHT - LEFT, 9):
            page.text(LEFT, y, part, 9)
            y += 12
        y += 6
    page.text(LEFT, 780, f"{issuer} · Stand 01/30", 7)
    return page


# ------------------------------------------------------------------------------------------------------------------ forms


def build_form(rng: random.Random) -> LogicalDoc:
    """A form issued by an organisation and filled in (typed) and signed by a person (§2.4)."""
    org = organisation(rng, vocab.INSURERS, legal_form=False)
    filler = person(rng)
    title, special = rng.choice(vocab.FORM_TITLES)
    member = f"{rng.randint(10, 99)}/{rng.randint(1000, 9999)}"
    signed = ids.random_date(rng, EPOCH, 720)
    page = Page()
    _letterhead(page, org)
    page.text(LEFT, 110, title, 14, bold=True)
    page.text(RIGHT, 110, f"Formular {rng.choice('ABCDEFG')}-{rng.randint(10, 99)} ({signed.month:02d}/{signed.year % 100:02d})",
              6.5, align="right")
    special_value = (f"{ids.fmt_de(ids.money(rng, 2000, 90000))} EUR" if special == "Erstattungsbetrag"
                     else ids.fmt_date_de(signed + timedelta(days=rng.randint(20, 90))))
    fields = [("Name, Vorname", f"{filler.last}, {filler.first}"), ("Straße, Hausnummer", filler.address.street),
              ("PLZ, Ort", f"{filler.address.zip_code} {filler.address.city}"), ("Mitglieds-Nr.", member),
              (special, special_value)]
    y = 150.0
    for label, value in fields:
        page.text(LEFT, y, f"{label}:", 9)
        page.text(200, y, value, 10, mono=True)
        page.rule(198, y + 3, RIGHT)
        y += 28
    page.text(LEFT, y + 10, "Ich versichere die Richtigkeit meiner Angaben.", 9)
    page.text(LEFT, y + 50, "Ort, Datum:", 9)
    page.text(140, y + 50, f"{filler.address.city}, {ids.fmt_date_de(signed)}", 10, mono=True)
    signature(page, 340, y + 46, rng.randint(0, 999))
    page.rule(330, y + 56, RIGHT)
    page.text(330, y + 66, "Unterschrift", 7)
    page.text(LEFT, 780, f"Bitte senden an: {org.name} · {org.address.one_line()}", 7)
    truth = _empty_truth()
    truth.update({"sender": filler.name, "recipient": org.name, "doc_date": signed.isoformat(), "doc_number": member})
    doc = LogicalDoc("form", ["APPLICATION"], [page], truth, summary=f"{title} for {org.name}")
    doc.visual["has_signature"] = True
    return doc


# --------------------------------------------------------------------------------------------------------------- receipts


def build_receipt(rng: random.Random) -> LogicalDoc:
    """A narrow till receipt with items, total and included VAT."""
    shop = organisation(rng, vocab.SHOPS)
    when = ids.random_date(rng, EPOCH, 720)
    items = [(name, ids.money(rng, low, high)) for name, (low, high) in rng.sample(vocab.RECEIPT_ITEMS, rng.randint(2, 6))]
    gross = sum((price for _, price in items), Decimal("0"))
    net = _q(gross / Decimal("1.19"))
    tax = gross - net
    page = Page(width=227.0, height=440.0 + 14 * len(items))
    page.text(20, 30, shop.name, 10, bold=True)
    page.text(20, 43, shop.address.street, 7.5)
    page.text(20, 53, f"{shop.address.zip_code} {shop.address.city}", 7.5)
    page.text(20, 63, f"USt-IdNr. {shop.vat_id}", 7)
    page.text(20, 85, f"Datum: {ids.fmt_date_de(when)}  {rng.randint(8, 19):02d}:{rng.randint(0, 59):02d}", 8)
    y = 110.0
    for name, price in items:
        page.text(20, y, name, 8)
        page.text(207, y, f"{ids.fmt_de(price)} A", 8, align="right")
        y += 14
    page.rule(20, y - 6, 207)
    page.text(20, y + 8, "SUMME EUR", 10, bold=True)
    page.text(207, y + 8, ids.fmt_de(gross), 10, bold=True, align="right")
    page.text(20, y + 26, rng.choice(("Kartenzahlung", "Bar")), 8)
    page.text(20, y + 46, "A = 19 % MwSt", 7.5)
    page.text(20, y + 58, f"Netto {ids.fmt_de(net)}  MwSt {ids.fmt_de(tax)}  Brutto {ids.fmt_de(gross)}", 7.5)
    page.text(20, y + 84, "Vielen Dank für Ihren Einkauf!", 8)
    truth = _empty_truth()
    truth.update({"sender": shop.name, "doc_date": when.isoformat(), "total_gross": float(gross), "total_net": float(net),
                  "tax_amount": float(tax), "currency": "EUR"})
    return LogicalDoc("receipt", ["RECEIPT"], [page], truth, summary=f"Receipt of {shop.name}")


def build_transfer(rng: random.Random) -> LogicalDoc:
    """A bank confirmation of a single transfer (§2.8): bank = sender, ordering customer = recipient."""
    bank = organisation(rng, vocab.BANKS, legal_form=False)
    customer = person(rng)
    payee = organisation(rng)
    when = ids.random_date(rng, EPOCH, 720)
    amount = ids.money(rng, 1000, 250000)
    reference = ids.document_number(rng, when.year, 0)
    page = Page()
    _letterhead(page, bank)
    _window(page, bank, _recipient_lines(customer))
    _info(page, [("Erstellt am", ids.fmt_date_de(when))])
    page.text(LEFT, 290, "Ausführungsbestätigung Überweisung", 12, bold=True)
    y = 320.0
    for label, value in (("Auftraggeber", customer.name), ("IBAN Auftraggeber", ids.group_iban(customer.iban)),
                         ("Empfänger", payee.name), ("IBAN Empfänger", ids.group_iban(payee.iban)),
                         ("Betrag", f"{ids.fmt_de(amount)} EUR"), ("Verwendungszweck", f"Rechnung {reference}"),
                         ("Ausführungsdatum", ids.fmt_date_de(when))):
        page.text(LEFT, y, f"{label}:", 9)
        page.text(200, y, value, 9)
        y += 16
    page.text(LEFT, y + 20, "Diese Bestätigung wurde maschinell erstellt und ist ohne Unterschrift gültig.", 8)
    _footer(page, bank, bank=True)
    truth = _empty_truth()
    truth.update({"sender": bank.name, "recipient": customer.name, "doc_date": when.isoformat(), "currency": "EUR",
                  "iban": customer.iban})
    doc = LogicalDoc("transfer", ["BANK_STATEMENT", "TRANSFER_CONFIRMATION"], [page], truth,
                     summary=f"Transfer confirmation of {bank.name} ({ids.fmt_de(amount)} EUR)")
    doc.facts.update({"customer": customer.name, "payee": payee.name, "payee_iban": payee.iban, "amount": float(amount),
                      "bank_code": bank.iban[4:12], "account": customer.iban[12:]})
    return doc


def build_enclosure(rng: random.Random) -> LogicalDoc:
    """An enclosure without letterhead (§2.6): a call protocol labelled with its own date."""
    when = ids.random_date(rng, EPOCH, 720)
    caller = person(rng)
    page = Page()
    page.text(LEFT, 110, f"Protokoll vom {ids.fmt_date_de(when)}", 12, bold=True)
    page.text(LEFT, 126, "Telefonat mit dem Kundenservice", 9.5)
    y = 150.0
    for line in (f"Teilnehmer: {caller.name}, Kundenservice", "Anlass: Rückfrage zur Vertragsänderung",
                 "Ergebnis: Unterlagen werden nachgereicht.", "Nächster Schritt: Rückruf nach Eingang der Unterlagen."):
        page.text(LEFT, y, line, 9.5)
        y += 15
    truth = _empty_truth()
    truth.update({"doc_date": when.isoformat()})
    return LogicalDoc("enclosure", ["OTHER"], [page], truth, summary="Call protocol (enclosure)")


def build_private_note(rng: random.Random, card: Optional[bool] = None) -> LogicalDoc:
    """A private notice or card without letterhead or address block (§2.7: no parties): a death notice or a greeting card."""
    greeting = rng.random() < 0.5 if card is None else card
    page = Page(width=420.0, height=298.0)
    if greeting:
        to, by = person(rng), person(rng)
        lines = [("Herzlichen Glückwunsch zum Geburtstag!", 15, True), ("", 10, False),
                 (f"Liebe{'' if to.female else 'r'} {to.first},", 11, False),
                 ("wir wünschen Dir alles Gute, Gesundheit und viele schöne Momente.", 10, False), ("", 10, False),
                 (f"Herzliche Grüße, {by.first} und Familie", 10, False)]
        summary = "Greeting card"
    else:
        died = person(rng)
        when = ids.random_date(rng, EPOCH, 720)
        lines = [("In stiller Trauer", 15, True), ("", 10, False), (died.name, 13, True),
                 (f"* {ids.fmt_date_de(when - timedelta(days=365 * rng.randint(60, 95)))}   "
                  f"† {ids.fmt_date_de(when)}", 10, False), ("", 10, False),
                 ("Wir nehmen Abschied in Dankbarkeit für die gemeinsame Zeit.", 10, False),
                 (f"Die Trauerfeier findet im engsten Familienkreis in {died.address.city} statt.", 9, False)]
        summary = "Death notice"
    y = 60.0
    for text, size, bold in lines:
        if text:
            width = Run(0, 0, text, size, bold).width()
            page.text(max(20.0, (page.width - width) / 2), y, text, size, bold=bold)  # centred
        y += size + 8
    return LogicalDoc("note", ["NOTE"], [page], _empty_truth(), summary=summary)


def build_contract(rng: random.Random, start: date, signed: date, signature_text: Optional[str] = None,
                   amendment: Optional[date] = None, rent: bool = True, waiver_years: Optional[int] = None) -> LogicalDoc:
    """
    A two-page rental contract: landlord letterhead, contract number, start
    printed with a two-digit year ('Mietbeginn: ab 1.10.24'), rent, and a
    signing line 'Ort, den d/m/yy' in a typewriter-like hand. With an
    amendment date the document is a later amendment ('Nachtrag') to the
    contract that started on `start`.

    Args:
        rng: Random source (parties, numbers).
        start: Contract start (effective date).
        signed: Signing date (the doc_date), or the amendment date.
        signature_text: Text printed for the signing date instead of 'd/m/yy' (e.g. an OCR misreading).
        amendment: Date of a later amendment; the document is then that amendment.
        rent: Print the rented flat, a rent table (components and the monthly total), the payee account and a
            deposit; False = a contract without payment terms.
        waiver_years: Print a mutual waiver of termination for this many years (a minimum term, not a notice period).

    Returns:
        LogicalDoc (doc_types CONTRACT; facts: effective_date, contract_number, start_text, signing_line and - with rent -
        components, rent_total, iban, bic, holder, reference, deposit, object).
    """
    landlord = organisation(rng, ["Beispiel Immobilien", "Muster Wohnbau"])
    tenant = person(rng)
    number = f"MV-{rng.randint(10000, 99999)}"
    doc_date = amendment or signed
    start_text = f"{start.day}.{start.month}.{start.year % 100:02d}"
    first, second = Page(), Page()
    _letterhead(first, landlord)
    title = "Nachtrag Nr. 1 zum Mietvertrag" if amendment else "Mietvertrag über Wohnraum"
    first.text(LEFT, 110, title, 14, bold=True)
    flat = person(rng).address
    location = rng.choice(("EG links", "EG rechts", "1. OG links", "2. OG rechts", "DG"))
    components = [("Grundmiete", ids.money(rng, 40000, 120000)), ("Pauschale für Betriebs- und Heizkosten", ids.money(rng, 9000, 30000)),
                  ("Pauschale für Strom", ids.money(rng, 3000, 9000))]
    total = sum((a for _, a in components), Decimal("0"))
    deposit = total * 2
    reference = f"Miete {flat.street} {location}"
    lines = [f"Vertragsnummer: {number}", f"zwischen {landlord.name}, {landlord.address.one_line()} (Vermieter)",
             f"und {tenant.name}, {tenant.address.one_line()} (Mieter)", ""]
    if rent:
        lines += [f"§ 1 Mietsache: Vermietet wird die Wohnung im Hause {flat.street}, {flat.zip_code} {flat.city}, {location}."]
    lines += [f"§ 2 Mietbeginn: Das Mietverhältnis beginnt ab {start_text} und läuft auf unbestimmte Zeit."]
    y = 140.0

    def put(text: str) -> None:
        nonlocal y
        for part in wrap(text, RIGHT - LEFT, 10) if text else [""]:
            if part:
                first.text(LEFT, y, part, 10)
            y += 14

    for line in lines:
        put(line)
    if rent:
        put("§ 3 Miete: Die Miete ist monatlich im Voraus bis zum dritten Werktag zu zahlen. Sie setzt sich zusammen aus:")
        for label, amount in components + [("Monatlich insgesamt zu zahlende Miete", total)]:
            first.text(LEFT + 20, y, label, 10, bold=label.startswith("Monatlich"))
            first.text(RIGHT, y, f"{ids.fmt_de(amount)} EUR", 10, bold=label.startswith("Monatlich"), align="right")
            y += 14
        y += 4
        for label, value in (("Kontoinhaber", landlord.name), ("IBAN", ids.group_iban(landlord.iban)), ("BIC", landlord.bic),
                             ("Verwendungszweck", reference)):
            first.text(LEFT + 20, y, f"{label}: {value}", 10)
            y += 14
        y += 4
        put(f"§ 4 Kaution: Die Kaution in Höhe von {ids.fmt_de(deposit)} EUR ist zu Beginn des Mietverhältnisses zu leisten.")
    else:
        put("§ 3 Hausordnung: Die Hausordnung ist Bestandteil dieses Vertrags.")
    waiver = (f"Die Vertragsparteien verzichten wechselseitig auf ihr Recht zur Kündigung des Mietvertrags für die Dauer "
              f"von {waiver_years} Jahren. Eine Kündigung ist erstmalig nach Ablauf dieses Zeitraums mit gesetzlicher Frist "
              f"zulässig.")
    if waiver_years:
        put(f"§ 5 Kündigungsverzicht: {waiver}")
    if amendment:
        put(f"§ 6 Nachtrag: Ab {ids.fmt_date_de(amendment)} wird ein Stellplatz mitvermietet.")
    _footer(first, landlord, (1, 2))
    second.text(LEFT, 80, "§ 5 Schlussbestimmungen: Änderungen bedürfen der Schriftform.", 10)
    signing = signature_text or f"{doc_date.day}/{doc_date.month}/{doc_date.year % 100:02d}"
    second.text(LEFT, 160, f"{landlord.address.city}, den {signing}", 11, mono=True)
    signature(second, LEFT, 200, rng.randint(0, 999))
    signature(second, 330, 200, rng.randint(0, 999))
    second.text(LEFT, 215, "Vermieter", 8)
    second.text(330, 215, "Mieter", 8)
    _footer(second, landlord, (2, 2))
    truth = _empty_truth()
    truth.update({"sender": landlord.name, "recipient": tenant.name, "doc_date": doc_date.isoformat(), "doc_number": number})
    doc = LogicalDoc("contract", ["CONTRACT"], [first, second], truth, summary=title)
    doc.visual["has_signature"] = True
    doc.facts.update({"effective_date": start.isoformat(), "start_text": f"ab {start_text}", "contract_number": number,
                      "signing_line": f"{landlord.address.city}, den {signing}"})
    if rent:
        doc.facts.update({"components": [(label, float(amount)) for label, amount in components], "rent_total": float(total),
                          "iban": landlord.iban, "bic": landlord.bic, "holder": landlord.name, "reference": reference,
                          "deposit": float(deposit), "object": {"street": flat.street, "zip_code": flat.zip_code,
                                                                "city": flat.city, "location_detail": location}})
    if waiver_years:
        doc.facts.update({"waiver_years": waiver_years, "waiver_clause": waiver})
    return doc


def build_permission_letter(rng: random.Random) -> LogicalDoc:
    """
    A short permission from a landlord to a tenant without letterhead: the
    tenant is named in the printed text, the landlord's signature line is
    blank, only the tenant signs by hand (§2.9: no sender is printed).
    """
    tenant = person(rng)
    when = ids.random_date(rng, EPOCH, 720)
    page = Page()
    page.text(LEFT, 110, "Genehmigung", 14, bold=True)
    title = "Der Mieterin" if tenant.female else "Dem Mieter"
    salutation = "Frau" if tenant.female else "Herrn"
    body = (f"{title} {salutation} {tenant.name}, {tenant.address.one_line()}, wird genehmigt, im Garten des Hauses ein "
            f"Gewächshaus aufzustellen. Die Genehmigung kann jederzeit widerrufen werden.")
    y = 150.0
    for part in wrap(body, RIGHT - LEFT, 10):
        page.text(LEFT, y, part, 10)
        y += 14
    page.rule(LEFT, y + 60, LEFT + 200)
    page.text(LEFT, y + 72, "Datum/Unterschrift Vermieter", 8)
    page.text(330, y + 52, ids.fmt_date_de(when), 11, hand=True)
    signature(page, 410, y + 50, rng.randint(0, 999))
    page.rule(330, y + 60, RIGHT)
    page.text(330, y + 72, "Datum/Unterschrift Mieter", 8)
    truth = _empty_truth()
    truth.update({"recipient": tenant.name})  # the only date is the tenant's handwritten one next to the signature (§3)
    doc = LogicalDoc("permission", ["OTHER", "LETTER"], [page], truth, summary="Permission of the landlord")
    doc.visual.update({"has_signature": True, "has_handwriting": True})
    doc.facts.update({"tenant": tenant.name, "tenant_surname": tenant.last})
    return doc


def build_landlord_confirmation(rng: random.Random, move_out: bool = False) -> LogicalDoc:
    """
    A landlord's confirmation of a move-in / move-out (Wohnungsgeberbestätigung
    nach § 19 BMG): a printed form filled in by hand (handwriting-like font)
    with the issuer, the event and its date, the flat address and floor, the
    persons and a signature; a form print number in the footer (§4: never the
    doc_number).
    """
    landlord, flat = person(rng), person(rng).address
    movers = [person(rng) for _ in range(rng.randint(1, 2))]
    event_date = ids.random_date(rng, EPOCH, 700)
    signed = event_date + timedelta(days=rng.randint(0, 10))
    floor = rng.choice(("EG", "1. OG links", "2. OG rechts", "DG"))
    page = Page()
    page.text(LEFT, 70, "Wohnungsgeberbestätigung", 15, bold=True)
    page.text(LEFT, 86, "nach § 19 des Bundesmeldegesetzes (BMG) - zur Vorlage bei der Meldebehörde", 8.5)
    y = 120.0

    def row(label: str, value: str, tick: Optional[bool] = None) -> None:
        nonlocal y
        box = "[x] " if tick else "[ ] " if tick is False else ""
        page.text(LEFT, y, f"{box}{label}", 9)
        if value:
            page.text(230, y, value, 11, hand=True)
        page.rule(228, y + 3, RIGHT)
        y += 26

    row("Name des Wohnungsgebers:", landlord.name)
    row("Einzug am:", "" if move_out else ids.fmt_date_de(event_date), tick=not move_out)
    row("Auszug am:", ids.fmt_date_de(event_date) if move_out else "", tick=move_out)
    row("Anschrift der Wohnung:", f"{flat.street}, {flat.zip_code} {flat.city}")
    row("Geschoss / Lage:", floor)
    row("Folgende Person(en) ist/sind ein- bzw. ausgezogen:", "")
    for mover in movers:
        page.text(230, y - 10, mover.name, 11, hand=True)
        y += 16
    row("Anschrift des Wohnungsgebers:", landlord.address.one_line().replace(" · ", ", "))
    row("Der Wohnungsgeber ist Eigentümer der Wohnung", "", tick=True)
    row("Ort, Datum:", f"{landlord.address.city}, {ids.fmt_date_de(signed)}")
    signature(page, 240, y + 4, rng.randint(0, 999))
    page.rule(228, y + 12, RIGHT)
    page.text(230, y + 24, "Unterschrift des Wohnungsgebers", 8)
    code = f"{rng.randint(32, 99)}.{rng.randint(1, 9)}.{rng.randint(1, 999):03d}"
    page.text(LEFT, 815, code, 6.5)
    truth = _empty_truth()
    truth.update({"sender": landlord.name, "doc_date": signed.isoformat()})
    doc = LogicalDoc("landlord_confirmation", ["CERTIFICATE"], [page], truth, summary="Wohnungsgeberbestätigung")
    doc.visual.update({"has_signature": True, "has_handwriting": True})
    doc.facts.update({"persons": [m.name for m in movers], "event": "MOVE_OUT" if move_out else "MOVE_IN",
                      "event_date": event_date.isoformat(), "form_code": code, "floor": floor,
                      "subject": {"street": flat.street, "zip_code": flat.zip_code, "city": flat.city}})
    return doc


def build_stack(rng: random.Random) -> List[LogicalDoc]:
    """A letter with its terms page, followed by a separate enclosure (segmentation test)."""
    letter = build_letter(rng, official=False)
    letter.pages.append(build_terms_page(letter.truth["sender"]))
    return [letter, build_enclosure(rng)]
