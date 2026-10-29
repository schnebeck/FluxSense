# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/samples.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Synthetic sample PDFs (reportlab) for tests and smoke runs:
                faulty-VAT invoice, two-page invoice with carry-over rows,
                three-page bank statement with carry-over rows and running
                balance, a letter, and an image-only (scanned) variant.
                All names, addresses and IBANs are fictitious test data.
------------------------------------------------------------------------------
"""

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Tuple

from fluxsense.pdflock import PYMUPDF_LOCK

SELLER = ("Muster Elektro GmbH", "Industriestraße 7", "00173 Stuttgart", "DE123456789")
BUYER = ("Erika Beispiel", "Lindenweg 3", "00115 Berlin")
TEST_IBAN = "DE44 9901 2345 0532 0130 00"
TEST_BIC = "XMPLDEFFXXX"


def _fmt(value: Decimal) -> str:
    """German amount format 1.234,56."""
    q = f"{value:,.2f}"
    return q.replace(",", "X").replace(".", ",").replace("X", ".")


@dataclass
class SampleTruth:
    """Ground truth of a generated sample (for tests)."""

    path: Path
    data: Dict[str, object]


def _styles() -> Tuple[object, object, object]:
    """Returns (normal, heading, small) paragraph styles."""
    from reportlab.lib.styles import getSampleStyleSheet
    sheet = getSampleStyleSheet()
    return sheet["Normal"], sheet["Heading1"], sheet["BodyText"]


def _table_style(header_rows: int = 1) -> object:
    """Grid style with a bold header."""
    from reportlab.lib import colors
    from reportlab.platypus import TableStyle
    return TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
    ])


def _address_block(normal: object) -> List[object]:
    """Seller letterhead and buyer address."""
    from reportlab.platypus import Paragraph, Spacer
    return [
        Paragraph(f"<b>{SELLER[0]}</b> · {SELLER[1]} · {SELLER[2]} · USt-IdNr. {SELLER[3]}", normal),
        Spacer(1, 12),
        Paragraph(f"{BUYER[0]}<br/>{BUYER[1]}<br/>{BUYER[2]}", normal),
        Spacer(1, 18),
    ]


def make_invoice_wrong_vat(path: Path) -> SampleTruth:
    """
    One-page invoice: net 300,00, printed VAT 59,00 (correct 19 % = 57,00),
    printed gross 357,00 (net + printed VAT would be 359,00).

    Args:
        path: Output file.

    Returns:
        SampleTruth.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table

    normal, heading, _ = _styles()
    story = _address_block(normal)
    story += [Paragraph("Rechnung", heading),
              Paragraph("Rechnungsnummer: RE-2026-0815<br/>Rechnungsdatum: 14.09.2026<br/>"
                        "Leistungsdatum: 10.09.2026<br/>Kundennummer: K-4711<br/>Fällig am: 28.09.2026", normal),
              Spacer(1, 12)]
    rows = [["Pos", "Beschreibung", "Menge", "Einzelpreis", "Gesamt"],
            ["1", "Installation Steckdosen", "4", "25,00", "100,00"],
            ["2", "Kabel NYM-J 3x1,5 (Meter)", "50", "2,00", "100,00"],
            ["3", "Arbeitszeit Elektriker (Std.)", "2", "50,00", "100,00"]]
    story.append(Table(rows, colWidths=[30, 220, 50, 80, 80], style=_table_style()))
    story.append(Spacer(1, 12))
    totals = [["Nettobetrag", "300,00 EUR"], ["zzgl. 19 % USt", "59,00 EUR"], ["Rechnungsbetrag", "357,00 EUR"]]
    story.append(Table(totals, colWidths=[300, 160], style=_table_style(0)))
    story.append(Spacer(1, 18))
    story.append(Paragraph(f"Bitte überweisen Sie den Betrag bis zum 28.09.2026 auf IBAN {TEST_IBAN}, BIC {TEST_BIC} "
                           "(Beispielbank). Verwendungszweck: RE-2026-0815", normal))
    SimpleDocTemplate(str(path), pagesize=A4, title="Rechnung RE-2026-0815").build(story)
    return SampleTruth(path, {"invoice_number": "RE-2026-0815", "net": Decimal("300.00"), "tax": Decimal("59.00"),
                              "gross": Decimal("357.00"), "iban": TEST_IBAN.replace(" ", "")})


def make_invoice_multipage(path: Path, items: int = 34) -> SampleTruth:
    """
    Two-page invoice whose line-item table continues on page 2 with a
    repeated header and 'Übertrag' carry-over rows.

    Args:
        path: Output file.
        items: Number of line items.

    Returns:
        SampleTruth with the line items.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

    normal, heading, _ = _styles()
    lines = []
    for i in range(1, items + 1):
        qty = Decimal(i % 5 + 1)
        price = Decimal("3.50") + Decimal(i) * Decimal("1.25")
        lines.append((str(i), f"Artikel {i:03d} Schraubensortiment Typ {chr(65 + i % 26)}", qty, price, qty * price))
    split = 20
    header = ["Pos", "Beschreibung", "Menge", "Einzelpreis", "Gesamt"]
    first = sum((line[4] for line in lines[:split]), Decimal("0"))
    net = sum((line[4] for line in lines), Decimal("0"))
    tax = (net * Decimal("0.19")).quantize(Decimal("0.01"))
    story = _address_block(normal)
    story += [Paragraph("Rechnung", heading),
              Paragraph("Rechnungsnummer: RE-2026-1001<br/>Rechnungsdatum: 01.10.2026<br/>Fällig am: 15.10.2026", normal),
              Spacer(1, 10)]
    rows1 = [header] + [[n, d, str(q), _fmt(p), _fmt(t)] for n, d, q, p, t in lines[:split]]
    rows1.append(["", "Übertrag", "", "", _fmt(first)])
    story.append(Table(rows1, colWidths=[30, 230, 45, 75, 80], style=_table_style()))
    story.append(Paragraph("Seite 1 von 2", normal))
    story.append(PageBreak())
    story.append(Paragraph(f"Rechnung RE-2026-1001 · {SELLER[0]}", normal))
    rows2 = [header, ["", "Übertrag", "", "", _fmt(first)]]
    rows2 += [[n, d, str(q), _fmt(p), _fmt(t)] for n, d, q, p, t in lines[split:]]
    story.append(Table(rows2, colWidths=[30, 230, 45, 75, 80], style=_table_style()))
    story.append(Spacer(1, 10))
    totals = [["Nettobetrag", f"{_fmt(net)} EUR"], ["zzgl. 19 % USt", f"{_fmt(tax)} EUR"],
              ["Rechnungsbetrag", f"{_fmt(net + tax)} EUR"]]
    story.append(Table(totals, colWidths=[300, 160], style=_table_style(0)))
    story.append(Paragraph(f"Zahlbar auf IBAN {TEST_IBAN} · Seite 2 von 2", normal))
    SimpleDocTemplate(str(path), pagesize=A4, title="Rechnung RE-2026-1001").build(story)
    return SampleTruth(path, {"lines": lines, "net": net, "tax": tax, "gross": net + tax, "split": split})


def make_invoice_long(path: Path, per_page: Tuple[int, ...] = (5, 9, 8, 9, 9, 8, 1), info_pages: int = 3) -> SampleTruth:
    """
    Long invoice: the item table runs over len(per_page) pages,
    the totals block follows the last item on its page, then shipping /
    box information pages of the same invoice follow.

    Args:
        path: Output file.
        per_page: Item rows per page.
        info_pages: Trailing information pages without items.

    Returns:
        SampleTruth with the line items (pos, description, qty, price, total, page) and totals.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

    normal, heading, _ = _styles()
    lines: List[Tuple[str, str, Decimal, Decimal, Decimal, int]] = []
    number = 0
    for page, count in enumerate(per_page, start=1):
        for _ in range(count):
            number += 1
            qty = Decimal(number % 7 + 1)
            price = Decimal("0.25") + Decimal(number) * Decimal("0.75")
            lines.append((str(number), f"PART: BX-{number:04d}-ND DESC: Bauteil {number:03d}", qty, price, qty * price, page))
    net = sum((line[4] for line in lines), Decimal("0"))
    tax = (net * Decimal("0.19")).quantize(Decimal("0.01"))
    header = ["Pos", "Beschreibung", "Menge", "Einzelpreis", "Betrag"]
    total_pages = len(per_page) + info_pages
    story = _address_block(normal)
    story += [Paragraph("Rechnung / Invoice", heading),
              Paragraph("Rechnungsnummer: RE-2031-4711<br/>Rechnungsdatum: 10.02.2031", normal), Spacer(1, 10)]
    for page in range(1, len(per_page) + 1):
        if page > 1:
            story.append(Paragraph("Rechnungsnummer: RE-2031-4711", normal))
        rows = [header] + [[n, d, str(q), _fmt(p), _fmt(t)] for n, d, q, p, t, pg in lines if pg == page]
        story.append(Table(rows, colWidths=[30, 230, 45, 75, 80], style=_table_style()))
        if page == len(per_page):
            totals = [["Zwischensumme / Charges subtotal", f"{_fmt(net)} EUR"], ["MwSt 19 %", f"{_fmt(tax)} EUR"],
                      ["Gesamt / Total due", f"{_fmt(net + tax)} EUR"]]
            story.append(Spacer(1, 10))
            story.append(Table(totals, colWidths=[300, 160], style=_table_style(0)))
            story.append(Paragraph(f"Zahlbar auf IBAN {TEST_IBAN}", normal))
        story.append(Paragraph(f"Seite {page} von {total_pages}", normal))
        story.append(PageBreak())
    for extra in range(1, info_pages + 1):
        page = len(per_page) + extra
        story.append(Paragraph("Versandinformationen / Shipping information", heading))
        story.append(Paragraph(f"Paket {extra}: Gewicht 1,2 kg, Sendungsnummer 1Z99{extra:04d}", normal))
        story.append(Paragraph(f"Seite {page} von {total_pages}", normal))
        if extra < info_pages:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4, title="Rechnung RE-2031-4711").build(story)
    return SampleTruth(path, {"lines": lines, "net": net, "tax": tax, "gross": net + tax, "pages": total_pages,
                              "totals_page": len(per_page)})


def make_bank_statement(path: Path, bookings: int = 48, per_page: int = 18) -> SampleTruth:
    """
    Three-page bank statement with running balance, repeated headers and
    'Übertrag' rows at page breaks.

    Args:
        path: Output file.
        bookings: Number of bookings.
        per_page: Bookings per page.

    Returns:
        SampleTruth with the bookings.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

    normal, heading, _ = _styles()
    opening = Decimal("1523.40")
    balance = opening
    start = date(2026, 9, 1)
    counterparties = ["REWE Markt", "Stadtwerke Musterstadt", "Gehalt Beispiel AG", "Amazon EU", "Miete Lindenweg",
                      "Tankstelle Nord", "Apotheke am Markt", "Versicherung Muster VVaG"]
    entries = []
    for i in range(bookings):
        day = start + timedelta(days=i * 29 // bookings)
        amount = Decimal("2450.00") if i % 16 == 2 else -(Decimal("7.35") + Decimal(i * 13 % 97) + Decimal(i % 7) / 4)
        balance += amount
        entries.append((day, counterparties[i % len(counterparties)], f"Ref {1000 + i}", amount, balance))
    header = ["Buchung", "Valuta", "Vorgang", "Betrag EUR", "Saldo EUR"]
    story = [Paragraph("<b>Musterbank eG</b> · Kontoauszug Nr. 9/2026", normal),
             Paragraph(f"Kontoinhaber: {BUYER[0]} · IBAN {TEST_IBAN} · Zeitraum 01.09.2026 - 30.09.2026", normal),
             Spacer(1, 8), Paragraph(f"Alter Kontostand: {_fmt(opening)} EUR", normal), Spacer(1, 6)]
    pages = [entries[i:i + per_page] for i in range(0, len(entries), per_page)]
    running = opening
    for number, chunk in enumerate(pages, start=1):
        rows = [header]
        if number > 1:
            rows.append(["", "", "Übertrag", "", _fmt(running)])
        for day, cp, ref, amount, bal in chunk:
            rows.append([day.strftime("%d.%m."), day.strftime("%d.%m."), f"{cp} {ref}", _fmt(amount), _fmt(bal)])
        running = chunk[-1][4]
        if number < len(pages):
            rows.append(["", "", "Übertrag", "", _fmt(running)])
        story.append(Table(rows, colWidths=[50, 50, 230, 80, 80], style=_table_style()))
        story.append(Paragraph(f"Seite {number} von {len(pages)}", normal))
        if number < len(pages):
            story.append(PageBreak())
            story.append(Paragraph("Musterbank eG · Kontoauszug Nr. 9/2026 (Fortsetzung)", normal))
    debits = sum((-e[3] for e in entries if e[3] < 0), Decimal("0"))
    credits = sum((e[3] for e in entries if e[3] > 0), Decimal("0"))
    story += [Spacer(1, 8), Paragraph(f"Neuer Kontostand: {_fmt(balance)} EUR", normal),
              Paragraph(f"Summe Belastungen: {_fmt(debits)} EUR · Summe Gutschriften: {_fmt(credits)} EUR", normal)]
    SimpleDocTemplate(str(path), pagesize=A4, title="Kontoauszug 9/2026").build(story)
    return SampleTruth(path, {"entries": entries, "opening": opening, "closing": balance, "pages": len(pages),
                              "debits": debits, "credits": credits})


def make_letter(path: Path) -> SampleTruth:
    """One-page letter with a deadline."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    normal, heading, _ = _styles()
    story = [Paragraph("<b>Hausverwaltung Sonnenhof GmbH</b> · Am Park 12 · 00345 Musterstadt", normal), Spacer(1, 12),
             Paragraph(f"{BUYER[0]}<br/>{BUYER[1]}<br/>{BUYER[2]}", normal), Spacer(1, 12),
             Paragraph("Musterstadt, 05.10.2026", normal), Spacer(1, 8),
             Paragraph("<b>Ankündigung Wartung der Heizungsanlage – Objekt Lindenweg 3</b>", normal), Spacer(1, 8),
             Paragraph("Sehr geehrte Frau Beispiel,<br/><br/>am 20.10.2026 wird die Heizungsanlage gewartet. Bitte gewähren "
                       "Sie dem Techniker zwischen 8 und 12 Uhr Zugang zu Ihrer Wohnung oder teilen Sie uns bis zum "
                       "15.10.2026 einen Ersatztermin mit. Unser Zeichen: HV-2026-117.<br/><br/>Mit freundlichen Grüßen<br/>"
                       "Hausverwaltung Sonnenhof", normal)]
    SimpleDocTemplate(str(path), pagesize=A4, title="Wartung Heizung").build(story)
    return SampleTruth(path, {"deadline": "2026-10-15", "reference": "HV-2026-117"})


def make_scanned(source: Path, target: Path, dpi: int = 150) -> Path:
    """
    Creates an image-only copy of a PDF (simulated scan without text layer).

    Args:
        source: Input PDF.
        target: Output PDF.
        dpi: Raster resolution.

    Returns:
        The target path.
    """
    import pymupdf

    with PYMUPDF_LOCK, pymupdf.open(source) as src, pymupdf.open() as out:
        for page in src:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, pixmap=pix)
        out.save(target)
    return target


def make_all(directory: Path) -> Dict[str, SampleTruth]:
    """
    Writes all samples into a directory.

    Args:
        directory: Output directory (created).

    Returns:
        name -> SampleTruth.
    """
    directory.mkdir(parents=True, exist_ok=True)
    samples = {
        "invoice_wrong_vat": make_invoice_wrong_vat(directory / "invoice_wrong_vat.pdf"),
        "invoice_multipage": make_invoice_multipage(directory / "invoice_multipage.pdf"),
        "invoice_long": make_invoice_long(directory / "invoice_long.pdf"),
        "bank_statement_3p": make_bank_statement(directory / "bank_statement_3p.pdf"),
        "letter": make_letter(directory / "letter.pdf"),
    }
    scanned = make_scanned(directory / "invoice_wrong_vat.pdf", directory / "invoice_wrong_vat_scanned.pdf")
    samples["invoice_wrong_vat_scanned"] = SampleTruth(scanned, samples["invoice_wrong_vat"].data)
    return samples


def build_cii_xml(invoice_number: str, issue_date: str, seller: str, buyer: str, net: str, tax: str, gross: str,
                  rate: str = "19", iban: str = "DE44990123450532013000", type_code: str = "380",
                  due_date: str = "") -> bytes:
    """
    Builds a minimal EN 16931 CII XML (ZUGFeRD / Factur-X) for tests.

    Args:
        invoice_number: BT-1.
        issue_date: BT-2 as YYYYMMDD.
        seller: Seller name.
        buyer: Buyer name.
        net: Tax basis total.
        tax: Tax total.
        gross: Grand total.
        rate: VAT rate.
        iban: Payee IBAN.
        type_code: BT-3 (380 invoice, 381 credit note).

    Returns:
        XML bytes.
    """
    terms = (f"\n   <ram:SpecifiedTradePaymentTerms><ram:DueDateDateTime><udt:DateTimeString format=\"102\">{due_date}"
             f"</udt:DateTimeString></ram:DueDateDateTime></ram:SpecifiedTradePaymentTerms>") if due_date else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rsm:CrossIndustryInvoice xmlns:rsm="urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100"
 xmlns:ram="urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100"
 xmlns:udt="urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100">
 <rsm:ExchangedDocument><ram:ID>{invoice_number}</ram:ID><ram:TypeCode>{type_code}</ram:TypeCode>
  <ram:IssueDateTime><udt:DateTimeString format="102">{issue_date}</udt:DateTimeString></ram:IssueDateTime></rsm:ExchangedDocument>
 <rsm:SupplyChainTradeTransaction>
  <ram:IncludedSupplyChainTradeLineItem><ram:AssociatedDocumentLineDocument><ram:LineID>1</ram:LineID></ram:AssociatedDocumentLineDocument>
   <ram:SpecifiedTradeProduct><ram:Name>Service</ram:Name></ram:SpecifiedTradeProduct>
   <ram:SpecifiedLineTradeAgreement><ram:NetPriceProductTradePrice><ram:ChargeAmount>{net}</ram:ChargeAmount></ram:NetPriceProductTradePrice></ram:SpecifiedLineTradeAgreement>
   <ram:SpecifiedLineTradeDelivery><ram:BilledQuantity unitCode="C62">1</ram:BilledQuantity></ram:SpecifiedLineTradeDelivery>
   <ram:SpecifiedLineTradeSettlement><ram:SpecifiedTradeSettlementLineMonetarySummation><ram:LineTotalAmount>{net}</ram:LineTotalAmount></ram:SpecifiedTradeSettlementLineMonetarySummation></ram:SpecifiedLineTradeSettlement>
  </ram:IncludedSupplyChainTradeLineItem>
  <ram:ApplicableHeaderTradeAgreement>
   <ram:SellerTradeParty><ram:Name>{seller}</ram:Name><ram:PostalTradeAddress><ram:CountryID>DE</ram:CountryID></ram:PostalTradeAddress>
    <ram:SpecifiedTaxRegistration><ram:ID schemeID="VA">DE123456789</ram:ID></ram:SpecifiedTaxRegistration></ram:SellerTradeParty>
   <ram:BuyerTradeParty><ram:Name>{buyer}</ram:Name></ram:BuyerTradeParty>
  </ram:ApplicableHeaderTradeAgreement>
  <ram:ApplicableHeaderTradeSettlement><ram:InvoiceCurrencyCode>EUR</ram:InvoiceCurrencyCode>
   <ram:SpecifiedTradeSettlementPaymentMeans><ram:TypeCode>58</ram:TypeCode>
    <ram:PayeePartyCreditorFinancialAccount><ram:IBANID>{iban}</ram:IBANID></ram:PayeePartyCreditorFinancialAccount></ram:SpecifiedTradeSettlementPaymentMeans>{terms}
   <ram:ApplicableTradeTax><ram:CalculatedAmount>{tax}</ram:CalculatedAmount><ram:TypeCode>VAT</ram:TypeCode>
    <ram:BasisAmount>{net}</ram:BasisAmount><ram:CategoryCode>S</ram:CategoryCode><ram:RateApplicablePercent>{rate}</ram:RateApplicablePercent></ram:ApplicableTradeTax>
   <ram:SpecifiedTradeSettlementHeaderMonetarySummation><ram:LineTotalAmount>{net}</ram:LineTotalAmount>
    <ram:TaxBasisTotalAmount>{net}</ram:TaxBasisTotalAmount><ram:TaxTotalAmount currencyID="EUR">{tax}</ram:TaxTotalAmount>
    <ram:GrandTotalAmount>{gross}</ram:GrandTotalAmount><ram:DuePayableAmount>{gross}</ram:DuePayableAmount>
   </ram:SpecifiedTradeSettlementHeaderMonetarySummation>
  </ram:ApplicableHeaderTradeSettlement>
 </rsm:SupplyChainTradeTransaction>
</rsm:CrossIndustryInvoice>""".encode("utf-8")


def embed_zugferd(source: Path, target: Path, xml: bytes) -> Path:
    """
    Writes a copy of a PDF with an embedded factur-x.xml (source untouched).

    Args:
        source: Input PDF.
        target: Output PDF.
        xml: CII XML.

    Returns:
        The target path.
    """
    import pymupdf

    with PYMUPDF_LOCK, pymupdf.open(source) as doc:
        doc.embfile_add("factur-x.xml", xml, filename="factur-x.xml", desc="Factur-X invoice")
        doc.save(target)
    return target


def make_phone_bill(path: Path, calls_per_page: int = 70) -> SampleTruth:
    """
    Two-page telecom bill with a dense itemised call list (digit-heavy text,
    a pattern that can overflow a 32k slot context).

    Args:
        path: Output file.
        calls_per_page: Itemised calls per page.

    Returns:
        SampleTruth.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

    normal, heading, _ = _styles()
    story = [Paragraph("<b>Fernmelde Muster AG</b> · Postfach 1 · 00105 Bonn · USt-IdNr. DE987654321", normal), Spacer(1, 8),
             Paragraph(f"{BUYER[0]}<br/>{BUYER[1]}<br/>{BUYER[2]}", normal), Spacer(1, 8),
             Paragraph("Rechnung Telefon und Internet", heading),
             Paragraph("Rechnungsnummer 2026 0815 4711 · Rechnungsdatum 05.10.2026 · Kundennummer 123 456 7890", normal),
             Paragraph("Grundgebühr 39,95 · Verbindungen 12,40 · Nettobetrag 52,35 · USt 19 % 9,95 · Rechnungsbetrag 62,30 EUR", normal),
             PageBreak(), Paragraph("Einzelverbindungsnachweis", heading)]
    rows = [["Datum", "Uhrzeit", "Zielrufnummer", "Dauer", "Betrag"]]
    for i in range(calls_per_page * 2):
        rows.append([f"{1 + i % 28:02d}.09.2026", f"{8 + i % 12:02d}:{i % 60:02d}:{(i * 7) % 60:02d}",
                     f"0{170 + i % 30} {1000000 + i * 7919}", f"00:{i % 59:02d}:{(i * 3) % 60:02d}", f"0,{i % 90 + 10:02d}"])
    story.append(Table(rows, colWidths=[70, 60, 140, 60, 60], style=_table_style()))
    SimpleDocTemplate(str(path), pagesize=A4, title="Telefonrechnung").build(story)
    return SampleTruth(path, {"gross": Decimal("62.30"), "net": Decimal("52.35"), "tax": Decimal("9.95")})
