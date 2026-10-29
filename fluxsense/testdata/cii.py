# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/cii.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    EN 16931 CII XML (ZUGFeRD / Factur-X) for generated invoices,
                built from the same data as the visible invoice: number,
                dates, parties, VAT ID, payee IBAN, all line items, the tax
                breakdown and the totals.
------------------------------------------------------------------------------
"""

from datetime import date
from decimal import Decimal
from typing import Any, List, Sequence, Tuple
from xml.sax.saxutils import escape

from fluxsense.testdata.parties import Organisation

_NS = ('xmlns:rsm="urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100" '
       'xmlns:ram="urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100" '
       'xmlns:udt="urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100"')


def _d(value: date) -> str:
    return f'<udt:DateTimeString format="102">{value.strftime("%Y%m%d")}</udt:DateTimeString>'


def _address(street: str, zip_code: str, city: str) -> str:
    return (f"<ram:PostalTradeAddress><ram:PostcodeCode>{escape(zip_code)}</ram:PostcodeCode>"
            f"<ram:LineOne>{escape(street)}</ram:LineOne><ram:CityName>{escape(city)}</ram:CityName>"
            f"<ram:CountryID>DE</ram:CountryID></ram:PostalTradeAddress>")


def build_cii(number: str, issued: date, due: date, seller: Organisation, buyer: Any,
              rows: Sequence[Tuple[str, str, str, Decimal, Decimal, Decimal]], net: Decimal, tax: Decimal,
              gross: Decimal) -> bytes:
    """
    Builds the CII XML of a generated invoice.

    Args:
        number: Invoice number (BT-1).
        issued: Invoice date (BT-2).
        due: Payment due date (BT-9).
        seller: Seller organisation.
        buyer: Buyer person or organisation.
        rows: (position, part number, description, quantity, unit price, line total).
        net: Sum of line totals = tax basis.
        tax: VAT total.
        gross: Grand total.

    Returns:
        XML bytes (UTF-8).
    """
    lines: List[str] = []
    for pos, sku, desc, qty, unit, total in rows:
        lines.append(
            f"<ram:IncludedSupplyChainTradeLineItem><ram:AssociatedDocumentLineDocument><ram:LineID>{pos}</ram:LineID>"
            f"</ram:AssociatedDocumentLineDocument><ram:SpecifiedTradeProduct><ram:SellerAssignedID>{escape(sku)}"
            f"</ram:SellerAssignedID><ram:Name>{escape(desc)}</ram:Name></ram:SpecifiedTradeProduct>"
            f"<ram:SpecifiedLineTradeAgreement><ram:NetPriceProductTradePrice><ram:ChargeAmount>{unit}</ram:ChargeAmount>"
            f"</ram:NetPriceProductTradePrice></ram:SpecifiedLineTradeAgreement><ram:SpecifiedLineTradeDelivery>"
            f'<ram:BilledQuantity unitCode="C62">{qty}</ram:BilledQuantity></ram:SpecifiedLineTradeDelivery>'
            f"<ram:SpecifiedLineTradeSettlement><ram:ApplicableTradeTax><ram:TypeCode>VAT</ram:TypeCode>"
            f"<ram:CategoryCode>S</ram:CategoryCode><ram:RateApplicablePercent>19</ram:RateApplicablePercent>"
            f"</ram:ApplicableTradeTax><ram:SpecifiedTradeSettlementLineMonetarySummation><ram:LineTotalAmount>{total}"
            f"</ram:LineTotalAmount></ram:SpecifiedTradeSettlementLineMonetarySummation></ram:SpecifiedLineTradeSettlement>"
            f"</ram:IncludedSupplyChainTradeLineItem>")
    buyer_address = buyer.address
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<rsm:CrossIndustryInvoice {_NS}>'
        f"<rsm:ExchangedDocumentContext><ram:GuidelineSpecifiedDocumentContextParameter><ram:ID>"
        f"urn:cen.eu:en16931:2017</ram:ID></ram:GuidelineSpecifiedDocumentContextParameter></rsm:ExchangedDocumentContext>"
        f"<rsm:ExchangedDocument><ram:ID>{escape(number)}</ram:ID><ram:TypeCode>380</ram:TypeCode>"
        f"<ram:IssueDateTime>{_d(issued)}</ram:IssueDateTime></rsm:ExchangedDocument>"
        f"<rsm:SupplyChainTradeTransaction>{''.join(lines)}<ram:ApplicableHeaderTradeAgreement>"
        f"<ram:SellerTradeParty><ram:Name>{escape(seller.name)}</ram:Name>"
        f"{_address(seller.address.street, seller.address.zip_code, seller.address.city)}"
        f'<ram:SpecifiedTaxRegistration><ram:ID schemeID="VA">{seller.vat_id}</ram:ID></ram:SpecifiedTaxRegistration>'
        f"</ram:SellerTradeParty><ram:BuyerTradeParty><ram:Name>{escape(buyer.name)}</ram:Name>"
        f"{_address(buyer_address.street, buyer_address.zip_code, buyer_address.city)}</ram:BuyerTradeParty>"
        f"</ram:ApplicableHeaderTradeAgreement><ram:ApplicableHeaderTradeDelivery/>"
        f"<ram:ApplicableHeaderTradeSettlement><ram:InvoiceCurrencyCode>EUR</ram:InvoiceCurrencyCode>"
        f"<ram:SpecifiedTradeSettlementPaymentMeans><ram:TypeCode>58</ram:TypeCode>"
        f"<ram:PayeePartyCreditorFinancialAccount><ram:IBANID>{seller.iban}</ram:IBANID>"
        f"</ram:PayeePartyCreditorFinancialAccount></ram:SpecifiedTradeSettlementPaymentMeans>"
        f"<ram:ApplicableTradeTax><ram:CalculatedAmount>{tax}</ram:CalculatedAmount><ram:TypeCode>VAT</ram:TypeCode>"
        f"<ram:BasisAmount>{net}</ram:BasisAmount><ram:CategoryCode>S</ram:CategoryCode>"
        f"<ram:RateApplicablePercent>19</ram:RateApplicablePercent></ram:ApplicableTradeTax>"
        f"<ram:SpecifiedTradePaymentTerms><ram:DueDateDateTime>{_d(due)}</ram:DueDateDateTime>"
        f"</ram:SpecifiedTradePaymentTerms><ram:SpecifiedTradeSettlementHeaderMonetarySummation>"
        f"<ram:LineTotalAmount>{net}</ram:LineTotalAmount><ram:TaxBasisTotalAmount>{net}</ram:TaxBasisTotalAmount>"
        f'<ram:TaxTotalAmount currencyID="EUR">{tax}</ram:TaxTotalAmount><ram:GrandTotalAmount>{gross}</ram:GrandTotalAmount>'
        f"<ram:DuePayableAmount>{gross}</ram:DuePayableAmount></ram:SpecifiedTradeSettlementHeaderMonetarySummation>"
        f"</ram:ApplicableHeaderTradeSettlement></rsm:SupplyChainTradeTransaction></rsm:CrossIndustryInvoice>")
    return xml.encode("utf-8")
