# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/zugferd.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    ZUGFeRD / Factur-X / XRechnung (CII) support, ported from
                KPaperFlux core/utils/zugferd_extractor.py with fixes
                (currency from InvoiceCurrencyCode, tax rows mapped to
                TaxBreakdownRow, decimals). The XML is authoritative ground
                truth: it short-circuits classification (BT-3 type code) and
                is compared field by field with the visual extraction.
------------------------------------------------------------------------------
"""

import copy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pymupdf as fitz
from lxml import etree

from fluxsense.pdflock import PYMUPDF_LOCK
from fluxsense.grounding.normalize import compact, normalize_text, parse_date, to_decimal
from fluxsense.log import get_logger
from fluxsense.models.result import ZugferdComparison

logger = get_logger("stages.zugferd")

XML_NAMES: Tuple[str, ...] = ("factur-x.xml", "zugferd-invoice.xml", "xrechnung.xml", "zugferd.xml")
NS: Dict[str, str] = {
    "rsm": "urn:un:unece:uncefact:data:standard:CrossIndustryInvoice:100",
    "ram": "urn:un:unece:uncefact:data:standard:ReusableAggregateBusinessInformationEntity:100",
    "udt": "urn:un:unece:uncefact:data:standard:UnqualifiedDataType:100",
}
TYPE_CODE_MAP: Dict[str, List[str]] = {"380": ["INVOICE"], "381": ["CREDIT_NOTE"], "383": ["INVOICE"],
                                       "384": ["INVOICE"], "386": ["INVOICE"], "389": ["INVOICE"]}


def embedded_xml(pdf_path: Path) -> Optional[bytes]:
    """
    Returns the embedded CII XML of a PDF, if any.

    Args:
        pdf_path: PDF file.

    Returns:
        XML bytes or None.
    """
    try:
        with PYMUPDF_LOCK, fitz.open(pdf_path) as doc:
            for i in range(doc.embfile_count()):
                name = str(doc.embfile_info(i).get("name", "")).lower()
                if name in XML_NAMES:
                    return doc.embfile_get(i)
    except (RuntimeError, ValueError) as e:
        logger.warning(f"Could not inspect embedded files of {pdf_path.name}: {e}")
    return None


def _format_date(value: Optional[str]) -> Optional[str]:
    """Converts YYYYMMDD (format 102) to YYYY-MM-DD."""
    if not value:
        return None
    value = value.strip()
    if len(value) == 8 and value.isdigit():
        return f"{value[:4]}-{value[4:6]}-{value[6:]}"
    return value


def _dec(value: Optional[str]) -> Optional[str]:
    """Normalizes an XML decimal string (keeps strings for JSON)."""
    if value is None or not str(value).strip():
        return None
    try:
        return str(Decimal(str(value).strip()))
    except InvalidOperation:
        return None


def parse_cii_xml(xml_bytes: bytes) -> Dict[str, Any]:
    """
    Parses UN/CEFACT CII XML into FluxSense structures.

    Args:
        xml_bytes: XML content.

    Returns:
        {"type_tags", "meta_header", "finance_body"} (finance_body FinanceBody-compatible).

    Raises:
        etree.XMLSyntaxError: If the XML is malformed.
    """
    root = etree.fromstring(xml_bytes)

    def t(xpath: str, context: Any = root) -> Optional[str]:
        """First text/attribute result of an XPath."""
        found = context.xpath(xpath, namespaces=NS)
        if not found:
            return None
        node = found[0]
        if isinstance(node, (str, bytes)):
            return str(node)
        return node.text if node.text is not None else None

    raw_type = (t("//rsm:ExchangedDocument/ram:TypeCode") or "").strip()
    type_tags = TYPE_CODE_MAP.get(raw_type, ["INVOICE"])
    invoice_number = t("//rsm:ExchangedDocument/ram:ID")
    invoice_date = _format_date(t("//rsm:ExchangedDocument/ram:IssueDateTime/udt:DateTimeString"))
    currency = t("//ram:ApplicableHeaderTradeSettlement/ram:InvoiceCurrencyCode") \
        or t("//ram:SpecifiedTradeSettlementHeaderMonetarySummation/ram:TaxTotalAmount/@currencyID") or "EUR"

    accounts = []
    for pm in root.xpath("//ram:SpecifiedTradeSettlementPaymentMeans", namespaces=NS):
        iban = t("ram:PayeePartyCreditorFinancialAccount/ram:IBANID", pm)
        if iban:
            accounts.append({k: v for k, v in {
                "iban": "".join(iban.split()).upper(),
                "bic": t("ram:PayeeSpecifiedCreditorFinancialInstitution/ram:BICID", pm),
                "bank_name": t("ram:PayeeSpecifiedCreditorFinancialInstitution/ram:Name", pm),
            }.items() if v})

    taxes = []
    for tn in root.xpath("//ram:ApplicableHeaderTradeSettlement/ram:ApplicableTradeTax", namespaces=NS):
        row = {"tax_basis_amount": _dec(t("ram:BasisAmount", tn)), "tax_amount": _dec(t("ram:CalculatedAmount", tn)),
               "tax_rate": _dec(t("ram:RateApplicablePercent", tn)), "tax_category": t("ram:CategoryCode", tn) or "S"}
        if row["tax_basis_amount"] is not None and row["tax_amount"] is not None and row["tax_rate"] is not None:
            taxes.append(row)

    summation = root.xpath("//ram:SpecifiedTradeSettlementHeaderMonetarySummation", namespaces=NS)
    totals: Dict[str, Optional[str]] = {}
    if summation:
        s = summation[0]
        totals = {
            "line_total_amount": _dec(t("ram:LineTotalAmount", s)),
            "charge_total_amount": _dec(t("ram:ChargeTotalAmount", s)),
            "allowance_total_amount": _dec(t("ram:AllowanceTotalAmount", s)),
            "tax_basis_total_amount": _dec(t("ram:TaxBasisTotalAmount", s)),
            "tax_total_amount": _dec(t("ram:TaxTotalAmount", s)),
            "grand_total_amount": _dec(t("ram:GrandTotalAmount", s)),
            "due_payable_amount": _dec(t("ram:DuePayableAmount", s)),
        }

    items = []
    for i, node in enumerate(root.xpath("//ram:IncludedSupplyChainTradeLineItem", namespaces=NS)):
        items.append({k: v for k, v in {
            "pos": t("ram:AssociatedDocumentLineDocument/ram:LineID", node) or str(i + 1),
            "description": t("ram:SpecifiedTradeProduct/ram:Name", node),
            "quantity": _dec(t("ram:SpecifiedLineTradeDelivery/ram:BilledQuantity", node)),
            "unit": t("ram:SpecifiedLineTradeDelivery/ram:BilledQuantity/@unitCode", node),
            "unit_price": _dec(t("ram:SpecifiedLineTradeAgreement/ram:NetPriceProductTradePrice/ram:ChargeAmount", node)),
            "total_price": _dec(t("ram:SpecifiedLineTradeSettlement/ram:SpecifiedTradeSettlementLineMonetarySummation/ram:LineTotalAmount", node)),
            "tax_rate": _dec(t("ram:SpecifiedLineTradeSettlement/ram:ApplicableTradeTax/ram:RateApplicablePercent", node)),
        }.items() if v is not None})

    def party(xpath: str) -> Dict[str, Any]:
        """Parses a trade party."""
        nodes = root.xpath(xpath, namespaces=NS)
        if not nodes:
            return {}
        p = nodes[0]
        data = {
            "company": t("ram:Name", p),
            "street": t("ram:PostalTradeAddress/ram:LineOne", p),
            "zip_code": t("ram:PostalTradeAddress/ram:PostcodeCode", p),
            "city": t("ram:PostalTradeAddress/ram:CityName", p),
            "country": t("ram:PostalTradeAddress/ram:CountryID", p),
            "vat_id": t("ram:SpecifiedTaxRegistration/ram:ID[@schemeID='VA']", p),
            "tax_id": t("ram:SpecifiedTaxRegistration/ram:ID[@schemeID='FC']", p),
        }
        return {k: v for k, v in data.items() if v}

    sender = party("//ram:SellerTradeParty")
    if accounts:
        sender["iban"] = accounts[0]["iban"]
    finance = {k: v for k, v in {
        "invoice_number": invoice_number, "invoice_date": invoice_date,
        "due_date": _format_date(t("//ram:SpecifiedTradePaymentTerms/ram:DueDateDateTime/udt:DateTimeString")),
        "currency": currency,
        "payment_reference": t("//ram:ApplicableHeaderTradeSettlement/ram:PaymentReference"),
        "order_number": t("//ram:BuyerOrderReferencedDocument/ram:IssuerAssignedID"),
        "service_date": _format_date(t("//ram:ActualDeliverySupplyChainEvent/ram:OccurrenceDateTime/udt:DateTimeString")),
        "customer_id": t("//ram:BuyerTradeParty/ram:ID"),
        "buyer_reference": t("//ram:ApplicableHeaderTradeAgreement/ram:BuyerReference"),
        "payment_terms": t("//ram:SpecifiedTradePaymentTerms/ram:Description"),
        "payment_accounts": accounts, "line_items": items,
        "monetary_summation": {k: v for k, v in totals.items() if v is not None}, "tax_breakdown": taxes,
    }.items() if v not in (None, [], {})}
    return {"type_tags": type_tags,
            "meta_header": {"sender": sender, "recipient": party("//ram:BuyerTradeParty"),
                            "doc_number": invoice_number, "doc_date": invoice_date},
            "finance_body": finance}


def read_zugferd(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """
    Detects and parses embedded ZUGFeRD XML.

    Args:
        pdf_path: PDF file.

    Returns:
        Parsed data or None.
    """
    xml = embedded_xml(pdf_path)
    if xml is None:
        return None
    try:
        return parse_cii_xml(xml)
    except etree.XMLSyntaxError as e:
        logger.warning(f"Embedded ZUGFeRD XML of {pdf_path.name} is malformed: {e}")
        return None


COMPARE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("finance_body.invoice_number", "identifier"), ("finance_body.invoice_date", "date"),
    ("finance_body.due_date", "date"), ("finance_body.currency", "code"),
    ("finance_body.monetary_summation.line_total_amount", "amount"),
    ("finance_body.monetary_summation.tax_basis_total_amount", "amount"),
    ("finance_body.monetary_summation.tax_total_amount", "amount"),
    ("finance_body.monetary_summation.grand_total_amount", "amount"),
    ("finance_body.monetary_summation.due_payable_amount", "amount"),
    ("meta_header.sender.company", "text"), ("meta_header.sender.vat_id", "identifier"),
    ("meta_header.sender.iban", "identifier"), ("meta_header.recipient.company", "text"),
)


def _get(data: Dict[str, Any], path: str) -> Any:
    """Dotted-path lookup."""
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def values_equal(kind: str, a: Any, b: Any) -> bool:
    """Compares two values of a kind after normalization."""
    if kind == "amount":
        da, db = to_decimal(a), to_decimal(b)
        return da is not None and db is not None and abs(da - db) < Decimal("0.005")
    if kind == "date":
        return parse_date(a) is not None and parse_date(a) == parse_date(b)
    if kind == "identifier":
        return compact(str(a)) == compact(str(b))
    if kind == "text":
        na, nb = normalize_text(str(a)), normalize_text(str(b))
        return bool(na) and (na == nb or na in nb or nb in na)
    return str(a).strip().upper() == str(b).strip().upper()


def compare(xml: Dict[str, Any], visual: Dict[str, Any]) -> ZugferdComparison:
    """
    Compares the XML ground truth with the visual extraction.

    Args:
        xml: parse_cii_xml() result.
        visual: Clean visual fields ({"meta_header", "bodies"}).

    Returns:
        ZugferdComparison (accuracy = matched / compared).
    """
    result = ZugferdComparison()
    xml_view = {"finance_body": xml.get("finance_body", {}), "meta_header": xml.get("meta_header", {})}
    vis_view = {"finance_body": (visual.get("bodies") or {}).get("finance_body") or {},
                "meta_header": visual.get("meta_header") or {}}
    for path, kind in COMPARE_FIELDS:
        expected = _get(xml_view, path)
        if expected in (None, ""):
            continue
        actual = _get(vis_view, path)
        if path == "meta_header.sender.company" and actual is None:
            actual = _get(vis_view, "meta_header.sender.name")
        result.compared += 1
        if actual in (None, ""):
            result.missing_in_visual += 1
            result.differences.append({"field": path, "xml": str(expected), "visual": None})
        elif values_equal(kind, expected, actual):
            result.matched += 1
        else:
            result.mismatched += 1
            result.differences.append({"field": path, "xml": str(expected), "visual": str(actual)})
    xml_items = len(xml.get("finance_body", {}).get("line_items", []))
    vis_items = len(vis_view["finance_body"].get("line_items") or [])
    if xml_items:
        result.compared += 1
        if xml_items == vis_items:
            result.matched += 1
        else:
            result.mismatched += 1
            result.differences.append({"field": "finance_body.line_items.count", "xml": str(xml_items), "visual": str(vis_items)})
    if result.compared:
        result.accuracy = round(result.matched / result.compared, 4)
    return result


def xml_leaves(node: Any, path: str = "") -> Dict[str, Any]:
    """Scalar leaves of XML-derived fields by dotted path (list items by index)."""
    out: Dict[str, Any] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            out.update(xml_leaves(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.update(xml_leaves(value, f"{path}.{index}"))
    elif node not in (None, ""):
        out[path] = node
    return out


def protect_xml_values(fields: Dict[str, Any], xml_fields: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """
    Values of a valid embedded XML are authoritative: any field the
    post-checks, heuristics or corrections changed or nulled is restored to
    its XML value. Fields the XML does not carry are left alone.

    Returns:
        (path, changed value, XML value) of every restored field.
    """
    restored: List[Tuple[str, Any, Any]] = []
    for path, value in xml_leaves(xml_fields).items():
        if path == "extraction_source":
            continue
        node: Any = fields
        parts = path.split(".")
        for part in parts[:-1]:
            if isinstance(node, list) and part.isdigit():
                while len(node) <= int(part):
                    node.append({})
                node = node[int(part)]
            elif isinstance(node, dict):
                nxt = node.get(part)
                if not isinstance(nxt, (dict, list)):
                    nxt = [] if parts[parts.index(part) + 1].isdigit() else {}
                    node[part] = nxt
                node = nxt
        last = parts[-1]
        current = node[int(last)] if isinstance(node, list) and last.isdigit() and int(last) < len(node) else \
            node.get(last) if isinstance(node, dict) else None
        if str(current) != str(value):
            if isinstance(node, dict):
                node[last] = copy.deepcopy(value)
            restored.append((path, current, value))
    return restored


def overlay(visual: Dict[str, Any], xml: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the visual fields with the XML values applied as ground truth
    (finance_body replaced, parties/doc number/date taken from the XML).

    Args:
        visual: Clean visual fields.
        xml: parse_cii_xml() result.

    Returns:
        New fields dict.
    """
    merged = {k: v for k, v in visual.items()}
    bodies = dict(merged.get("bodies") or {})
    bodies["finance_body"] = copy.deepcopy(xml.get("finance_body", {}))  # never share (and mutate) the parsed XML
    merged["bodies"] = bodies
    meta = dict(merged.get("meta_header") or {})
    for key in ("sender", "recipient"):
        if xml["meta_header"].get(key):
            party = dict(meta.get(key) or {})
            party.update(copy.deepcopy(xml["meta_header"][key]))
            meta[key] = party
    for key in ("doc_number", "doc_date"):
        if xml["meta_header"].get(key):
            meta[key] = xml["meta_header"][key]
    merged["meta_header"] = meta
    merged["extraction_source"] = "ZUGFERD_NATIVE"
    return merged
