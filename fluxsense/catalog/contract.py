# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/catalog/contract.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Field contract (docs/field_contract.md) as data: the
                party table (§2), the doc_date (§3) and doc_number (§4)
                label lists in priority order with their 'never' lists, the
                type -> contract-row mapping, tolerant label regexes and the
                adjacency search ('same line, or directly below'). Catalog
                rules, prompt text and post-checks are generated from here;
                tests/unit/test_contract.py checks these rows against the
                markdown document so that both cannot drift apart.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from fluxsense.grounding.normalize import compact, parse_date

CONTRACT_VERSION: str = "1.8"


@dataclass(frozen=True)
class FieldRow:
    """One row of a §3/§4 table: first cell, meaning cell, backticked labels in priority order."""

    row: str
    meaning: str
    labels: Tuple[str, ...]


@dataclass(frozen=True)
class PartyRow:
    """One row of the §2 party table."""

    row: str
    sender: str
    recipient: str


PARTY_ROWS: Dict[str, PartyRow] = {
    "2.1": PartyRow("Letter (DIN 5008 or similar)", "letterhead, return-address line or signing organisation",
                    "the DIN 5008 address window; else the printed addressee"),
    "2.2": PartyRow("Invoice, credit note, receipt", "seller (BT-27)", "buyer (BT-44)"),
    "2.3": PartyRow("Bank / credit-card statement", "bank or card issuer", "account holder (address window)"),
    "2.4": PartyRow("Form filled in by a person (Antrag, Formular, Kündigung, Vollmacht, Erklärung)",
                    "the person who fills in and signs (whoever it is; typed or handwritten entries count)",
                    "the organisation that issued and receives the form"),
    "2.5": PartyRow("Pre-addressed cover or return sheet, not filled in", "the issuing organisation", "the printed address"),
    "2.6": PartyRow("Enclosure without letterhead (log, protocol, screenshot, web page)",
                    "empty unless a sender is printed as such", "empty unless printed"),
    "2.7": PartyRow("Private notices, cards, notes (death notices, greeting cards, memos)", "empty", "empty"),
    "2.8": PartyRow("Bank confirmation of a single transaction (Überweisungsbeleg, Auftragsbestätigung, Ausführungsbestätigung, "
                    "Auslandsüberweisung, Einzahlungsbeleg, Lastschriftbeleg)", "the bank that issued it",
                    "the account holder / ordering customer; the payee of the transfer is a counterparty, not the recipient"),
}
# §2.8 / §7: labels of the ordering customer (never the sender) and of the payee (a counterparty, never the recipient).
ORDERING_CUSTOMER_LABELS: Tuple[str, ...] = ("Auftraggeber", "Kontoinhaber")
PAYEE_LABELS: Tuple[str, ...] = ("Zahlungsempfänger", "Empfänger", "Begünstigter", "Empfängername")
TRANSFER_STRONG_RE = re.compile(r"(überweisungsbeleg|ueberweisungsbeleg|ausführungsbestätigung|ausfuehrungsbestaetigung|"
                                r"auslandsüberweisung|auslandsueberweisung|einzahlungsbeleg|lastschriftbeleg)", re.IGNORECASE)
TRANSFER_CONTEXT_RE = re.compile(r"(überweisung|ueberweisung|lastschrift|einzahlung|auftraggeber|begünstigter|zahlungsauftrag|"
                                 r"empfänger[- ]?iban|iban des empfängers|zahlungsempfänger)", re.IGNORECASE)
PARTY_NEVER: Tuple[str, ...] = ("Sehr geehrte ...", "Privat Versicherte/ Selbstzahler", "Patient/ Patientin")
# Never a party either: info-column labels and postal service / franking marks.
INFO_COLUMN_LABELS: Tuple[str, ...] = (
    "Ihr Ansprechpartner", "Ihre Ansprechpartnerin", "Ansprechpartner", "Ansprechpartnerin", "Sachbearbeiter",
    "Sachbearbeiterin", "Ihr Zeichen", "Unser Zeichen", "Durchwahl", "Telefon", "Telefax", "Datum", "Bearbeiter")
SERVICE_ADDRESS_WORDS: Tuple[str, ...] = (
    "PREMIUMADRESS", "Premiumadresse", "Werbeantwort", "Entgelt bezahlt", "Entgelt zahlt Empfänger",
    "Postvertriebsstück", "Dialogpost", "Infopost", "DV-Freimachung", "Frankit", "Port payé", "Deutsche Post",
    "Bitte freimachen", "Rücksendeadresse")
_NOT_A_PARTY_RE = re.compile(
    r"^\s*(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s*") for w in INFO_COLUMN_LABELS + SERVICE_ADDRESS_WORDS) + r")\b",
    re.IGNORECASE)
# Form-field labels next to which the person filling in a form (§2.4 sender) is printed (typed or handwritten).
FORM_NAME_LABELS: Tuple[str, ...] = (
    "Vorname und Name", "Vor- und Nachname", "Vor- und Zuname", "Name, Vorname", "Vorname, Name", "Name und Vorname",
    "Name des Versicherungsnehmers", "Versicherungsnehmer", "Antragsteller", "Antragstellerin", "Kontoinhaber",
    "Domaininhaber", "Name")


def not_a_party(value: object) -> bool:
    """True for info-column labels ('Ihr Ansprechpartner') and postal service marks ('PREMIUMADRESS') taken as a party."""
    return bool(value) and _NOT_A_PARTY_RE.match(str(value)) is not None
TITLE_WORDS: Tuple[str, ...] = ("Herrn", "Frau", "Dr.")

DOC_DATE_ROWS: Dict[str, FieldRow] = {
    "invoice": FieldRow("Invoice, credit note, dunning, utility bill", "invoice date",
                        ("Rechnungsdatum", "Re.-Dat", "Liefer=Re.Dat", "Invoice date", "Datum")),
    "receipt": FieldRow("Receipt", "transaction/receipt date", ("Datum",)),
    "letter": FieldRow("Letter, official / legal correspondence", "date of the letter", ("Datum", "<Ort>, den <Datum>")),
    "statement": FieldRow("Bank / credit-card statement", "statement or settlement date",
                          ("Auszugsdatum", "Abrechnungsdatum", "erstellt am", "Rechnungsdatum")),
    "form": FieldRow("Form filled in by a person", "date next to the signature (handwritten allowed)", ("Datum", "Ort, Datum")),
    "contract": FieldRow("Contract, policy, certificate, report", "issue date", ("ausgestellt am", "Datum", "Stand")),
    "enclosure": FieldRow("Enclosure without letterhead", "only if labelled as its own date", ("Protokoll vom", "Erstellt am")),
}
DOC_DATE_NEVER: Tuple[str, ...] = ("Auftragsdatum", "Leistungsdatum", "Lieferdatum", "Datum fertig")
# Same meaning as the §3 'never' labels ('order date (Auftragsdatum)').
DOC_DATE_NEVER_SYNONYMS: Tuple[str, ...] = ("Bestelldatum",)
# Unlabelled forms that the §3 rows name in prose ('date line', 'receipt print date').
DATE_LINE_GROUPS: Tuple[str, ...] = ("letter", "receipt")
# §3 'date of the letter' (date line, info column) and 'Datum in the header block': labels on the first page only.
FIRST_PAGE_DATE_GROUPS: Tuple[str, ...] = ("letter", "invoice")

DOC_NUMBER_ROWS: Dict[str, FieldRow] = {
    "invoice": FieldRow("Invoice, credit note, utility bill", "invoice / credit-note number (BT-1)",
                        ("Rechnungsnummer", "Rech-Nr.", "Rechnungs-Nr.", "Gutschrift-Nr.", "Invoice No")),
    "letter": FieldRow("Letter, official / legal correspondence", "the sender's own reference",
                       ("Unser Zeichen", "Aktenzeichen", "Geschäftszeichen", "Vorgangsnummer", "Schadennummer", "Vertragsnummer",
                        "Versicherungsschein-Nr.", "Versicherungsnummer", "Kundennummer")),
    "statement": FieldRow("Bank / credit-card statement", "printed statement number", ("Auszug", "Kontoauszug Nr.", "Abrechnung Nr.")),
    "form": FieldRow("Form", "the form's own number, else the number of the contract/order it refers to",
                     ("Antragsnummer", "Versicherungsschein-Nr.", "Vertragsnummer", "Mitglieds-Nr.", "Mitgliedsnummer",
                      "Unser Zeichen", "Auftragsnummer", "Kundennummer")),
    "contract": FieldRow("Contract, policy", "contract / policy number", ("Vertragsnummer", "Versicherungsschein-Nr.", "Police")),
    "certificate": FieldRow("Certificate, report", "report / certificate number",
                            ("Bericht-Nr.", "Zertifikat-Nr.", "Report No", "No.", "Nr.")),
    "enclosure": FieldRow("Enclosure without letterhead", "empty unless labelled as its own number", ("Unser Zeichen",)),
}
DOC_NUMBER_NEVER: Tuple[str, ...] = ("RECHNUNG RK7", "1/2")
# §3: 'Rechnungsdatum' is a statement date on credit-card statements only.
CARD_ONLY_DATE_LABELS: Tuple[str, ...] = ("Rechnungsdatum",)
# §4: these labels count only in the document's header area (the first lines of its first page).
HEADER_ONLY_NUMBER_LABELS: Tuple[str, ...] = ("No.", "Nr.")
HEADER_AREA_LINES: int = 15
# §4: letters from a court, an arbitration / ombudsman body or a public authority - their case / proceedings number.
AUTHORITY_NUMBER_LABELS: Tuple[str, ...] = ("Verfahren", "…verfahren <nr>", "Az.", "Aktenzeichen")
AUTHORITY_RE = re.compile(r"(gericht|schlichtung|ombuds|behörde|behoerde|staatsanwalt|ministerium|polizei|jobcenter|"
                          r"bundesagentur|verwaltung|(?<![a-zäöüß])amt\b|[a-zäöüß]amt\b)", re.IGNORECASE)
# The enclosure row has no list ('empty unless labelled as its own number'): labels that name a document's OWN number.
OWN_NUMBER_LABELS: Tuple[str, ...] = ("Protokoll-Nr.", "Dokument-Nr.", "Bericht-Nr.", "Report No")
# Labels of numbers that are never a statement's doc_number (§4: account/card/customer numbers on statements).
STATEMENT_FOREIGN_NUMBER_LABELS: Tuple[str, ...] = ("Kundennummer", "Kontonummer", "Kartennummer")


# Labels of the other core fields (§5-§8 describe them in prose); a value printed after one of them on the same
# line (or on the line below a label standing alone) has provenance 'label'.
FIELD_LABELS: Dict[str, Tuple[str, ...]] = {
    "grand_total_amount": ("Rechnungsbetrag", "Gesamtbetrag", "Endbetrag", "Gesamtsumme", "Bruttobetrag", "Brutto", "Summe",
                           "Gesamt", "Total", "zu zahlen", "Zahlbetrag", "Betrag"),
    "due_payable_amount": ("zu zahlen", "Zahlbetrag", "offener Betrag", "fälliger Betrag", "Rechnungsbetrag", "Gesamtbetrag",
                           "Restbetrag", "Amount due"),
    "tax_basis_total_amount": ("Nettobetrag", "Netto", "Summe netto", "Zwischensumme", "Net", "Warenwert"),
    "line_total_amount": ("Nettobetrag", "Netto", "Summe netto", "Zwischensumme", "Summe Positionen", "Warenwert"),
    "tax_total_amount": ("MwSt", "MwSt.", "USt", "USt.", "Umsatzsteuer", "Mehrwertsteuer", "VAT", "Steuer"),
    "due_date": ("zahlbar bis", "fällig am", "Fälligkeit", "Fälligkeitsdatum", "buchen wir am", "Abbuchung am", "Zahlungsziel",
                 "due date"),
    "payment_due_date": ("wird am", "belastet am", "fällig am", "zahlbar bis", "Abbuchung am", "Lastschrift am"),
    "deadline": ("bis spätestens", "spätestens", "Frist"),
    "iban": ("IBAN",),
    "opening_balance": ("alter Kontostand", "Kontostand alt", "Saldo alt", "alter Saldo", "Anfangssaldo", "Saldovortrag",
                        "Saldo letzte Abrechnung", "vorheriger Saldo", "Vortrag", "previous balance"),
    "closing_balance": ("neuer Kontostand", "Kontostand neu", "Saldo neu", "neuer Saldo", "Endsaldo", "new balance", "Kontosaldo",
                        "offener Gesamtbetrag", "Saldo"),
    "period_start": ("Zeitraum", "Abrechnungszeitraum"),
    "period_end": ("Zeitraum", "Abrechnungszeitraum"),
    "statement_date": DOC_DATE_ROWS["statement"].labels,
    "statement_number": DOC_NUMBER_ROWS["statement"].labels,
    "invoice_number": DOC_NUMBER_ROWS["invoice"].labels,
    "invoice_date": DOC_DATE_ROWS["invoice"].labels,
}


# §6: 'bis zum' / 'bis' label a letter's deadline only on a line (or the line after it) that asks the recipient to act;
# an offer's or campaign's 'gilt bei Buchung bis zum <Datum>' is no deadline.
DEADLINE_CONTEXT_LABELS: Tuple[str, ...] = ("bis zum", "bis")
DEADLINE_DUTY_RE = re.compile(r"(zahl|überweis|ueberweis|einreich|zurücksend|zuruecksend|zurückgeb|zurueckgeb|vorleg|mitteil|"
                              r"melden|widersprech|widerspruch|einspruch|antwort|rückmeld|rueckmeld|stellungnahme|nachreich|"
                              r"ausgleich|beglich|begleich|nachweis|unterlagen|kündig|kuendig|einleg|bestätig|bestaetig|"
                              r"übersend|uebersend|zusend|vorzuleg|einzureich|zu begleichen|zu zahlen)", re.IGNORECASE)


# Labels of a party block ('Empfänger' with the name below or after it).
PARTY_LABELS: Dict[str, Tuple[str, ...]] = {
    "recipient": ("Rechnungsempfänger", "Leistungsempfänger", "Rechnungsadresse", "Empfänger", "Kontoinhaber", "Auftraggeber",
                  "Kunde", "Customer"),
    "sender": ("Absender", "Rechnungssteller", "Verkäufer", "Lieferant", "Aussteller", "Auftragnehmer", "Contractor",
               "Ein Angebot von", "Anbieter"),
}
# Labels after which a printed date is the previous statement's date (§8: period_start = the day after it).
PREVIOUS_STATEMENT_LABELS: Tuple[str, ...] = ("alter Kontostand vom", "Saldo letzte Abrechnung vom", "letzter Kontoauszug vom",
                                              "Saldovortrag vom", "alter Saldo vom")


@dataclass(frozen=True)
class TypeContract:
    """Contract rows that apply to one catalog type (None = the contract has no row for it)."""

    date_group: Optional[str]
    number_group: Optional[str]
    party_case: Optional[str]


TYPE_CONTRACT: Dict[str, TypeContract] = {
    "INVOICE": TypeContract("invoice", "invoice", "2.2"),
    "CREDIT_NOTE": TypeContract("invoice", "invoice", "2.2"),
    "UTILITY_BILL": TypeContract("invoice", "invoice", "2.2"),
    "DUNNING": TypeContract("invoice", None, "2.2"),
    "RECEIPT": TypeContract("receipt", None, "2.2"),
    "OFFICIAL_LETTER": TypeContract("letter", "letter", "2.1"),
    "LEGAL_CORRESPONDENCE": TypeContract("letter", "letter", "2.1"),
    "LETTER": TypeContract("letter", "letter", "2.1"),
    "BANK_STATEMENT": TypeContract("statement", "statement", "2.3"),
    "CREDIT_CARD_STATEMENT": TypeContract("statement", "statement", "2.3"),
    "APPLICATION": TypeContract("form", "form", "2.4"),
    "CONTRACT": TypeContract("contract", "contract", "2.1"),
    "INSURANCE_POLICY": TypeContract("contract", "contract", "2.1"),
    "CERTIFICATE": TypeContract("contract", "certificate", "2.1"),
    "TECHNICAL_DOC": TypeContract("contract", "certificate", "2.1"),
    "OTHER": TypeContract("enclosure", "enclosure", "2.6"),
    "NOTE": TypeContract(None, None, "2.7"),
    "TRANSFER_CONFIRMATION": TypeContract(None, None, "2.8"),
}
# Groups whose doc_number is removed when it is not printed next to a listed label (others are only flagged).
STRICT_NUMBER_GROUPS: Tuple[str, ...] = ("statement", "form", "enclosure")


def contract_for(type_name: str) -> TypeContract:
    """Contract rows of a catalog type."""
    return TYPE_CONTRACT.get(str(type_name).upper(), TypeContract(None, None, None))


def date_labels(type_name: str) -> Tuple[str, ...]:
    """§3 labels of a type in priority order ('Rechnungsdatum' on credit-card statements only)."""
    group = contract_for(type_name).date_group
    labels = DOC_DATE_ROWS[group].labels if group else ()
    if group == "statement" and str(type_name).upper() != "CREDIT_CARD_STATEMENT":
        labels = tuple(label for label in labels if label not in CARD_ONLY_DATE_LABELS)
    return labels


def number_labels(type_name: str, authority: bool = False) -> Tuple[str, ...]:
    """
    §4 labels of a type in priority order (enclosures: plus own-number
    labels; letters from an authority / court / ombudsman: plus its case
    number labels right after 'Unser Zeichen').
    """
    group = contract_for(type_name).number_group
    if group == "enclosure":
        return DOC_NUMBER_ROWS[group].labels + OWN_NUMBER_LABELS
    labels = DOC_NUMBER_ROWS[group].labels if group else ()
    if group == "letter" and authority:
        labels = labels[:1] + tuple(a for a in AUTHORITY_NUMBER_LABELS if a not in labels) + labels[1:]
    return labels


def is_authority(sender_text: str, header_lines: Sequence[str]) -> bool:
    """True if the sender is a court, an arbitration / ombudsman body or a public authority (sender name or header)."""
    return bool(AUTHORITY_RE.search(sender_text or "")) or any(AUTHORITY_RE.search(line) for line in header_lines)


def header_area(lines: Sequence[str]) -> List[str]:
    """The header area: the first non-empty lines of the document's first page."""
    return [line for line in lines if line.strip()][:HEADER_AREA_LINES]


def own_number_hits(type_name: str, lines: Sequence[str], first_page_lines: Optional[Sequence[str]] = None,
                    sender_text: str = "") -> List["LabelHit"]:
    """
    §4 own-number candidates of a type in priority order: listed labels
    anywhere, header-only labels ('No.', 'Nr.') in the header area, and the
    case-number labels of authority letters.
    """
    header = header_area(first_page_lines if first_page_lines is not None else lines)
    letterhead = [line for line in header[:4] if not any(ch.isdigit() for ch in line)]
    labels = number_labels(type_name, is_authority(sender_text, letterhead))
    hits = [h for h in find_labelled("identifier", labels, lines) if h.label not in HEADER_ONLY_NUMBER_LABELS]
    if any(label in HEADER_ONLY_NUMBER_LABELS for label in labels):
        hits += [h for h in find_labelled("identifier", labels, header) if h.label in HEADER_ONLY_NUMBER_LABELS]
    unique: List[LabelHit] = []
    for hit in sorted(hits, key=lambda h: (h.priority, h.line)):
        if not any(u.value == hit.value for u in unique):  # equivalent labels ('Mitglieds-Nr.' / 'Mitgliedsnummer')
            unique.append(hit)
    return unique


_NUMBER_SUFFIX_RE = re.compile(r"(?:s?\s*-?\s*(?:nr\.?|nummer)|\s+no\.?)$", re.IGNORECASE)
_ANY_NUMBER_SUFFIX = r"s?\s*-?\s*(?:nr\b\.?|nummer|no\b\.?|number)"


def label_regex(label: str, compound_head: bool = False) -> str:
    """
    Tolerant regex of a printed label: case-insensitive, optional dots,
    flexible spaces/hyphens, interchangeable number suffixes ('-Nr.',
    'nummer', 'No'), never inside a longer word ('Datum' never matches inside
    'Rechnungsdatum'). compound_head: a one-word label also matches as the
    head of a German compound ('Auszug' in 'Kontoauszug', 'Police' in
    'Versicherungspolice') - used for number labels, never for dates
    ('Geburtsdatum' is no 'Datum').
    """
    if label == "<Ort>, den <Datum>":
        return r"(?<![\wäöüß])[A-ZÄÖÜ][\wäöüß-]+,\s*den(?![\wäöüß])"
    if label == "…verfahren <nr>":
        return r"[a-zäöüß]*verfahren(?![a-zäöüß])"
    stem, suffix = label, ""
    match = _NUMBER_SUFFIX_RE.search(label)
    if match and match.start() > 2:
        stem, suffix = label[:match.start()], _ANY_NUMBER_SUFFIX
    parts = []
    if stem.lower().startswith("unser "):
        parts.append(r"unsere?")  # 'Unser Zeichen' / 'Unsere Zeichen'
        stem = stem[5:]
    for char in stem.lower():
        if char == ".":
            parts.append(r"\.?")
        elif char == "-":
            parts.append(r"\s*-?\s*")
        elif char == " ":
            parts.append(r"\s*")
        elif char in "=,":
            parts.append(r"\s*" + re.escape(char) + r"?\s*")
        else:
            parts.append(re.escape(char))
    compound = compound_head and " " not in label and "-" not in label and len(label) > 5
    head = r"(?<![\d])" if compound else r"(?<![\wäöüß])"
    return head + "".join(parts) + suffix + r"(?![a-zäöüß])"


_COMPILED: Dict[Tuple[str, bool], "re.Pattern[str]"] = {}


def compiled(label: str, compound_head: bool = False) -> "re.Pattern[str]":
    """Compiled label regex (cached)."""
    pattern = _COMPILED.get((label, compound_head))
    if pattern is None:
        pattern = _COMPILED[(label, compound_head)] = re.compile(label_regex(label, compound_head),
                                                                 0 if label.startswith("<") else re.IGNORECASE)
    return pattern


DATE_TOKEN_RE = re.compile(r"\d{4}-\d{2}-\d{2}|[0-3]?\d\.\s?[01]?\d\.\s?(?:\d{4}|\d{2})(?!\d)|[0-3]?\d/[01]?\d/(?:\d{4}|\d{2})(?!\d)"
                           r"|[0-3]?\d\.\s*[A-Za-zäöüÄÖÜ]{3,}\.?\s*\d{4}|[0-3]?\d\s+[A-Za-zäöüÄÖÜ]{3,}\.?\s+(?:\d{4}|\d{2})(?!\d)")
_SEPARATOR_RE = re.compile(r"^\s*[:.#]?\s*(?:vom\s+|am\s+)?")


@dataclass
class LabelHit:
    """A value printed next to a listed label."""

    label: str
    priority: int
    value: str
    line: int
    start: int = 0
    prefix: str = ""


def _date_after(fragment: str) -> Optional[str]:
    """ISO date printed directly after a label (only separators in between)."""
    rest = _SEPARATOR_RE.sub("", fragment, count=1)
    match = DATE_TOKEN_RE.match(rest)
    parsed = parse_date(match.group(0)) if match else None
    return parsed.isoformat() if parsed else None


def _identifier_after(fragment: str) -> Optional[str]:
    """Identifier printed directly after a label: tokens up to the first word without digits or a date."""
    rest = re.sub(r"^\s*(?:nr|no|nummer)\b\.?\s*[:.#]?", "", _SEPARATOR_RE.sub("", fragment, count=1), flags=re.IGNORECASE)
    picked: List[str] = []
    for token in rest.split()[:6]:
        token = token.strip(",;:")
        if not token or not re.fullmatch(r"[\wÄÖÜäöüß./-]+", token) or DATE_TOKEN_RE.fullmatch(token):
            break
        if not any(ch.isdigit() for ch in token) and not (len(token) <= 3 and token.isupper()) and (picked or len(token) > 4):
            break
        picked.append(token.rstrip("."))
    while picked and not any(ch.isdigit() for ch in picked[-1]):
        picked.pop()
    value = " ".join(picked)
    return value if any(ch.isdigit() for ch in value) else None


def find_labelled(kind: str, labels: Sequence[str], lines: Sequence[str], above: bool = False) -> List[LabelHit]:
    """
    Values next to listed labels: on the same line directly after the label,
    or - if the label stands alone on its line - on the line directly below
    (above=True: also the line directly above, for signature lines 'Ort, Datum').

    Args:
        kind: 'date' or 'identifier'.
        labels: Labels in priority order.
        lines: Text lines in reading order.
        above: Also accept the value on the line above a label that stands alone.

    Returns:
        Hits sorted by label priority, then reading order.
    """
    extract = _date_after if kind == "date" else _identifier_after
    neighbour = _date_line if kind == "date" else _identifier_after
    hits: List[LabelHit] = []
    for priority, label in enumerate(labels):
        pattern = compiled(label, compound_head=kind != "date")
        for index, line in enumerate(lines):
            for match in pattern.finditer(line):
                if kind == "date" and any(s <= match.start() < e for s, e in never_spans(line)):
                    continue
                value = extract(line[match.end():])
                rest = line[match.end():]
                # alone: nothing after the label, or only further column labels of the same header line
                # ('Unsere Zeichen, Nachricht' with the values on the line below)
                alone = not line[:match.start()].strip() and (
                    not rest.strip(" :.\t") or _HEADER_REST_RE.fullmatch(rest) is not None)
                if value is None and alone and index + 1 < len(lines):
                    value = neighbour(lines[index + 1].strip())
                if value is None and alone and above and index > 0:
                    value = neighbour(lines[index - 1].strip())
                if value:
                    hits.append(LabelHit(label, priority, value, index, match.start(), line[:match.start()]))
            if line.count("|") >= 2:
                for value in _table_values(kind, pattern, lines, index):
                    hits.append(LabelHit(label, priority, value, index, 0, ""))
    return sorted(hits, key=lambda h: (h.priority, h.line))


_HEADER_REST_RE = re.compile(r"\s*[,/&+]\s*(?:[A-Za-zÄÖÜäöüß.-]+\s*){1,3}[:.]?\s*")
_SEPARATOR_ROW_RE = re.compile(r"^\s*\|?\s*:?-{2,}")


def _cells(line: str) -> List[str]:
    """Cells of a pipe-table row ('| a | b |')."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _table_values(kind: str, pattern: "re.Pattern[str]", lines: Sequence[str], index: int) -> List[str]:
    """
    Values of a label inside a pipe-table row (Docling / OvisOCR3 tables):
    the label cell's own rest, the next cell of the row ('| Datum | 17/11/2031 |')
    or the cell of the same column in the next row (header row -> value row).
    """
    extract = _date_after if kind == "date" else _identifier_after
    cells = _cells(lines[index])
    values: List[str] = []
    for column, cell in enumerate(cells):
        match = pattern.match(cell)
        if match is None or (kind == "date" and any(s <= 0 < e for s, e in never_spans(cell))):
            continue
        rest = cell[match.end():]
        value = extract(rest)
        if value is None and not rest.strip(" :.") and column + 1 < len(cells):
            value = extract(cells[column + 1])
        if value is None and not rest.strip(" :."):
            below = index + 1
            while below < len(lines) and _SEPARATOR_ROW_RE.match(lines[below]):
                below += 1
            if below < len(lines) and lines[below].count("|") >= 2:
                row = _cells(lines[below])
                following = _cells(lines[below + 1]) if below + 1 < len(lines) and lines[below + 1].count("|") >= 2 else []
                # a label in the header of a multi-row table names a column (bookings, instalments), not a document field
                if len(row) == len(cells) and len(following) != len(cells):
                    value = extract(row[column])
        if value:
            values.append(value)
    return values


_DATE_LINE_RE = re.compile(r"\s*(?:(?:\d{5}\s+)?[A-ZÄÖÜ][\wäöüß. /-]{1,30}?,\s*(?:den\s+)?|\d{5}\s+[A-ZÄÖÜ][\wäöüß. /-]{1,30}?\s+)?("
                           + DATE_TOKEN_RE.pattern
                           + r")(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\s*")
# Signature line of a form: '<Ort> <Datum> <Name>' ('Musterstadt 28.11.2031 Erika Muster').
_SIGNATURE_LINE_RE = re.compile(r"\s*(?:(?:\d{5}\s+)?[A-ZÄÖÜ][\wäöüß.-]{1,30},?\s+(?:den\s+)?)?(" + DATE_TOKEN_RE.pattern
                                + r")\s+([A-ZÄÖÜ][a-zäöüß]+(?:[- ][A-ZÄÖÜ][a-zäöüß]+){1,3})\s*")


SIGNATURE_CUE_RE = re.compile(r"(unterschrift|unterzeichn|signatur|signature|signed)", re.IGNORECASE)
_PLACE_DATE_RE = re.compile(r"\s*(?:\d{5}\s+)?[A-ZÄÖÜ][\wäöüß.-]{1,30}(?:\s+[A-ZÄÖÜ][\wäöüß.-]{1,30})?,?\s+(?:den\s+)?("
                            + DATE_TOKEN_RE.pattern + r")\s*")


_QUARTER_RE = re.compile(r"(?<![\d.])([1-4])\.?\s*(?:quartal|quarter)\s*(\d{4})|\bq([1-4])\s*[/ ]?\s*(\d{4})", re.IGNORECASE)
_HALF_RE = re.compile(r"(?<![\d.])([12])\.?\s*halbjahr\s*(\d{4})", re.IGNORECASE)


def printed_period_pairs(text: str) -> List[Tuple[date, date]]:
    """
    §8 printed periods of a text: 'vom 01.08.2031 bis 30.08.2031',
    '01.06.2031 - 30.06.2031', and named periods normalised to their first
    and last day ('3. Quartal 2031', 'Q3/2031', '1. Halbjahr 2031').
    """
    pattern = re.compile(rf"({DATE_TOKEN_RE.pattern})\s*(?:-|–|bis(?:\s+zum)?)\s*({DATE_TOKEN_RE.pattern})", re.IGNORECASE)
    pairs: List[Tuple[date, date]] = []
    for match in pattern.finditer(text):
        start, end = parse_date(match.group(1)), parse_date(match.group(2))
        if start and end and start <= end:
            pairs.append((start, end))
    for match in _QUARTER_RE.finditer(text):
        quarter, year = int(match.group(1) or match.group(3)), int(match.group(2) or match.group(4))
        first = date(year, 3 * quarter - 2, 1)
        last = date(year + 1, 1, 1) - timedelta(days=1) if quarter == 4 else date(year, 3 * quarter + 1, 1) - timedelta(days=1)
        pairs.append((first, last))
    for match in _HALF_RE.finditer(text):
        half, year = int(match.group(1)), int(match.group(2))
        pairs.append((date(year, 1, 1), date(year, 6, 30)) if half == 1 else (date(year, 7, 1), date(year, 12, 31)))
    return list(dict.fromkeys(pairs))


def previous_statement_days(lines: Sequence[str]) -> List[date]:
    """§8: the day after each printed previous statement date ('alter Kontostand vom 27.04.31' -> 2031-04-28)."""
    out: List[date] = []
    for hit in find_labelled("date", PREVIOUS_STATEMENT_LABELS, lines):
        parsed = parse_date(hit.value)
        if parsed and parsed + timedelta(days=1) not in out:
            out.append(parsed + timedelta(days=1))
    return out


def derived_period_value(path: str, value: object, lines: Sequence[str]) -> bool:
    """
    §8 period values that are normalised from printed text rather than
    copied verbatim: the first / last day of a printed named period
    ('3. Quartal 2031') and the day after a printed previous statement date.
    """
    target = parse_date(value)
    if target is None or not path.endswith(("period_start", "period_end")):
        return False
    named = [p for line in lines for p in printed_period_pairs(line)]
    if any((p[0] if path.endswith("period_start") else p[1]) == target for p in named):
        return True
    return path.endswith("period_start") and target in previous_statement_days(lines)


def form_signature_dates(lines: Sequence[str], window: int = 2) -> List[str]:
    """
    §3 forms: dates next to the signature only - after 'Ort, Datum', on a
    signature line '<Ort> <Datum> <Name>', or a 'Datum' / place-date line
    within a few lines of a signature cue ('Unterschrift'). A 'Datum' that
    answers a form question ('Wann ereignete sich der Schaden? Datum: ...')
    is no signature date.
    """
    cues = [i for i, line in enumerate(lines) if SIGNATURE_CUE_RE.search(line)]

    def near(index: int) -> bool:
        return any(abs(index - c) <= window for c in cues)

    found: List[str] = []
    for hit in find_labelled("date", ("Ort, Datum",), lines, above=True):
        found.append(hit.value)
    for index, line in enumerate(lines):
        match = compiled("Ort, Datum").search(line)
        if match and not line[match.end():].strip(" :") and not line[:match.start()].strip():
            # a signature field: the place and date are written on the line above (or below) its 'Ort, Datum' caption
            for neighbour in (lines[index - 1] if index else "", lines[index + 1] if index + 1 < len(lines) else ""):
                written = _PLACE_DATE_RE.fullmatch(neighbour)
                if written and parse_date(written.group(1)):
                    found.append(parse_date(written.group(1)).isoformat())  # type: ignore[union-attr]
        rest = re.sub(r"^[\s:]+", "", line[match.end():]) if match else ""
        place_date = _PLACE_DATE_RE.fullmatch(rest) if rest else None
        if place_date and parse_date(place_date.group(1)):
            found.append(parse_date(place_date.group(1)).isoformat())  # type: ignore[union-attr]
    for hit in find_labelled("date", ("Datum",), lines, above=True):
        if near(hit.line):
            found.append(hit.value)
    for index, line in enumerate(lines):
        signed = signature_line(line.strip())
        if signed:
            found.append(signed[0])
            continue
        match = _PLACE_DATE_RE.fullmatch(line) or _DATE_LINE_RE.fullmatch(line)
        parsed = parse_date(match.group(1)) if match else None
        if parsed and near(index):
            found.append(parsed.isoformat())
    return list(dict.fromkeys(found))


def signature_line(line: str) -> Optional[Tuple[str, str]]:
    """(ISO date, name) of a form signature line '<Ort> <Datum> <Name>', else None."""
    match = _SIGNATURE_LINE_RE.fullmatch(line)
    parsed = parse_date(match.group(1)) if match else None
    return (parsed.isoformat(), match.group(2)) if parsed and match else None


def _date_line(line: str) -> Optional[str]:
    """ISO date of a line that consists of a date (optionally after a place), else None."""
    match = _DATE_LINE_RE.fullmatch(line)
    parsed = parse_date(match.group(1)) if match else None
    return parsed.isoformat() if parsed else None


def date_line_dates(lines: Sequence[str]) -> List[str]:
    """Dates standing alone on a line, optionally after a place ('Berlin, 3. Mai 2021', '03.05.2021')."""
    found: List[str] = []
    for line in lines:
        match = _DATE_LINE_RE.fullmatch(line)
        parsed = parse_date(match.group(1)) if match else None
        if parsed and parsed.isoformat() not in found:
            found.append(parsed.isoformat())
    return found


def same_value(kind: str, left: object, right: object) -> bool:
    """Value equality per kind (dates by calendar day, identifiers compact; a printed value may extend the field)."""
    if kind == "date":
        return parse_date(left) is not None and parse_date(left) == parse_date(right)
    a, b = compact(str(left)), compact(str(right))
    return bool(a) and bool(b) and (a == b or (len(b) >= 3 and b in a))


def adjacent_label(kind: str, value: object, labels: Sequence[str], lines: Sequence[str],
                   above: bool = False) -> Optional[LabelHit]:
    """The highest-priority listed label next to which this value is printed (None if none)."""
    return next((h for h in find_labelled(kind, labels, lines, above) if same_value(kind, h.value, value)), None)


def never_regex(label: str) -> "re.Pattern[str]":
    """
    Tolerant regex of a §3 'never' label, abbreviations included
    ('Auftr.-Dat.', 'Leistungsdat.', 'Liefer Datum', 'Datum fertig'); the
    invoice label 'Liefer=Re.Dat' is not a delivery date.
    """
    pattern = _NEVER_COMPILED.get(label)
    if pattern is None:
        lower = label.lower()
        if lower.endswith("datum") and " " not in lower:
            body = rf"{re.escape(lower[:5])}[\wäöüß]*\.?\s*-?\s*dat(?:um|\.)?(?!\s*=)"
        else:
            head, _, tail = lower.partition(" ")
            body = rf"{re.escape(head[:3])}[\wäöüß]*\.?\s*{re.escape(tail)}"
        pattern = _NEVER_COMPILED[label] = re.compile(r"(?<![\wäöüß])" + body + r"(?![a-zäöüß=])", re.IGNORECASE)
    return pattern


_NEVER_COMPILED: Dict[str, "re.Pattern[str]"] = {}


def never_spans(line: str) -> List[Tuple[int, int]]:
    """Character spans of §3 'never' labels on a line."""
    return [m.span() for label in DOC_DATE_NEVER + DOC_DATE_NEVER_SYNONYMS for m in never_regex(label).finditer(line)]


def never_date_label(value: object, lines: Sequence[str], snippet: Optional[str] = None) -> Optional[str]:
    """The §3 'never' label (order/service/delivery date) the date is printed next to (in the text or its snippet)."""
    target = parse_date(value)
    if target is None:
        return None
    for label in DOC_DATE_NEVER + DOC_DATE_NEVER_SYNONYMS:
        for line in list(lines) + ([snippet] if snippet else []):
            for match in never_regex(label).finditer(line):
                if _date_after(line[match.end():]) == target.isoformat():
                    return label
    return None


def labelled_dates(type_name: str, lines: Sequence[str]) -> List[date]:
    """Dates next to the type's §3 labels in priority order (date lines appended for letters/receipts)."""
    contract = contract_for(type_name)
    hits = find_labelled("date", date_labels(type_name), lines, above=contract.date_group == "form")
    values = [h.value for h in hits] + (date_line_dates(lines) if contract.date_group in DATE_LINE_GROUPS else [])
    found: List[date] = []
    for value in values:
        parsed = parse_date(value)
        if parsed and parsed not in found:
            found.append(parsed)
    return found


_PERSON_NAME_RE = re.compile(r"^[A-ZÄÖÜ][a-zäöüß]+(?:[- ][A-ZÄÖÜ][a-zäöüß.]+){1,3}$")


# Form fields that name the SUBJECT of a form (the insured, deceased, patient or member), not the person filling it in.
FORM_SUBJECT_LABELS: Tuple[str, ...] = (
    "Versicherte", "Versicherter", "Verstorbene", "Verstorbener", "Mitglied", "Patient", "Patientin", "versicherte Person",
    "Erblasser", "Erblasserin", "Sterbefall", "Kind")
_SUBJECT_RE = re.compile(r"(?<![\wäöüß])(?:" + "|".join(re.escape(w).replace(r"\ ", r"\s+") for w in FORM_SUBJECT_LABELS)
                         + r")(?:\(r\)|\(in\)|in|s|es|n)?(?![a-zäöüß])", re.IGNORECASE)


def subject_context(lines: Sequence[str], index: int) -> bool:
    """True if a form line belongs to a subject block (the line or the two lines above name the insured/deceased/patient)."""
    return any(_SUBJECT_RE.search(lines[i]) for i in range(max(0, index - 2), index + 1))


def _name_of(candidate: str) -> Optional[str]:
    """A person name at the start of a form value."""
    name = re.split(r"\s{2,}|\t|\|", candidate.strip())[0].strip(" ,;")
    return name if _PERSON_NAME_RE.match(name) and not not_a_party(name) else None


def form_filler(lines: Sequence[str]) -> Optional[Tuple[str, str]]:
    """
    §2.4: the person who fills in and signs a form: the name on a signature
    line ('<Ort> <Datum> <Name>'), else the name entered in a name field
    ('Vorname und Name des Ansprechpartners / Domaininhabers: Erika Muster',
    a label standing alone with the name below, or a table header cell with
    the name in the cell below). Name fields of the form's SUBJECT (insured,
    deceased, patient, member) are skipped. Returns (label, name) or None.
    """
    for index, line in enumerate(lines):
        signed = signature_line(line.strip())
        if signed and not subject_context(lines, index):
            return "signature line", signed[1]
    for label in FORM_NAME_LABELS:
        pattern = re.compile(label_regex(label), re.IGNORECASE)
        for index, line in enumerate(lines):
            if subject_context(lines, index):
                continue
            if line.count("|") >= 2:
                cells = _cells(line)
                below = lines[index + 1] if index + 1 < len(lines) else ""
                row = _cells(below) if below.count("|") >= 2 else []
                for column, cell in enumerate(cells):
                    if pattern.match(cell) and len(row) == len(cells):
                        name = _name_of(row[column])
                        if name:
                            return label, name
                continue
            match = pattern.search(line)
            if match is None or line[:match.start()].strip(" -•*"):
                continue
            rest = line[match.end():]
            colon = re.match(r"^[^:\n]{0,70}?:\s*(.*)$", rest)
            candidates = [colon.group(1)] if colon else []
            if not candidates and len(rest.strip()) <= 70 and index + 1 < len(lines):
                candidates.append(lines[index + 1])
            for candidate in candidates:
                name = _name_of(candidate)
                if name:
                    return label, name
    return None


DOC_TITLE_WORDS = frozenset({
    "rechnung", "invoice", "report", "testing", "test", "bericht", "prüfbericht", "pruefbericht", "zeugnis", "kontoauszug",
    "protokoll", "antrag", "bescheinigung", "zertifikat", "certificate", "angebot", "mahnung", "gutschrift", "vertrag",
    "police", "erklärung", "erklaerung", "formular", "lieferschein", "quittung", "kündigung", "kuendigung", "safety", "check",
    "checks", "list", "liste", "bescheid", "mitteilung", "abrechnung", "auftragsbestätigung", "bestellung", "nachweis",
    "datenblatt", "anlage", "information", "informationen", "übersicht", "uebersicht", "statement", "summary"})
_ADDRESS_ONLY_RE = re.compile(
    r"^\s*(?:[\wÄÖÜäöüß.-]+\s+){0,3}?[\wÄÖÜäöüß.-]*(?:straße|strasse|str\.|weg|allee|platz|gasse|ring|damm|ufer|chaussee|pfad)"
    r"\s*\d+\s*[a-zA-Z]?\s*$|^\s*(?:postfach|pf)\s*\d|^\s*(?:D-)?\d{5}\s+[A-ZÄÖÜ][\wäöüß -]*$", re.IGNORECASE)
_DEPARTMENT_RE = re.compile(r"^\s*(?:niederlassung|abteilung|abt\.|referat|filiale|geschäftsstelle|geschaeftsstelle|fachbereich|"
                            r"dezernat|bereich|kundenservice|kundenbetreuung|service\s*-?\s*center|team)\b", re.IGNORECASE)


def is_document_title(value: object) -> bool:
    """True for a document heading taken as a party ('PRÜFBERICHT', 'Rechnung', 'Kontoauszug')."""
    tokens = [t for t in re.findall(r"[a-zäöüß]+", str(value or "").lower()) if len(t) > 1]
    return bool(tokens) and all(t in DOC_TITLE_WORDS for t in tokens)


def is_address_only(value: object) -> bool:
    """True for a street, P.O. box or ZIP/city line taken as a party name ('Lindenallee 13', '00345 Musterstadt')."""
    return bool(value) and _ADDRESS_ONLY_RE.match(str(value)) is not None


def is_department(value: object) -> bool:
    """True for a department / branch line ('Niederlassung Kundenservice Nord', 'Abteilung Leistung')."""
    return bool(value) and _DEPARTMENT_RE.match(str(value)) is not None


# Field labels and form vocabulary that never occur in a party name ('Ihre Kundennummer', 'Seite 2 von 3').
PARTY_FORM_VOCABULARY = frozenset({
    "kundennummer", "kundennr", "nummer", "nr", "datum", "seite", "betreff", "zeichen", "rechnung", "rechnungsnummer",
    "report", "telefon", "telefax", "fax", "e-mail", "email", "iban", "bic", "kontonummer", "vertragsnummer", "aktenzeichen",
    "auftragsnummer", "mitgliedsnummer", "versicherungsnummer", "bearbeiter", "durchwahl", "unterschrift", "seite",
    "anlage", "anlagen", "betrag", "summe", "uhrzeit", "container", "bedingungen", "versicherungsbedingungen",
    "geschäftsbedingungen", "agb", "avb", "ahb"})
_TITLES = frozenset({"herr", "herrn", "frau", "dr", "prof", "familie", "eheleute"})
_CO_LINE_RE = re.compile(r"^\s*(?:c/o|c\.\s*o\.|z\.\s*hd\.?|zu\s+händen|attn\.?)(?:\s|$)", re.IGNORECASE)
_LEGAL_FORM_RE = re.compile(r"\b(gmbh|ag|kg|ohg|se|eg|e\.\s?v\.|ug|mbh|inc|ltd|llc|kgaa|gbr)\b", re.IGNORECASE)


def party_problem(value: object, person: bool = False) -> Optional[str]:
    """
    Party-name plausibility gate (applied to every party value):
    returns why a value cannot be a sender/recipient name, else None.

    Args:
        value: Company or person name.
        person: True for a person-name field (a single bare token is no full name there).
    """
    text = " ".join(str(value or "").split())
    if not text:
        return None
    if not_a_party(text):
        return "info-column label / postal service mark"
    if is_document_title(text):
        return "document heading / title"
    if is_address_only(text):
        return "street / P.O. box / city only"
    if _CO_LINE_RE.match(text):
        return "c/o line"
    if DATE_TOKEN_RE.search(text) or re.search(r"\b\d{4}-\d{2}-\d{2}\b", text):
        return "contains a date"
    words = re.findall(r"[a-zäöüß-]+", text.lower())
    if any(w.strip("-") in PARTY_FORM_VOCABULARY for w in words):
        return "field label / form vocabulary"
    alnum = [ch for ch in text if ch.isalnum()]
    digits = sum(ch.isdigit() for ch in alnum)
    tokens = text.split()
    mixed = sum(1 for t in tokens if re.search(r"\d", t) and re.search(r"[A-Za-z]", t))
    if alnum and (digits / len(alnum) > 0.3 or mixed >= 2):
        return "ID / code string"
    names = [t for t in tokens if re.search(r"[A-Za-zÄÖÜäöüß]", t) and t.lower().strip(".") not in _TITLES]
    if person and len(names) == 1 and not _LEGAL_FORM_RE.search(text):
        return "single name token (bare surname or first name)"
    return None


def is_bank_transfer_confirmation(text: str) -> bool:
    """§2.8: a bank's confirmation of a single transaction (a strong term, or 'Auftragsbestätigung' in transfer context)."""
    head = text[:3000]
    if TRANSFER_STRONG_RE.search(head):
        return True
    return bool(re.search(r"auftragsbestätigung|auftragsbestaetigung", head, re.IGNORECASE)) and \
        len(set(m.lower() for m in TRANSFER_CONTEXT_RE.findall(head))) >= 2


def labelled_values(labels: Sequence[str], lines: Sequence[str]) -> List[str]:
    """Text printed after a label ('Auftraggeber: Erika Muster') on its line, or on the next line if the label stands alone."""
    values: List[str] = []
    for label in labels:
        pattern = compiled(label)
        for index, line in enumerate(lines):
            for cell in (_cells(line) if line.count("|") >= 2 else [line]):
                match = pattern.search(cell)
                if match is None or cell[:match.start()].strip(" -•*"):
                    continue
                rest = re.sub(r"^[\s:./-]+", "", cell[match.end():])
                if not rest and index + 1 < len(lines):
                    rest = lines[index + 1].strip().strip("|").strip()
                if rest:
                    values.append(rest)
    return values


def is_name_fragment(value: object) -> bool:
    """
    A single word without legal form that is no acronym ('Velmora'): from the
    layout alone it may be a truncated or OCR-fragmented name, never an
    accepted party (acronyms such as 'BKV' and names with a legal form pass).
    """
    tokens = [t for t in str(value or "").split() if t.lower().strip(".") not in _TITLES]
    if len(tokens) != 1 or _LEGAL_FORM_RE.search(tokens[0]):
        return False
    word = tokens[0].strip(".,;:")
    return not re.fullmatch(r"[A-ZÄÖÜ0-9&-]{2,}", word)


def geometry_policy(type_name: str, cover_sheet: bool = False) -> Tuple[bool, bool]:
    """
    Role mapping by type: (address window -> recipient, letterhead / return address -> sender).
    Letters, invoices, receipts, statements, contracts: both. Certificates / reports: letterhead only. Forms filled in by
    a person (§2.4): the window may give the receiving organisation, never the sender (the filler); a
    pre-addressed cover sheet (§2.5) follows the letter rule.
    Enclosures and private notes (§2.6 / §2.7, OTHER, NOTE): neither.
    """
    name = str(type_name).upper()
    rows = contract_for(name)
    if name in ("OTHER", "NOTE") or rows.party_case in ("2.6", "2.7"):
        return False, False
    if rows.party_case == "2.4":
        # the window of a return form shows the receiving organisation (recipient); the filler is never set by geometry
        return (True, True) if cover_sheet else (True, False)
    if name in ("CERTIFICATE", "TECHNICAL_DOC"):
        return False, True
    return True, True


def stamp_exempt(type_name: str) -> bool:
    """True for forms filled in by a person: their doc_date may be handwritten (date next to the signature)."""
    return contract_for(type_name).date_group == "form"


def _join(labels: Sequence[str]) -> str:
    """Quoted label list."""
    return ", ".join(f"'{label}'" for label in labels)


def type_prompt_rules(type_name: str) -> str:
    """Contract rules of one type for the extraction prompt (its §2, §3 and §4 rows)."""
    contract = contract_for(type_name)
    parts: List[str] = []
    if contract.date_group:
        row = DOC_DATE_ROWS[contract.date_group]
        extra = " - handwritten allowed, the only exception to the stamp/handwriting rule" if contract.date_group == "form" else ""
        parts.append(f"meta_header.doc_date = {row.meaning}{extra}; labels in priority order: {_join(row.labels)}"
                     + ("; else the date line" if contract.date_group in DATE_LINE_GROUPS else "")
                     + ("; else the printed period end" if contract.date_group == "statement" else ""))
    if contract.number_group:
        row = DOC_NUMBER_ROWS[contract.number_group]
        labels = number_labels(type_name)
        parts.append(f"meta_header.doc_number = {row.meaning}" + (f"; labels in priority order: {_join(labels)}" if labels else "")
                     + ("; copy only the number ('Auszug 3' -> '3')" if contract.number_group == "statement" else "")
                     + (f"; if the sender is a court, an arbitration or ombudsman body or a public authority, its case / "
                        f"proceedings number counts ({_join(AUTHORITY_NUMBER_LABELS)})" if contract.number_group == "letter" else "")
                     + (f"; {_join(HEADER_ONLY_NUMBER_LABELS)} only in the document's header area"
                        if contract.number_group == "certificate" else ""))
    if contract.party_case:
        party = PARTY_ROWS[contract.party_case]
        parts.append(f"sender = {party.sender}; recipient = {party.recipient}")
    return f"Field contract v{CONTRACT_VERSION}: {'. '.join(parts)}." if parts else ""


def general_prompt_rules() -> str:
    """Contract rules shared by all types: label priority and the §3/§4 'never' lists."""
    return (
        f"**Dates:** a value next to a listed label beats any unlabelled value; earlier labels beat later ones. "
        f"Never a doc_date: {_join(DOC_DATE_NEVER)}, a service period, a transaction date, a form-version date and NEVER a "
        f"date inside a stamp ('ÜBERWIESEN', 'bezahlt', 'Eingang', 'erhalten') or a handwritten date (those belong to the "
        f"audit) - the only exception is the date next to the signature of a form filled in by a person. For direct-debit "
        f"invoices the due_date is the printed debit date ('buchen wir am ...', 'Abbuchung am ...'). A deadline exists only "
        f"if the recipient must act by a printed date.\n"
        f"11. **Document numbers:** a value next to a listed number label beats any other code. Never a doc_number: headings "
        f"(e.g. '{DOC_NUMBER_NEVER[0]}'), page numbers ('{DOC_NUMBER_NEVER[1]}'), form codes in margins or footers, ZIP codes, "
        f"account/card/customer numbers on statements, third parties' case numbers when the sender's own reference is "
        f"printed, transaction IDs. Copy the number without its label words.")


def party_prompt_rules() -> str:
    """The §2 party table as prompt bullets."""
    rows = "\n".join(f"  - {row.row}: sender = {row.sender}; recipient = {row.recipient}." for row in PARTY_ROWS.values())
    return (f"- Who is sender and who is recipient (field contract v{CONTRACT_VERSION}):\n{rows}\n"
            f"- Salutations ({_join(PARTY_NEVER[:1])}), greetings, form headings ({_join(PARTY_NEVER[1:])}), table cells and log "
            f"entries are never parties. Person names are given without salutation or title ({_join(TITLE_WORDS)} are dropped). "
            f"Info-column labels ({_join(INFO_COLUMN_LABELS[:2])}) and postal service marks ({_join(SERVICE_ADDRESS_WORDS[:2])}) "
            f"are never parties either. On forms, the filler's name is the value of a name field "
            f"({_join(FORM_NAME_LABELS[:2])}, 'Versicherungsnehmer', 'Kontoinhaber'), typed or handwritten. "
            f"sender and recipient are never the same party.")
