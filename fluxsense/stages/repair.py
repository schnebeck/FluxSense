# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/stages/repair.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Deterministic repairs after extraction: fills transaction rows
                without amount (running-balance difference, else the matching
                text-layer line by date + description, else a zoomed crop of
                the row) and normalizes every date field to ISO 8601.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from fluxsense.catalog.contract import (DATE_LINE_GROUPS, DATE_TOKEN_RE, FIELD_LABELS, FIRST_PAGE_DATE_GROUPS,
                                       ORDERING_CUSTOMER_LABELS, PAYEE_LABELS, STATEMENT_FOREIGN_NUMBER_LABELS, contract_for,
                                       date_labels, date_line_dates, find_labelled, form_signature_dates, labelled_dates,
                                       labelled_values,
                                       never_date_label, number_labels, own_number_hits, previous_statement_days,
                                       printed_period_pairs, same_value, stamp_exempt,
                                       valid_tax_id)
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.grounding.normalize import compact, extract_dates, normalize_text, parse_date, to_decimal
from fluxsense.log import get_logger
from fluxsense.models.result import CheckResult, CheckStatus, GroundingStatus, Severity

logger = get_logger("stages.repair")

_SIGNED_AMOUNT_RE = re.compile(r"(?<![\w.,])([-−+]?\s?\d{1,3}(?:[.\u00a0 ]\d{3})*,\d{2}|[-−+]?\s?\d+,\d{2}|[-−+]?\s?\d{1,3}(?:,\d{3})*\.\d{2})(?![\d.,])(\s?-|\s[SH]\b)?")
DATE_FIELD_NAMES: Tuple[str, ...] = ("date", "period_start", "period_end", "valid_until", "deadline")


@dataclass
class RowRepair:
    """Outcome of the row amount repair."""

    filled: List[int] = field(default_factory=list)
    methods: Dict[int, str] = field(default_factory=dict)
    unresolved: List[int] = field(default_factory=list)


def _signed(token: str, suffix: Optional[str]) -> Optional[Decimal]:
    """Signed Decimal of an amount token with optional trailing '-' / S / H marker."""
    value = to_decimal(token.replace(" ", ""))
    if value is None:
        return None
    marker = (suffix or "").strip().upper()
    if marker in ("-", "S") and value > 0:
        return -value
    if marker == "H":
        return abs(value)
    return value


def line_amounts(line: str) -> List[Decimal]:
    """Signed amounts of a text line in order."""
    return [v for v in (_signed(m.group(1), m.group(2)) for m in _SIGNED_AMOUNT_RE.finditer(line)) if v is not None]


def _row_lines(layer: TextLayer, row: Dict[str, Any]) -> List[str]:
    """Text lines of the row's page that contain its date and share description words."""
    page = layer.page(row.get("source_page")) if isinstance(row.get("source_page"), int) else None
    pages = [page] if page else layer.pages
    booking = parse_date(row.get("booking_date"))
    words = [w for w in normalize_text(str(row.get("description") or "")).split() if len(w) >= 3][:4]
    hits: List[str] = []
    for candidate in pages:
        for line in candidate.all_text().splitlines():
            if booking and not (booking.strftime("%d.%m.") in line or booking.isoformat() in line):
                continue
            norm = normalize_text(line)
            if words and sum(1 for w in words if w in norm) < max(1, len(words) // 2):
                continue
            hits.append(line)
    return hits


def fill_row_amounts(rows: List[Dict[str, Any]], opening: Optional[Any], layer: TextLayer) -> RowRepair:
    """
    Fills missing transaction amounts deterministically.

    Order: (1) balance - previous balance (or opening balance), (2) the
    single non-balance amount on the matching text line.

    Args:
        rows: Transactions (modified in place).
        opening: Opening balance.
        layer: Text layer.

    Returns:
        RowRepair with filled and unresolved row indices.
    """
    repair = RowRepair()
    previous = to_decimal(opening)
    for i, row in enumerate(rows):
        balance = to_decimal(row.get("balance"))
        amount = to_decimal(row.get("amount"))
        if amount is None:
            if balance is not None and previous is not None:
                row["amount"] = str(balance - previous)
                repair.filled.append(i)
                repair.methods[i] = "balance_difference"
            else:
                lines = _row_lines(layer, row)
                candidates = []
                if len(lines) == 1:
                    candidates = [a for a in line_amounts(lines[0]) if balance is None or abs(a) != abs(balance)]
                if len(candidates) == 1:
                    row["amount"] = str(candidates[0])
                    repair.filled.append(i)
                    repair.methods[i] = "text_line"
                else:
                    repair.unresolved.append(i)
        amount = to_decimal(row.get("amount"))
        if balance is not None:
            previous = balance
        elif previous is not None and amount is not None:
            previous = previous + amount
    return repair


def repair_checks(table: str, repair: RowRepair, rows: List[Dict[str, Any]]) -> List[CheckResult]:
    """Reports filled and unresolved rows as findings."""
    checks: List[CheckResult] = []
    if repair.filled:
        checks.append(CheckResult(check_id="table.amount_filled", rule="repair", status=CheckStatus.FAIL, severity=Severity.INFO,
                                  message=f"{len(repair.filled)} missing amounts filled deterministically "
                                          f"({', '.join(sorted(set(repair.methods.values())))})",
                                  rows=repair.filled, table=table,
                                  pages=sorted({rows[i]["source_page"] for i in repair.filled if isinstance(rows[i].get("source_page"), int)})))
    return checks


def normalize_dates(node: Any, path: str = "") -> List[str]:
    """
    Normalizes every date-like field (name contains 'date' or is a period/deadline
    field) to ISO 8601 in place.

    Args:
        node: Fields (dict/list).
        path: Current path.

    Returns:
        Paths that were rewritten.
    """
    changed: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            sub = f"{path}.{key}" if path else key
            if isinstance(value, str) and any(n in key for n in DATE_FIELD_NAMES):
                parsed = parse_date(value)
                if parsed is not None and parsed.isoformat() != value:
                    node[key] = parsed.isoformat()
                    changed.append(sub)
            else:
                changed += normalize_dates(value, sub)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            changed += normalize_dates(value, f"{path}.{i}")
    return changed


def zoom_targets_for_rows(rows: List[Dict[str, Any]], indices: List[int], limit: int) -> List[Tuple[int, Dict[str, Any]]]:
    """Rows to read on a zoomed crop (capped)."""
    return [(i, rows[i]) for i in indices[:limit] if isinstance(rows[i].get("source_page"), int)]



REFERENCE_LABEL_RE = re.compile(r"(zeichen|aktenzeichen|az\.?|nr\.?|nummer|number|no\.|rechnung|vorgang|schaden|kunden|"
                                r"vertrag|referenz|reference|ticket|fall|police|versicherungsschein|bescheid|steuer)", re.IGNORECASE)


def is_form_code(value: Any, snippet: Optional[str], page: Optional[PageText]) -> bool:
    """
    True if a doc_number candidate looks like a print/form code: no reference
    label in its evidence snippet and printed in the footer (bottom 8 %) or a
    side margin (outer 6 %) of the page.

    Args:
        value: Extracted doc_number.
        snippet: Evidence snippet.
        page: Page with word boxes.

    Returns:
        True if the value should not be used as document number.
    """
    if not value or page is None or not page.words:
        return False
    if snippet and REFERENCE_LABEL_RE.search(snippet.replace(str(value), "")):
        return False
    from fluxsense.vision.zoom import find_words
    rect = find_words(page.words, str(value))
    if rect is None:
        return False
    x0, y0, x1, y1 = rect
    in_footer = y0 > page.height * 0.92
    in_margin = x1 < page.width * 0.06 or x0 > page.width * 0.94
    return in_footer or in_margin


_YEARLESS_RE = re.compile(r"^\s*(\d{1,2})\.(\d{1,2})\.?\s*$")


def complete_statement_years(body: Dict[str, Any]) -> int:
    """
    Completes year-less booking/value dates ('01.09.') from the statement
    period (period_end, else statement_date, else period_start); a month after
    the period end month belongs to the previous year (Dec/Jan statements).

    Args:
        body: statement_body (modified in place).

    Returns:
        Number of completed dates.
    """
    anchor = parse_date(body.get("period_end")) or parse_date(body.get("statement_date")) or parse_date(body.get("period_start"))
    if anchor is None:
        return 0
    count = 0
    for row in body.get("transactions") or []:
        for key in ("booking_date", "value_date"):
            match = _YEARLESS_RE.match(str(row.get(key) or ""))
            if not match:
                continue
            day, month = int(match.group(1)), int(match.group(2))
            year = anchor.year - 1 if month > anchor.month else anchor.year
            try:
                row[key] = date(year, month, day).isoformat()
                count += 1
            except ValueError:
                logger.debug(f"cannot complete date {row.get(key)!r}")
    return count


PERIOD_CUE_RE = re.compile(r"(bis|zeitraum|vom|period|from|to\b|[–-]\s*\d)", re.IGNORECASE)
PREVIOUS_CUE_RE = re.compile(r"(letzte|vorherige|vorige|previous|alter|seit)", re.IGNORECASE)
NEGATIVE_SNIPPET_RE = re.compile(r"(\d\s*-\s*$|\d\s*-(?=\s)|\d\s+S\b|\bSoll\b|^\s*[-−])", re.IGNORECASE)


def printed_period_dates(text: str) -> List[Tuple[date, date]]:
    """Printed periods of a text (dates, quarters, half years; see catalog.contract.printed_period_pairs)."""
    return printed_period_pairs(text)


def drop_form_deadline(fields: Dict[str, Any], type_name: str) -> List[str]:
    """A form's signature date (its doc_date) is never a due date or deadline."""
    if contract_for(type_name).date_group != "form":
        return []
    doc_date = parse_date((fields.get("meta_header") or {}).get("doc_date"))
    notes: List[str] = []
    bodies = fields.get("bodies") or {}
    for body_key, key in (("correspondence_body", "deadline"), ("finance_body", "due_date")):
        body = bodies.get(body_key)
        if doc_date and isinstance(body, dict) and parse_date(body.get(key)) == doc_date:
            notes.append(f"{body_key}.{key} {body.pop(key)} removed (the form's signature date is no due date)")
    return notes


def fix_statement_semantics(body: Dict[str, Any], evidence: List[Any], text: str = "") -> List[str]:
    """
    Statement rules: (1) balances take the sign printed in their evidence
    snippet ('512,30 -', 'S', 'Soll'); (2) without a printed period cue,
    period_end is the statement date and period_start is removed unless it
    refers to a previous statement.

    Args:
        body: statement_body (modified in place).
        evidence: Evidence entries of the document.

    Returns:
        Notes about changes.
    """
    notes: List[str] = []
    by_path = {e.path: e for e in evidence}
    for key in ("opening_balance", "closing_balance"):
        ev = by_path.get(f"bodies.statement_body.{key}")
        value = to_decimal(body.get(key))
        if ev is None or value is None or not ev.evidence:
            continue
        if NEGATIVE_SNIPPET_RE.search(ev.evidence) and value > 0:
            body[key] = str(-value)
            ev.original_value, ev.value = ev.value, body[key]
            notes.append(f"{key} sign taken from the printed '{ev.evidence}'")
    statement_date = parse_date(body.get("statement_date"))
    end_ev = by_path.get("bodies.statement_body.period_end")
    periods = printed_period_dates(text)
    if statement_date and body.get("period_end") and end_ev is not None and end_ev.evidence and not periods \
            and not PERIOD_CUE_RE.search(end_ev.evidence) and parse_date(body["period_end"]) != statement_date:
        notes.append(f"period_end {body['period_end']} -> statement date {statement_date} (no printed period)")
        body["period_end"] = statement_date.isoformat()
    start_ev = by_path.get("bodies.statement_body.period_start")
    if body.get("period_start") and start_ev is not None and start_ev.evidence \
            and parse_date(body["period_start"]) not in {start for start, _ in periods} \
            and not PERIOD_CUE_RE.search(start_ev.evidence) and not PREVIOUS_CUE_RE.search(start_ev.evidence):
        notes.append(f"period_start {body['period_start']} removed (no printed period or previous statement date)")
        body.pop("period_start")
    notes += fill_statement_period(body, text)
    return notes


def fill_statement_period(body: Dict[str, Any], text: str) -> List[str]:
    """
    §8 fills of an empty period: a unique printed period (ending on the
    statement date when that is known; of several such periods the one
    enclosing the others); else period_end = the statement
    date and period_start = the day after the printed previous statement
    date.
    """
    notes: List[str] = []
    statement_date = parse_date(body.get("statement_date"))
    periods = printed_period_pairs(text)
    if statement_date:
        periods = [p for p in periods if p[1] == statement_date] or ([] if len(periods) > 1 else periods)
        widest = min(periods, key=lambda p: p[0]) if periods else None
        if widest and all(widest[0] <= p[0] for p in periods):
            periods = [widest]  # sub-periods (interest, partial periods) end inside the enclosing statement period
    if len(set(periods)) == 1:
        start, end = periods[0]
        for key, value in (("period_start", start), ("period_end", end)):
            if not body.get(key):
                body[key] = value.isoformat()
                notes.append(f"{key} {value} from the printed period (§8)")
        return notes
    if periods:
        return notes
    if statement_date and not body.get("period_end"):
        body["period_end"] = statement_date.isoformat()
        notes.append(f"period_end = statement date {statement_date} (no printed period, §8)")
    previous = previous_statement_days(text.splitlines())
    if not body.get("period_start") and len(previous) == 1:
        body["period_start"] = previous[0].isoformat()
        notes.append(f"period_start {previous[0]} = the day after the printed previous statement date (§8)")
    return notes


def _inside(word: Any, bbox: Tuple[float, float, float, float]) -> bool:
    """True if a word's center lies in a box."""
    cx, cy = (word.x0 + word.x1) / 2, (word.y0 + word.y1) / 2
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def sender_only_in_table(sender: Optional[Dict[str, Any]], layer: TextLayer, pages: List[int]) -> bool:
    """
    True if the sender name/company is printed on the document's pages only
    inside table cells (e.g. a merchant or transaction-ID column), i.e. not in a
    letterhead, sender block or signature.

    Args:
        sender: meta_header.sender.
        layer: Text layer (word boxes, table regions).
        pages: Pages of the logical document.

    Returns:
        True if every occurrence is inside a table.
    """
    from fluxsense.vision.zoom import find_words
    name = (sender or {}).get("company") or (sender or {}).get("name")
    if not name:
        return False
    occurrences = 0
    for number in pages:
        page = layer.page(number)
        if page is None or not page.words:
            continue
        free = [w for w in page.words if not w.cell and not any(t.bbox and _inside(w, t.bbox) for t in page.tables)]
        if find_words(free, str(name)) is not None:
            return False
        if find_words(page.words, str(name)) is not None:
            occurrences += 1
    return occurrences > 0


STAMP_LINE_RE = re.compile(r"(überwiesen|ueberwiesen|bezahlt|eingang|eingegangen|erhalten|gebucht|paid|received)", re.IGNORECASE)
VALIDITY_DATE_RE = re.compile(r"(?:gültig|gueltig|laufzeit|vertragsende|valid)\w*\s+(?:bis|until|zum)\s+(?:zum\s+)?"
                              r"([0-3]?\d\.[01]?\d\.(?:\d{4}|\d{2})|\d{1,2}\.\s*[A-Za-zäöüÄÖÜ]+\.?\s*\d{4})", re.IGNORECASE)
DEBIT_DATE_RE = re.compile(r"(?:buchen wir|abbuchung|abgebucht|einzug|ziehen wir)[^\n.]{0,80}?(?:am|zum|ab dem)\s+"
                           r"([0-3]?\d\.[01]?\d\.(?:\d{4}|\d{2})|\d{1,2}\.\s*[A-Za-zäöüÄÖÜ]+\.?\s*\d{4})", re.IGNORECASE)


def labelled_invoice_dates(text: str) -> List[date]:
    """Dates printed next to an invoice-date label of the field contract (priority order, 'never' labels excluded)."""
    return labelled_dates("INVOICE", text.splitlines())


def stamp_dates(text: str, audit: Optional[Dict[str, Any]]) -> List[date]:
    """Dates inside stamps: visual-audit stamp fields/content and text lines with stamp words."""
    found: List[date] = []
    for stamp in (audit or {}).get("layer_stamps") or []:
        values = [stamp.get("raw_content") or ""] + [str(f.get("normalized_value") or f.get("raw_value") or "")
                                                      for f in stamp.get("form_fields") or []]
        for value in values:
            parsed = parse_date(value)
            if parsed:
                found.append(parsed)
            dates, _ = extract_dates(value)
            found += list(dates)
    for line in text.splitlines():
        if STAMP_LINE_RE.search(line):
            dates, _ = extract_dates(line)
            found += list(dates)
    return sorted(set(found))


def fix_document_dates(fields: Dict[str, Any], evidence: List[Any], text: str, audit: Optional[Dict[str, Any]],
                       type_name: str = "INVOICE", first_page_text: Optional[str] = None) -> List[str]:
    """
    doc_date post-checks of the field contract (§3): a value next to a listed
    label of the type beats any unlabelled value (earlier labels first); a
    date next to a 'never' label (order/service/delivery date) or inside a
    stamp is replaced by the labelled date, else removed. Forms are exempt
    from the stamp rule (their date may be handwritten next to the
    signature) and an unlabelled form date is kept. For direct-debit
    invoices the due date is the printed debit date (§6).

    Args:
        fields: Clean fields (modified in place).
        evidence: Evidence entries.
        text: Text of the document's pages.
        audit: Visual audit result (stamps).
        type_name: Catalog type whose contract row applies.

    Returns:
        Notes about the changes.
    """
    notes: List[str] = []
    finance = (fields.get("bodies") or {}).get("finance_body")
    meta = fields.get("meta_header") or {}
    rows = contract_for(type_name)
    lines = text.splitlines()
    by_path = {e.path: e for e in evidence}
    if rows.date_group:
        exempt = stamp_exempt(type_name)
        stamps = set() if exempt else set(stamp_dates(text, audit))
        explicit: List[date] = []
        first_page = (first_page_text if first_page_text is not None else text).splitlines()[:60]
        label_lines = first_page if rows.date_group in FIRST_PAGE_DATE_GROUPS else lines
        hits = form_signature_dates(lines) if rows.date_group == "form" else \
            [h.value for h in find_labelled("date", date_labels(type_name), label_lines, above=exempt)]
        for value in hits:
            parsed = parse_date(value)
            if parsed and parsed not in explicit:
                explicit.append(parsed)
        # a date standing on its own line of the first page is the document's date line (all groups but enclosures)
        date_lines = [parse_date(d) for d in date_line_dates(lines if rows.date_group in DATE_LINE_GROUPS else first_page)] \
            if rows.date_group != "enclosure" else []
        targets = [("meta_header.doc_date", meta)]
        if rows.date_group == "invoice" and isinstance(finance, dict):
            targets.append(("bodies.finance_body.invoice_date", finance))
        for path, holder in targets:
            key = path.split(".")[-1]
            current = parse_date(holder.get(key))
            if current is None or current in explicit:
                continue
            ev = by_path.get(path)
            never = never_date_label(current, lines, ev.evidence if ev else None)
            stamped = current in stamps
            candidates = [d for d in explicit if d not in stamps]
            unlabelled = not exempt and current not in date_lines and bool(candidates)
            if not (stamped or never or unlabelled):
                continue
            if not candidates:
                candidates = [d for d in date_lines if d and d not in stamps and d != current]
            reason = "stamp date" if stamped else f"next to '{never}'" if never else "unlabelled, a labelled date is printed"
            if candidates:
                holder[key] = candidates[0].isoformat()
                notes.append(f"{path}: {current} ({reason}) -> labelled date {candidates[0]}")
                if ev is not None:
                    ev.original_value, ev.value, ev.action = ev.value, holder[key], "replaced"
            else:
                holder.pop(key)
                notes.append(f"{path}: {current} removed ({reason}; never a doc_date)")
                if ev is not None:
                    ev.original_value, ev.action = ev.value, "nulled"
    if isinstance(finance, dict):
        match = DEBIT_DATE_RE.search(text)
        debit = parse_date(match.group(1)) if match else None
        if debit and parse_date(finance.get("due_date")) != debit:
            notes.append(f"due_date {finance.get('due_date')} -> printed debit date {debit}")
            finance["due_date"] = debit.isoformat()
    if isinstance(finance, dict) and finance.get("due_date"):
        due = parse_date(finance["due_date"])
        validity = [m for m in VALIDITY_DATE_RE.finditer(text) if parse_date(m.group(1)) == due]
        labelled_due = any(parse_date(h.value) == due for h in find_labelled("date", FIELD_LABELS["due_date"], lines))
        if due and validity and not labelled_due:
            notes.append(f"due_date {finance.pop('due_date')} removed: a validity / term end ('{validity[0].group(0)[:40]}'), "
                         f"not a §6 due date")
    stamps_all = set() if stamp_exempt(type_name) else set(stamp_dates(text, audit))
    statement = (fields.get("bodies") or {}).get("statement_body")
    for path, holder, key in (("bodies.finance_body.due_date", finance, "due_date"),
                              ("bodies.statement_body.payment_due_date", statement, "payment_due_date")):
        if not isinstance(holder, dict) or parse_date(holder.get(key)) not in stamps_all:
            continue
        current = holder.pop(key)
        labelled = [h for h in find_labelled("date", FIELD_LABELS["due_date"], lines) if parse_date(h.value) not in stamps_all]
        if labelled:
            holder[key] = labelled[0].value
        notes.append(f"{path}: {current} is a stamp date -> {holder.get(key) or 'removed'}")
        ev = by_path.get(path)
        if ev is not None:
            ev.original_value, ev.value, ev.action = ev.value, holder.get(key), "replaced" if holder.get(key) else "nulled"
    return notes


OWN_REFERENCE_RE = re.compile(
    r"(?<![A-Za-zäöü])(?<!ihr )(?<!ihre )(?<!Ihr )(?<!Ihre )"
    r"(rech(?:nungs)?\s*[-.]?\s*n(?:umme)?r\.?|rechnungsnummer|invoice\s+no\.?|unser(?:e)?\s+zeichen|unser\s+az\.?|"
    r"vertrags\s*-?\s*n(?:umme)?r\.?|vertragsnummer|versicherungsschein\s*-?\s*n(?:umme)?r\.?|police\s*-?\s*n(?:umme)?r\.?)"
    r"\s*[:.]?\s*([A-Z0-9][A-Z0-9./-]*(?:\s[A-Z0-9][A-Z0-9./-]*){0,5})", re.IGNORECASE)
FINANCE_LABEL_RE = re.compile(r"rech|invoice", re.IGNORECASE)


def own_references(text: str) -> List[Tuple[str, str]]:
    """(label, value) pairs of the sender's own references printed in the text (values need >= 3 digits)."""
    found: List[Tuple[str, str]] = []
    for line in text.splitlines():
        for match in OWN_REFERENCE_RE.finditer(line):
            value = match.group(2).strip(" .:-")
            tokens = value.split()
            while tokens and not any(ch.isdigit() for ch in tokens[-1]) and len(tokens) > 1:
                tokens.pop()
            value = " ".join(tokens)
            if sum(ch.isdigit() for ch in value) >= 3:
                found.append((match.group(1), value))
    return found


THIRD_PARTY_PREFIX_RE = re.compile(r"(?:\bihr(?:e|em|er)?|\byour)\s*$", re.IGNORECASE)


def party_names(fields: Dict[str, Any], role: str = "sender") -> str:
    """Company and name of a party as one text."""
    party = (fields.get("meta_header") or {}).get(role) or {}
    return " ".join(str(party.get(k)) for k in ("company", "name") if isinstance(party, dict) and party.get(k))


def contract_references(type_name: str, text: str, sender_text: str = "") -> List[Any]:
    """
    The type's own-number candidates (§4): values next to its listed labels
    in priority order (header-only and authority labels included);
    'Ihr Zeichen'-style third-party labels excluded, values need >= 3 digits
    (statements: >= 1).
    """
    min_digits = 1 if contract_for(type_name).number_group == "statement" else 3
    return [h for h in own_number_hits(type_name, text.splitlines(), sender_text=sender_text)
            if not THIRD_PARTY_PREFIX_RE.search(h.prefix) and sum(ch.isdigit() for ch in h.value) >= min_digits]


def fix_document_number(fields: Dict[str, Any], evidence: List[Any], text: str, finance: bool,
                        type_name: Optional[str] = None) -> List[str]:
    """
    Prefers the sender's own labelled reference (§4) over heading/form codes
    and third-party case numbers for meta_header.doc_number (and the
    invoice_number of invoices): earlier labels beat later ones; a value
    next to an earlier or equal label is kept. Types without a contract row
    use the generic own-reference labels.

    Args:
        fields: Clean fields (modified in place).
        evidence: Evidence entries.
        text: Text of the document's pages.
        finance: True for invoice-like documents (Rechnungsnummer preferred).
        type_name: Catalog type whose contract row applies (None = generic labels).

    Returns:
        Notes about the changes.
    """
    rows = contract_for(type_name or "")
    by_path = {e.path: e for e in evidence}
    notes: List[str] = []
    meta = fields.setdefault("meta_header", {})
    fin = (fields.get("bodies") or {}).get("finance_body")
    if rows.number_group == "statement":
        return []  # the statement number field is the doc_number (catalog doc_number_from)
    if type_name and rows.number_group:
        hits = contract_references(type_name, text, party_names(fields))
        if not hits:
            return []
        best = hits[0]
        targets = [("meta_header.doc_number", meta)]
        if rows.number_group == "invoice" and isinstance(fin, dict):
            targets.append(("bodies.finance_body.invoice_number", fin))
        for path, holder in targets:
            key = path.split(".")[-1]
            current = holder.get(key)
            if current and compact(str(current)) == compact(best.value):
                continue
            own = next((h for h in hits if same_value("identifier", h.value, current)), None) if current else None
            if own is not None and own.priority <= best.priority:
                continue
            holder[key] = best.value
            notes.append(f"{path}: {current!r} -> {best.value!r} (printed next to '{best.label}')")
            ev = by_path.get(path)
            if ev is not None:
                ev.original_value, ev.value, ev.action = ev.value, best.value, "replaced"
        return notes
    refs = own_references(text)
    if not refs:
        return []
    if finance:
        refs = sorted(refs, key=lambda r: 0 if FINANCE_LABEL_RE.search(r[0]) else 1)
    label, value = refs[0]
    targets = [("meta_header.doc_number", meta)]
    if finance and isinstance(fin, dict) and FINANCE_LABEL_RE.search(label):
        targets.append(("bodies.finance_body.invoice_number", fin))
    for path, holder in targets:
        key = path.split(".")[-1]
        current = holder.get(key)
        if current and compact(str(current)) == compact(value):
            continue
        ev = by_path.get(path)
        if current and ev is not None and ev.evidence and OWN_REFERENCE_RE.search(ev.evidence):
            continue  # the model already took a labelled own reference
        holder[key] = value
        notes.append(f"{path}: {current!r} -> {value!r} (printed next to '{label}')")
        if ev is not None:
            ev.original_value, ev.value, ev.action = ev.value, value, "replaced"
    return notes


PAGE_NUMBER_VALUE_RE = re.compile(r"^\s*(?:seite|page|blatt)?\s*\d{1,3}\s*(?:/|von|of)\s*\d{1,3}\s*$", re.IGNORECASE)
LABEL_PREFIX_RE = re.compile(r"^\s*(?:kontoauszug(?:\s*nr\.?)?|auszug(?:\s*nr\.?)?|abrechnung(?:\s*nr\.?)?|"
                             r"rechnungs?-?\s*(?:nummer|nr\.?)|rech-?\s*nr\.?|invoice\s+(?:no\.?|number)|unser(?:e)?\s+zeichen|"
                             r"aktenzeichen|az\.?|vertrags-?\s*(?:nummer|nr\.?)|kunden-?\s*(?:nummer|nr\.?)|nr\.?|no\.?)\s*[:.#]?\s*",
                             re.IGNORECASE)


MONTHS = (r"jan(?:uar)?|feb(?:ruar)?|m[äa]r(?:z)?|maerz|apr(?:il)?|mai|may|jun[ie]?|jul[iy]?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
          r"okt(?:ober)?|oct(?:ober)?|nov(?:ember)?|dez(?:ember)?|dec(?:ember)?|march|june|july")
PERIOD_VALUE_RE = re.compile(
    rf"^\s*(?:[1-4]\.?\s*(?:quartal|quarter|q)\b|q[1-4]\b|(?:[0-3]?\d\.?\s*)?(?:{MONTHS})\.?\b|(?:kw|woche|week)\s*\d{{1,2}})"
    rf"[\s./-]*(?:\d{{2}}|\d{{4}})?\s*$", re.IGNORECASE)


def is_date_or_period(value: str) -> bool:
    """True if a 'number' is a date or a period ('23. Juli 1998', '3. Quartal 2024', '27 MAI 09', '01.02.2024', 'Q3/2024')."""
    text = value.strip()
    if re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}|[0-3]?\d\.[01]?\d\.(?:\d{4}|\d{2}))", text):
        return parse_date(text) is not None
    return PERIOD_VALUE_RE.match(text) is not None


def clean_document_numbers(fields: Dict[str, Any]) -> List[str]:
    """
    doc_number / invoice_number / statement_number hygiene: strips a printed
    label from the value, removes page numbering ('1/2') and values without
    any digit (words such as a title are not numbers).

    Returns:
        Notes about changes.
    """
    notes: List[str] = []
    holders = [("meta_header", fields.get("meta_header"), "doc_number"),
               ("finance_body", (fields.get("bodies") or {}).get("finance_body"), "invoice_number"),
               ("statement_body", (fields.get("bodies") or {}).get("statement_body"), "statement_number")]
    for name, holder, key in holders:
        if not isinstance(holder, dict) or not holder.get(key):
            continue
        value = str(holder[key])
        stripped = LABEL_PREFIX_RE.sub("", value).strip()
        if PAGE_NUMBER_VALUE_RE.match(stripped) or not any(ch.isdigit() for ch in stripped):
            holder.pop(key)
            notes.append(f"{name}.{key} {value!r} removed (page numbering or no digits)")
        elif is_date_or_period(stripped):
            holder.pop(key)
            notes.append(f"{name}.{key} {value!r} removed (a date or period, not a number)")
        elif stripped != value:
            holder[key] = stripped
            notes.append(f"{name}.{key} label stripped: {value!r} -> {stripped!r}")
    return notes


def drop_bank_statement_due_date(fields: Dict[str, Any], type_tags: List[str]) -> List[str]:
    """A bank account statement has no payment due date (only credit card statements do)."""
    body = (fields.get("bodies") or {}).get("statement_body")
    if isinstance(body, dict) and body.get("payment_due_date") and "CREDIT_CARD_STATEMENT" not in type_tags:
        return [f"payment_due_date {body.pop('payment_due_date')} removed (bank account statement)"]
    return []


LOG_ENTRY_RE = re.compile(r"^\s*(?:\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})[ ,T]+\d{1,2}:\d{2}")
HEADING_LINE_RE = re.compile(r"^\s*[A-ZÄÖÜ](?:[A-ZÄÖÜ ]){3,}\s+(\S+)\s*$")


def _table_cells(fields: Dict[str, Any]) -> List[str]:
    """Text cells of the extracted tables (transactions, line items)."""
    bodies = fields.get("bodies") or {}
    rows = list(((bodies.get("statement_body") or {}).get("transactions") or [])) + \
        list(((bodies.get("finance_body") or {}).get("line_items") or []))
    cells: List[str] = []
    for row in rows:
        values = row.values() if isinstance(row, dict) else row if isinstance(row, list) else []
        cells += [str(v) for v in values if isinstance(v, str)]
    return cells


def contract_never_number(fields: Dict[str, Any], number: str, type_name: str, text: str) -> Optional[str]:
    """
    §4 'never' checks of a doc_number that is not printed next to one of the
    type's listed labels: a heading code ('RECHNUNG RK7'), a transaction ID
    or other table cell, a customer/account/card number of a statement.

    Returns:
        The reason, or None if none applies.
    """
    lines = text.splitlines()
    labels = number_labels(type_name)
    if labels and any(same_value("identifier", h.value, number)
                      for h in own_number_hits(type_name, lines, sender_text=party_names(fields))):
        return None
    key = compact(number)
    if len(key) < 2:
        return None
    if contract_for(type_name).number_group == "statement" and \
            any(same_value("identifier", h.value, number) for h in find_labelled("identifier", STATEMENT_FOREIGN_NUMBER_LABELS, lines)):
        return "is a customer/account/card number on a statement"
    if len(key) >= 4 and any(key in compact(cell) for cell in _table_cells(fields)):
        return "is a table cell / transaction ID"
    occurrences = [line for line in lines if key in compact(line)]
    if occurrences and all((m := HEADING_LINE_RE.match(line)) and compact(m.group(1)) == key for line in occurrences):
        return "is part of a heading"
    if len(key) >= 6 and occurrences and all(LOG_ENTRY_RE.match(line) for line in occurrences):
        return "is a log entry / transaction ID"
    return None


def never_number_reason(fields: Dict[str, Any], number: str, type_name: str, text: str) -> Optional[str]:
    """
    Why a document / statement number is never one (§4): a postcode, the
    statement's own account / card number or IBAN, or a 'never' case of
    contract_never_number (customer / account / card number next to its
    label, table cell, heading, log entry). None if no reason applies.
    """
    meta = fields.get("meta_header") or {}
    statement = (fields.get("bodies") or {}).get("statement_body") or {}
    key = compact(number)
    if number_labels(type_name) and any(same_value("identifier", h.value, number) for h in
                                        own_number_hits(type_name, text.splitlines(), sender_text=party_names(fields))):
        return None  # printed next to one of the type's own number labels
    foreign = [compact(str(f)) for f in [statement.get(k) for k in ("account_number", "card_number_masked", "iban")] +
               [(meta.get("recipient") or {}).get("iban")] if f]
    if re.fullmatch(r"\d{5}", key):
        return "looks like a postcode"
    if key and any(key == f or (len(key) >= 6 and key in f) for f in foreign):
        return "is an account/card number"
    return contract_never_number(fields, number, type_name, text)


def enforce_doc_semantics(fields: Dict[str, Any], evidence: List[Any], spec: Any, text: str,
                          has_letterhead: bool) -> List[str]:
    """
    Applies the catalog's doc_number / doc_date / due-date semantics of a type.

    Args:
        fields: Clean fields (modified in place).
        evidence: Evidence entries.
        spec: Primary TypeSpec.
        text: Text of the document's pages.
        has_letterhead: The first page has a letterhead / address window.

    Returns:
        Notes about removed or replaced values.
    """
    notes: List[str] = []
    meta = fields.setdefault("meta_header", {})
    by_path = {e.path: e for e in evidence}
    number = meta.get("doc_number")
    if spec.doc_number_from:
        source = fields
        for part in spec.doc_number_from.split("."):
            source = source.get(part) if isinstance(source, dict) else None
        if source:
            cleaned = LABEL_PREFIX_RE.sub("", str(source)).strip()
            if cleaned and cleaned != number:
                notes.append(f"doc_number {number!r} -> {cleaned!r} ({spec.doc_number_from})")
                meta["doc_number"] = number = cleaned
    if number:
        # A number that is merely not next to a listed label is kept and flagged by the acceptance (§0.5): removing it
        # lost correct values printed next to labels the contract does not list.
        reason = never_number_reason(fields, str(number), spec.name, text)
        if reason:
            meta.pop("doc_number")
            notes.append(f"doc_number {number!r} removed ({reason})")
    statement = (fields.get("bodies") or {}).get("statement_body")
    if isinstance(statement, dict) and statement.get("statement_number"):
        own = str(statement["statement_number"])
        reason = never_number_reason(fields, own, spec.name, text)
        if reason:
            statement.pop("statement_number")
            notes.append(f"statement_number {own!r} removed ({reason}, §4)")
    if spec.doc_date_labels and not has_letterhead and meta.get("doc_date"):
        ev = by_path.get("meta_header.doc_date")
        date_text = str(meta["doc_date"])
        parsed = parse_date(date_text)
        printed = [parsed.strftime("%d.%m.%Y"), parsed.strftime("%d.%m.%y")] if parsed else [date_text]
        pattern = "|".join(f"(?:{label})" for label in spec.doc_date_labels)
        labelled = bool(ev and ev.evidence and re.search(pattern, ev.evidence, re.IGNORECASE)) or any(
            re.search(rf"(?:{pattern})[^\n]{{0,25}}{re.escape(p)}", text, re.IGNORECASE) for p in printed)
        if not labelled:
            notes.append(f"doc_date {meta.pop('doc_date')!r} removed (enclosure without letterhead, date not labelled)")
    if spec.due_date_labels:
        pattern = "|".join(f"(?:{label})" for label in spec.due_date_labels)
        statement = (fields.get("bodies") or {}).get("statement_body")
        if isinstance(statement, dict) and statement.get("payment_due_date") and not re.search(pattern, text, re.IGNORECASE):
            notes.append(f"payment_due_date {statement.pop('payment_due_date')!r} removed (no printed debit/due date)")
    return notes


PAYOUT_CUE_RE = re.compile(r"(erstatt|gutschr|überweisen\s+wir|ueberweisen\s+wir|zahlen\s+wir|auszahl|auf\s+ihr(?:em)?\s+konto|"
                           r"ihre\s+bankverbindung|ihr\s+konto|refund|paid\s+to)", re.IGNORECASE)


def _remap_accounts(evidence: List[Any], mapping: Dict[int, Optional[int]]) -> None:
    """Moves payment_accounts evidence paths to the new indices (None = retired)."""
    prefix = "bodies.finance_body.payment_accounts."
    for ev in evidence:
        if not ev.path.startswith(prefix):
            continue
        head, _, rest = ev.path[len(prefix):].partition(".")
        if not head.isdigit() or int(head) not in mapping:
            continue
        target = mapping[int(head)]
        if target is None:
            ev.original_value, ev.action, ev.status = ev.value, "nulled", GroundingStatus.REMOVED
        else:
            ev.path = f"{prefix}{target}.{rest}" if rest else f"{prefix}{target}"


def order_payee_accounts(fields: Dict[str, Any], evidence: List[Any], text: str, credit_note: bool) -> List[str]:
    """
    Field contract §7: all printed payee accounts are kept, the first in
    reading order is the primary one. A credit note keeps only the account
    the credit is paid to (an IBAN printed near a payout cue such as
    'erstatten', 'gutschreiben', 'auf Ihr Konto'), else none.

    Returns:
        Notes about changes.
    """
    finance = (fields.get("bodies") or {}).get("finance_body")
    accounts = finance.get("payment_accounts") if isinstance(finance, dict) else None
    if not isinstance(accounts, list) or not accounts:
        return []
    lines = text.splitlines()
    flat = compact(text)

    def position(account: Any) -> int:
        iban = compact(str(account.get("iban") or "")) if isinstance(account, dict) else ""
        found = flat.find(iban) if iban else -1
        return found if found >= 0 else len(flat) + 1

    def payout(account: Any) -> bool:
        iban = compact(str(account.get("iban") or "")) if isinstance(account, dict) else ""
        for index, line in enumerate(lines):
            if iban and iban in compact(line):
                if re.search(r"\b(?:unser[e]?|our)\b", line, re.IGNORECASE):
                    continue  # the issuer's own bank details
                window = " ".join(lines[max(0, index - 1):index + 1])
                if PAYOUT_CUE_RE.search(window):
                    return True
        return False

    notes: List[str] = []
    indexed = list(enumerate(accounts))
    if credit_note:
        kept = [(i, a) for i, a in indexed if payout(a)]
        if len(kept) != len(indexed):
            notes.append(f"credit note: {len(indexed) - len(kept)} account(s) without payout cue removed (§7)")
        indexed = kept
    ordered = sorted(indexed, key=lambda item: position(item[1]))
    mapping: Dict[int, Optional[int]] = {i: None for i, _ in enumerate(accounts)}
    for new, (old, _account) in enumerate(ordered):
        mapping[old] = new
    if [i for i, _ in ordered] != list(range(len(accounts))):
        if not notes:
            notes.append("payee accounts ordered by reading order (§7: the first is primary)")
        _remap_accounts(evidence, mapping)
        finance["payment_accounts"] = [a for _, a in ordered]
        if not finance["payment_accounts"]:
            finance.pop("payment_accounts")
    return notes


def _party_text(party: Any) -> str:
    """Company and name of a party."""
    return " ".join(str(party.get(k)) for k in ("company", "name") if isinstance(party, dict) and party.get(k))


def fix_transfer_confirmation(fields: Dict[str, Any], text: str) -> List[str]:
    """
    Field contract §2.8 for bank confirmations of a single transaction: the
    transfer amount is no invoice total (finance totals and tax removed); the
    ordering customer ('Auftraggeber', 'Kontoinhaber') is never the sender
    but the recipient; the payee is a counterparty, never the recipient.

    Returns:
        Notes about changes.
    """
    from fluxsense.layout.din5008 import contained
    notes: List[str] = []
    finance = (fields.get("bodies") or {}).get("finance_body")
    if isinstance(finance, dict):
        for key in ("monetary_summation", "tax_breakdown"):
            if finance.get(key):
                finance.pop(key)
                notes.append(f"finance_body.{key} removed (a transfer amount is no invoice total, §2.8)")
    lines = text.splitlines()
    ordering = labelled_values(ORDERING_CUSTOMER_LABELS, lines)
    payee = labelled_values(PAYEE_LABELS, lines)
    meta = fields.setdefault("meta_header", {})
    sender, recipient = meta.get("sender") or {}, meta.get("recipient") or {}
    if _party_text(sender) and ordering and contained(_party_text(sender), ordering):
        if not _party_text(recipient) or (payee and contained(_party_text(recipient), payee)):
            meta["recipient"] = sender
            notes.append(f"ordering customer {_party_text(sender)!r} moved from sender to recipient (§2.8)")
        else:
            notes.append(f"sender {_party_text(sender)!r} removed: the ordering customer is never the sender (§2.8)")
        meta.pop("sender")
    recipient = meta.get("recipient") or {}
    if _party_text(recipient) and payee and contained(_party_text(recipient), payee) \
            and not (ordering and contained(_party_text(recipient), ordering)):
        meta.pop("recipient")
        notes.append(f"recipient {_party_text(recipient)!r} removed: the payee is a counterparty (§2.8)")
    return notes


def sanitize_bank_identifiers(fields: Dict[str, Any]) -> List[str]:
    """
    §7: bank identifiers are never a tax / VAT ID or an IBAN unless they pass
    that field's format check - a BLZ as tax_id and an account number failing
    mod-97 as IBAN are removed (also where no evidence record checked them).
    """
    from fluxsense.grounding.snap import iban_valid
    notes: List[str] = []
    meta = fields.get("meta_header") or {}
    for role in ("sender", "recipient"):
        party = meta.get(role)
        if not isinstance(party, dict):
            continue
        for key, vat in (("tax_id", False), ("vat_id", True)):
            if party.get(key) and not valid_tax_id(party[key], vat):
                notes.append(f"{role}.{key} {party.pop(key)!r} removed (no valid {'VAT ID' if vat else 'tax number'})")
        if party.get("iban") and not iban_valid(str(party["iban"])):
            notes.append(f"{role}.iban {party.pop('iban')!r} removed (fails mod-97)")
    bodies = fields.get("bodies") or {}
    holders = [bodies.get("statement_body")] + list(((bodies.get("finance_body") or {}).get("payment_accounts") or []))
    for holder in holders:
        if isinstance(holder, dict) and holder.get("iban") and not iban_valid(str(holder["iban"])):
            notes.append(f"iban {holder.pop('iban')!r} removed (fails mod-97)")
    return notes


def fill_tax_basis(fields: Dict[str, Any]) -> List[str]:
    """
    §5: total_net is BT-109 (tax basis). Without document-level allowances
    or charges the printed net sum is both BT-106 and BT-109, so a net the
    model put only into the line total is copied (not computed) into the
    tax basis - if it is consistent with net + tax = gross where those are
    printed.
    """
    finance = (fields.get("bodies") or {}).get("finance_body")
    ms = finance.get("monetary_summation") if isinstance(finance, dict) else None
    if not isinstance(ms, dict) or ms.get("tax_basis_total_amount") not in (None, "") or ms.get("line_total_amount") in (None, ""):
        return []
    if any(to_decimal(ms.get(k)) not in (None, Decimal(0)) for k in ("allowance_total_amount", "charge_total_amount")):
        return []
    net, tax, gross = to_decimal(ms.get("line_total_amount")), to_decimal(ms.get("tax_total_amount")), to_decimal(ms.get("grand_total_amount"))
    if net is None or (tax is not None and gross is not None and abs(net + tax - gross) > Decimal("0.011")):
        return []
    ms["tax_basis_total_amount"] = ms["line_total_amount"]
    return [f"tax_basis_total_amount = printed net {ms['line_total_amount']} (no allowances / charges: BT-106 = BT-109, §5)"]


STAMP_TEXT_RE = re.compile(r"^\s*(ÜBERWIESEN|UEBERWIESEN|BEZAHLT|EINGEGANGEN|EINGANG|ERHALTEN|GEBUCHT|ERLEDIGT|PAID|RECEIVED)"
                           r"(?:\s*[:.]?\s*(?:am\s+)?([0-3]?\d\.\s*(?:[01]?\d\.|[A-ZÄÖÜ]{3,}\.?)\s*\d{2,4}))?\s*$")
STAMP_TYPE: Dict[str, str] = {"ÜBERWIESEN": "PAID", "UEBERWIESEN": "PAID", "BEZAHLT": "PAID", "PAID": "PAID", "GEBUCHT": "PAID",
                              "ERLEDIGT": "PAID"}


def stamps_from_text(audit: Dict[str, Any], text: str) -> int:
    """
    A stamp the visual audit missed but the OCR / text layer printed as a
    stamp line of its own (an upper-case stamp word, optionally with a date:
    'ÜBERWIESEN 12. MAI 2031') is added to the audit's stamps. Sentences
    ('Sie erhalten ...') never match.

    Returns:
        Number of stamps added.
    """
    if not isinstance(audit, dict) or audit.get("layer_stamps"):
        return 0
    stamps = []
    for line in text.splitlines():
        match = STAMP_TEXT_RE.match(line)
        if match:
            word = match.group(1)
            stamps.append({"raw_content": line.strip(), "type": STAMP_TYPE.get(word, "RECEIVED"), "location": None, "page": None,
                           "source": "text_layer",
                           "form_fields": ([{"label": word, "raw_value": match.group(2), "normalized_value": match.group(2),
                                             "value_type": "DATE"}] if match.group(2) else [])})
    if stamps:
        audit["layer_stamps"] = stamps
        audit.setdefault("audit_summary", {})["was_stamp_interference"] = True
    return len(stamps)
