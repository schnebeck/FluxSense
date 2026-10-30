# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/acceptance.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Field contract §0.5: provenance and auto-acceptance of every
                filled scalar field. provenance = check (deterministic check:
                mod-97, sums, balance, ZUGFeRD XML) > label (printed next to a
                listed label of the type: same line or directly below) >
                geometry (DIN 5008 region) > model. auto_accepted is true
                only for label/geometry/check values that are grounded (or
                zoom-verified) and touched by no failing check; semantic
                rules (sender == recipient, a party that is a salutation /
                form heading / table cell / log entry, doc_date after a stamp
                date, doc_number without a listed label) force a review.
                Party geometry also comes from text patterns when no word
                boxes place the DIN 5008 regions: an address block under a
                title line ('Herrn', 'Frau'), a return-address line, the
                legal footer and the closing block. Non-core fields
                (addresses, phones, line details) are informational.
------------------------------------------------------------------------------
"""

import re
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from fluxsense.catalog.contract import (DATE_TOKEN_RE, FIELD_LABELS, FIRST_PAGE_DATE_GROUPS, PARTY_LABELS, compiled, contract_for,
                                        date_labels, date_line_dates, find_labelled, form_filler, is_address_only, is_department,
                                        is_document_title, not_a_party, number_labels, own_number_hits, same_value, signature_line,
                                        subject_context, HEADER_ONLY_NUMBER_LABELS, ORDERING_CUSTOMER_LABELS, PAYEE_LABELS,
                                        form_signature_dates, geometry_policy, is_name_fragment, labelled_values, party_problem,
                                        derived_period_value, previous_statement_days, printed_period_pairs,
                                        valid_tax_id)
from fluxsense.extract.base import TextLayer
from fluxsense.grounding.harness import _get, holds
from fluxsense.grounding.matcher import build_index, ground_value
from fluxsense.grounding.normalize import compact, extract_amounts, extract_dates, normalize_text, parse_date, to_decimal
from fluxsense.grounding.snap import iban_valid
from fluxsense.layout import din5008
from fluxsense.models.result import CheckResult, CheckStatus, FieldEvidence, GroundingStatus
from fluxsense.stages.identity import is_role_label, unfilled_form
from fluxsense.stages.repair import sender_only_in_table, stamp_dates
from fluxsense.validate.contract_terms import payment_checked_paths
from fluxsense.validate.dates import CHECK_CONSISTENCY, CHECK_YEAR

LABEL, GEOMETRY, CHECK, MODEL = "label", "geometry", "check", "model"
ACCEPTED_PROVENANCE: Tuple[str, ...] = (LABEL, GEOMETRY, CHECK)
ACCEPTED_STATUS: Tuple[GroundingStatus, ...] = (GroundingStatus.GROUNDED, GroundingStatus.VERIFIED)
TABLE_FIELDS: Tuple[str, ...] = ("line_items", "transactions", "items")
FIN = "bodies.finance_body"
MS = f"{FIN}.monetary_summation"
ST = "bodies.statement_body"
# Core fields that get an evidence record even when the extraction produced none (e.g. filled by the layout rules).
CORE_PATHS: Tuple[Tuple[str, str], ...] = (
    ("meta_header.doc_date", "date"), ("meta_header.doc_number", "identifier"),
    ("meta_header.sender.company", "text"), ("meta_header.sender.name", "text"), ("meta_header.sender.iban", "iban"),
    ("meta_header.recipient.company", "text"), ("meta_header.recipient.name", "text"),
    (f"{FIN}.invoice_number", "identifier"), (f"{FIN}.invoice_date", "date"), (f"{FIN}.due_date", "date"),
    (f"{MS}.grand_total_amount", "amount"), (f"{MS}.due_payable_amount", "amount"), (f"{MS}.tax_basis_total_amount", "amount"),
    (f"{MS}.line_total_amount", "amount"), (f"{MS}.tax_total_amount", "amount"),
    (f"{ST}.iban", "iban"), (f"{ST}.statement_date", "date"), (f"{ST}.statement_number", "identifier"),
    (f"{ST}.period_start", "date"), (f"{ST}.period_end", "date"), (f"{ST}.opening_balance", "amount"),
    (f"{ST}.closing_balance", "amount"), (f"{ST}.payment_due_date", "date"), ("bodies.correspondence_body.deadline", "date"),
)
PARTY_NAME_KEYS: Tuple[str, ...] = ("company", "name")
CORE_SET = frozenset(path for path, _ in CORE_PATHS)
CORE_ROW_RE = re.compile(r"^bodies\.finance_body\.(?:payment_accounts\.\d+\.iban|tax_breakdown\.\d+\.(?:tax_basis_amount|tax_amount|tax_rate))$")
TITLE_LINE_RE = re.compile(r"^\s*(?:herrn?|frau|firma|an|eheleute|familie)\s*$", re.IGNORECASE)
ZIP_LINE_RE = re.compile(r"^\s*(?:D-)?\d{5}\s+[A-ZÄÖÜ]")
RETURN_LINE_RE = re.compile(r"[,·•|–-]\s*(?:D-)?\d{5}\s+[A-ZÄÖÜ]")
LEGAL_FOOTER_RE = re.compile(r"(vorstand|aufsichtsrat|geschäftsführ|geschaeftsfuehr|sitz der gesellschaft|registergericht|amtsgericht|"
                             r"\bhrb\b|\bhra\b|handelsregister|ust-?id|steuernummer)", re.IGNORECASE)
CLOSING_RE = re.compile(r"(mit freundlichen grüßen|mit freundlichen gruessen|freundliche grüße|hochachtungsvoll|viele grüße|"
                        r"kind regards|best regards)", re.IGNORECASE)


def fieldread_values(layer: TextLayer, pages: Sequence[int]) -> Set[str]:
    """Entries and ISO dates of the FORM FIELDS readings ('label: entry -> iso') of the pages."""
    values: Set[str] = set()
    for number in pages:
        page = layer.page(number)
        for line in ((page.sources.get("fieldread") or "") if page else "").splitlines():
            entry = line.split(": ", 1)[1] if ": " in line else ""
            if not entry or entry.startswith("("):
                continue
            entry = re.sub(r"\s*\(readings: .*\)$", "", entry)
            for part in entry.split(" -> "):
                values.add(part.strip())
            values |= {p.strip() for p in re.split(r"[;\n]", entry) if p.strip()}
    return values


def party_name_text(party: Any) -> str:
    """Company and name of a party as one text."""
    if not isinstance(party, dict):
        return ""
    return " ".join(str(party.get(k)) for k in PARTY_NAME_KEYS if party.get(k))


def is_core(path: str) -> bool:
    """True for the core fields of the contract (KPI); other fields (addresses, phones, line details) are informational."""
    return path in CORE_SET or CORE_ROW_RE.match(path) is not None
LOG_LINE_RE = re.compile(r"^\s*(?:\d{1,2}:\d{2}|\d{1,2}\.\d{1,2}\.\d{2,4}|\d{4}-\d{2}-\d{2})\b")
DOC_DATE_PATHS: Tuple[str, ...] = ("meta_header.doc_date", f"{FIN}.invoice_date", f"{ST}.statement_date")
DUE_DATE_PATHS: Tuple[str, ...] = (f"{FIN}.due_date", f"{ST}.payment_due_date", "bodies.correspondence_body.deadline")
NUMBER_PATHS: Tuple[str, ...] = ("meta_header.doc_number", f"{FIN}.invoice_number", f"{ST}.statement_number")
TOLERANCE = Decimal("0.011")


def _is_row(path: str) -> bool:
    """True for table-row cells (assessed as tables, not as scalar fields)."""
    return any(f".{name}." in f".{path}." for name in TABLE_FIELDS)


def _dec(value: Any) -> Optional[Decimal]:
    """Decimal of a field value (None if not a number)."""
    return to_decimal(value) if value is not None else None


def _close(a: Optional[Decimal], b: Optional[Decimal]) -> bool:
    """Equal within a cent."""
    return a is not None and b is not None and abs(a - b) <= TOLERANCE


def checked_paths(fields: Dict[str, Any]) -> Set[str]:
    """
    Paths confirmed by a deterministic check: net + tax = gross, sum of line
    items = line total / tax basis, tax breakdown = tax total, amount due =
    confirmed gross, opening + bookings = closing balance.
    """
    paths: Set[str] = set()
    bodies = fields.get("bodies") or {}
    fin = bodies.get("finance_body") or {}
    ms = fin.get("monetary_summation") or {}
    basis, tax, gross = _dec(ms.get("tax_basis_total_amount")), _dec(ms.get("tax_total_amount")), _dec(ms.get("grand_total_amount"))
    if basis is not None and tax is not None and _close(basis + tax, gross):
        paths |= {f"{MS}.tax_basis_total_amount", f"{MS}.tax_total_amount", f"{MS}.grand_total_amount"}
    totals = [_dec(r.get("total_price")) for r in fin.get("line_items") or [] if isinstance(r, dict)]
    if totals and all(t is not None for t in totals):
        items = sum(totals, Decimal(0))  # type: ignore[arg-type]
        if _close(items, _dec(ms.get("line_total_amount"))):
            paths.add(f"{MS}.line_total_amount")
        if _close(items, basis) and not ms.get("allowance_total_amount") and not ms.get("charge_total_amount"):
            paths.add(f"{MS}.tax_basis_total_amount")
    breakdown = [r for r in fin.get("tax_breakdown") or [] if isinstance(r, dict)]
    for i, row in enumerate(breakdown):
        row_basis, row_tax, rate = _dec(row.get("tax_basis_amount")), _dec(row.get("tax_amount")), _dec(row.get("tax_rate"))
        if row_basis is not None and row_tax is not None and rate is not None and rate > 0 and \
                abs(row_basis * rate / 100 - row_tax) <= Decimal("0.021"):
            paths |= {f"{FIN}.tax_breakdown.{i}.tax_basis_amount", f"{FIN}.tax_breakdown.{i}.tax_amount",
                      f"{FIN}.tax_breakdown.{i}.tax_rate"}
    bases = [_dec(r.get("tax_basis_amount")) for r in breakdown]
    if breakdown and all(b is not None for b in bases) and _close(sum(bases, Decimal(0)), basis):  # type: ignore[arg-type]
        paths.add(f"{MS}.tax_basis_total_amount")
        paths |= {f"{FIN}.tax_breakdown.{i}.tax_basis_amount" for i in range(len(breakdown))}
    amounts = [_dec(r.get("tax_amount")) for r in breakdown]
    if breakdown and all(a is not None for a in amounts) and _close(sum(amounts, Decimal(0)), tax):  # type: ignore[arg-type]
        paths.add(f"{MS}.tax_total_amount")
        paths |= {f"{FIN}.tax_breakdown.{i}.tax_amount" for i in range(len(breakdown))}
    due = _dec(ms.get("due_payable_amount"))
    if f"{MS}.grand_total_amount" in paths and _close(due, gross):
        paths.add(f"{MS}.due_payable_amount")
    statement = bodies.get("statement_body") or {}
    rows = [r for r in statement.get("transactions") or [] if isinstance(r, dict)]
    opening, closing = _dec(statement.get("opening_balance")), _dec(statement.get("closing_balance"))
    bookings = [_dec(r.get("amount")) for r in rows]
    if rows and opening is not None and all(b is not None for b in bookings) and \
            _close(opening + sum(bookings, Decimal(0)), closing):  # type: ignore[arg-type]
        paths |= {f"{ST}.opening_balance", f"{ST}.closing_balance"}
    paths |= payment_checked_paths(fields)  # §10: contract payment components add up to the printed total
    return paths - inconsistent_amounts(fields)


def inconsistent_amounts(fields: Dict[str, Any]) -> Set[str]:
    """
    Amounts that contradict the arithmetic a 'check' would claim. A net
    amount (tax basis / line total) equal to the gross total while tax is
    printed is the gross price, not a net (§0.2: never compute it); net +
    tax != gross marks net and tax; a tax-breakdown row whose basis x rate
    is not its tax marks that row.
    """
    out: Set[str] = set()
    fin = (fields.get("bodies") or {}).get("finance_body") or {}
    ms = fin.get("monetary_summation") or {}
    tax, gross = _dec(ms.get("tax_total_amount")), _dec(ms.get("grand_total_amount"))
    for key in ("tax_basis_total_amount", "line_total_amount"):
        net = _dec(ms.get(key))
        if net is None or tax is None or tax <= 0 or gross is None:
            continue
        if _close(net, gross):
            out.add(f"{MS}.{key}")
        elif key == "tax_basis_total_amount" and not _close(net + tax, gross):
            out |= {f"{MS}.{key}", f"{MS}.tax_total_amount"}
    for i, row in enumerate(r for r in fin.get("tax_breakdown") or [] if isinstance(r, dict)):
        row_basis, row_tax, rate = _dec(row.get("tax_basis_amount")), _dec(row.get("tax_amount")), _dec(row.get("tax_rate"))
        if row_basis is not None and row_tax is not None and rate is not None and rate > 0 and \
                abs(row_basis * rate / 100 - row_tax) > Decimal("0.021"):
            out |= {f"{FIN}.tax_breakdown.{i}.tax_basis_amount", f"{FIN}.tax_breakdown.{i}.tax_amount"}
    return out


def _value_in(kind: str, value: Any, fragment: str) -> bool:
    """True if the value is printed in a text fragment."""
    if kind == "amount":
        target = _dec(value)
        return target is not None and abs(target) in {abs(a) for a in extract_amounts(fragment)}
    if kind == "date":
        target = parse_date(value)
        return target is not None and target in extract_dates(fragment)[0]
    key = compact(str(value))
    return len(key) >= 2 and key in compact(fragment)


def _has_kind(kind: str, fragment: str) -> bool:
    """True if a fragment carries any value of the kind (an amount, a date)."""
    if kind == "amount":
        return bool(extract_amounts(fragment))
    if kind == "date":
        return bool(extract_dates(fragment)[0])
    return bool(fragment.strip(" :.\t|"))


def _cells(line: str) -> List[str]:
    """Cells of a pipe-table row."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def label_near(kind: str, value: Any, labels: Sequence[str], lines: Sequence[str]) -> Optional[str]:
    """
    The first label after which the value is printed: on the same line, on
    the next line when the label's line carries no other value of the kind
    ('Kontosaldo per 30.09.2031' / 'EUR + 1.234,56'), or - in pipe tables -
    in the next cell or the same column of the next row.
    """
    for label in labels:
        pattern = compiled(label)
        for index, line in enumerate(lines):
            for match in pattern.finditer(line):
                rest = line[match.end():]
                if _value_in(kind, value, rest):
                    return label
                if line.count("|") < 2 and not _has_kind(kind, rest) and index + 1 < len(lines) \
                        and _value_in(kind, value, lines[index + 1]):
                    return label
            if line.count("|") >= 2 and index + 1 < len(lines) and lines[index + 1].count("|") >= 2:
                cells, row = _cells(line), _cells(lines[index + 1])
                if len(cells) == len(row) and any(pattern.match(c) and not _has_kind(kind, c) and _value_in(kind, value, row[i])
                                                  for i, c in enumerate(cells)):
                    return label
    return None


def printed_periods(lines: Sequence[str]) -> List[Tuple[str, str]]:
    """Printed periods (dates, quarters, half years) as ISO date pairs."""
    return [(a.isoformat(), b.isoformat()) for line in lines for a, b in printed_period_pairs(line)]


def _snippet_label(kind: str, ev: FieldEvidence, labels: Sequence[str], text_norm: str) -> Optional[str]:
    """A listed label inside the model's evidence snippet - only if the snippet itself is printed in the text."""
    if not ev.evidence or not labels or normalize_text(ev.evidence) not in text_norm:
        return None
    hit = next((h for h in find_labelled("date" if kind == "date" else "identifier", labels, [ev.evidence])
                if same_value("date" if kind == "date" else "identifier", h.value, ev.value)), None)
    return hit.label if hit else None


class Assessor:
    """Determines provenance and acceptance of the filled fields of one logical document."""

    def __init__(self, fields: Dict[str, Any], layer: TextLayer, pages: List[int], type_name: str,
                 checks: Sequence[CheckResult], audit: Optional[Dict[str, Any]] = None,
                 xml_fields: Optional[Dict[str, Any]] = None,
                 geometry_hint: Optional[Set[Tuple[str, str]]] = None) -> None:
        """
        Args:
            fields: Clean fields of the document.
            layer: Text layer.
            pages: Pages of the document.
            type_name: Catalog type whose contract rows apply.
            checks: Plausibility / grounding checks of the document.
            audit: Visual audit (stamps).
            xml_fields: Fields read from an embedded ZUGFeRD XML (None if none).
            geometry_hint: (path, value) pairs placed in a DIN 5008 region by an earlier run whose word boxes are
                no longer available (offline replay only).
        """
        self.fields, self.layer, self.pages, self.type_name = fields, layer, pages, type_name
        self.contract = contract_for(type_name)
        self.page_lines = {p: layer.page(p).best_text().splitlines() for p in pages if layer.page(p)}
        self.first_lines = self.page_lines.get(pages[0], []) if pages else []
        self.lines = [line for p in pages for line in self.page_lines.get(p, [])]
        self.fieldread = fieldread_values(layer, pages)
        self._independent: Optional[Dict[int, Any]] = None
        self.text = "\n".join(self.lines)
        self.text_norm = normalize_text(self.text)
        self.checks = [c for c in checks if c.status == CheckStatus.FAIL and c.field]
        self.checked = checked_paths(fields)
        self.inconsistent = inconsistent_amounts(fields)
        self.outranked: Set[str] = set()
        self.xml_fields = xml_fields
        self.stamps = stamp_dates("", audit)
        self.geometry_hint = geometry_hint or set()
        first, last = layer.page(pages[0]) if pages else None, layer.page(pages[-1]) if pages else None
        self.hints = din5008.analyze(first, last)
        self.policy = geometry_policy(type_name, unfilled_form(audit))
        self.has_boxes = bool(first and first.words)
        self.periods = printed_periods(self.lines)
        meta = fields.get("meta_header") or {}
        statement = (fields.get("bodies") or {}).get("statement_body") or {}
        self.doc_date = parse_date(meta.get("doc_date")) or parse_date(statement.get("statement_date"))

    # ------------------------------------------------------------ provenance

    def _date_line(self, value: Any) -> bool:
        """The date stands on its own line (place optional) on the first page; forms: anywhere, also on the signature line."""
        if self.contract.date_group == "enclosure" or (self.contract.date_group is None and not self.policy[0]):
            return False  # enclosures and types without letter structure have no date line
        lines = self.lines if self.contract.date_group == "form" else self.first_lines
        if any(same_value("date", d, value) for d in date_line_dates(lines)):
            return True
        return self.contract.date_group == "form" and any(
            (signed := signature_line(line.strip())) is not None and same_value("date", signed[0], value) for line in lines)

    def _statement_date(self, value: Any) -> Optional[str]:
        """Label provenance of a statement date: listed label, printed period end, or the header date line."""
        labels = date_labels(self.type_name) if self.contract.date_group == "statement" else FIELD_LABELS["statement_date"]
        if any(same_value("date", h.value, value) for h in find_labelled("date", labels, self.lines)):
            return "listed date label"
        if any(same_value("date", end, value) for _, end in self.periods):
            return "printed period end (§3)"
        if any(same_value("date", d, value) for d in date_line_dates(self.first_lines)):
            return "date line"
        return None

    def _label(self, ev: FieldEvidence) -> Optional[str]:
        """Label provenance per field."""
        path, name, kind = ev.path, ev.path.split(".")[-1], ev.kind
        parts = path.split(".")
        if path in ("meta_header.sender.name", "meta_header.sender.company") and self.contract.party_case == "2.4":
            filler = form_filler(self.lines)
            if filler and din5008.contained(filler[1], [str(ev.value)]) and din5008.contained(str(ev.value), [filler[1]]):
                return f"form field '{filler[0]}'"
        if len(parts) == 3 and parts[0] == "meta_header" and parts[1] in PARTY_LABELS and parts[2] in PARTY_NAME_KEYS:
            return self._party_label(parts[1], str(ev.value))
        if path == "meta_header.doc_date":
            if self.contract.date_group == "statement":
                return self._statement_date(ev.value)
            if self.contract.date_group == "form":  # §3: the date next to the signature only
                return "date next to the signature" if any(same_value("date", d, ev.value)
                                                          for d in form_signature_dates(self.lines)) else None
            labels = date_labels(self.type_name)
            label_lines = self.first_lines if self.contract.date_group in FIRST_PAGE_DATE_GROUPS else self.lines
            hit = find_labelled("date", labels, label_lines, above=self.contract.date_group == "form")
            if any(same_value("date", h.value, ev.value) for h in hit):
                return "listed date label"
            if self._date_line(ev.value):
                return "date line"
            return _snippet_label("date", ev, labels, self.text_norm)
        if path in NUMBER_PATHS:
            if path == "meta_header.doc_number" and self.contract.number_group not in (None, "statement"):
                sender = (self.fields.get("meta_header") or {}).get("sender") or {}
                sender_text = " ".join(str(sender.get(k)) for k in PARTY_NAME_KEYS if isinstance(sender, dict) and sender.get(k))
                hits = [h for h in own_number_hits(self.type_name, self.lines, self.first_lines, sender_text)
                        if sum(ch.isdigit() for ch in h.value) >= 3]
                own = next((h for h in hits if same_value("identifier", h.value, ev.value)), None)
                if own is not None and own.priority <= hits[0].priority:
                    return "listed number label"
                if own is not None:
                    self.outranked.add(ev.path)  # §0.4: an earlier label prints another number
                    return None
                labels = tuple(label for label in number_labels(self.type_name) if label not in HEADER_ONLY_NUMBER_LABELS)
                return _snippet_label("identifier", ev, labels, self.text_norm)
            labels = number_labels(self.type_name) if path == "meta_header.doc_number" else FIELD_LABELS[name]
            if path == "meta_header.doc_number" and self.contract.number_group == "statement":
                labels = FIELD_LABELS["statement_number"]
            if any(same_value("identifier", h.value, ev.value) for h in find_labelled("identifier", labels, self.lines)):
                return "listed number label"
            return _snippet_label("identifier", ev, labels, self.text_norm)
        if name == "statement_date":
            return self._statement_date(ev.value)
        if name == "invoice_date":
            labels = FIELD_LABELS[name]
            if any(same_value("date", h.value, ev.value) for h in find_labelled("date", labels, self.lines)):
                return "listed date label"
            if self._date_line(ev.value):
                return "date line"
            return _snippet_label("date", ev, labels, self.text_norm)
        if path == f"{ST}.period_start" and any(same_value("date", start, ev.value) for start, _ in self.periods):
            return "printed period"
        if path == f"{ST}.period_end":
            if any(same_value("date", end, ev.value) for _, end in self.periods):
                return "printed period"
            if not self.periods and self._statement_date(ev.value):
                return "statement date (§8, no printed period)"
            return None
        if name in ("period_start",):
            return None
        labels = FIELD_LABELS.get(name) or (FIELD_LABELS["iban"] if kind == "iban" else ())
        return label_near(kind, ev.value, labels, self.lines) if labels else None

    def _party_label(self, role: str, value: str) -> Optional[str]:
        """A party name after a party label ('Empfänger', 'Absender') on the same line or the line below."""
        labels = PARTY_LABELS[role]
        if self.contract.party_case == "2.8":  # bank transfer confirmation: 'Empfänger' is the payee (a counterparty)
            labels = ORDERING_CUSTOMER_LABELS if role == "recipient" else ()
        for label in labels:
            pattern = compiled(label)
            for index, line in enumerate(self.lines):
                cells = _cells(line) if line.count("|") >= 2 else [line.strip()]
                for column, cell in enumerate(cells):
                    match = pattern.match(cell)
                    if match is None:
                        continue
                    rest = cell[match.end():].strip(" :")
                    if not rest and len(cells) > 1:
                        rest = cells[column + 1] if column + 1 < len(cells) else ""  # '| Customer: | Name ... |'
                    below = self.lines[index + 1] if index + 1 < len(self.lines) else ""
                    if (rest and din5008.contained(value, [rest])) or (not rest and din5008.contained(value, [below])):
                        return f"party label '{label}'"
        return None

    def _text_regions(self, role: str) -> List[str]:
        """Party regions from text patterns (no word boxes needed)."""
        lines = self.first_lines
        regions: List[str] = []
        top = [line for line in lines if line.strip()][:3]
        if role == "recipient":
            for index, line in enumerate(lines[:40]):
                if TITLE_LINE_RE.match(line):
                    block = lines[index + 1:index + 5]
                    if any(ZIP_LINE_RE.match(b) for b in block):
                        regions += [b for b in block if not ZIP_LINE_RE.match(b)][:3]
            return regions
        if self.contract.party_case != "2.4":
            regions += top  # the letterhead lines at the top of the first page
        regions += [line for line in lines[:15] if RETURN_LINE_RE.search(line)]
        regions += [line for p in self.pages for line in self.page_lines.get(p, []) if LEGAL_FOOTER_RE.search(line)]
        for index, line in enumerate(self.lines):
            if CLOSING_RE.search(line):
                regions += self.lines[index:index + 4]
        return regions

    def _geometry(self, ev: FieldEvidence) -> Optional[str]:
        """Party values inside the DIN 5008 region of their role (word boxes, else text patterns)."""
        parts = ev.path.split(".")
        if len(parts) != 3 or parts[0] != "meta_header" or parts[1] not in ("sender", "recipient"):
            return None
        window_ok, letterhead_ok = self.policy
        allowed = window_ok if parts[1] == "recipient" else letterhead_ok
        form_issuer = parts[1] == "recipient" and self.contract.party_case == "2.4" and parts[2] == "company"
        if not allowed and not form_issuer:
            return None  # no geometry for this role and type (enclosures, notes, forms, certificate windows)
        regions: List[str] = []
        if allowed and parts[1] == "recipient":
            regions = list(self.hints.address_window)
        elif allowed:
            regions = list(self.hints.letterhead) + list(self.hints.info_column) + list(self.hints.closing)
            regions += [self.hints.return_address] if self.hints.return_address else []
        value = str(ev.value)
        if regions and din5008.contained(value, regions):
            return "DIN 5008 region"
        if allowed and (ev.path, value) in self.geometry_hint:
            return "DIN 5008 region (earlier run)"
        if parts[2] in PARTY_NAME_KEYS:
            text_regions = self._text_regions(parts[1]) if allowed else []
            if text_regions and din5008.contained(value, text_regions):
                return "address block / return address / footer / closing (text layout)"
        if form_issuer and din5008.contained(value, self._form_issuer_regions()):
            return "form issuer header / legal footer (§2.4)"
        return None

    def _form_issuer_regions(self) -> List[str]:
        """§2.4 forms: the issuing organisation (the recipient) printed in the form header (first lines) and legal footer."""
        top = [line for line in self.first_lines if line.strip()][:3]
        return top + [line for p in self.pages for line in self.page_lines.get(p, []) if LEGAL_FOOTER_RE.search(line)]

    def provenance(self, ev: FieldEvidence) -> Tuple[str, str]:
        """(provenance, detail) of one filled value."""
        if ev.path in self.checked:
            return CHECK, "arithmetic/balance check"
        if ev.kind == "iban" and iban_valid(str(ev.value)):
            return CHECK, "IBAN mod-97"
        if self.xml_fields is not None:
            xml_value = _get(self.xml_fields, ev.path)
            if xml_value is not None and compact(str(xml_value)) == compact(str(ev.value)):
                return CHECK, "ZUGFeRD XML"
        if ev.path == f"{MS}.due_payable_amount" and self._labelled_gross(ev.value):
            return CHECK, "equals the labelled grand total"
        if ev.path == f"{ST}.period_start" and self._previous_statement_day(ev.value):
            return CHECK, "day after the printed previous statement date (§8)"
        label = self._label(ev)
        if label:
            return LABEL, label
        if ev.status == GroundingStatus.GROUNDED and self._field_reading(ev):
            return LABEL, "read from the crop of its printed form label (§12)"
        geometry = self._geometry(ev)
        if geometry:
            return GEOMETRY, geometry
        return MODEL, ""

    def _field_reading(self, ev: FieldEvidence) -> bool:
        """True if the value is a form-field reading of its printed label (§12)."""
        if ev.matched_source == "fieldread":
            return True
        if ev.kind == "date":
            parsed = parse_date(ev.value)
            return parsed is not None and parsed.isoformat() in self.fieldread
        return bool(ev.value) and compact(str(ev.value)) in {compact(v) for v in self.fieldread}

    def _single_reading(self, ev: FieldEvidence) -> bool:
        """
        §12: a form-field reading without an independent second reading - the
        same value in another text source of the page (native text layer, OCR
        / layout text). The full-page extraction is no second reading: it saw
        the FORM FIELDS block.
        """
        if not self.fieldread or not self._field_reading(ev):
            return False
        if self._independent is None:
            from fluxsense.extract.base import PageText
            pages = []
            for number in self.pages:
                page = self.layer.page(number)
                if page is not None:
                    pages.append(PageText(page=number, width=page.width, height=page.height,
                                          sources={k: v for k, v in page.sources.items() if k != "fieldread"}))
            self._independent = build_index(TextLayer(pages=pages, engine="independent"), self.pages)
        return ground_value(ev.kind, ev.value, self._independent).status != GroundingStatus.GROUNDED

    def _labelled_gross(self, value: Any) -> bool:
        """The amount due equals the grand total, and that total is confirmed or printed next to a total label."""
        ms = ((self.fields.get("bodies") or {}).get("finance_body") or {}).get("monetary_summation") or {}
        gross = ms.get("grand_total_amount")
        if not _close(_dec(value), _dec(gross)):
            return False
        return f"{MS}.grand_total_amount" in self.checked or \
            label_near("amount", gross, FIELD_LABELS["grand_total_amount"], self.lines) is not None

    def _previous_statement_day(self, value: Any) -> bool:
        """period_start = the day after a printed previous statement date (contract §8)."""
        return parse_date(value) in previous_statement_days(self.lines)

    # --------------------------------------------------------- semantic rules

    def _party_reasons(self, ev: FieldEvidence, provenance: str) -> List[str]:
        """A party name that is a salutation, heading, address line, department, form subject, table cell or log entry."""
        parts = ev.path.split(".")
        if len(parts) != 3 or parts[0] != "meta_header" or parts[2] not in PARTY_NAME_KEYS:
            return []
        value = str(ev.value)
        if is_role_label(value):
            return ["party is a form heading / role label"]
        if not_a_party(value):
            return ["party is an info-column label / postal service mark"]
        if is_document_title(value):
            return ["party is a document heading / title"]
        if is_address_only(value):
            return ["party is a street / P.O. box / city, no name"]
        party = (self.fields.get("meta_header") or {}).get(parts[1]) or {}
        problem = party_problem(value, person=parts[2] == "name" and not party.get("company"))
        if problem:
            return [f"implausible party name ({problem})"]
        if ev.action == "synthesized" and provenance == GEOMETRY and is_name_fragment(party_name_text(party)):
            return ["layout fragment (a single word without legal form, filled from the layout only)"]
        if is_department(value) and (parts[2] == "company" or not party.get("company") or is_department(party.get("company"))):
            return ["party is a department / branch line, not the organisation"]
        occurrences = [i for i, line in enumerate(self.lines) if din5008.contained(value, [line])]
        if occurrences and all(din5008.SALUTATION_RE.match(self.lines[i]) for i in occurrences):
            return ["party is a salutation or greeting"]
        if occurrences and all(LOG_LINE_RE.match(self.lines[i]) for i in occurrences):
            return ["party is a log entry"]
        form_field = provenance == LABEL and parts[1] == "sender" and self.contract.party_case == "2.4"
        if parts[1] == "sender" and self.contract.party_case == "2.4" and not form_field and occurrences \
                and any(subject_context(self.lines, i) for i in occurrences):
            return ["form subject (insured / deceased / patient), not the person who fills in"]
        if provenance == MODEL and sender_only_in_table({parts[2]: value}, self.layer, self.pages):
            return ["party is a table cell"]
        return []

    def _identifier_reasons(self, ev: FieldEvidence) -> List[str]:
        """§7 / §2.8: tax / VAT IDs and IBANs must pass their format check; transfer confirmations have no invoice totals
        and their parties follow §2.8."""
        name = ev.path.split(".")[-1]
        reasons: List[str] = []
        if name in ("tax_id", "vat_id") and not valid_tax_id(ev.value, name == "vat_id"):
            reasons.append(f"not a valid {'VAT ID' if name == 'vat_id' else 'tax number'} (bank identifier?)")
        if ev.kind == "iban" and not iban_valid(str(ev.value)):
            reasons.append("IBAN fails mod-97")
        if self.contract.party_case != "2.8":
            return reasons
        if ev.path.startswith(f"{MS}.") or ".tax_breakdown." in ev.path:
            reasons.append("a transfer amount is no invoice total (§2.8)")
        parts = ev.path.split(".")
        if len(parts) == 3 and parts[0] == "meta_header" and parts[2] in PARTY_NAME_KEYS:
            ordering = labelled_values(ORDERING_CUSTOMER_LABELS, self.lines)
            payee = labelled_values(PAYEE_LABELS, self.lines)
            value = str(ev.value)
            if parts[1] == "sender" and ordering and din5008.contained(value, ordering):
                reasons.append("the ordering customer is never the sender (§2.8)")
            if parts[1] == "recipient" and payee and din5008.contained(value, payee) \
                    and not (ordering and din5008.contained(value, ordering)):
                reasons.append("the payee is a counterparty, not the recipient (§2.8)")
        return reasons

    def _same_party_suspects(self, evidence: List[FieldEvidence], provenances: Dict[int, str]) -> Set[str]:
        """
        sender == recipient: one of them is wrong. The role whose name has the
        weaker provenance (model < geometry < label / check) is the suspect;
        with equal evidence both are.
        """
        rank = {CHECK: 3, LABEL: 3, GEOMETRY: 2, MODEL: 0}
        best: Dict[str, int] = {"sender": -1, "recipient": -1}
        for ev in evidence:
            parts = ev.path.split(".")
            if len(parts) == 3 and parts[0] == "meta_header" and parts[1] in best and parts[2] in PARTY_NAME_KEYS:
                best[parts[1]] = max(best[parts[1]], rank.get(provenances[id(ev)], 0))
        if best["sender"] == best["recipient"]:
            return {"sender", "recipient"}
        return {"sender"} if best["sender"] < best["recipient"] else {"recipient"}

    def _same_party(self) -> bool:
        """sender == recipient (by name)."""
        meta = self.fields.get("meta_header") or {}
        names = []
        for role in ("sender", "recipient"):
            party = meta.get(role) or {}
            names.append(" ".join(str(party.get(k)) for k in PARTY_NAME_KEYS if isinstance(party, dict) and party.get(k)))
        a, b = names
        return bool(a and b) and (normalize_text(a) == normalize_text(b) or
                                  (din5008.contained(a, [b]) and din5008.contained(b, [a])))

    def reasons(self, ev: FieldEvidence, provenance: str, same_party: bool, derived: bool = False) -> List[str]:
        """Why a filled value is not auto-accepted (empty = accepted)."""
        reasons: List[str] = []
        if provenance not in ACCEPTED_PROVENANCE:
            reasons.append("model reading only (no listed label, layout region or check)")
        if ev.status not in ACCEPTED_STATUS and not derived:
            reasons.append(f"not grounded ({ev.status.value})")
        failing = [c.check_id for c in self.checks if c.field and (c.field == ev.path or ev.path.startswith(f"{c.field}.")
                                                                   or c.field.startswith(f"{ev.path}."))]
        if derived:  # normalised from printed text (§8): not verbatim, so the text matcher cannot find it
            failing = [f for f in failing if f != "grounding.ungrounded"]
        if ev.path in self.inconsistent:
            reasons.append("amount arithmetic inconsistent")
        if ev.path in self.outranked:
            reasons.append("an earlier listed label prints another number (§0.4)")
        reasons += self._identifier_reasons(ev)
        if self._single_reading(ev):
            reasons.append("handwritten form field (single reading)")
        named = (CHECK_CONSISTENCY, CHECK_YEAR, "fieldread.disagreement", "fieldread.ambiguous", "grounding.iban_printed_invalid")
        date_reasons = list(dict.fromkeys(c.message for c in self.checks if c.check_id in named and c.field == ev.path))
        if date_reasons:  # §9 / §12: name the conflicting value, the ambiguous readings or the year decision
            reasons += date_reasons
            failing = [f for f in failing if f not in named]
        if din5008.GEOMETRY_CONFLICT in failing:
            reasons.append("geometry/model conflict")
            failing = [f for f in failing if f != din5008.GEOMETRY_CONFLICT]
        if failing:
            reasons.append(f"failing check: {', '.join(sorted(set(failing)))}")
        if same_party and ev.path.startswith(("meta_header.sender.", "meta_header.recipient.")):
            reasons.append("sender == recipient")
        reasons += self._party_reasons(ev, provenance)
        if ev.path in DUE_DATE_PATHS and parse_date(ev.value) in self.stamps:
            reasons.append("due date is a stamp date")
        if ev.path in DUE_DATE_PATHS and self.contract.date_group == "form" and self.doc_date \
                and parse_date(ev.value) == self.doc_date:
            reasons.append("a form's signature date is never a due date")
        if ev.path in DOC_DATE_PATHS and self.stamps:
            current = parse_date(ev.value)
            earlier = [d for d in self.stamps if current and d < current]
            if earlier:
                reasons.append(f"doc_date after the stamp date {min(earlier)}")
        if ev.path == "meta_header.doc_number" and number_labels(self.type_name) and provenance not in (LABEL, CHECK):
            reasons.append(f"doc_number not printed next to a listed label of {self.type_name}")
        return reasons

    # ------------------------------------------------------------------ run

    def _synthesize(self, evidence: List[FieldEvidence]) -> List[FieldEvidence]:
        """Evidence records for filled core fields that have no holding record (e.g. layout-filled parties)."""
        held = {e.path for e in evidence if e.status != GroundingStatus.REMOVED and holds(self.fields, e)}
        wanted = list(CORE_PATHS)
        accounts = ((self.fields.get("bodies") or {}).get("finance_body") or {}).get("payment_accounts") or []
        wanted += [(f"{FIN}.payment_accounts.{i}.iban", "iban") for i in range(len(accounts))]
        created: List[FieldEvidence] = []
        index = None
        for path, kind in wanted:
            value = _get(self.fields, path)
            if value in (None, "") or path in held or isinstance(value, (dict, list)):
                continue
            if index is None:
                index = build_index(self.layer, self.pages)
            match = ground_value(kind, value, index)
            created.append(FieldEvidence(path=path, kind=kind, value=value, status=match.status, matched_source=match.source,
                                         matched_page=match.page, matched_text=match.text, action="synthesized"))
        return created

    def run(self, evidence: List[FieldEvidence]) -> Dict[str, int]:
        """
        Sets provenance / auto_accepted / review_reasons on every evidence
        record in place (synthesized records for core fields are appended).

        Returns:
            Counts of core fields (auto_accepted, flagged) and of informational fields.
        """
        evidence += self._synthesize(evidence)
        live = [ev for ev in evidence if not (_is_row(ev.path) or ev.status == GroundingStatus.REMOVED or not holds(self.fields, ev))]
        provenances = {id(ev): self.provenance(ev)[0] for ev in live}
        suspect = self._same_party_suspects(live, provenances) if self._same_party() else set()
        counts = {"auto_accepted": 0, "flagged": 0, "informational_auto_accepted": 0, "informational_flagged": 0}
        for ev in evidence:
            if id(ev) not in provenances:
                ev.provenance, ev.auto_accepted, ev.review_reasons = None, None, []
                continue
            provenance = provenances[id(ev)]
            derived = provenance in (CHECK, LABEL) and derived_period_value(ev.path, ev.value, self.lines)
            same_party = ev.path.split(".")[1] in suspect if ev.path.startswith("meta_header.") else False
            reasons = self.reasons(ev, provenance, same_party, derived)
            ev.provenance, ev.auto_accepted, ev.review_reasons = provenance, not reasons, reasons
            prefix = "" if is_core(ev.path) else "informational_"
            counts[f"{prefix}auto_accepted" if not reasons else f"{prefix}flagged"] += 1
        return counts


def assess(fields: Dict[str, Any], evidence: List[FieldEvidence], layer: TextLayer, pages: List[int], type_name: str,
           checks: Sequence[CheckResult], audit: Optional[Dict[str, Any]] = None,
           xml_fields: Optional[Dict[str, Any]] = None,
           geometry_hint: Optional[Set[Tuple[str, str]]] = None) -> Dict[str, int]:
    """Convenience wrapper around Assessor.run()."""
    return Assessor(fields, layer, pages, type_name, checks, audit, xml_fields, geometry_hint).run(evidence)
