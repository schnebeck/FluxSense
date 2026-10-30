# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/text/correct.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Evidence-based corrections after the plausibility checks -
                the image decides. Words of OCR pages: a dictionary word
                replaces an unknown one only as the unique candidate at edit
                distance <= 2 that typical OCR confusions explain ('rn' /
                'm', 'li' / 'll', 'U' / 'Ü'); names, company names, ALL-CAPS,
                tokens with digits, e-mails and URLs, words under five
                letters and words read the same way more than once in the
                document (a consistent reading is printed so: surnames) are never
                touched. Off by default (text.spell_check).
                Fixed formats: a character that violates its position class
                (a letter at a digit position) is replaced when exactly one
                candidate passes the checksum. Values that fail a check
                without such a forced position (digit / digit confusions)
                and arithmetic mismatches (qty x price != line total, net +
                tax != gross) are re-read on a native-resolution crop that
                never shows the old value: the re-read replaces the value
                only when it differs and makes the check pass; a re-read
                that confirms the printed value keeps it and flags a
                document error; anything else keeps it for review. A value
                corrected by a check stays visible as corrected and is
                auto-accepted only when re-read and check agree. Review
                confirmations by a person use the same records (hooks).
------------------------------------------------------------------------------
"""

import copy
import difflib
import re
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from fluxsense.config import Settings
from fluxsense.extract.base import PageText, TextLayer
from fluxsense.formats import FormatSpec, format_for_field, get_format
from fluxsense.formats.builtin import CARD, EAN, IBAN, ISBN10, ISBN13, STEUER_ID, VAT_ID
from fluxsense.grounding.normalize import to_decimal
from fluxsense.grounding.snap import damerau
from fluxsense.llm.base import LLMBackend, LLMRequest
from fluxsense.log import get_logger
from fluxsense.models.result import CheckResult, CheckStatus, CorrectionRecord, FieldEvidence, Severity, TextCorrection

logger = get_logger("text.correct")

DICTIONARY, FORMAT, REREAD, ARITHMETIC, HUMAN = "dictionary", "format", "reread", "arithmetic", "human"
CHECK_CORRECTED = "format.corrected"
CHECK_PRINTED_INVALID = "format.printed_invalid"
CHECK_REREAD_CONFLICT = "format.reread_conflict"
CHECK_ARITHMETIC_PRINTED = "arithmetic.printed_inconsistent"
CHECK_ARITHMETIC_CONFLICT = "arithmetic.reread_conflict"
REVIEW_CHECKS: Tuple[str, ...] = (CHECK_CORRECTED, CHECK_PRINTED_INVALID, CHECK_REREAD_CONFLICT, CHECK_ARITHMETIC_PRINTED,
                                  CHECK_ARITHMETIC_CONFLICT)

# Character confusions of OCR in words (segment of the reading -> segment of the dictionary word).
OCR_WORD_PAIRS: FrozenSet[Tuple[str, str]] = frozenset(
    pair for a, b in (("rn", "m"), ("ri", "n"), ("cl", "d"), ("li", "h"), ("vv", "w"), ("c", "e"), ("l", "i"), ("l", "t"),
                      ("I", "l"), ("i", "j"), ("1", "l"), ("0", "o"), ("n", "u"), ("h", "b"), ("f", "t"), ("ii", "ü"),
                      ("u", "ü"), ("a", "ä"), ("o", "ö"), ("U", "Ü"), ("A", "Ä"), ("O", "Ö"), ("B", "ß"), ("li", "ll"),
                      ("il", "ll"), ("e", "o"), ("fi", "fl"))
    for pair in ((a, b), (b, a)))
_WORD_RE = re.compile(r"(?<![\w@/.\-])([A-Za-zÄÖÜäöüß]{5,})(?![\w@\-])")
_SKIP_SPAN_RE = re.compile(r"\S+@\S+|(?:https?://|www\.)\S+|\S+\.(?:de|com|org|net|eu|at|ch)\b\S*", re.IGNORECASE)
_TITLE_RE = re.compile(r"\b(?:Herrn?|Frau|Dr\.|Prof\.|Familie|Firma|Eheleute)\s+([A-ZÄÖÜ][\w-]*)(?:\s+([A-ZÄÖÜ][\w-]*))?")
# Fixed formats corrected inside the page text: the IBAN anywhere, the others only after their label on the line.
TEXT_FORMATS: Tuple[Tuple[FormatSpec, Optional["re.Pattern[str]"]], ...] = (
    (IBAN, None),
    (VAT_ID, re.compile(r"ust\.?-?\s*id|umsatzsteuer-?id|vat\s*(?:id|no|reg)", re.IGNORECASE)),
    (STEUER_ID, re.compile(r"steuer-?id|identifikationsnummer|idnr", re.IGNORECASE)),
    (EAN, re.compile(r"\b(?:ean|gtin)\b", re.IGNORECASE)),
    (ISBN13, re.compile(r"\bisbn\b", re.IGNORECASE)),
    (ISBN10, re.compile(r"\bisbn\b", re.IGNORECASE)),
    (CARD, re.compile(r"karte|kartennummer|card", re.IGNORECASE)),
)


# --------------------------------------------------------------------------------------------------------- words


def ocr_explained(word: str, candidate: str) -> bool:
    """True if a dictionary word differs from the OCR word only by typical OCR confusions (edit distance <= 2)."""
    if word == candidate or damerau(word, candidate, 2) > 2:
        return False
    ops = difflib.SequenceMatcher(a=word, b=candidate, autojunk=False).get_opcodes()
    return all(tag == "equal" or (word[i0:i1], candidate[j0:j1]) in OCR_WORD_PAIRS for tag, i0, i1, j0, j1 in ops)


def protected_words(texts: Iterable[str]) -> Set[str]:
    """Words of names that are never spell-corrected (party names, companies, signers, streets, towns)."""
    words: Set[str] = set()
    for text in texts:
        words |= {w for w in re.findall(r"[A-Za-zÄÖÜäöüß]+", str(text or "")) if len(w) >= 2}
    return words


def repeated_words(texts: Iterable[str]) -> Set[str]:
    """Words read the same way more than once in a document (a consistent reading is printed so: surnames, firm names)."""
    seen = Counter(m.group(1) for text in texts for m in _WORD_RE.finditer(text or ""))
    return {word for word, count in seen.items() if count > 1}


def word_corrections(page: int, text: str, lookup: Callable[[Sequence[str]], Dict[str, Optional[List[str]]]],
                     protected: Optional[Set[str]] = None) -> List[TextCorrection]:
    """
    Dictionary corrections of one OCR page text: unknown words with exactly
    one dictionary suggestion that OCR confusions explain. Words under five
    letters and words that occur more than once in the same form are kept.

    Args:
        page: Page number.
        text: The page's clean text.
        lookup: Dictionary lookup (word -> None if known, else suggestions).
        protected: Name words that are never corrected.

    Returns:
        Corrections with offsets in the text.
    """
    protected = protected or set()
    skip = [m.span() for m in _SKIP_SPAN_RE.finditer(text)]
    titled = {w for m in _TITLE_RE.finditer(text) for w in m.groups() if w}
    repeated = repeated_words([text])
    tokens: List[Tuple[int, str]] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(1)
        if word.isupper() or word in protected or word in titled or word in repeated \
                or any(s <= match.start() < e for s, e in skip):
            continue
        tokens.append((match.start(), word))
    if not tokens:
        return []
    results = lookup([w for _, w in tokens])
    out: List[TextCorrection] = []
    for offset, word in tokens:
        suggestions = results.get(word)
        if not suggestions:
            continue  # known (None), or no suggestion, or no dictionary
        explained = list(dict.fromkeys(s for s in suggestions if ocr_explained(word, s)))
        if len(explained) == 1:
            out.append(TextCorrection(page=page, offset=offset, before=word, after=explained[0], evidence_kind=DICTIONARY,
                                      reason=f"unique dictionary word explained by OCR confusions ({word} -> {explained[0]})"))
    return out


# ------------------------------------------------------------------------------------------------------- formats


def map_printed(printed: str, spec: FormatSpec, repaired: str) -> str:
    """The repaired normalised value written back into the printed form (spaces and separators kept)."""
    chars = list(printed)
    index = 0
    for position, char in enumerate(chars):
        if not spec.normalise(char):
            continue  # a separator the normaliser drops
        if index < len(repaired):
            chars[position] = repaired[index] if char.upper() != repaired[index] else char
            index += 1
    return "".join(chars)


def format_corrections(page: int, text: str, extra: Sequence[FormatSpec] = ()) -> List[TextCorrection]:
    """
    Fixed-format repairs inside an OCR page text: a span with a format's
    shape whose characters violate their position class gets the unique
    repair that passes the checksum. The IBAN is found anywhere, the other
    formats only after their label on the same line; formats supplied by the
    application are found anywhere.
    """
    out: List[TextCorrection] = []
    taken: List[Tuple[int, int]] = []
    for spec, label in list(TEXT_FORMATS) + [(s, None) for s in extra]:
        for start, end, printed in spec.find(text):
            if any(s < end and start < e for s, e in taken) or spec.valid(printed):
                continue
            if label is not None:
                line_start = text.rfind("\n", 0, start) + 1
                if not label.search(text[line_start:start]):
                    continue
            repairs = spec.forced_repairs(printed)
            if len(repairs) != 1:
                continue
            after = map_printed(printed, spec, repairs[0])
            taken.append((start, end))
            out.append(TextCorrection(page=page, offset=start, before=printed, after=after, evidence_kind=FORMAT,
                                      reason=f"{spec.name}: character at a wrong position class, unique repair passes the "
                                             f"check"))
    return out


# -------------------------------------------------------------------------------------------------------- re-read


REREAD_PROMPT = ("This is a high-resolution crop of page {page} of a scanned document. It shows the {label}. Read it "
                 "character by character exactly as printed - do not correct, complete or calculate anything. If it is not "
                 "visible, answer null. Return ONLY JSON: {{\"printed\": \"...\"}}")
REREAD_SCHEMA: Dict[str, Any] = {"type": "object", "additionalProperties": False, "required": ["printed"],
                                 "properties": {"printed": {"type": ["string", "null"], "maxLength": 80}}}
LABELS: Dict[str, str] = {"iban": "IBAN", "bic": "BIC", "vat_id": "VAT identification number (USt-IdNr.)",
                          "tax_number": "tax number", "steuer_id": "tax identification number", "card": "card number",
                          "postcode_de": "postcode", "postcode_at": "postcode", "postcode_ch": "postcode",
                          "ean": "EAN / GTIN", "isbn10": "ISBN", "isbn13": "ISBN"}
MONEY_LABELS: Dict[str, str] = {"tax_basis_total_amount": "net total amount (tax basis)", "line_total_amount": "sum of the line items",
                                "tax_total_amount": "VAT amount", "grand_total_amount": "gross total amount",
                                "quantity": "quantity", "unit_price": "unit price", "total_price": "line total"}
MS = "bodies.finance_body.monetary_summation"


@dataclass
class _Option:
    """One value that may be wrong: where it is, how to read it, how to try a replacement."""

    path: str
    kind: str
    value: Any
    page: int
    label: str
    evidence: Optional[str] = None
    bbox: Optional[List[float]] = None


def _get(data: Dict[str, Any], path: str) -> Any:
    node: Any = data
    for part in path.split("."):
        if isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        elif isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _set(data: Dict[str, Any], path: str, value: Any) -> None:
    from fluxsense.grounding.harness import set_path
    set_path(data, path, value)


class FieldCorrector:
    """Corrects fixed-format fields and arithmetic mismatches of OCR pages by their checks and crop re-reads."""

    def __init__(self, backend: LLMBackend, settings: Settings) -> None:
        """
        Args:
            backend: Vision-language backend.
            settings: Settings.
        """
        self.backend = backend
        self.settings = settings

    def reread(self, pdf_path: Path, layer: TextLayer, option: _Option) -> Optional[str]:
        """
        Reads a value again on a native-resolution crop of its place; the
        crop is found with the current reading, the prompt never shows it.
        None if the place is unknown or the value not visible.
        """
        from fluxsense.llm.images import ImagePart, encode_data_uri
        from fluxsense.vision.zoom import ZoomTarget, expand, locate, render_crop
        page = layer.page(option.page)
        if page is None:
            return None
        rect, _ = locate(page, ZoomTarget(id="R", page=option.page, label=option.label, value=option.value, kind=option.kind,
                                          evidence=option.evidence, bbox=option.bbox))
        if rect is None:
            return None
        llm = self.settings.llm
        image, dpi = render_crop(pdf_path, option.page, expand(rect, page.width, page.height, 60.0, 14.0), llm.zoom_dpi,
                                 llm.zoom_max_px)
        part = ImagePart(label="CROP", data_uri=encode_data_uri(image, llm.image_format), width=image.width, height=image.height)
        answer = self.backend.generate_json(LLMRequest(
            prompt=REREAD_PROMPT.format(page=option.page, label=option.label), label=f"Reread p{option.page} {option.path[-40:]}",
            model=self.settings.models.vision, images=[part], schema=REREAD_SCHEMA, schema_name="reread", max_tokens=60))
        printed = answer.get("printed") if isinstance(answer, dict) else None
        return str(printed).strip() if isinstance(printed, (str, int, float)) and str(printed).strip() else None

    # ------------------------------------------------------------------ formats

    def _formats(self, pdf_path: Path, layer: TextLayer, fields: Dict[str, Any], evidence: List[FieldEvidence],
                 extra: Sequence[FormatSpec], skip: Set[str]) -> Tuple[List[CheckResult], List[CorrectionRecord]]:
        from fluxsense.grounding.harness import holds
        checks: List[CheckResult] = []
        records: List[CorrectionRecord] = []
        for ev in evidence:
            if ev.value in (None, "") or ev.path in skip or not holds(fields, ev):
                continue
            party = _get(fields, ev.path.rsplit(".", 1)[0]) if "." in ev.path else None
            country = party.get("country") if isinstance(party, dict) else None
            name = ev.path.rsplit(".", 1)[-1]
            spec = get_format(name, extra) if any(s.name == name for s in extra) else format_for_field(ev.path, ev.value, country)
            number = ev.matched_page or ev.page
            page = layer.page(number) if number else None
            if spec is None or page is None or not page.scanned() or spec.valid(ev.value):
                continue
            value = spec.normalise(ev.value)
            forced = spec.forced_repairs(value)
            if not forced and spec.shape(value) is None:
                continue  # neither the format's shape nor a forced repair: not this format
            option = _Option(ev.path, ev.kind, ev.value, number, LABELS.get(spec.name, spec.name), ev.evidence, ev.bbox)
            raw = self.reread(pdf_path, layer, option) if self.settings.text.format_reread else None
            read = spec.normalise(raw) if raw else None
            unique = forced[0] if len(forced) == 1 else None
            if read is not None and read != value and spec.valid(read) and (unique is None or unique == read):
                self._apply(fields, ev, page, spec, value, read, REREAD, records,
                            f"re-read at native resolution ({raw!r}) passes the {spec.name} check")
            elif unique is not None and (read is None or read == unique):
                self._apply(fields, ev, page, spec, value, unique, FORMAT, records,
                            f"character at a wrong position class, the only repair passing the {spec.name} check")
                if read is None:  # the check alone corrected it: stays flagged
                    checks.append(CheckResult(check_id=CHECK_CORRECTED, rule="format", status=CheckStatus.FAIL,
                                              severity=Severity.WARNING, field=ev.path, expected=unique, actual=value,
                                              message=f"{spec.name} {value} corrected to {unique} by its format (position "
                                                      f"class and check) - not confirmed by a re-read, verify"))
            elif read is not None and read == value:
                checks.append(CheckResult(check_id=CHECK_PRINTED_INVALID, rule="format", status=CheckStatus.FAIL,
                                          severity=Severity.WARNING, field=ev.path, actual=value,
                                          message=f"{spec.name} {value} fails its check as printed (confirmed by a re-read at "
                                                  f"native resolution) - a document error, verify with the issuer"))
            else:
                checks.append(CheckResult(check_id=CHECK_REREAD_CONFLICT, rule="format", status=CheckStatus.FAIL,
                                          severity=Severity.WARNING, field=ev.path, actual=value, expected=read,
                                          message=f"{spec.name} {value} fails its check; the re-read "
                                                  f"{'is ' + repr(raw) if raw else 'found nothing'} - kept for review"))
        return checks, records

    def _apply(self, fields: Dict[str, Any], ev: FieldEvidence, page: PageText, spec: FormatSpec, before: str, after: str,
               kind: str, records: List[CorrectionRecord], reason: str) -> None:
        value = map_printed(str(ev.value), spec, after)  # the value keeps its own formatting (compact IBAN, 'KD-...')
        _set(fields, ev.path, value)
        ev.original_value, ev.value, ev.action, ev.resolution = ev.value, value, "corrected", f"{kind}+check"
        records.append(CorrectionRecord(trigger_checks=[f"format.{spec.name}"], pages=[page.page], scope=ev.path, applied=True,
                                        note=f"{ev.path}: {ev.original_value} -> {value} ({reason})"))
        printed = next((p for _, _, p in spec.find(page.best_text()) if spec.normalise(p) == before), None)
        if printed is not None:
            page.corrections.append(TextCorrection(page=page.page, before=printed, after=map_printed(printed, spec, after),
                                                   reason=reason, evidence_kind=kind, field=ev.path))

    # --------------------------------------------------------------- arithmetic

    def _resolve(self, pdf_path: Path, layer: TextLayer, fields: Dict[str, Any], options: List[_Option],
                 passes: Callable[[Dict[str, Any]], bool], scope: str) -> Tuple[List[CheckResult], List[CorrectionRecord]]:
        """
        Re-reads every value of a failing equation; a single re-read that
        differs and makes the equation hold replaces its value.
        """
        from fluxsense.grounding.harness import parse_printed
        readings: List[Tuple[_Option, Optional[Decimal]]] = []
        for option in options:
            parsed = parse_printed("amount", self.reread(pdf_path, layer, option))
            readings.append((option, to_decimal(parsed) if parsed is not None else None))
        fixes: List[Tuple[_Option, Decimal]] = []
        for option, read in readings:
            current = to_decimal(option.value)
            if read is None or current is None or abs(read - current) < Decimal("0.005"):
                continue
            trial = copy.deepcopy(fields)
            _set(trial, option.path, float(read))
            if passes(trial):
                fixes.append((option, read))
        anchor = options[-1].path
        if len(fixes) == 1:
            option, read = fixes[0]
            _set(fields, option.path, float(read))
            page = layer.page(option.page)
            if page is not None:
                page.corrections.append(TextCorrection(page=option.page, before=_printed_amount(page, option.value) or "",
                                                       after=_format_like(_printed_amount(page, option.value), read),
                                                       reason=f"re-read makes {scope} hold", evidence_kind=ARITHMETIC,
                                                       field=option.path))
                page.corrections[:] = [c for c in page.corrections if c.before]
            return [], [CorrectionRecord(trigger_checks=[scope], pages=[option.page], scope=option.path, applied=True,
                                         note=f"{option.path}: {option.value} -> {read} (re-read at native resolution; "
                                              f"{scope} now holds)")]
        confirmed = all(read is not None and to_decimal(o.value) is not None and abs(read - to_decimal(o.value)) < Decimal("0.005")
                        for o, read in readings)
        if confirmed:
            return [CheckResult(check_id=CHECK_ARITHMETIC_PRINTED, rule="arithmetic", status=CheckStatus.FAIL,
                                severity=Severity.WARNING, field=anchor,
                                message=f"{scope}: the re-read confirms every printed value - the document's own "
                                        f"arithmetic does not add up")], []
        return [CheckResult(check_id=CHECK_ARITHMETIC_CONFLICT, rule="arithmetic", status=CheckStatus.FAIL,
                            severity=Severity.WARNING, field=anchor,
                            message=f"{scope}: re-reads give no single consistent correction - kept for review")], []

    def _arithmetic(self, pdf_path: Path, layer: TextLayer, fields: Dict[str, Any], evidence: List[FieldEvidence],
                    findings: Sequence[CheckResult], evaluate: Callable[[Dict[str, Any]], Any]
                    ) -> Tuple[List[CheckResult], List[CorrectionRecord]]:
        checks: List[CheckResult] = []
        records: List[CorrectionRecord] = []
        by_path = {e.path: e for e in evidence}

        def still_failing(trial: Dict[str, Any], check_id: str, rows: Sequence[int] = ()) -> bool:
            report = evaluate(trial)
            for c in report.checks:
                if c.check_id == check_id and c.status == CheckStatus.FAIL and (not rows or set(rows) & set(c.rows)):
                    return True
            return False

        for finding in findings:
            if finding.check_id == "invoice.net_plus_tax":
                options = []
                for name in ("tax_basis_total_amount", "tax_total_amount", "grand_total_amount"):
                    ev = by_path.get(f"{MS}.{name}")
                    page = layer.page(ev.matched_page or ev.page or 0) if ev else None
                    if ev is not None and ev.value is not None and page is not None and page.scanned():
                        options.append(_Option(ev.path, "amount", ev.value, page.page, MONEY_LABELS[name], None, ev.bbox))
                if len(options) >= 2:
                    found = self._resolve(pdf_path, layer, fields, options,
                                          lambda trial: not still_failing(trial, "invoice.net_plus_tax"), "net + tax = gross")
                    checks += found[0]
                    records += found[1]
            elif finding.check_id == "invoice.qty_x_price":
                items = _get(fields, "bodies.finance_body.line_items") or []
                for row in finding.rows[:5]:
                    item = items[row] if row < len(items) and isinstance(items[row], dict) else None
                    page = layer.page(int(item.get("source_page") or 0)) if item else None
                    if item is None or page is None or not page.scanned():
                        continue
                    options = [_Option(f"bodies.finance_body.line_items.{row}.{key}", "amount", item[key], page.page,
                                       f"{MONEY_LABELS[key]} of item {row + 1} ({str(item.get('description') or '')[:40]})")
                               for key in ("quantity", "unit_price", "total_price") if item.get(key) is not None]
                    if len(options) == 3:
                        found = self._resolve(pdf_path, layer, fields, options,
                                              lambda trial, r=row: not still_failing(trial, "invoice.qty_x_price", [r]),
                                              f"quantity x unit price = line total (item {row + 1})")
                        checks += found[0]
                        records += found[1]
        return checks, records

    def run(self, pdf_path: Path, layer: TextLayer, fields: Dict[str, Any], evidence: List[FieldEvidence],
            findings: Sequence[CheckResult], evaluate: Callable[[Dict[str, Any]], Any], extra: Sequence[FormatSpec] = (),
            skip: Iterable[str] = ()) -> Tuple[List[CheckResult], List[CorrectionRecord]]:
        """
        Format and arithmetic corrections of one logical document (in place).

        Args:
            pdf_path: PDF file.
            layer: Text layer (pending text corrections are stored on its pages).
            fields: Fields (changed in place).
            evidence: Evidence records (changed in place).
            findings: Failing plausibility checks.
            evaluate: fields -> plausibility report.
            extra: Formats supplied by the application for this call (applied to the field of the same name).
            skip: Field paths already handled (e.g. an IBAN flagged as printed with a wrong checksum).

        Returns:
            (check results, correction records).
        """
        checks, records = self._formats(pdf_path, layer, fields, evidence, extra, set(skip))
        if self.settings.text.format_reread:
            more_checks, more_records = self._arithmetic(pdf_path, layer, fields, evidence, findings, evaluate)
            checks += more_checks
            records += more_records
        if records:
            logger.info(f"text corrections: {len(records)} field(s) corrected by check / re-read")
        return checks, records


def _printed_amount(page: PageText, value: Any) -> Optional[str]:
    """The printed form of an amount in the page text ('1.234,56' / '1234.56' / ...), None if not found."""
    from fluxsense.vision.zoom import value_variants
    text = page.best_text()
    for variant in value_variants("amount", value):
        if re.search(rf"(?<![\d,.]){re.escape(variant)}(?![\d])", text):
            return variant
    return None


def _format_like(printed: Optional[str], value: Decimal) -> str:
    """An amount written like a printed one (German or English separators)."""
    if printed is None:
        return f"{value:.2f}"
    en = f"{abs(value):,.2f}"
    de = en.replace(",", "X").replace(".", ",").replace("X", ".")
    if "," in printed and (printed.rfind(",") > printed.rfind(".")):
        return de if "." in printed else de.replace(".", "")
    return en if "," in printed else en.replace(",", "")


# --------------------------------------------------------------------------------------------------------- hooks


def confirm_text(page: Any, before: str, after: str, offset: Optional[int] = None, note: str = "") -> TextCorrection:
    """
    Review hook: a person confirms a text correction of a page output
    (PageTextOut); it is applied with the highest priority and recorded as
    evidence_kind 'human'.
    """
    from fluxsense.text.levels import apply_corrections
    correction = TextCorrection(page=page.page, offset=offset, before=before, after=after, evidence_kind=HUMAN,
                                reason=note or "confirmed in review")
    corrections = list(page.corrections) + [correction]
    page.corrected = apply_corrections(page.clean if page.clean is not None else page.text, corrections)
    page.corrections = corrections
    return correction


def confirm_field(doc: Any, path: str, value: Any, note: str = "") -> FieldEvidence:
    """
    Review hook: a person confirms (or corrects) a field value of a logical
    document; the value is set, its evidence becomes provenance 'human',
    auto-accepted, and the change is recorded.
    """
    from fluxsense.grounding.harness import set_path
    old = _get(doc.fields, path)
    set_path(doc.fields, path, value)
    ev = next((e for e in doc.evidence if e.path == path), None)
    if ev is None:
        ev = FieldEvidence(path=path, value=value)
        doc.evidence.append(ev)
    if old is not None and old != value:
        ev.original_value = old
    ev.value, ev.action, ev.provenance, ev.auto_accepted, ev.review_reasons = value, "confirmed", HUMAN, True, []
    doc.corrections.append(CorrectionRecord(trigger_checks=["review"], scope=path, applied=True,
                                            note=f"{path}: {old!r} -> {value!r} confirmed by a person"
                                                 + (f" ({note})" if note else "")))
    return ev
