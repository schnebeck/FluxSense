# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/formats/spec.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    The format spec type of the registry: variants as
                per-position character classes ('9' digit, 'A' letter, 'X'
                letter or digit, 'C' digit or check 'X', '\\c' a literal
                character), a checksum / semantic check, a normaliser and
                the OCR confusion map. A spec says whether a value has the
                shape (matches) and is valid (valid), and proposes repairs:
                forced ones where a character violates its position class
                (a letter at a digit position) and free single confusions
                or transpositions checked by the checksum. Formats supplied
                by an application (e.g. a sender's customer-number pattern)
                use the same type.
------------------------------------------------------------------------------
"""

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

DIGIT, LETTER, ALNUM, CHECK = "9", "A", "X", "C"

# OCR confusions between letters and digits (both directions: the position class decides which way is forced).
LETTER_DIGIT: Tuple[Tuple[str, str], ...] = (("O", "0"), ("Q", "0"), ("D", "0"), ("I", "1"), ("L", "1"), ("S", "5"),
                                              ("B", "8"), ("Z", "2"), ("G", "6"), ("T", "7"), ("A", "4"))
# Visual confusions among digits (only free repairs, never forced: the checksum must decide).
DIGIT_DIGIT: Tuple[Tuple[str, str], ...] = (("0", "6"), ("0", "8"), ("6", "5"), ("6", "8"), ("8", "3"), ("1", "7"), ("3", "8"),
                                            ("5", "6"))


def confusion_map(pairs: Sequence[Tuple[str, str]], both: bool = True) -> Dict[str, str]:
    """Character -> confusable characters of the given pairs (both directions, or first -> second only)."""
    out: Dict[str, str] = {}
    for a, b in pairs:
        out[a] = out.get(a, "") + b
        if both:
            out[b] = out.get(b, "") + a
    return out


OCR_CONFUSIONS: Dict[str, str] = confusion_map(LETTER_DIGIT)


def compact_upper(value: object) -> str:
    """Uppercase letters and digits only (spaces, dots, dashes dropped)."""
    from fluxsense.grounding.normalize import compact
    return compact(str(value or ""))


def parse_pattern(pattern: str) -> List[Tuple[str, str]]:
    """Position classes of a pattern: ('class', '9'|'A'|'X'|'C') or ('literal', char) for '\\c' and other characters."""
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\" and i + 1 < len(pattern):
            out.append(("literal", pattern[i + 1]))
            i += 2
            continue
        out.append(("class", char) if char in (DIGIT, LETTER, ALNUM, CHECK) else ("literal", char))
        i += 1
    return out


def fits(char: str, position: Tuple[str, str]) -> bool:
    """True if a character satisfies a position."""
    kind, value = position
    if kind == "literal":
        return char == value
    if value == DIGIT:
        return char.isdigit()
    if value == LETTER:
        return char.isalpha()
    if value == CHECK:
        return char.isdigit() or char == "X"
    return char.isalnum()


@dataclass(frozen=True)
class FormatSpec:
    """
    A fixed format.

    Attributes:
        name: Registry name ('iban', 'ean13', 'vat_id', ... or an application's own name).
        patterns: Variants as position-class patterns of the normalised value.
        checksum: Check of a normalised value that has a variant's shape (checksum, calendar, ranges); None = shape only.
        normaliser: Value -> normalised form (default: uppercase letters and digits).
        fallback: Shape check for values no variant covers (e.g. IBANs of countries without a BBAN pattern).
        confusions: Character -> confusable characters for forced repairs.
        description: Human-readable description.
    """

    name: str
    patterns: Tuple[str, ...]
    checksum: Optional[Callable[[str], bool]] = None
    normaliser: Callable[[object], str] = compact_upper
    fallback: Optional[Callable[[str], bool]] = None
    confusions: Mapping[str, str] = field(default_factory=lambda: OCR_CONFUSIONS)
    description: str = ""

    @classmethod
    def custom(cls, name: str, patterns: Sequence[str], checksum: Optional[Callable[[str], bool]] = None,
               normaliser: Optional[Callable[[object], str]] = None, description: str = "") -> "FormatSpec":
        """A format supplied by an application (per-call hint; nothing is stored)."""
        return cls(name=name, patterns=tuple(patterns), checksum=checksum, normaliser=normaliser or compact_upper,
                   description=description)

    def normalise(self, value: object) -> str:
        """The normalised value."""
        return self.normaliser(value)

    def _variants(self, length: int) -> List[List[Tuple[str, str]]]:
        return [parsed for parsed in (parse_pattern(p) for p in self.patterns) if len(parsed) == length]

    def shape(self, value: object) -> Optional[str]:
        """The variant pattern the normalised value has (None if none)."""
        text = self.normalise(value)
        for pattern in self.patterns:
            parsed = parse_pattern(pattern)
            if len(parsed) == len(text) and all(fits(c, p) for c, p in zip(text, parsed)):
                return pattern
        return None

    def matches(self, value: object) -> bool:
        """True if the value has the shape of a variant (or the fallback shape) - checksum not checked."""
        text = self.normalise(value)
        return bool(text) and (self.shape(text) is not None or bool(self.fallback and self.fallback(text)))

    def valid(self, value: object) -> bool:
        """True if the value has the shape and passes the checksum / semantic check."""
        text = self.normalise(value)
        return self.matches(text) and (self.checksum is None or self.checksum(text))

    def forced_positions(self, value: object) -> List[int]:
        """Positions that violate their class in every variant of the value's length (empty: shape fine or no variant)."""
        text = self.normalise(value)
        variants = self._variants(len(text))
        if not variants:
            return []
        per_variant = [[i for i, (c, p) in enumerate(zip(text, parsed)) if not fits(c, p)] for parsed in variants]
        return min(per_variant, key=len) if all(per_variant) else []

    def forced_repairs(self, value: object, limit: int = 256) -> List[str]:
        """
        Repairs forced by the position classes: every character that violates
        its class is replaced by a confusable character of the right class
        ('O' at a digit position -> '0'); the candidates that are valid.
        """
        text = self.normalise(value)
        out: List[str] = []
        for parsed in self._variants(len(text)):
            wrong = [i for i, (c, p) in enumerate(zip(text, parsed)) if not fits(c, p)]
            if not wrong:
                continue
            options = [[alt for alt in self.confusions.get(text[i], "") if fits(alt, parsed[i])] for i in wrong]
            if not all(options):
                continue
            for count, combo in enumerate(product(*options)):
                if count >= limit:
                    break
                chars = list(text)
                for i, alt in zip(wrong, combo):
                    chars[i] = alt
                candidate = "".join(chars)
                if candidate not in out and self.valid(candidate):
                    out.append(candidate)
        return out

    def free_repairs(self, value: object, confusions: Mapping[str, str], start: int = 0,
                     transpositions: bool = True) -> List[str]:
        """Valid values one confusion (or one adjacent transposition) away, in position order."""
        text = self.normalise(value)
        out: List[str] = []
        for i in range(start, len(text)):
            for alt in confusions.get(text[i], ""):
                candidate = text[:i] + alt + text[i + 1:]
                if candidate not in out and self.valid(candidate):
                    out.append(candidate)
            if transpositions and i + 1 < len(text) and text[i] != text[i + 1]:
                candidate = text[:i] + text[i + 1] + text[i] + text[i + 2:]
                if candidate not in out and self.valid(candidate):
                    out.append(candidate)
        return out

    def find(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Spans of a text that have one of the variants' shapes after
        normalisation, also with confusable characters at class positions
        ('DE44 99O1 ...'): (start, end, printed text). Separators inside a
        span (spaces, dots, dashes) are allowed only where the printed form
        has them.
        """
        found: List[Tuple[int, int, str]] = []
        for pattern in self.patterns:
            regex = _span_regex(parse_pattern(pattern), self.confusions)
            for match in regex.finditer(text):
                span = (match.start(), match.end(), match.group(0))
                if not any(s <= span[0] < e for s, e, _ in found):
                    found.append(span)
        return sorted(found)


def _span_regex(parsed: List[Tuple[str, str]], confusions: Mapping[str, str]) -> "re.Pattern[str]":
    """A regex for printed spans of a pattern: each position its class plus confusable characters, optional single
    separators between positions."""
    parts: List[str] = []
    for kind, value in parsed:
        if kind == "literal":
            parts.append(re.escape(value))
            continue
        allowed = {DIGIT: "0-9", LETTER: "A-ZÄÖÜa-zäöü", ALNUM: "A-Za-z0-9", CHECK: "0-9X"}[value]
        extra = "".join(sorted({c for c, alts in confusions.items() for a in alts
                                if fits(a, (kind, value)) and not fits(c, (kind, value))}))
        parts.append(f"[{allowed}{re.escape(extra)}]")
    return re.compile(r"(?<![A-Za-z0-9])" + r"[ .\-/]?".join(parts) + r"(?![A-Za-z0-9])")
