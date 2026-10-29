# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/evidence.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Evidence wire schema. Builds, from the clean Pydantic body
                models, a compact JSON schema in which every copyable scalar
                is requested as {"v": value, "p": page, "e": "verbatim
                snippet"} and big tables use short row keys. unwrap() maps
                the wire answer back to the clean (KPaperFlux-compatible)
                structure plus a separate evidence list.
------------------------------------------------------------------------------
"""

import types
import typing
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from fluxsense.catalog.registry import TableSpec
from fluxsense.grounding.matcher import (KIND_AMOUNT, KIND_CODE, KIND_DATE, KIND_IBAN, KIND_IDENTIFIER, KIND_INTEGER,
                                         KIND_TEXT)
from fluxsense.models.result import FieldEvidence

CODE_FIELDS: frozenset = frozenset({
    "currency", "country", "language", "unit", "tax_category", "statement_type", "category", "frequency",
    "contract_type", "transaction_type", "anchor_type", "anchor_scope", "ref_type", "summary",
    "requested_action", "next_billing_date", "is_recurring", "source_page",
})
DATE_FIELDS: frozenset = frozenset({"service_period_start", "service_period_end", "period_start", "period_end",
                                    "valid_until", "deadline", "pay_period_start", "pay_period_end"})
IDENT_FIELDS: frozenset = frozenset({"bic", "vat_id", "tax_id", "pos", "zip_code", "house_number", "phone", "email",
                                     "card_number_masked", "reference", "tax_class", "ref_value", "article_number"})

ROW_PAGE_KEY: str = "_p"
# Kinds requested with evidence wrappers; names/addresses are plain values (grounded by fuzzy text search).
WRAP_KINDS: frozenset = frozenset({KIND_AMOUNT, KIND_DATE, KIND_IBAN, KIND_IDENTIFIER, KIND_INTEGER})

# EN 16931 business term names (shown to the model in the schema descriptions).
BT_LABELS: Dict[str, str] = {
    "BT-1": "Invoice number", "BT-2": "Invoice issue date", "BT-5": "Invoice currency code", "BT-7": "Tax point / service date",
    "BT-9": "Payment due date", "BT-10": "Buyer reference", "BT-11": "Project reference", "BT-13": "Purchase order reference",
    "BT-14": "Order date", "BT-19": "Buyer accounting reference", "BT-20": "Payment terms", "BT-46": "Buyer identifier / customer number",
    "BT-83": "Remittance information / payment reference", "BT-106": "Sum of invoice line net amounts",
    "BT-107": "Sum of charges", "BT-108": "Sum of allowances", "BT-109": "Invoice total without VAT (tax basis)",
    "BT-110": "Invoice total VAT amount", "BT-112": "Invoice total with VAT (grand total)", "BT-115": "Amount due for payment",
    "BT-116": "VAT category taxable amount", "BT-117": "VAT category tax amount", "BT-118": "VAT category code",
    "BT-119": "VAT category rate (percent)", "BT-126": "Invoice line identifier", "BT-129": "Invoiced quantity",
    "BT-130": "Unit of measure code", "BT-131": "Invoice line net amount", "BT-146": "Item net price",
    "BT-152": "Invoiced item VAT rate (percent)", "BT-153": "Item name", "BT-155": "Item seller identifier",
}


def field_kind(name: str, annotation: Any) -> str:
    """
    Infers the grounding kind of a field from its name and type.

    Args:
        name: Field name.
        annotation: Python type (Optional already removed).

    Returns:
        One of the KIND_* constants.
    """
    if name in CODE_FIELDS or annotation is bool:
        return KIND_CODE
    if annotation is Decimal or annotation is float:
        return KIND_AMOUNT
    if annotation is int:
        return KIND_INTEGER
    if "iban" in name:
        return KIND_IBAN
    if "date" in name or name in DATE_FIELDS:
        return KIND_DATE
    if name in IDENT_FIELDS or name.endswith(("_number", "_id", "_reference")):
        return KIND_IDENTIFIER
    return KIND_TEXT


def _strip_optional(annotation: Any) -> Any:
    """Removes Optional[...] / X | None."""
    origin = typing.get_origin(annotation)
    if origin in (Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return args[0] if len(args) == 1 else annotation
    return annotation


@dataclass
class Scalar:
    """Plan node for a scalar field."""

    kind: str
    wrapped: bool
    json_type: str
    description: str = ""
    enum: Optional[List[str]] = None


@dataclass
class Obj:
    """Plan node for an object."""

    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rows:
    """Plan node for a list of objects (table). Big tables use tuple rows."""

    item: Obj
    columns: Tuple[str, ...] = ()
    big_table: Optional[TableSpec] = None
    description: str = ""


@dataclass
class Primitives:
    """Plan node for a list of strings."""

    description: str = ""


def _describe(info: Any) -> str:
    """Field description including its BT alias."""
    parts = []
    alias = getattr(info, "alias", None)
    if alias and str(alias).startswith("BT-"):
        parts.append(f"{alias} {BT_LABELS.get(str(alias), '')}".strip())
    if getattr(info, "description", None):
        parts.append(str(info.description))
    return " ".join(parts)


def build_plan(model: Type[BaseModel], in_row: bool = False, tables: Optional[Dict[str, TableSpec]] = None) -> Obj:
    """
    Builds the wire plan of a Pydantic model.

    Args:
        model: Clean model class.
        in_row: True inside table rows (no evidence wrappers).
        tables: field name -> TableSpec for big tables of this model.

    Returns:
        The object plan.
    """
    plan = Obj()
    for name, info in model.model_fields.items():
        annotation = _strip_optional(info.annotation)
        origin = typing.get_origin(annotation)
        description = _describe(info)
        if origin in (list, List):
            (item_type,) = typing.get_args(annotation) or (str,)
            if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                spec = (tables or {}).get(name)
                item_plan = build_plan(item_type, in_row=True)
                columns: Tuple[str, ...] = ()
                if spec:
                    columns = tuple(c for c in spec.columns if c in item_plan.fields)
                    item_plan = Obj({c: item_plan.fields[c] for c in columns})
                plan.fields[name] = Rows(item=item_plan, columns=columns, big_table=spec,
                                         description=f"rows as arrays {spec.column_hint()}" if spec else description)
            else:
                plan.fields[name] = Primitives(description=description)
            continue
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            plan.fields[name] = build_plan(annotation, in_row=in_row)
            continue
        kind = field_kind(name, annotation)
        if annotation is bool:
            json_type = "boolean"
        elif annotation is int:
            json_type = "integer"
        elif annotation in (Decimal, float):
            json_type = "number"
        else:
            json_type = "string"
        plan.fields[name] = Scalar(kind=kind, wrapped=(not in_row and kind in WRAP_KINDS), json_type=json_type,
                                   description=description)
    return plan


# Caps only stop runaway generation; they are far above real values. A value that reaches its cap is treated as
# possibly truncated (completed from the text layer or flagged) - identifiers are never cut silently.
MAX_LENGTH_BY_KIND: Dict[str, int] = {KIND_IDENTIFIER: 96, KIND_IBAN: 48, KIND_DATE: 24, KIND_CODE: 80, KIND_TEXT: 400}


def _scalar_schema(node: Scalar) -> Dict[str, Any]:
    """JSON schema of a bare (nullable) scalar; strings are length-capped per kind (no runaway values)."""
    schema: Dict[str, Any] = {"type": [node.json_type, "null"]}
    if node.json_type == "string":
        schema["maxLength"] = MAX_LENGTH_BY_KIND.get(node.kind, 240)
    if node.enum:
        schema["enum"] = node.enum + [None]
    if node.description:
        schema["description"] = node.description
    return schema


def wire_schema(node: Any, evidence_chars: int = 40, ask_bbox: bool = False, with_page: bool = True,
                with_line: bool = False) -> Dict[str, Any]:
    """
    Converts a plan into the wire JSON schema.

    Args:
        node: Plan node.
        evidence_chars: maxLength of evidence snippets.
        ask_bbox: Also request a bounding box per value.
        with_page: Request page numbers (False for single-page documents).
        with_line: Request a pointer to the numbered text line ("l", pointer extraction).

    Returns:
        JSON schema dict.
    """
    if isinstance(node, Obj):
        return {"type": "object", "additionalProperties": False,
                "properties": {k: wire_schema(v, evidence_chars, ask_bbox, with_page, with_line) for k, v in node.fields.items()}}
    if isinstance(node, Rows):
        if node.columns:
            cells = []
            for column in node.columns:
                cell = _scalar_schema(node.item.fields[column])
                cell["description"] = f"{column} {cell.get('description', '')}".strip()
                cells.append(cell)
            schema = {"type": "array", "items": {"type": "array", "prefixItems": cells}}
            schema["description"] = node.description
            return schema
        item = wire_schema(node.item, evidence_chars, ask_bbox, with_page, with_line)
        if with_page:
            item["properties"][ROW_PAGE_KEY] = {"type": ["integer", "null"], "description": "page"}
        schema: Dict[str, Any] = {"type": "array", "items": item}
        if node.description:
            schema["description"] = node.description
        return schema
    if isinstance(node, Primitives):
        return {"type": "array", "items": {"type": "string"}}
    if isinstance(node, Scalar):
        if not node.wrapped:
            return _scalar_schema(node)
        props: Dict[str, Any] = {"v": _scalar_schema(node)}
        if with_page:
            props["p"] = {"type": ["integer", "null"]}
        if with_line:
            props["l"] = {"type": ["string", "null"], "maxLength": 12, "description": "line id, e.g. p1:L37"}
        props["e"] = {"type": ["string", "null"], "maxLength": evidence_chars}
        if ask_bbox:
            props["b"] = {"type": ["array", "null"], "items": {"type": "number"}, "maxItems": 4}
        return {"type": "object", "additionalProperties": False, "required": ["v"], "properties": props}
    raise TypeError(f"Unknown plan node {node!r}")


@dataclass
class Unwrapped:
    """Clean data plus the evidence layer."""

    data: Any
    evidence: List[FieldEvidence] = field(default_factory=list)
    table_rows: Dict[str, List[Tuple[int, Dict[str, Any]]]] = field(default_factory=dict)


def unwrap(node: Any, wire: Any, path: str, out: Unwrapped, evidence_chars: int = 60,
           row_page: Optional[int] = None) -> Any:
    """
    Maps a wire answer to clean data, collecting evidence.

    Args:
        node: Plan node.
        wire: Wire value.
        path: Dotted path of this node.
        out: Collector.
        evidence_chars: Max evidence length kept.
        row_page: Page of the enclosing table row.

    Returns:
        The clean value (None if absent).
    """
    if wire is None:
        return None
    if isinstance(node, Obj):
        if not isinstance(wire, dict):
            return None
        clean: Dict[str, Any] = {}
        for name, child in node.fields.items():
            if name not in wire:
                continue
            value = unwrap(child, wire[name], f"{path}.{name}" if path else name, out, evidence_chars, row_page)
            if value is not None and value != [] and value != {}:
                clean[name] = value
        return clean
    if isinstance(node, Rows):
        if not isinstance(wire, list):
            return None
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(wire):
            if node.columns and isinstance(item, list):
                item = {c: v for c, v in zip(node.columns, item)}
            if not isinstance(item, dict):
                continue
            page = item.get("source_page", item.get(ROW_PAGE_KEY))
            row_path = f"{path}.{len(rows)}"
            clean_row = unwrap(node.item, item, row_path, out if not node.big_table else Unwrapped(None),
                               evidence_chars, page if isinstance(page, int) else None)
            if not clean_row:
                continue
            rows.append(clean_row)
            if node.big_table:
                out.table_rows.setdefault(path, []).append((index, clean_row))
        return rows
    if isinstance(node, Primitives):
        return [str(v) for v in wire if v not in (None, "")] if isinstance(wire, list) else None
    if isinstance(node, Scalar):
        page = row_page
        snippet: Optional[str] = None
        bbox: Optional[List[float]] = None
        line: Optional[str] = None
        value = wire
        if isinstance(wire, dict):
            value = wire.get("v")
            page = wire.get("p") if isinstance(wire.get("p"), int) else row_page
            snippet = str(wire["e"])[:evidence_chars] if wire.get("e") else None
            line = str(wire["l"]) if wire.get("l") else None
            raw_bbox = wire.get("b")
            if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
                bbox = [float(x) for x in raw_bbox]
        if value is None or value == "":
            return None
        if node.kind != KIND_CODE:
            out.evidence.append(FieldEvidence(path=path, kind=node.kind, value=value, page=page,
                                              evidence=snippet, bbox=bbox, line=line))
        return value
    raise TypeError(f"Unknown plan node {node!r}")
