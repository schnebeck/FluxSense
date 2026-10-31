# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/catalog/registry.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Document type catalog: one declarative TypeSpec per type
                (body schema, type-specific questions, tables, critical
                fields, plausibility rules, audit depth). Extensible at
                runtime and via a TOML file with custom types.
------------------------------------------------------------------------------
"""

import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

from fluxsense.log import get_logger
from fluxsense.models.semantic import BODY_MODELS

logger = get_logger("catalog")

AUDIT_FULL: str = "FULL_AUDIT"
AUDIT_STAMP: str = "STAMP_ONLY"
AUDIT_NONE: str = "NONE"
AUDIT_LEVEL: Dict[str, int] = {AUDIT_NONE: 0, AUDIT_STAMP: 1, AUDIT_FULL: 2}


@dataclass(frozen=True)
class TableSpec:
    """
    A logical table inside a body.

    Attributes:
        name: Table name in the output (e.g. "transactions").
        body_key: Body containing the list.
        field_name: List field inside the body.
        columns: Row fields requested from the model, in tuple order. Rows are
            sent as compact JSON arrays (prefixItems) to save output tokens.
        amount_fields: Row fields holding money amounts (grounding, checks).
        date_fields: Row fields holding dates.
    """

    name: str
    body_key: str
    field_name: str
    columns: Tuple[str, ...] = ()
    amount_fields: Tuple[str, ...] = ()
    date_fields: Tuple[str, ...] = ()

    def column_hint(self) -> str:
        """Human-readable tuple layout for prompts."""
        return "[" + ", ".join(self.columns) + "]"

    @property
    def path(self) -> str:
        """Dotted path inside the fields object."""
        return f"bodies.{self.body_key}.{self.field_name}"


@dataclass(frozen=True)
class TypeSpec:
    """
    Definition of one document type.

    Attributes:
        name: Type name (DocType value or custom).
        description: One-line description for prompts.
        body_key: Primary body key.
        extra_bodies: Further bodies extracted with this type.
        questions: Type-specific questions the extraction must answer.
        tables: Logical tables of this type.
        rules: Plausibility rule ids applied to this type.
        critical_fields: Path patterns ('*' = any segment) of critical values.
        required_fields: Paths that must be filled (completeness rule).
        audit_mode: Visual audit depth (stamps/signatures).
        zugferd: True if an embedded ZUGFeRD XML is authoritative for this type.
        kpf_type: KPaperFlux DocType used in the output (None = same name).
        doc_number_rule: What meta_header.doc_number means for this type (prompt text).
        doc_number_from: Field path whose value is the doc_number (e.g. the statement number).
        doc_number_labels: Regexes; a doc_number must be printed next to one of them (else removed).
        doc_date_labels: Regexes; for enclosures without letterhead a doc_date must be labelled (else removed).
        due_date_labels: Regexes; a due/payment date must be printed next to one of them (else removed).
        fold_audit: False for types whose extraction degrades when the visual audit is folded into it (statements,
            payslips): they get no stamp-audit floor and keep the text-layer stamps only.
    """

    name: str
    description: str
    body_key: Optional[str]
    extra_bodies: Tuple[str, ...] = ()
    questions: Tuple[str, ...] = ()
    tables: Tuple[TableSpec, ...] = ()
    rules: Tuple[str, ...] = ()
    critical_fields: Tuple[str, ...] = ()
    required_fields: Tuple[str, ...] = ()
    audit_mode: str = AUDIT_NONE
    zugferd: bool = False
    kpf_type: Optional[str] = None
    doc_number_rule: str = ""
    doc_number_from: Optional[str] = None
    doc_number_labels: Tuple[str, ...] = ()
    doc_date_labels: Tuple[str, ...] = ()
    due_date_labels: Tuple[str, ...] = ()
    fold_audit: bool = True

    @property
    def output_type(self) -> str:
        """Type name in KPaperFlux's DocType vocabulary."""
        return self.kpf_type or self.name

    def body_keys(self) -> List[str]:
        """Primary plus extra body keys."""
        keys = [self.body_key] if self.body_key else []
        return keys + [k for k in self.extra_bodies if k not in keys]


class Catalog:
    """Registry of TypeSpecs."""

    def __init__(self) -> None:
        """Creates an empty catalog."""
        self._types: Dict[str, TypeSpec] = {}

    def register(self, spec: TypeSpec) -> None:
        """Adds or replaces a type."""
        for key in spec.body_keys():
            if key not in BODY_MODELS:
                raise ValueError(f"Type {spec.name}: unknown body '{key}'")
        self._types[spec.name.upper()] = spec

    def get(self, name: str) -> TypeSpec:
        """Returns the spec for a type name (falls back to OTHER)."""
        return self._types.get(str(name).upper().split(".")[-1]) or self._types["OTHER"]

    def has(self, name: str) -> bool:
        """True if the type is registered."""
        return str(name).upper() in self._types

    def names(self) -> List[str]:
        """All registered type names."""
        return list(self._types.keys())

    def specs_for(self, type_tags: List[str]) -> List[TypeSpec]:
        """Specs for the given type tags (known types only, OTHER if none)."""
        specs = [self._types[t.upper()] for t in type_tags if t.upper() in self._types]
        return specs or [self._types["OTHER"]]

    def output_tags(self, type_tags: List[str]) -> List[str]:
        """Maps type tags to the KPaperFlux vocabulary (order kept, duplicates removed)."""
        mapped = [self.get(t).output_type if self.has(t) else t for t in type_tags]
        return list(dict.fromkeys(mapped))

    def audit_mode(self, type_tags: List[str], floor: str = AUDIT_NONE) -> str:
        """
        Highest audit mode over all tags.

        Args:
            type_tags: Document types.
            floor: Minimum mode for types that allow a folded audit (e.g. STAMP_ONLY: stamps and seals on every type).

        Returns:
            The audit mode.
        """
        best = AUDIT_NONE
        for spec in self.specs_for(type_tags):
            mode = spec.audit_mode
            if spec.fold_audit and AUDIT_LEVEL[floor] > AUDIT_LEVEL[mode]:
                mode = floor
            if AUDIT_LEVEL[mode] > AUDIT_LEVEL[best]:
                best = mode
        return best


_PY_TYPES: Dict[str, Any] = {"string": str, "number": Decimal, "integer": int, "boolean": bool, "date": str}


def _dynamic_model(name: str, fields: Dict[str, Dict[str, Any]]) -> Type[BaseModel]:
    """Builds a pydantic body model from a TOML field table."""
    definitions: Dict[str, Any] = {}
    for field_name, spec in fields.items():
        py_type = _PY_TYPES.get(str(spec.get("type", "string")), str)
        definitions[field_name] = (Optional[py_type], Field(None, description=str(spec.get("description", ""))))
    return create_model(name, __config__=ConfigDict(extra="forbid"), **definitions)


def load_custom_types(catalog: Catalog, path: Path) -> int:
    """
    Loads custom types from TOML:

        [[types]]
        name = "LEASING_CONTRACT"
        description = "Vehicle leasing contract"
        body_key = "leasing_body"            # new body (needs [types.fields]) or existing one
        questions = ["What is the monthly rate?"]
        rules = ["completeness", "dates"]
        required_fields = ["bodies.leasing_body.monthly_rate"]
        [types.fields]
        monthly_rate = { type = "number", description = "Monthly leasing rate" }

    Args:
        catalog: Catalog to extend.
        path: TOML file.

    Returns:
        Number of registered types.
    """
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    count = 0
    for entry in data.get("types", []):
        body_key = entry.get("body_key")
        if entry.get("fields"):
            model_name = "".join(part.title() for part in str(body_key).split("_"))
            BODY_MODELS[body_key] = _dynamic_model(model_name, entry["fields"])
        catalog.register(TypeSpec(
            name=str(entry["name"]).upper(),
            description=str(entry.get("description", "")),
            body_key=body_key,
            questions=tuple(entry.get("questions", [])),
            rules=tuple(entry.get("rules", ["completeness", "dates"])),
            critical_fields=tuple(entry.get("critical_fields", [])),
            required_fields=tuple(entry.get("required_fields", [])),
            audit_mode=str(entry.get("audit_mode", AUDIT_NONE)),
        ))
        count += 1
    logger.info(f"Loaded {count} custom types from {path}")
    return count
