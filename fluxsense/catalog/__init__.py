# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/catalog/__init__.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Document type catalog (questions, schema, tables, rules).
------------------------------------------------------------------------------
"""

from fluxsense.catalog.builtin import default_catalog
from fluxsense.catalog.registry import Catalog, TableSpec, TypeSpec

__all__ = ["Catalog", "TableSpec", "TypeSpec", "default_catalog"]
