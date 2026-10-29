# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/test_code_quality.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Quality gate (ported from KPaperFlux tests/test_code_quality.py): no silent or bare except handlers, mandatory header + SPDX line, no Qt imports, type hints on all functions.
------------------------------------------------------------------------------
"""

import ast
from pathlib import Path
from typing import List, Tuple

import pytest

ROOT = Path(__file__).parent.parent
SCAN_DIRS = ("fluxsense", "tools", "tests")
HEADER_KEYS = ("Project:        FluxSense", "Producer:       thorsten.schnebeck@gmx.net", "Generator:", "Description:")


def python_files() -> List[Path]:
    """All project Python files (hidden folders and the venv excluded)."""
    files: List[Path] = []
    for name in SCAN_DIRS:
        directory = ROOT / name
        if directory.exists():
            files += [p for p in directory.rglob("*.py") if not any(part.startswith(".") for part in p.parts)]
    return sorted(files)


class SilentFailureVisitor(ast.NodeVisitor):
    """Finds except handlers whose body is only 'pass' / '...', and bare excepts."""

    def __init__(self, filename: str) -> None:
        """Args: filename: Reported file name."""
        self.filename = filename
        self.violations: List[Tuple[str, int, str]] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Checks one handler."""
        silent = all(isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                                                    and stmt.value.value is Ellipsis) for stmt in node.body)
        if silent:
            self.violations.append((self.filename, node.lineno, "silent except"))
        if node.type is None:
            self.violations.append((self.filename, node.lineno, "bare except"))
        self.generic_visit(node)


def test_no_silent_or_bare_except() -> None:
    """Quality Gate: no 'except: pass' and no bare 'except:'."""
    violations: List[Tuple[str, int, str]] = []
    for path in python_files():
        visitor = SilentFailureVisitor(str(path.relative_to(ROOT)))
        visitor.visit(ast.parse(path.read_text(encoding="utf-8")))
        violations += visitor.violations
    if violations:
        pytest.fail("Quality Gate Failed:\n" + "\n".join(f"  - {f}:{line} -> {kind}" for f, line, kind in violations))


def test_file_headers_and_spdx() -> None:
    """Every file starts with the SPDX line and carries the FluxSense header block."""
    missing = []
    for path in python_files():
        text = path.read_text(encoding="utf-8")
        if not text.startswith("# SPDX-" + "License-Identifier: GPL-3.0-or-later"):  # split: no SPDX tag in this line
            missing.append(f"{path.relative_to(ROOT)}: SPDX line")
        head = text[:1500]
        missing += [f"{path.relative_to(ROOT)}: {key}" for key in HEADER_KEYS if key not in head]
    assert not missing, "Header violations:\n" + "\n".join(missing)


def test_no_qt_dependency() -> None:
    """FluxSense must not import Qt."""
    offenders = []
    for path in python_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            if any(n.split(".")[0] in ("PyQt5", "PyQt6", "PySide2", "PySide6") for n in names):
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_type_hints_on_package_functions() -> None:
    """Every function in the package has annotated arguments and a return annotation."""
    offenders = []
    for path in (ROOT / "fluxsense").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [a for a in node.args.args + node.args.kwonlyargs if a.arg not in ("self", "cls")]
                if node.returns is None or any(a.annotation is None for a in args):
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")
    assert not offenders, "Missing type hints:\n" + "\n".join(offenders)
