# Contributing

Thanks for your interest in FluxSense.

- Open an issue before larger changes, so the design can be discussed first.
- Code: Python 3.12, type hints on every function, English names and
  comments, f-strings, `pathlib`, no bare or silent `except` (the quality
  gate `tests/test_code_quality.py` enforces this), the file header block
  of the existing modules with SPDX lines.
- Tests: `pip install -e .[dev]` and `pytest`. Unit tests must not need a
  network or a model server; use the fakes in `tests/conftest.py`.
  Live tests (`-m live`) are optional and skipped by default.
- Test data must be synthetic (fictional names, IBANs from the public test
  ranges, no real documents).
- Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, ...).
- By contributing you agree that your contribution is licensed under
  GPL-3.0-or-later.
