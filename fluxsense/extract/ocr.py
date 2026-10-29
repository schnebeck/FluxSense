# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/extract/ocr.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    ocrmypdf fallback. Always works on a temporary copy; the
                input file is never modified (unlike KPaperFlux, which
                replaced the vault file with the sandwich PDF).
------------------------------------------------------------------------------
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

from fluxsense.config import ExtractSettings
from fluxsense.extract.base import SOURCE_OCR, PageText
from fluxsense.extract.native import read_pages
from fluxsense.log import get_logger

logger = get_logger("extract.ocr")


class OcrError(Exception):
    """Raised when ocrmypdf fails."""


def build_command(settings: ExtractSettings, source: Path, target: Path) -> List[str]:
    """
    Builds the ocrmypdf command line (options ported from KPaperFlux).

    Args:
        settings: Extraction settings.
        source: Temporary input copy.
        target: Output PDF in the temp dir.

    Returns:
        The argv list.
    """
    return [
        settings.ocrmypdf_binary,
        "--force-ocr",
        "--rotate-pages",
        "--deskew",
        "--jobs", str(settings.ocr_jobs),
        "--optimize", "0",
        "--output-type", "pdf",
        "-l", "+".join(settings.ocr_languages),
        str(source),
        str(target),
    ]


def run_ocrmypdf(pdf_path: Path, settings: ExtractSettings) -> List[PageText]:
    """
    OCRs a copy of the PDF and reads text/words/tables from the result.

    Args:
        pdf_path: Original PDF (read only).
        settings: Extraction settings.

    Returns:
        One PageText per page with source 'ocr'.

    Raises:
        OcrError: If ocrmypdf is missing or fails.
    """
    with tempfile.TemporaryDirectory(prefix="fluxsense-ocr-") as tmp:
        tmp_dir = Path(tmp)
        source = tmp_dir / "input.pdf"
        target = tmp_dir / "ocr.pdf"
        shutil.copyfile(pdf_path, source)
        cmd = build_command(settings, source, target)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=settings.ocr_timeout_s, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise OcrError(f"ocrmypdf could not run: {e}") from e
        if proc.returncode != 0 or not target.exists():
            raise OcrError(f"ocrmypdf failed (rc={proc.returncode}): {proc.stderr.strip()[-500:]}")
        return read_pages(target, SOURCE_OCR, with_tables=True)
