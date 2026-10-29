# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/llm/images.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Page rendering and image normalization for vision requests:
                render PDF pages with PyMuPDF, downscale to a maximum long
                side, encode as PNG/JPEG data URIs.
------------------------------------------------------------------------------
"""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import pymupdf as fitz
from PIL import Image

from fluxsense.pdflock import PYMUPDF_LOCK
from fluxsense.log import get_logger

logger = get_logger("llm.images")

DEFAULT_RENDER_DPI: int = 200


@dataclass(frozen=True)
class ImagePart:
    """An image to send to a vision model, with a context label."""

    label: str
    data_uri: str
    width: int
    height: int


def fit_image(img: Image.Image, max_px: int) -> Image.Image:
    """
    Downscales an image so that its long side is at most ``max_px``.

    Args:
        img: Source image.
        max_px: Maximum long side in pixels (<= 0 disables scaling).

    Returns:
        The (possibly) resized RGB/L image.
    """
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    long_side = max(img.size)
    if max_px > 0 and long_side > max_px:
        scale = max_px / long_side
        new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def encode_data_uri(img: Image.Image, image_format: str = "png", quality: int = 90) -> str:
    """
    Encodes an image as base64 data URI.

    Args:
        img: Image (already fitted).
        image_format: "png" or "jpeg".
        quality: JPEG quality.

    Returns:
        A data URI string.
    """
    buffer = io.BytesIO()
    fmt = image_format.lower()
    if fmt in ("jpg", "jpeg"):
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        img.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def render_page(pdf_path: Path, page_index: int, dpi: int = DEFAULT_RENDER_DPI) -> Image.Image:
    """
    Renders one PDF page (0-based) to a PIL image.

    Args:
        pdf_path: PDF file (opened read-only).
        page_index: 0-based page index.
        dpi: Render resolution.

    Returns:
        The rendered image.

    Raises:
        IndexError: If the page does not exist.
    """
    with PYMUPDF_LOCK, fitz.open(pdf_path) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"Page {page_index + 1} out of range (1..{doc.page_count})")
        png = doc.load_page(page_index).get_pixmap(dpi=dpi).tobytes("png")
    return Image.open(io.BytesIO(png)).convert("RGB")


def page_images(pdf_path: Path, pages: Sequence[int], labels: Sequence[str], max_px: int,
                image_format: str = "png", dpi: int = DEFAULT_RENDER_DPI) -> List[ImagePart]:
    """
    Renders 1-based pages and returns fitted, encoded image parts.

    Args:
        pdf_path: PDF file.
        pages: 1-based page numbers.
        labels: One label per page.
        max_px: Maximum long side in pixels.
        image_format: "png" or "jpeg".
        dpi: Render resolution (before downscaling).

    Returns:
        List of ImagePart (pages that fail to render are skipped with a warning).
    """
    parts: List[ImagePart] = []
    for page, label in zip(pages, labels):
        try:
            img = fit_image(render_page(pdf_path, page - 1, dpi), max_px)
        except (IndexError, RuntimeError, ValueError) as e:
            logger.warning(f"Rendering page {page} of {pdf_path.name} failed: {e}")
            continue
        parts.append(ImagePart(label=label, data_uri=encode_data_uri(img, image_format), width=img.width, height=img.height))
    return parts
