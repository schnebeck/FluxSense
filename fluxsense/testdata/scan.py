# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/testdata/scan.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    "Scan" variants of generated PDFs: pages rasterised at
                150-300 dpi, slightly skewed, speckled and blurred, without
                a text layer; optional synthetic stamp and handwriting-like
                overlays. Deterministic per random source (noise comes from
                a seeded tile, not from the C library's generator).
------------------------------------------------------------------------------
"""

import io
import math
import random
from pathlib import Path
from typing import Optional

import pymupdf
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from fluxsense.pdflock import PYMUPDF_LOCK


def _noise_tile(rng: random.Random, size: int = 256) -> Image.Image:
    """A grey speckle tile (values 215-255) from the seeded random source."""
    raw = bytes(255 - (b % 41) if b < 24 else 255 - (b % 6) for b in rng.randbytes(size * size))
    return Image.frombytes("L", (size, size), raw)


def _speckle(image: Image.Image, rng: random.Random) -> Image.Image:
    tile = _noise_tile(rng)
    noise = Image.new("L", image.size, 255)
    for x in range(0, image.size[0], tile.size[0]):
        for y in range(0, image.size[1], tile.size[1]):
            noise.paste(tile, (x, y))
    return ImageChops.multiply(image, Image.merge("RGB", (noise, noise, noise)))


def stamp(image: Image.Image, text: str, rng: random.Random) -> Image.Image:
    """Overlays a rotated rubber-stamp box (synthetic, red) between the info block and the heading."""
    scale = image.size[0] / 595.0
    box = Image.new("RGBA", (int(170 * scale), int(48 * scale)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(box)
    colour = (190, 30, 40, 200)
    draw.rectangle((2, 2, box.size[0] - 3, box.size[1] - 3), outline=colour, width=max(2, int(2 * scale)))
    font = ImageFont.load_default(size=max(10, int(15 * scale)))
    draw.text((int(10 * scale), int(14 * scale)), text, fill=colour, font=font)
    box = box.rotate(rng.uniform(-12, 12), expand=True, resample=Image.Resampling.BICUBIC)
    position = (int(image.size[0] * rng.uniform(0.55, 0.62)), int(image.size[1] * rng.uniform(0.235, 0.26)))
    image.paste(box, position, box)
    return image


def handwriting(image: Image.Image, rng: random.Random) -> Image.Image:
    """Overlays a handwriting-like scribble (a synthetic wavy pen line with a tick) in the lower margin."""
    scale = image.size[0] / 595.0
    draw = ImageDraw.Draw(image)
    x0, y0 = 70 * scale, image.size[1] * rng.uniform(0.82, 0.9)
    points = []
    for i in range(60):
        t = i / 59
        points.append((x0 + t * 180 * scale, y0 + math.sin(t * 18 + rng.random()) * 5 * scale + rng.uniform(-1, 1) * scale))
    draw.line(points, fill=(20, 40, 140), width=max(2, int(1.4 * scale)), joint="curve")
    draw.line([(x0 + 200 * scale, y0), (x0 + 207 * scale, y0 + 8 * scale), (x0 + 222 * scale, y0 - 12 * scale)],
              fill=(20, 40, 140), width=max(2, int(1.6 * scale)))
    return image


def make_scan(source: Path, target: Path, rng: random.Random, stamp_text: Optional[str] = None,
              handwritten: bool = False) -> Path:
    """
    Writes an image-only "scan" of a PDF.

    Args:
        source: Digital PDF.
        target: Output PDF (no text layer).
        rng: Random source for dpi, skew, noise and overlay positions.
        stamp_text: Stamp overlay text on page 1 (None: no stamp).
        handwritten: Handwriting-like overlay on page 1.

    Returns:
        The target path.
    """
    dpi = rng.choice((150, 200, 300))
    with PYMUPDF_LOCK, pymupdf.open(source) as src, pymupdf.open() as out:
        for index, page in enumerate(src):
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            if index == 0 and stamp_text:
                image = stamp(image, stamp_text, rng)
            if index == 0 and handwritten:
                image = handwriting(image, rng)
            image = image.rotate(rng.uniform(-0.9, 0.9), resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))
            image = _speckle(image, rng).filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
            buffer = io.BytesIO()
            coloured = index == 0 and (stamp_text or handwritten)
            (image if coloured else image.convert("L")).save(buffer, format="JPEG", quality=82)
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=buffer.getvalue())
        for name in src.embfile_names():
            out.embfile_add(name, src.embfile_get(name), filename=name)
        out.set_metadata({"producer": "FluxSense test-document generator (scan variant)"})
        out.save(target, garbage=3, deflate=True)
    return target
