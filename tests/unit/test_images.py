# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_images.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for page rendering, downscaling and data-URI encoding.
------------------------------------------------------------------------------
"""

import base64
import io
from typing import Any, Dict

from PIL import Image

from fluxsense.llm.images import encode_data_uri, fit_image, page_images, render_page


def test_fit_image_downscales_long_side() -> None:
    """The long side is limited, the aspect ratio kept, small images untouched."""
    img = Image.new("RGB", (3000, 1500), "white")
    fitted = fit_image(img, 1000)
    assert fitted.size == (1000, 500)
    assert fit_image(Image.new("RGB", (400, 300)), 1000).size == (400, 300)
    assert fit_image(Image.new("RGBA", (10, 10)), 0).mode == "RGB"


def test_encode_data_uri_png_and_jpeg() -> None:
    """Data URIs carry the right MIME type and decode back to the image."""
    img = Image.new("RGB", (20, 10), "red")
    png = encode_data_uri(img, "png")
    assert png.startswith("data:image/png;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(png.split(",", 1)[1])))
    assert decoded.size == (20, 10)
    assert encode_data_uri(img, "jpeg").startswith("data:image/jpeg;base64,")


def test_page_images_render_and_label(samples: Dict[str, Any]) -> None:
    """Pages are rendered, fitted to max_px and labelled; bad pages skipped."""
    path = samples["bank_statement_3p"].path
    parts = page_images(path, [1, 3, 9], ["P1", "P3", "P9"], max_px=800)
    assert [p.label for p in parts] == ["P1", "P3"]
    assert max(parts[0].width, parts[0].height) == 800
    assert render_page(path, 0, dpi=72).size[0] > 0
