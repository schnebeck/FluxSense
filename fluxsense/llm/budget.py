# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/llm/budget.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Token budget for llama.cpp slots: conservative prompt-size
                estimator (Qwen tokenizers split numbers into single digits;
                Qwen3-VL needs ~(w/32)*(h/32) tokens per image) and graceful
                degradation (smaller images, fewer images, shorter auxiliary
                text) so that requests fit the per-slot context.
------------------------------------------------------------------------------
"""

import math
import re
from typing import Any, Callable, List, Optional, Sequence

from fluxsense.llm.base import LLMRequest
from fluxsense.llm.images import ImagePart

CHARS_PER_TOKEN: float = 3.2
MESSAGE_OVERHEAD: int = 16
IMAGE_OVERHEAD: int = 12
SAFETY_TOKENS: int = 512
TRUNCATION_NOTE: str = "\n[... auxiliary text truncated to fit the context ...]\n"
_DIGITS = re.compile(r"\d")
_SPACES = re.compile(r"[ \t]{2,}")

Rerender = Callable[[int], List[ImagePart]]


def text_tokens(text: str) -> int:
    """
    Estimates tokens of a text: every digit is one token, the rest ~3.2 chars per token.

    Args:
        text: Text.

    Returns:
        Estimated tokens.
    """
    if not text:
        return 0
    digits = len(_DIGITS.findall(text))
    rest = len(text) - digits
    return int(digits + rest / CHARS_PER_TOKEN) + 1


def image_tokens(width: int, height: int) -> int:
    """Qwen3-VL image tokens: one per 32x32 patch (after 2x2 merge of 16 px patches)."""
    return math.ceil(width / 32) * math.ceil(height / 32) + IMAGE_OVERHEAD


def estimate_request(request: LLMRequest) -> int:
    """
    Estimated prompt tokens of a request (system + user + follow-ups + images).

    Args:
        request: The request.

    Returns:
        Estimated tokens.
    """
    total = text_tokens(request.system or "") + text_tokens(request.prompt) + MESSAGE_OVERHEAD * (2 + len(request.followups))
    for turn in request.followups:
        content = turn.get("content")
        total += text_tokens(content if isinstance(content, str) else str(content))
    total += sum(image_tokens(img.width, img.height) for img in request.images)
    return total


def available(request: LLMRequest, context_tokens: int) -> int:
    """Prompt tokens available after reserving the output budget."""
    return context_tokens - (request.max_tokens or 0) - SAFETY_TOKENS


def compact_markdown(text: str) -> str:
    """Removes alignment padding of markdown tables (Docling pads cells with spaces)."""
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            line = _SPACES.sub(" ", line)
            if re.fullmatch(r"\s*\|[\s:|-]+\|\s*", line):
                line = "|---|"
        lines.append(line)
    return "\n".join(lines)


def strip_markdown_tables(text: str, placeholder: str = "[table: see PRE-PARSED TABLES]") -> str:
    """Replaces markdown table blocks by a placeholder (tables are sent once, stitched)."""
    out: List[str] = []
    in_table = False
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            if not in_table:
                out.append(placeholder)
                in_table = True
            continue
        in_table = False
        out.append(line)
    return "\n".join(out)


def truncate_prompt(request: LLMRequest, budget: int, tail: str = "") -> bool:
    """
    Truncates the user prompt so the request fits the budget.

    Args:
        request: Request (modified).
        budget: Available prompt tokens.
        tail: Text to keep at the end (closing instruction).

    Returns:
        True if the prompt was truncated.
    """
    overflow = estimate_request(request) - budget
    if overflow <= 0:
        return False
    ratio = max(0.05, 1.0 - overflow / max(1, text_tokens(request.prompt)) - 0.02)
    keep = max(1500, int(len(request.prompt) * ratio))
    request.prompt = request.prompt[:keep] + TRUNCATION_NOTE + tail
    return True


def _page_of(image: Any) -> Optional[int]:
    """Page number of an image part from its label ('PAGE 3'), None if it has none."""
    last = str(getattr(image, "label", "")).split()[-1:] or [""]
    return int(last[0]) if last[0].isdigit() else None


def fit(request: LLMRequest, context_tokens: int, rerender: Optional[Rerender] = None,
        sizes: tuple = (1024, 768), tail: str = "", protect: Sequence[int] = ()) -> List[str]:
    """
    Degrades a request until it fits: smaller images, then fewer images
    (the first kept; the last kept unless protected pages are given, which
    go only after all others; the image nearest the middle goes first),
    then shorter auxiliary text.

    Args:
        request: Request (modified in place).
        context_tokens: Context size per slot.
        rerender: Callback re-rendering the same pages at a given max_px.
        sizes: Image sizes to try.
        tail: Closing instruction kept when truncating.
        protect: Page numbers whose images are dropped only after all others (e.g. the totals page).

    Returns:
        Notes describing the degradation steps.
    """
    notes: List[str] = []
    budget = available(request, context_tokens)
    if estimate_request(request) <= budget:
        return notes
    if rerender is not None and request.images:
        current = max(max(i.width, i.height) for i in request.images)
        for size in sizes:
            if size >= current:
                continue
            request.images = rerender(size)
            notes.append(f"images at {size}px")
            if estimate_request(request) <= budget:
                return notes
    images = list(request.images)
    original = len(images)
    while estimate_request(request) > budget and len(images) > 2:
        middle = range(1, len(images) - 1)
        # with protected pages the last page may go too (an invoice's usage list), never the first one
        candidates = range(1, len(images)) if protect else middle
        free = [i for i in candidates if _page_of(images[i]) not in protect] or list(middle)
        images.pop(min(free, key=lambda i: abs(i - len(images) // 2)))
        request.images = images
    if len(images) < original:
        notes.append(f"images {original}->{len(images)}")
    if truncate_prompt(request, budget, tail):
        notes.append("aux text truncated")
    return notes


def shrink_for_retry(request: LLMRequest) -> None:
    """
    Emergency reduction after a server-side context overflow: halve the
    images (first and last kept) and cut the prompt to 60 %.

    Args:
        request: Request (modified).
    """
    images = list(request.images)
    if len(images) > 2:
        keep = max(2, len(images) // 2)
        request.images = images[: keep - 1] + images[-1:]
    elif len(images) == 2:
        request.images = images[:1]
    request.prompt = request.prompt[: int(len(request.prompt) * 0.6)] + TRUNCATION_NOTE
