# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/llm/json_repair.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Provider-independent, tiered JSON parsing and repair for LLM
                responses (think blocks, code fences, trailing garbage,
                trailing/missing commas, truncated output). Ported from the
                KPaperFlux 4-step repair (gemini_provider.py).
------------------------------------------------------------------------------
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from fluxsense.log import get_logger

logger = get_logger("llm.json_repair")

_OPENERS: str = "{["
_CLOSER_FOR: Dict[str, str] = {"{": "}", "[": "]"}
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """
    Removes reasoning blocks (<think>...</think>) emitted by Qwen-style chat
    templates; an unterminated block at the start is dropped up to the first
    JSON opener.

    Args:
        text: Raw model output.

    Returns:
        The text without reasoning blocks.
    """
    cleaned = _THINK_RE.sub("", text)
    lowered = cleaned.lower()
    if "<think>" in lowered:
        start = lowered.find("<think>")
        rest = cleaned[start + len("<think>"):]
        positions = [p for p in (rest.find("{"), rest.find("[")) if p != -1]
        cleaned = cleaned[:start] + (rest[min(positions):] if positions else "")
    return cleaned.replace("</think>", "")


def _try_load(candidate: Optional[str]) -> Optional[Any]:
    """
    Attempts to decode a JSON candidate string.

    Args:
        candidate: The string to decode.

    Returns:
        The decoded object, or None if decoding failed or yielded null.
    """
    if not candidate:
        return None
    try:
        return json.loads(candidate, strict=False)
    except json.JSONDecodeError as e:
        logger.debug(f"JSON candidate rejected: {e}")
        return None


def strip_code_fences(text: str) -> str:
    """
    Removes a surrounding Markdown code fence (```json ... ```) if present.

    Args:
        text: Raw model output.

    Returns:
        The fenced content, or the original text if no fence is found.
    """
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    return text


def _find_start(text: str) -> int:
    """Returns the index of the first JSON container opener, or -1."""
    positions = [p for p in (text.find(c) for c in _OPENERS) if p != -1]
    return min(positions) if positions else -1


def _balanced_end(text: str, start: int) -> int:
    """
    Finds the index of the bracket closing the container opened at ``start``.
    The scan is string-aware (brackets inside JSON strings are ignored).

    Args:
        text: The full text.
        start: Index of the opening bracket.

    Returns:
        Index of the matching closer, or -1 if the container is never closed.
    """
    stack: List[str] = []
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            stack.append(_CLOSER_FOR[char])
        elif char in "}]":
            if not stack or stack[-1] != char:
                return -1
            stack.pop()
            if not stack:
                return idx
    return -1


def heuristic_repair(text: str) -> str:
    """
    Fixes common syntactic LLM mistakes.

    Args:
        text: A JSON candidate string.

    Returns:
        The repaired candidate.
    """
    text = re.sub(r",\s*([\]}])", r"\1", text)      # Trailing commas
    text = re.sub(r"}\s*\n\s*\"", "},\n\"", text)    # Missing comma between object and key
    text = re.sub(r"\]\s*\n\s*\"", "],\n\"", text)   # Missing comma between array and key
    text = re.sub(r"}\s*\n\s*{", "},\n{", text)      # Missing comma between list objects
    return text


def close_truncated(text: str) -> str:
    """
    Closes a truncated JSON document: terminates an open string, drops a
    dangling comma/colon and appends the missing closing brackets.

    Args:
        text: A (possibly truncated) JSON candidate starting with an opener.

    Returns:
        The closed candidate (unchanged if nothing was open).
    """
    stack: List[str] = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _OPENERS:
            stack.append(_CLOSER_FOR[char])
        elif char in "}]" and stack and stack[-1] == char:
            stack.pop()

    if not stack and not in_string:
        return text

    result = text
    if in_string:
        if escaped:
            result = result[:-1]
        result += '"'
    result = re.sub(r"[,:\s]+$", "", result)
    # A dangling key without value ("key") inside an object is invalid; add null.
    if stack and stack[-1] == "}" and re.search(r'[{,]\s*"[^"]*"$', result):
        result += ": null"
    return result + "".join(reversed(stack))


def parse_json_response(text: Optional[str]) -> Tuple[Optional[Any], Optional[str]]:
    """
    Parses an LLM response into a JSON object/list using tiered repair.

    Tiers:
        0. Direct parse of the (fence-stripped) text.
        1. Slice from the first opener to the last matching closer.
        2. Balanced-bracket slice (drops trailing garbage after the document).
        3. Heuristic syntax repair (trailing/missing commas).
        4. Truncation fix (close open strings and containers).

    Args:
        text: Raw model output.

    Returns:
        Tuple of (parsed object or None, error message or None).
    """
    if not text or not text.strip():
        return None, "Empty response"

    cleaned = strip_code_fences(strip_think(text.replace("\x00", ""))).strip()

    direct = _try_load(cleaned)
    if isinstance(direct, (dict, list)):
        return direct, None

    start = _find_start(cleaned)
    if start == -1:
        return None, "No JSON object found (missing '{')"

    closer = _CLOSER_FOR[cleaned[start]]
    candidates: List[str] = []

    last = cleaned.rfind(closer)
    if last > start:
        candidates.append(cleaned[start:last + 1])

    balanced = _balanced_end(cleaned, start)
    if balanced != -1:
        candidates.append(cleaned[start:balanced + 1])

    base = candidates[-1] if candidates else cleaned[start:]
    candidates.append(heuristic_repair(base))
    candidates.append(close_truncated(heuristic_repair(cleaned[start:])))
    candidates.append(heuristic_repair(close_truncated(cleaned[start:])))

    for candidate in candidates:
        result = _try_load(candidate)
        if isinstance(result, (dict, list)):
            return result, None

    return None, "JSON Syntax Error after repair attempts"
