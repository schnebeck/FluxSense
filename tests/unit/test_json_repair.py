# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_json_repair.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for the tiered JSON repair (fences, think blocks, garbage, commas, truncation).
------------------------------------------------------------------------------
"""

from fluxsense.llm.json_repair import close_truncated, parse_json_response, strip_think


def test_direct_and_fenced() -> None:
    """Plain and fenced JSON parse directly."""
    assert parse_json_response('{"a": 1}') == ({"a": 1}, None)
    assert parse_json_response('```json\n{"a": [1, 2]}\n```')[0] == {"a": [1, 2]}


def test_think_blocks_removed() -> None:
    """Qwen reasoning blocks are stripped, also unterminated ones."""
    assert parse_json_response('<think>we need {x}</think>{"ok": true}')[0] == {"ok": True}
    assert parse_json_response('<think>long reasoning without end {"a": 1}')[0] == {"a": 1}
    assert strip_think("a<think>b</think>c") == "ac"


def test_trailing_garbage_and_prose() -> None:
    """Prose before and garbage after the document is ignored."""
    assert parse_json_response('Sure! {"a": {"b": 2}} hope this helps }')[0] == {"a": {"b": 2}}


def test_trailing_and_missing_commas() -> None:
    """Heuristic comma repair."""
    assert parse_json_response('{"a": [1, 2,], "b": 3,}')[0] == {"a": [1, 2], "b": 3}
    assert parse_json_response('{"a": {"x": 1}\n"b": 2}')[0] == {"a": {"x": 1}, "b": 2}


def test_truncated_output_closed() -> None:
    """Truncated output is closed (open string, dangling key, open containers)."""
    assert parse_json_response('{"rows": [[1, 2], [3, 4')[0] == {"rows": [[1, 2], [3, 4]]}
    assert parse_json_response('{"a": {"b": "tru')[0] == {"a": {"b": "tru"}}
    assert close_truncated('{"a": 1, "b"') == '{"a": 1, "b": null}'


def test_lists_and_errors() -> None:
    """Top-level lists work; non-JSON yields an error message."""
    assert parse_json_response("[1, 2]")[0] == [1, 2]
    assert parse_json_response("")[0] is None
    result, error = parse_json_response("no json here")
    assert result is None and "No JSON" in error
