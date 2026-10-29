# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/config_import.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    One-off, read-only import of identity profiles from the
                KPaperFlux QSettings INI file (~/.config/kpaperflux/
                kpaperflux.conf). Parses the QSettings string quoting without
                any Qt dependency.
------------------------------------------------------------------------------
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

from pydantic import ValidationError

from fluxsense.log import get_logger
from fluxsense.models.identity import IdentityProfile

logger = get_logger("config.import")

KPAPERFLUX_CONF: Path = Path.home() / ".config" / "kpaperflux" / "kpaperflux.conf"

_ESCAPES: Dict[str, str] = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "a": "\a", "b": "\b", "f": "\f", "v": "\v"}


def unquote_qsettings_value(raw: str) -> str:
    """
    Decodes a QSettings INI value (optionally double-quoted, C-style escapes,
    \\x{HHHH} unicode escapes).

    Args:
        raw: The raw value right of '='.

    Returns:
        The decoded string.
    """
    value = raw.strip()
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    out = []
    i = 0
    while i < len(value):
        char = value[i]
        if char == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            if nxt == "x":
                match = re.match(r"x([0-9a-fA-F]{1,4})", value[i + 1:])
                if match:
                    out.append(chr(int(match.group(1), 16)))
                    i += 1 + len(match.group(0))
                    continue
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
        out.append(char)
        i += 1
    return "".join(out)


def read_ini_section(path: Path, section: str) -> Dict[str, str]:
    """
    Reads one section of a QSettings INI file (read-only).

    Args:
        path: INI file.
        section: Section name without brackets.

    Returns:
        Key -> decoded value.
    """
    result: Dict[str, str] = {}
    current: Optional[str] = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
            continue
        if current != section or "=" not in line:
            continue
        key, raw = line.split("=", 1)
        result[key.strip()] = unquote_qsettings_value(raw)
    return result


def _parse_profile(raw: Optional[str], label: str) -> Optional[IdentityProfile]:
    """Parses one JSON profile string, returning None if empty or invalid."""
    if not raw or not raw.strip() or raw.strip() == "{}":
        return None
    try:
        return IdentityProfile.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"KPaperFlux {label} profile could not be parsed: {e}")
        return None


def import_kpaperflux_identities(path: Path = KPAPERFLUX_CONF) -> Tuple[Optional[IdentityProfile], Optional[IdentityProfile]]:
    """
    Reads the private and business identity profiles from KPaperFlux.

    Args:
        path: KPaperFlux config file.

    Returns:
        Tuple (private, business); entries are None if not configured.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    identity = read_ini_section(path, "Identity")
    return (
        _parse_profile(identity.get("private_profile"), "private"),
        _parse_profile(identity.get("business_profile"), "business"),
    )
