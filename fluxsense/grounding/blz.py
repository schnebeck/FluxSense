# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/grounding/blz.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    German bank codes (Bankleitzahlen) for IBAN plausibility.
                No list is bundled (the Bundesbank file has its own terms):
                `fluxsense blz-update` downloads it into the user data dir.
                Without a list the bank-code check is simply skipped.
------------------------------------------------------------------------------
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional

import httpx

from fluxsense.log import get_logger

logger = get_logger("grounding.blz")

USER_BLZ_FILE: Path = Path.home() / ".local" / "share" / "fluxsense" / "blz.txt"
BUNDESBANK_PAGE: str = ("https://www.bundesbank.de/de/aufgaben/unbarer-zahlungsverkehr/serviceangebot/bankleitzahlen/"
                        "download-bankleitzahlen-602592")


def _parse(text: str) -> FrozenSet[str]:
    """8-digit codes at line starts (compact list or the Bundesbank fixed-width file)."""
    return frozenset(m.group(1) for m in re.finditer(r"^(\d{8})", text, re.MULTILINE))


@lru_cache(maxsize=1)
def load_blz() -> FrozenSet[str]:
    """
    Loads the known German bank codes downloaded by `fluxsense blz-update`.

    Returns:
        Set of 8-digit BLZ (empty if unavailable -> check skipped).
    """
    if USER_BLZ_FILE.exists():
        return _parse(USER_BLZ_FILE.read_text(encoding="latin-1"))
    logger.debug("no BLZ list (run 'fluxsense blz-update'); bank-code check skipped")
    return frozenset()


def known_german_blz(iban: str) -> Optional[bool]:
    """True/False for German IBANs if a BLZ list is available, else None."""
    codes = load_blz()
    if not codes or not iban.startswith("DE") or len(iban) != 22:
        return None
    return iban[4:12] in codes


def update_blz(url: Optional[str] = None, target: Path = USER_BLZ_FILE) -> int:
    """
    Downloads the current Bundesbank BLZ file and stores the unique codes.

    Args:
        url: Direct file URL (default: discovered on the Bundesbank download page).
        target: Output file.

    Returns:
        Number of codes stored.

    Raises:
        httpx.HTTPError: On download errors.
        ValueError: If no file link or no codes were found.
    """
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        if url is None:
            page = client.get(BUNDESBANK_PAGE).text
            match = re.search(r'href="([^"]+blz-aktuell-txt-data\.txt)"', page)
            if not match:
                raise ValueError("BLZ download link not found on the Bundesbank page")
            url = "https://www.bundesbank.de" + match.group(1) if match.group(1).startswith("/") else match.group(1)
        codes = _parse(client.get(url).content.decode("latin-1"))
    if not codes:
        raise ValueError("no bank codes in the downloaded file")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(sorted(codes)) + "\n", encoding="utf-8")
    load_blz.cache_clear()
    return len(codes)
