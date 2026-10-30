# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/text/dictionary.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Small spelling-dictionary wrapper for the OCR word
                correction: hunspell dictionaries (de_DE, en_US, ...) through
                the 'hunspell' or 'enchant-2' command in ispell pipe mode
                ('-a'), one process per batch of words. Without a usable
                command or dictionary the wrapper is unavailable and the word
                correction is skipped - nothing fails.
------------------------------------------------------------------------------
"""

import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from fluxsense.log import get_logger

logger = get_logger("text.dictionary")

COMMANDS: Tuple[str, ...] = ("hunspell", "enchant-2")
TIMEOUT_S: float = 30.0


class WordDictionary(Protocol):
    """What the word correction needs from a dictionary."""

    def lookup(self, words: Sequence[str]) -> Dict[str, Optional[List[str]]]:
        """word -> None if known, else its suggestions (possibly empty)."""


def parse_ispell(output: str, words: Sequence[str]) -> Dict[str, Optional[List[str]]]:
    """
    Parses ispell pipe-mode output for words sent one per line: after the
    version line, one block per input line (ended by an empty line) whose
    first result says '*' / '+' / '-' (known), '& word n off: suggestions'
    or '# word off' (unknown, no suggestions).
    """
    lines = output.splitlines()
    if lines and lines[0].startswith("@(#)"):
        lines = lines[1:]
    blocks: List[List[str]] = [[]]
    for line in lines:
        if line.strip():
            blocks[-1].append(line)
        elif blocks[-1]:
            blocks.append([])
    results: Dict[str, Optional[List[str]]] = {}
    for word, block in zip(words, [b for b in blocks if b]):
        tag = block[0][:1]
        if tag in ("*", "+", "-"):
            results[word] = None
        elif tag == "&":
            results[word] = [s.strip() for s in block[0].partition(":")[2].split(",") if s.strip()]
        elif tag == "#":
            results[word] = []
    return results


@dataclass
class Dictionary:
    """hunspell dictionaries through a spell-checker command (unavailable: lookups return nothing)."""

    languages: Tuple[str, ...] = ("de_DE", "en_US")
    command: Optional[str] = None
    _checked: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def system(cls, languages: Sequence[str]) -> "Dictionary":
        """The first installed spell-checker command with the given dictionaries."""
        command = next((c for c in COMMANDS if shutil.which(c)), None)
        return cls(languages=tuple(languages), command=command)

    @property
    def available(self) -> bool:
        """True if a command exists and at least one dictionary works."""
        return self.command is not None and any(self._usable(lang) for lang in self.languages)

    def _run(self, language: str, words: Sequence[str]) -> Optional[str]:
        if self.command is None:
            return None
        payload = "".join(f"^{w}\n" for w in words)
        try:
            done = subprocess.run([self.command, "-a", "-d", language], input=payload, capture_output=True, text=True,
                                  timeout=TIMEOUT_S, check=False)
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning(f"spell checker {self.command} failed: {e}")
            return None
        if done.returncode != 0:
            return None
        return done.stdout

    def _usable(self, language: str) -> bool:
        if language not in self._checked:
            self._checked[language] = self._run(language, ["Test"]) not in (None, "")
            if not self._checked[language]:
                logger.info(f"spelling dictionary {language} not available - word correction skipped for it")
        return self._checked[language]

    def lookup(self, words: Sequence[str]) -> Dict[str, Optional[List[str]]]:
        """
        word -> None if any dictionary knows it, else the suggestions of all
        dictionaries (an empty dict when no dictionary is available).
        """
        unique = list(dict.fromkeys(w for w in words if w))
        merged: Dict[str, Optional[List[str]]] = {}
        for language in self.languages:
            pending = [w for w in unique if merged.get(w, []) is not None]
            if not pending or not self._usable(language):
                continue
            output = self._run(language, pending)
            if output is None:
                continue
            for word, result in parse_ispell(output, pending).items():
                if result is None:
                    merged[word] = None
                elif merged.get(word) is not None or word not in merged:
                    merged[word] = list(dict.fromkeys((merged.get(word) or []) + result))
        return merged
