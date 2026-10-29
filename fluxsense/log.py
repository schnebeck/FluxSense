# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/log.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Logging helpers (namespaced loggers, CLI setup).
------------------------------------------------------------------------------
"""

import logging
import sys

ROOT_LOGGER: str = "fluxsense"


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger below the FluxSense root logger.

    Args:
        name: Sub-logger name (e.g. "llm.openai").

    Returns:
        The logger instance.
    """
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def setup_logging(level: str = "INFO") -> None:
    """
    Configures stderr logging for CLI usage.

    Args:
        level: Log level name.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root = logging.getLogger(ROOT_LOGGER)
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
