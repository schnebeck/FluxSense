# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           tests/unit/test_config.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Tests for settings (TOML, env overrides, secrets), the read-only KPaperFlux identity import and the QSettings unquoting.
------------------------------------------------------------------------------
"""

import json
from pathlib import Path

import pytest

from fluxsense.config import Settings, load_settings, resolve_api_key, save_settings
from fluxsense.config_import import import_kpaperflux_identities, read_ini_section, unquote_qsettings_value
from fluxsense.models.identity import IdentityProfile


def test_defaults_toml_and_env(tmp_path: Path) -> None:
    """TOML overrides defaults; FLUXSENSE_<SECTION>__<KEY> overrides TOML (typed)."""
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nbase_url = "http://x/v1"\nmax_concurrency = 2\n[models]\nprimary = "a"\n', encoding="utf-8")
    env = {"FLUXSENSE_LLM__MAX_CONCURRENCY": "3", "FLUXSENSE_REVIEW__TEXT": "true", "FLUXSENSE_EXTRACT__OCR_LANGUAGES": '["deu"]'}
    settings = load_settings(path, environ=env)
    assert settings.llm.base_url == "http://x/v1" and settings.models.primary == "a"
    assert settings.llm.max_concurrency == 3 and settings.review.text is True
    assert settings.extract.ocr_languages == ["deu"]
    assert Settings().models.primary == "qwen3.6-35b-a3b"


def test_save_never_contains_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The key is resolved from env/command at runtime and never written."""
    monkeypatch.setenv("FLUXSENSE_LLM_API_KEY", "secret-123")
    settings = Settings()
    assert resolve_api_key(settings.llm) == "secret-123"
    path = save_settings(settings, tmp_path / "c.toml")
    assert "secret-123" not in path.read_text(encoding="utf-8")
    assert oct(path.stat().st_mode)[-3:] == "600"


def test_key_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """api_key_command output is used when the env var is absent."""
    monkeypatch.delenv("FLUXSENSE_LLM_API_KEY", raising=False)
    settings = Settings()
    settings.llm.api_key_command = "echo from-command"
    assert resolve_api_key(settings.llm) == "from-command"


def test_qsettings_unquote() -> None:
    """QSettings INI quoting and escapes are decoded."""
    assert unquote_qsettings_value(r'"{\"name\": \"A\\B\"}"') == '{"name": "A\\B"}'
    assert unquote_qsettings_value(r'"line1\nline2"') == "line1\nline2"
    assert unquote_qsettings_value(r'"M\xfc"') == "Mü"
    assert unquote_qsettings_value("plain") == "plain"


def test_import_kpaperflux_read_only(tmp_path: Path) -> None:
    """Profiles are read from a KPaperFlux-style conf without modifying it."""
    private = IdentityProfile(name="Erika Beispiel", address_keywords=["00115", "Berlin"]).model_dump_json()
    quoted = '"' + private.replace("\\", "\\\\").replace('"', '\\"') + '"'
    conf = tmp_path / "kpaperflux.conf"
    conf.write_text(f"[General]\nlanguage=de\n\n[Identity]\nprivate_profile={quoted}\nbusiness_profile=\"{{}}\"\n", encoding="utf-8")
    before = conf.read_bytes()
    priv, bus = import_kpaperflux_identities(conf)
    assert priv is not None and priv.name == "Erika Beispiel" and priv.address_keywords == ["00115", "Berlin"]
    assert bus is None
    assert conf.read_bytes() == before
    assert read_ini_section(conf, "General") == {"language": "de"}
    with pytest.raises(FileNotFoundError):
        import_kpaperflux_identities(tmp_path / "missing.conf")
    assert json.loads(private)["name"] == "Erika Beispiel"



def test_save_identity_keeps_other_sections(tmp_path: Path) -> None:
    """The KPaperFlux import writes only [identity] and keeps the rest of the file."""
    from fluxsense.config import IdentitySettings, save_identity
    path = tmp_path / "c.toml"
    path.write_text('[llm]\nmax_concurrency = 2\n', encoding="utf-8")
    save_identity(IdentitySettings(private=IdentityProfile(name="Erika Beispiel")), path)
    settings = load_settings(path, environ={})
    assert settings.llm.max_concurrency == 2 and settings.identity.private.name == "Erika Beispiel"
    assert "[models]" not in path.read_text(encoding="utf-8")
