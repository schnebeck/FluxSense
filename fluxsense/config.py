# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: (C) 2026 Thorsten Schnebeck <thorsten.schnebeck@gmx.net>
"""
------------------------------------------------------------------------------
Project:        FluxSense
File:           fluxsense/config.py
Version:        0.1.0
Producer:       thorsten.schnebeck@gmx.net
Generator:      Claude Opus 5.5
Description:    Settings model. Precedence: defaults < TOML file
                (~/.config/fluxsense/config.toml or $FLUXSENSE_CONFIG) <
                environment (FLUXSENSE_<SECTION>__<KEY>). Secrets are never
                stored: the API key comes from an env var or an optional
                command.
------------------------------------------------------------------------------
"""

import os
import shlex
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import tomli_w
from pydantic import BaseModel, Field

from fluxsense.log import get_logger
from fluxsense.models.identity import IdentityProfile

logger = get_logger("config")

ENV_PREFIX: str = "FLUXSENSE_"
ENV_CONFIG_PATH: str = "FLUXSENSE_CONFIG"
DEFAULT_CONFIG_PATH: Path = Path.home() / ".config" / "fluxsense" / "config.toml"


class LlmSettings(BaseModel):
    """OpenAI-compatible endpoint (llama-swap / llama.cpp)."""

    base_url: str = Field("http://localhost:8080/v1", description="OpenAI-compatible endpoint (llama-server default port)")
    api_key_env: str = "FLUXSENSE_LLM_API_KEY"
    api_key_command: str = Field("", description="Optional command printing the API key (e.g. a password-manager CLI)")
    timeout_s: float = 900.0
    connect_timeout_s: float = 10.0
    transport_retries: int = 2
    logical_retries: int = 2
    temperature: float = 0.0
    seed: int = 42
    max_output_tokens: int = 12000
    max_concurrency: int = Field(4, description="Parallel in-flight requests (llama-server slots of the batch model)")
    context_tokens: int = Field(32768, description="Context window per slot; prompts are fitted to it")
    classify_image_px: int = Field(768, description="Long side of thumbnails for classification")
    max_image_px: int = Field(1280, description="Long side of page images for holistic extraction")
    zoom_dpi: int = Field(300, description="Render DPI of region crops (escalation level 2)")
    zoom_max_px: int = Field(1280, description="Long side limit of region crops")
    render_dpi: int = 200
    max_images_per_call: int = 6
    image_format: str = Field("png", description="png | jpeg")
    disable_thinking: bool = True
    structured_output: bool = True
    minify_output: bool = Field(True, description="Ask for single-line JSON (saves output tokens)")


class ModelSettings(BaseModel):
    """
    Model roles: the model ids your server exposes (/v1/models). The
    defaults are generic model names - rename them to the ids of
    your models ('fluxsense doctor' lists what is missing). primary and
    vision need a vision-language model; ovis_ocr is only used by the
    optional 'ovisocr3' text engine, review_text only by the optional text
    review, docling_vlm only by the 'docling-vlm' engine.
    """

    primary: str = "qwen3.6-35b-a3b"
    vision: str = "qwen3.6-35b-a3b"
    review_text: str = "qwen3.8-flash-next"
    docling_vlm: str = "granite-docling"
    ovis_ocr: str = "ovisocr3"


class ExtractSettings(BaseModel):
    """Text layer extraction."""

    engine: str = Field("auto", description="auto (docling-serve if reachable, else native+ocrmypdf) | native | ocrmypdf | docling | docling-vlm | docling-serve")
    scan_fallback: str = Field("ocrmypdf", description="Engine used by 'auto' for pages without a text layer")
    min_chars_per_page: int = 50
    ocr_languages: List[str] = Field(default_factory=lambda: ["deu", "eng"])
    ocrmypdf_binary: str = "ocrmypdf"
    ocr_jobs: int = 4
    ocr_timeout_s: float = 900.0
    docling_serve_url: str = Field("", description="docling-serve base URL (empty: engine disabled)")
    docling_serve_pipeline: str = Field("standard", description="standard | vlm (server preset)")
    docling_serve_vlm_preset: str = "default"
    docling_serve_ocr_lang: List[str] = Field(default_factory=lambda: ["de", "en"])
    docling_concurrency: int = Field(2, description="Parallel docling-serve conversions in bench")
    ovis_max_px: int = Field(2880, description="Long side of page renders for OvisOCR3 (2000 px lost IBAN digits)")
    ovis_max_tokens: int = 16384
    ovis_concurrency: int = Field(2, description="Parallel OvisOCR3 page requests")
    ovis_pages: str = Field("all", description="all | weak: run OvisOCR3 on every page, or only on pages without a good "
                                               "native text layer (digital pages keep native text + PDF geometry)")
    docling_timeout_s: float = 900.0
    docling_ocr_engine: str = Field("tesseract", description="tesseract | easyocr | rapidocr")
    detect_tables: bool = True


class PipelineSettings(BaseModel):
    """Stage behaviour."""

    ai_retries: int = Field(2, description="Validation/correction loops in stage 1.1 and stage 2")
    preflight_min_pages: int = Field(6, description="Pre-flight LLM call only for documents with at least N pages")
    signature_min_pages: int = Field(2, description="Per-page signature pass (stack segmentation) from N pages")
    header_scan_min_pages: int = 11
    max_pages_stage2: int = 80
    output_budget_tokens: int = Field(7000, description="Max expected output for a single-call extraction")
    tokens_per_row: int = Field(45, description="Estimated output tokens per table row (tuple rows)")
    chunk_pages: int = Field(3, description="Pages per chunk for chunked table extraction")
    statement_chunk_pages: int = Field(5, description="Statements with more pages are extracted page-chunked upfront")
    statement_chunk_rows: int = Field(40, description="Statements with more estimated rows are extracted page-chunked upfront")
    carry_rows: int = Field(3, description="Rows of the previous chunk given as context")
    repair_text: str = Field("auto", description="auto (only after OCR) | always | never")
    visual_audit: bool = True
    zugferd_visual_audit: bool = False
    targeted_correction: bool = True
    max_text_chars: int = 40000
    custom_types_file: str = ""
    verify_mode: str = Field("inline", description="inline | deferred (verify calls batched after all documents, one model load)")
    deferred_rounds: int = Field(4, description="Max queue/replay rounds in deferred mode")
    fieldread: bool = Field(True, description="Form fields of scanned form pages read by their printed label (one crop "
                                              "per field, §12)")
    fieldread_max_fields: int = Field(24, description="Maximum fields read per page")
    handwriting_probe: bool = Field(True, description="Handwriting level (none | signature | entries | full) of scanned "
                                                      "pages the Stage 1.0 signature pass did not describe: one call per "
                                                      "file (routes the form field reading, §12)")
    signatures: bool = Field(True, description="Signature places with status and printed signer (§12); OCR text of "
                                               "scribbles is removed from scanned pages")
    pointer_extraction: bool = Field(False, description="EXPERIMENT: numbered text lines on native pages; the model "
                                     "points to the line and the text layer supplies the characters")


class ReviewSettings(BaseModel):
    """Optional LLM plausibility review (off by default for throughput)."""

    vision: bool = False
    text: bool = False
    vision_max_pages: int = 2


class GroundingSettings(BaseModel):
    """Hallucination guard."""

    enabled: bool = True
    max_escalations: int = Field(5, description="Max zoom/verification calls per logical document")
    null_unverified_critical: bool = Field(True, description="Null critical values that stay ungrounded")
    verify_weak_critical: bool = Field(True, description="Also zoom-verify weakly grounded critical values")
    second_model: bool = Field(False, description="Escalation level 3: text-only second opinion")
    self_consistency: bool = Field(False, description="Second images-only extraction for critical fields")
    evidence_max_chars: int = 40


class PlausibilitySettings(BaseModel):
    """Deterministic checks."""

    country: str = "DE"
    amount_tolerance: float = 0.02
    future_days_tolerance: int = 1
    min_year: int = 1990


class HookSettings(BaseModel):
    """Pre-hooks (e.g. wake-on-LAN for a suspended AI server)."""

    wake_command: str = ""
    wake_timeout_s: float = 240.0
    keepalive_file: str = Field("", description="Optional keep-awake flag file of the AI server; 'doctor' reports it (empty: off)")
    wol_url: str = Field("", description="Optional WOL-Server base URL for device requests (empty: disabled)")
    wol_device: str = Field("", description="WOL-Server device id or name of the AI server")
    wol_token_command: str = Field("", description="Command printing the WOL-Server API token (never stored)")
    wol_ttl_s: int = Field(900, description="Request lifetime without heartbeat (heartbeat every ttl/3)")
    wol_release_after_idle_s: float = Field(600.0, description="Release the request after this long without analyses")
    wol_verify_tls: Union[bool, str] = Field(True, description="TLS verification: true, false or a CA bundle path")
    wol_client: str = Field("", description="Client name shown on the WOL-Server (default fluxsense@<hostname>)")


class IdentitySettings(BaseModel):
    """User identities for direction / tenant context."""

    private: IdentityProfile = Field(default_factory=IdentityProfile)
    business: IdentityProfile = Field(default_factory=IdentityProfile)


class TextSettings(BaseModel):
    """Text levels and evidence-based corrections (raw / clean / corrected)."""

    spell_check: bool = Field(True, description="Dictionary correction of OCR words (hunspell dictionaries through the "
                                                "'hunspell' or 'enchant-2' command; skipped when none is installed)")
    spell_languages: List[str] = Field(default_factory=lambda: ["de_DE", "en_US"])
    format_reread: bool = Field(True, description="Fixed-format fields (IBAN, BIC, VAT ID, ...) that fail their check and "
                                                  "arithmetic mismatches are re-read on a native-resolution crop")


class ServerSettings(BaseModel):
    """REST server."""

    host: str = "127.0.0.1"
    port: int = 8765
    max_upload_mb: int = 200


class Settings(BaseModel):
    """Root settings object."""

    llm: LlmSettings = Field(default_factory=LlmSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    extract: ExtractSettings = Field(default_factory=ExtractSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    review: ReviewSettings = Field(default_factory=ReviewSettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)
    plausibility: PlausibilitySettings = Field(default_factory=PlausibilitySettings)
    hooks: HookSettings = Field(default_factory=HookSettings)
    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    text: TextSettings = Field(default_factory=TextSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    def public_dump(self) -> Dict[str, Any]:
        """Settings without identity data (for result metadata and logs)."""
        data = self.model_dump(mode="json")
        data.pop("identity", None)
        return data


def config_path() -> Path:
    """Returns the active config file path."""
    override = os.environ.get(ENV_CONFIG_PATH)
    return Path(override).expanduser() if override else DEFAULT_CONFIG_PATH


def _parse_env_value(raw: str) -> Any:
    """Parses an env override as TOML value (numbers, bools, lists), else as string."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


def _apply_env(data: Dict[str, Any], environ: Dict[str, str]) -> Dict[str, Any]:
    """
    Applies FLUXSENSE_<SECTION>__<KEY>[__<SUBKEY>] overrides.

    Args:
        data: Settings dict (modified in place).
        environ: Environment mapping.

    Returns:
        The updated dict.
    """
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        path = key[len(ENV_PREFIX):].lower().split("__")
        node = data
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                break
        else:
            node[path[-1]] = _parse_env_value(raw)
    return data


def load_settings(path: Optional[Path] = None, environ: Optional[Dict[str, str]] = None) -> Settings:
    """
    Loads settings from TOML and environment.

    Args:
        path: Explicit config file (defaults to config_path()).
        environ: Environment mapping (defaults to os.environ).

    Returns:
        The validated Settings.
    """
    file_path = path or config_path()
    data: Dict[str, Any] = {}
    if file_path.exists():
        with file_path.open("rb") as handle:
            data = tomllib.load(handle)
        logger.debug(f"Loaded config from {file_path}")
    _apply_env(data, dict(os.environ) if environ is None else environ)
    return Settings.model_validate(data)


def save_settings(settings: Settings, path: Optional[Path] = None) -> Path:
    """
    Writes settings as TOML (never contains secrets: the key is not a setting).

    Args:
        settings: Settings to persist.
        path: Target file (defaults to config_path()).

    Returns:
        The written path.
    """
    file_path = path or config_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data = settings.model_dump(mode="json", exclude_none=True)
    file_path.write_text(tomli_w.dumps(data), encoding="utf-8")
    file_path.chmod(0o600)
    return file_path


_KEY_CACHE: Dict[str, Optional[str]] = {}


def resolve_api_key(llm: LlmSettings) -> Optional[str]:
    """
    Resolves the API key from the env var or by running api_key_command.

    Args:
        llm: LLM settings.

    Returns:
        The key, or None if neither source yields one.
    """
    env_value = os.environ.get(llm.api_key_env)
    if env_value:
        return env_value.strip()
    command = llm.api_key_command.strip()
    if not command:
        return None
    if command in _KEY_CACHE:
        return _KEY_CACHE[command]
    key: Optional[str] = None
    try:
        proc = subprocess.run(shlex.split(command), capture_output=True, text=True, timeout=30, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            key = proc.stdout.strip().splitlines()[0].strip()
        else:
            logger.warning(f"API key command failed (rc={proc.returncode}): {proc.stderr.strip()[:200]}")
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning(f"API key command could not be executed: {e}")
    _KEY_CACHE[command] = key
    return key


def save_identity(identity: IdentitySettings, path: Optional[Path] = None) -> Path:
    """
    Writes only the [identity] section into the config file, keeping all
    other sections as they are (defaults are not pinned into the file).

    Args:
        identity: Identity settings.
        path: Target file (defaults to config_path()).

    Returns:
        The written path.
    """
    file_path = path or config_path()
    data: Dict[str, Any] = {}
    if file_path.exists():
        with file_path.open("rb") as handle:
            data = tomllib.load(handle)
    data["identity"] = identity.model_dump(mode="json", exclude_none=True)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(tomli_w.dumps(data), encoding="utf-8")
    file_path.chmod(0o600)
    return file_path
