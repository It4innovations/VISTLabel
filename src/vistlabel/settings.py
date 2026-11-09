# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Static application configuration.

Loaded from ``config/app.yaml`` (see ``config/app.example.yaml``) with
environment-variable overrides for secrets or machine-specific paths.

Only ``config/app.yaml`` (human-edited, static) is read here. Mutable project
state is never modelled in this module -- see the ``persistence`` package.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config/app.yaml")
CONFIG_PATH_ENV_VAR = "VISTLABEL_CONFIG_FILE"


class WorkspaceSettings(BaseModel):
    root: Path = Path("./workspace")


class UploadSettings(BaseModel):
    allowed_extensions: list[str] = Field(default_factory=lambda: [".mp4"])
    maximum_file_size_gb: float = 50.0
    chunk_size_mb: int = 8
    minimum_free_disk_space_gb: float = 20.0
    normalize_variable_frame_rate: bool = True
    normalized_frame_rate: int = Field(default=30, ge=1)


class RuntimeSettings(BaseModel):
    cuda_device: str = "auto"
    default_dtype: str = "float16"
    maximum_concurrent_gpu_jobs: int = 1
    default_backend: str = "fake"
    #: Directory containing ffmpeg/ffprobe binaries, checked before PATH.
    #: Lets a workstation with ffmpeg installed outside PATH run without a
    #: system-wide PATH edit.
    ffmpeg_dir: Path | None = None


class Sam21ModelSettings(BaseModel):
    enabled: bool = False
    variant: str = "sam2.1_hiera_large"
    #: A local *directory* (a Hugging Face snapshot: config.json,
    #: model.safetensors, ...) produced by scripts/download_models.py --
    #: not the single .pt file Meta's own repository distributes.
    checkpoint: Path | None = None
    code_revision: str | None = None


class Sam3ModelSettings(BaseModel):
    enabled: bool = False
    #: Reported as the model name. The ``sam3`` backend actually runs
    #: ``facebook/sam3`` -- SAM 3.1 ships without a transformers integration
    #: (see segmentation/sam3.py) -- so this must not claim "3.1".
    variant: str = "sam3"
    #: Hugging Face snapshot *directory* (config.json + model.safetensors),
    #: as produced by ``scripts/download_models.py --model sam3``.
    checkpoint: Path | None = None
    code_revision: str | None = None
    require_startup_diagnostic: bool = True


class YoloSamModelSettings(BaseModel):
    enabled: bool = False
    yolo_checkpoint: Path | None = None
    class_mapping: Path | None = None
    refiner_backend: Literal["sam21", "sam3"] = "sam21"
    detection_stride_frames: int = 30


class ModelsSettings(BaseModel):
    sam21: Sam21ModelSettings = Field(default_factory=Sam21ModelSettings)
    sam3: Sam3ModelSettings = Field(default_factory=Sam3ModelSettings)
    yolo_sam: YoloSamModelSettings = Field(default_factory=YoloSamModelSettings)


class TrackingSettings(BaseModel):
    default_chunk_frames: int = 200
    checkpoint_after_chunk: bool = True
    #: Every Nth propagated frame is flagged for review regardless of any
    #: other signal (manual review sampling). None/0 disables it.
    review_sampling_stride: int | None = None
    #: Bounded LRU cache size for open backend sessions across all
    #: projects; the least recently used session is evicted beyond this.
    max_open_backend_sessions: int = 4


class ExportSettings(BaseModel):
    image_format: Literal["jpg", "png"] = "jpg"
    jpeg_quality: int = 95
    yolo_multipart_policy: Literal["strict_error", "minimal_bridge", "split_components"] = (
        "minimal_bridge"
    )
    always_include_coco_exact_annotations: bool = True
    annotation_type: Literal["segmentation", "bbox"] = "segmentation"
    dataset_layout: Literal["ultralytics", "coco"] = "ultralytics"


class LoggingSettings(BaseModel):
    level: str = "INFO"
    json_output: bool = True


class AuthSettings(BaseModel):
    """Authentication must be added before exposing the service beyond a
    controlled network. Disabled by default -- binding to localhost (the
    other default) is the real protection for a single-workstation
    install; this is the integration point for the case where VISTLabel is
    reachable from more than just its own host (a shared workstation,
    a small internal network) without a full reverse-proxy/SSO deployment
    in front of it yet. A production deployment beyond a trusted network
    should still prefer a real reverse-proxy-terminated auth scheme.
    """

    enabled: bool = False
    #: Shared API keys checked against the `Authorization: Bearer <key>`
    #: header. Not a user/session system -- a single shared secret (or a
    #: few, for simple key rotation) is the appropriate amount of
    #: complexity for "add an integration point," not a full identity
    #: provider integration.
    api_keys: list[str] = Field(default_factory=list)


class AppSettings(BaseSettings):
    """Root application settings.

    Values are layered: ``config/app.yaml`` provides the base, environment
    variables (prefix ``VISTLABEL_``, ``__`` as the nested delimiter) always
    win over the file.
    """

    model_config = SettingsConfigDict(
        env_prefix="VISTLABEL_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    schema_version: int = 1
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    upload: UploadSettings = Field(default_factory=UploadSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Environment variables must win over values loaded from the YAML
        # file (passed in as init kwargs), so they are listed first.
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


def resolve_config_path(config_path: Path | None = None) -> Path:
    if config_path is not None:
        return config_path
    env_value = os.environ.get(CONFIG_PATH_ENV_VAR)
    if env_value:
        return Path(env_value)
    return DEFAULT_CONFIG_PATH


def load_settings(config_path: Path | None = None) -> AppSettings:
    """Load application settings from YAML plus environment overrides.

    Missing configuration file is not an error: a fresh checkout must boot
    with safe defaults (all real model backends disabled), and the
    application must start without any model checkpoints.
    """
    path = resolve_config_path(config_path)
    raw: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is not None:
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}: expected a YAML mapping at the top level")
            raw = loaded
    _reject_renamed_keys(path, raw)
    return AppSettings(**raw)


#: Configuration keys that were renamed, mapped to their current name. The
#: nested settings models ignore unknown keys, so an upgraded instance whose
#: config still uses the old name would silently run with defaults -- for
#: ``sam31`` that means a configured, enabled backend quietly reporting
#: itself disabled. Fail loudly instead.
_RENAMED_MODEL_KEYS = {"sam31": "sam3"}


def _reject_renamed_keys(path: Path, raw: dict[str, Any]) -> None:
    models = raw.get("models")
    if not isinstance(models, dict):
        return
    for old, new in _RENAMED_MODEL_KEYS.items():
        if old in models:
            raise ValueError(
                f"{path}: models.{old} was renamed to models.{new} "
                f"(the backend runs SAM 3, not SAM 3.1). Rename the key -- it is "
                "otherwise ignored and the backend would report itself disabled."
            )
