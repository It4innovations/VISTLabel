# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vistlabel.settings import AppSettings, load_settings


def test_defaults_load_without_a_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "does-not-exist.yaml"

    settings = load_settings(missing_path)

    assert settings.runtime.default_backend == "fake"
    assert settings.models.sam21.enabled is False
    assert settings.models.sam3.enabled is False
    assert settings.models.yolo_sam.enabled is False


def test_yaml_file_values_are_applied(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text(
        """
        workspace:
          root: /custom/workspace
        runtime:
          default_backend: sam21
        models:
          sam21:
            enabled: true
            variant: sam2.1_hiera_base_plus
        """,
        encoding="utf-8",
    )

    settings = load_settings(config_path)

    assert settings.workspace.root == Path("/custom/workspace")
    assert settings.runtime.default_backend == "sam21"
    assert settings.models.sam21.enabled is True
    assert settings.models.sam21.variant == "sam2.1_hiera_base_plus"


def test_environment_variable_overrides_yaml_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text("workspace:\n  root: /from-yaml\n", encoding="utf-8")
    monkeypatch.setenv("VISTLABEL_WORKSPACE__ROOT", "/from-env")

    settings = load_settings(config_path)

    assert settings.workspace.root == Path("/from-env")


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "app.yaml"
    config_path.write_text("not_a_real_setting: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_settings(config_path)


def test_resolve_config_path_prefers_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VISTLABEL_CONFIG_FILE", "/env/app.yaml")

    from vistlabel.settings import resolve_config_path

    assert resolve_config_path() == Path("/env/app.yaml")


def test_app_settings_forbids_typos_via_kwargs() -> None:
    with pytest.raises(ValidationError):
        AppSettings(schema_versoin=1)  # type: ignore[call-arg]


def test_legacy_sam31_config_key_is_rejected_not_ignored(tmp_path: Path) -> None:
    """Nested settings models ignore unknown keys, so an upgraded instance
    still saying models.sam31 would silently run with defaults and report a
    configured backend as disabled. That must fail loudly instead."""
    config = tmp_path / "app.yaml"
    config.write_text(
        "schema_version: 1\nmodels:\n  sam31:\n    enabled: true\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="models.sam31 was renamed to models.sam3"):
        load_settings(config)


def test_current_sam3_config_key_loads(tmp_path: Path) -> None:
    config = tmp_path / "app.yaml"
    config.write_text(
        "schema_version: 1\nmodels:\n  sam3:\n    enabled: true\n", encoding="utf-8"
    )

    settings = load_settings(config)

    assert settings.models.sam3.enabled is True
