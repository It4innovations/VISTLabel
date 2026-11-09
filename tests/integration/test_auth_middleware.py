# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""The authentication integration point -- disabled by default (existing
behavior for every other test in this suite), real when enabled."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from vistlabel.main import create_app
from vistlabel.settings import AppSettings


def test_auth_disabled_by_default_allows_requests_without_a_header(tmp_path: Path) -> None:
    settings = AppSettings(workspace={"root": tmp_path / "workspace"})
    client = TestClient(create_app(settings))
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/projects").status_code == 200


def test_auth_enabled_rejects_requests_without_a_key(tmp_path: Path) -> None:
    settings = AppSettings(
        workspace={"root": tmp_path / "workspace"}, auth={"enabled": True, "api_keys": ["s3cr3t"]}
    )
    client = TestClient(create_app(settings))
    response = client.get("/api/v1/projects")
    assert response.status_code == 401


def test_auth_enabled_rejects_an_invalid_key(tmp_path: Path) -> None:
    settings = AppSettings(
        workspace={"root": tmp_path / "workspace"}, auth={"enabled": True, "api_keys": ["s3cr3t"]}
    )
    client = TestClient(create_app(settings))
    response = client.get(
        "/api/v1/projects", headers={"Authorization": "Bearer wrong-key"}
    )
    assert response.status_code == 401


def test_auth_enabled_accepts_a_valid_key(tmp_path: Path) -> None:
    settings = AppSettings(
        workspace={"root": tmp_path / "workspace"}, auth={"enabled": True, "api_keys": ["s3cr3t"]}
    )
    client = TestClient(create_app(settings))
    response = client.get("/api/v1/projects", headers={"Authorization": "Bearer s3cr3t"})
    assert response.status_code == 200


def test_auth_enabled_still_exempts_the_health_endpoint(tmp_path: Path) -> None:
    settings = AppSettings(
        workspace={"root": tmp_path / "workspace"}, auth={"enabled": True, "api_keys": ["s3cr3t"]}
    )
    client = TestClient(create_app(settings))
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_auth_enabled_supports_multiple_keys_for_rotation(tmp_path: Path) -> None:
    settings = AppSettings(
        workspace={"root": tmp_path / "workspace"},
        auth={"enabled": True, "api_keys": ["old-key", "new-key"]},
    )
    client = TestClient(create_app(settings))
    for key in ("old-key", "new-key"):
        response = client.get("/api/v1/projects", headers={"Authorization": f"Bearer {key}"})
        assert response.status_code == 200
