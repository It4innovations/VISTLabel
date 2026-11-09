# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_expected_sections(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded", "error")
    assert set(body) >= {"status", "package_version", "filesystem", "ffmpeg", "python", "cuda"}
    # The workspace fixture points at a fresh tmp_path, so it must be writable.
    assert body["filesystem"]["status"] == "ok"


def test_version_reports_package_version(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/version")

    assert response.status_code == 200
    assert "version" in response.json()


def test_models_lists_the_fake_backend(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/models")

    assert response.status_code == 200
    body = response.json()
    backend_ids = {backend["backend_id"] for backend in body["backends"]}
    assert "fake" in backend_ids
    fake_backend = next(b for b in body["backends"] if b["backend_id"] == "fake")
    assert fake_backend["available"] is True
    assert "video_propagation" in fake_backend["capabilities"]


def test_metrics_reports_operational_counters(api_client: TestClient) -> None:
    response = api_client.get("/api/v1/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["uptime_seconds"] >= 0.0
    assert body["project_count"] == 0
    assert body["total_annotated_frame_count"] == 0
    assert body["jobs_by_status"]["succeeded"] == 0
    assert body["backend_availability"]["fake"] is True


def test_javascript_is_served_with_an_executable_mime_type(api_client: TestClient) -> None:
    """Browsers enforce strict MIME checking on ES modules: a .js served as
    text/plain is refused and the whole frontend dies silently. On Windows
    the stdlib mimetypes module seeds from the registry, where .js is often
    mapped to text/plain, so main.py pins the type explicitly."""
    response = api_client.get("/static/js/upload.js")

    assert response.status_code == 200
    media_type = response.headers["content-type"].split(";")[0].strip()
    assert media_type in {"text/javascript", "application/javascript"}


def test_stylesheets_are_served_as_css(api_client: TestClient) -> None:
    response = api_client.get("/static/css/app.css")

    assert response.status_code == 200
    assert response.headers["content-type"].split(";")[0].strip() == "text/css"
