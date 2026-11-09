# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""End-to-end HTTP wiring for the real Sam21Backend: proves
the registry/API/session-manager plumbing works with a real model, not
just the backend in isolation (see
tests/model_contract/test_sam21_backend_contract.py for that).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.video.ffmpeg import FfmpegBinaries

pytestmark = pytest.mark.gpu

CHECKPOINT_DIR = Path("models/sam2.1-hiera-base-plus")


@pytest.fixture
def sam21_api_client(tmp_path: Path) -> TestClient:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("'gpu' extra not installed (pip install -e '.[gpu]')")
    if not (CHECKPOINT_DIR / "model.safetensors").is_file():
        pytest.skip(
            f"checkpoint not downloaded at {CHECKPOINT_DIR} "
            "(python scripts/download_models.py --model sam21-base-plus)"
        )

    from vistlabel.main import create_app
    from vistlabel.settings import AppSettings

    settings = AppSettings(
        workspace={"root": tmp_path / "workspace"},
        models={"sam21": {"enabled": True, "checkpoint": str(CHECKPOINT_DIR)}},
    )
    return TestClient(create_app(settings))


def test_models_endpoint_reports_sam21_available(
    sam21_api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    response = sam21_api_client.get("/api/v1/models")

    assert response.status_code == 200
    sam21 = next(b for b in response.json()["backends"] if b["backend_id"] == "sam21")
    assert sam21["available"] is True
    assert sam21["checkpoint_sha256"] is not None


def test_segment_with_sam21_produces_and_saves_a_real_mask(
    sam21_api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(sam21_api_client, "sam21 api test")["project_id"]
    object_id = sam21_api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": 3, "label_name": "pipette"},
    ).json()["object_id"]

    segment_response = sam21_api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "sam21",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 32, "y": 32, "positive": True}],
        },
    )
    assert segment_response.status_code == 200, segment_response.text
    prediction = segment_response.json()
    assert prediction["component_count"] >= 1
    mask_png = prediction["visible_mask_png_base64"]

    save_response = sam21_api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "sam21_interactive",
                "backend_id": "sam21",
            },
        },
    )
    assert save_response.status_code == 200, save_response.text
    assert save_response.json()["objects"][0]["backend_provenance"]["backend_id"] == "sam21"
