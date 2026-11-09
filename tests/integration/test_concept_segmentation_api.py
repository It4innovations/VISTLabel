# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Text proposals stay transient until explicitly accepted as annotations."""

from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.annotations.png_mask import decode_mask_png_base64
from vistlabel.segmentation.base import BackendCapability, ConceptProposal
from vistlabel.video.ffmpeg import FfmpegBinaries


def test_concept_search_returns_masks_without_creating_annotations(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = create_project_with_video(api_client, "concept preview")["project_id"]
    backend = api_client.app.state.registry.get("fake")
    descriptor = replace(
        backend.descriptor(),
        capabilities=frozenset({BackendCapability.TEXT_PROMPTS}),
        checkpoint_sha256="test-checkpoint",
        code_revision="test-revision",
    )
    monkeypatch.setattr(backend, "descriptor", lambda: descriptor)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True
    mask[30:40, 10:20] = True

    def search(session_id, frame_index, text, exemplar):
        assert frame_index == 2
        assert text == "pipette"
        assert exemplar is None
        return [ConceptProposal("match-1", 2, mask, 0.9, text, (10, 10, 19, 39))]

    monkeypatch.setattr(backend, "find_matching_objects", search)
    base = f"/api/v1/projects/{project_id}"
    response = api_client.post(
        f"{base}/segment-concept",
        json={"backend_id": "fake", "frame_index": 2, "text": "  pipette  "},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["checkpoint_sha256"] == "test-checkpoint"
    assert result["code_revision"] == "test-revision"
    proposal = result["proposals"][0]
    np.testing.assert_array_equal(
        decode_mask_png_base64(proposal["visible_mask_png_base64"], expected_shape=(64, 64)), mask
    )
    assert api_client.get(f"{base}/objects").json()["objects"] == []
    assert api_client.get(f"{base}/frames/2/annotations").json()["objects"] == []

    monkeypatch.setattr(backend, "find_matching_objects", lambda *args: iter(()))
    empty = api_client.post(
        f"{base}/segment-concept",
        json={"backend_id": "fake", "frame_index": 2, "text": "nothing"},
    )
    assert empty.status_code == 200
    assert empty.json()["proposals"] == []


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"text": "   "}, 422),
        ({"text": "x" * 501}, 422),
        ({"frame_index": -1}, 422),
        ({"frame_index": 999}, 404),
        ({"backend_id": "unknown"}, 404),
        ({"backend_id": "sam3"}, 409),
        ({"backend_id": "fake"}, 422),
    ],
)
def test_concept_search_rejects_invalid_or_unsupported_requests_before_opening_session(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries,
    monkeypatch: pytest.MonkeyPatch, payload: dict, status: int,
) -> None:
    project_id = create_project_with_video(api_client, "invalid concept")["project_id"]

    def unexpected_open(*args):
        pytest.fail("invalid search must not initialize a model session")

    monkeypatch.setattr(api_client.app.state.backend_sessions, "get_or_open", unexpected_open)
    response = api_client.post(
        f"/api/v1/projects/{project_id}/segment-concept",
        json={"frame_index": 0, "text": "pipette", **payload},
    )
    assert response.status_code == status, response.text
