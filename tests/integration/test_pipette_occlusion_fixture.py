# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Pipette-occlusion acceptance criteria, exercised against ground-truth
masks extracted directly from the synthetic fixture (not a model's guess)
-- proves the persistence/occlusion-state pipeline itself is correct. See
tests/model_contract/test_sam21_pipette_occlusion.py (GPU-marked) for
whether the real SAM 2.1 model perceives the same thing.
"""

from __future__ import annotations

import av
import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import FIXTURES_DIR, create_project_with_video
from vistlabel.video.ffmpeg import FfmpegBinaries

PIPETTE_FIXTURE = FIXTURES_DIR / "pipette_occlusion_video.mp4"
# Must match scripts/generate_pipette_fixture.py.
FINGER_ENTER_FRAME = 7
FINGER_EXIT_FRAME = 13


def _decode_all_frames(video_path) -> list[np.ndarray]:
    frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frames.append(np.array(frame.to_image()))
    return frames


def _pipette_ground_truth_mask(frame: np.ndarray) -> np.ndarray:
    """The pipette is drawn light gray (~200,200,200); the finger and
    background are both much darker, so a simple brightness threshold
    recovers exactly the *visible* pipette pixels -- occluded pixels are
    genuinely not light gray in the frame, matching the modal/visible-mask
    definition."""
    return (frame[:, :, 0] > 150) & (frame[:, :, 1] > 150) & (frame[:, :, 2] > 150)


def test_fixture_video_exists_and_is_tiny() -> None:
    assert PIPETTE_FIXTURE.is_file()
    assert PIPETTE_FIXTURE.stat().st_size < 50_000


def test_ground_truth_masks_show_expected_occlusion_pattern() -> None:
    """Sanity check on the fixture generator itself, independent of the
    application: components should be 1 outside the occlusion window and
    2 inside it."""
    from vistlabel.persistence.rle import count_connected_components

    frames = _decode_all_frames(PIPETTE_FIXTURE)
    for index, frame in enumerate(frames):
        mask = _pipette_ground_truth_mask(frame)
        components = count_connected_components(mask)
        if FINGER_ENTER_FRAME <= index <= FINGER_EXIT_FRAME:
            assert components == 2, f"frame {index}: expected 2 components, got {components}"
        else:
            assert components == 1, f"frame {index}: expected 1 component, got {components}"


def test_one_logical_object_survives_the_whole_occlusion_sequence(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """The acceptance criteria: one object_id throughout, visible
    mask has >=2 components during occlusion, finger pixels never counted as
    part of the pipette (by construction of the ground-truth mask above),
    component-count changes don't create a new object (there is only ever
    one object_id in this test), and the object "reappears" (returns to 1
    component) with the same identity after the finger passes."""
    project = create_project_with_video(
        api_client, "pipette occlusion fixture", video_path=PIPETTE_FIXTURE
    )
    project_id = project["project_id"]
    assert project["source"]["frame_count"] == 20

    object_id = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": 3, "label_name": "pipette"},
    ).json()["object_id"]

    from vistlabel.annotations.png_mask import encode_mask_png_base64

    frames = _decode_all_frames(PIPETTE_FIXTURE)
    revision = 0
    for frame_index, frame in enumerate(frames):
        mask = _pipette_ground_truth_mask(frame)
        response = api_client.patch(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations",
            json={
                "expected_revision": revision,
                "upsert_object": {
                    "object_id": object_id,
                    "label_id": 3,
                    "visible_mask_png_base64": encode_mask_png_base64(mask),
                    "source": "ground_truth_fixture",
                    "manually_modified": True,
                },
            },
        )
        assert response.status_code == 200, response.text
        saved = response.json()
        entry = saved["objects"][0]

        # One logical object throughout -- never a second object_id.
        assert len(saved["objects"]) == 1
        assert entry["object_id"] == object_id

        if FINGER_ENTER_FRAME <= frame_index <= FINGER_EXIT_FRAME:
            assert entry["connected_component_count"] == 2
            assert entry["occlusion_state"] == "partially_occluded"
        else:
            assert entry["connected_component_count"] == 1
            assert entry["occlusion_state"] == "visible"

        revision = 0  # each frame_index is a fresh FrameAnnotation, starts at revision 0
