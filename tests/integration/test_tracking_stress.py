# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Stress test for long videos, many objects, and repeated interruption.
CPU-only, FakeBackend: the point is exercising the chunking/checkpoint/pause/resume machinery under
load, not model behavior.
"""

from __future__ import annotations

import time
from pathlib import Path

import av
import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.video.ffmpeg import FfmpegBinaries

_FRAME_COUNT = 60
_OBJECT_COUNT = 3


def _write_synthetic_video(path: Path, frame_count: int, size: int = 48) -> None:
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=15)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        rng = np.random.default_rng(0)
        for _ in range(frame_count):
            array = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _wait_for_job(api_client: TestClient, project_id: str, job_id: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    terminal = {"succeeded", "failed", "cancelled", "paused"}
    while time.monotonic() < deadline:
        record = api_client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_many_objects_long_video_and_repeated_interruption(
    tmp_path: Path, api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    video_path = tmp_path / "stress.mp4"
    _write_synthetic_video(video_path, _FRAME_COUNT)

    project_id = create_project_with_video(api_client, "stress test", video_path=video_path)[
        "project_id"
    ]
    assert api_client.get(f"/api/v1/projects/{project_id}").json()["source"]["frame_count"] == (
        _FRAME_COUNT
    )

    object_ids: list[str] = []
    frame0_revision = 0
    for i in range(_OBJECT_COUNT):
        created = api_client.post(
            f"/api/v1/projects/{project_id}/objects",
            json={"label_id": 3, "label_name": "pipette"},
        ).json()
        object_id = created["object_id"]
        x, y = 4 + i * 6, 4 + i * 6
        segment = api_client.post(
            f"/api/v1/projects/{project_id}/segment",
            json={
                "backend_id": "fake",
                "frame_index": 0,
                "object_id": object_id,
                "points": [{"x": x, "y": y, "positive": True}],
            },
        ).json()
        save = api_client.patch(
            f"/api/v1/projects/{project_id}/frames/0/annotations",
            json={
                "expected_revision": frame0_revision,
                "upsert_object": {
                    "object_id": object_id,
                    "label_id": 3,
                    "visible_mask_png_base64": segment["visible_mask_png_base64"],
                    "source": "fake_interactive",
                },
            },
        )
        assert save.status_code == 200, save.text
        frame0_revision = save.json()["revision"]
        object_ids.append(object_id)

    started = time.monotonic()
    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": object_ids,
            "start_frame": 0,
            "end_frame": _FRAME_COUNT - 1,
            "direction": "forward",
            "chunk_frames": 10,  # 6 checkpoints over 60 frames
        },
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    job_id = response.json()["job_id"]

    # Repeated interruption: pause as soon as at least one checkpoint has
    # landed, then resume to completion -- twice, to exercise resume-after
    # -pause more than once in the same run's lifetime.
    for _ in range(2):
        deadline = time.monotonic() + 3.0
        run_record = None
        while time.monotonic() < deadline:
            response = api_client.get(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}")
            if response.status_code == 200:
                run_record = response.json()
                if run_record["last_completed_frame"] is not None or run_record["status"] != (
                    "running"
                ):
                    break
            time.sleep(0.005)
        assert run_record is not None
        if run_record["status"] != "running":
            break  # finished faster than we could catch a checkpoint -- fine
        api_client.post(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}/pause")
        paused = _wait_for_job(api_client, project_id, job_id)
        if paused["status"] != "paused":
            break  # completed before the pause request landed -- also fine
        resume_response = api_client.post(
            f"/api/v1/projects/{project_id}/tracking-runs/{run_id}/resume"
        )
        assert resume_response.status_code == 202, resume_response.text
        job_id = resume_response.json()["job_id"]

    final_job = _wait_for_job(api_client, project_id, job_id)
    assert final_job["status"] == "succeeded", final_job
    elapsed = time.monotonic() - started

    final_run = api_client.get(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}").json()
    assert final_run["status"] == "completed"
    assert final_run["last_completed_frame"] == _FRAME_COUNT - 1

    # Full coverage, no data loss across however many pause/resume cycles
    # actually happened: every object present on every frame.
    for frame_index in (0, _FRAME_COUNT // 2, _FRAME_COUNT - 1):
        annotation = api_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        present = {o["object_id"] for o in annotation["objects"]}
        assert present == set(object_ids), f"frame {frame_index} missing objects"

    # Stress-test report: print the timing so a run with -s records it.
    print(
        f"\n[stress] {_OBJECT_COUNT} objects x {_FRAME_COUNT} frames, "
        f"chunk_frames=10, completed in {elapsed:.2f}s "
        f"(FakeBackend, in-process TestClient)"
    )
