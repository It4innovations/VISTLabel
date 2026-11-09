# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Path traversal prevention -- end-to-end regression tests for two real,
reproduced vulnerabilities found during a security audit and fixed in the
same change:

1. ``DELETE /api/v1/projects/{project_id}`` with a percent-encoded ``..``
   project_id resolved to the workspace root itself, wiping every project
   in one unauthenticated call.
2. ``DELETE /api/v1/projects/{project_id}/exports/{export_id}`` with a
   percent-encoded ``..`` export_id resolved one level up, wiping the
   entire project directory instead of one export.

Both required the ``..`` to be percent-encoded (``%2e%2e``) in the request
line: a *literal* ``..`` in a URL is dot-segment-normalized away by
``httpx``/browsers before the request is ever sent, so it never reaches
the server as a literal path parameter at all -- these tests send the
percent-encoded form directly to prove the server-side fix, not rely on a
client that would mask the bug either way.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.settings import AppSettings
from vistlabel.video.ffmpeg import FfmpegBinaries


def _wait_for_job(api_client: TestClient, project_id: str, job_id: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    terminal = {"succeeded", "failed", "cancelled", "paused"}
    while time.monotonic() < deadline:
        record = api_client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_delete_project_with_traversal_id_does_not_wipe_the_workspace(
    api_client: TestClient, app_settings: AppSettings, ffmpeg_binaries: FfmpegBinaries
) -> None:
    victim_a = create_project_with_video(api_client, "victim a")["project_id"]
    victim_b = create_project_with_video(api_client, "victim b")["project_id"]

    response = api_client.delete("/api/v1/projects/%2e%2e")
    assert response.status_code == 404

    assert api_client.get(f"/api/v1/projects/{victim_a}").status_code == 200
    assert api_client.get(f"/api/v1/projects/{victim_b}").status_code == 200
    assert app_settings.workspace.root.is_dir()


def test_delete_project_with_traversal_id_variants_are_all_rejected(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    victim = create_project_with_video(api_client, "victim")["project_id"]
    for malicious_id in ("%2e%2e", "prj_%2e%2e", "not-a-real-project-id"):
        response = api_client.delete(f"/api/v1/projects/{malicious_id}")
        assert response.status_code == 404, (malicious_id, response.status_code)

    assert api_client.get(f"/api/v1/projects/{victim}").status_code == 200


def test_delete_export_with_traversal_id_does_not_wipe_the_project(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "export traversal victim")[
        "project_id"
    ]

    response = api_client.delete(f"/api/v1/projects/{project_id}/exports/%2e%2e")
    assert response.status_code == 204  # idempotent no-op, not an error leak

    # The whole project -- not just its (nonexistent) exports -- must
    # still be there afterward.
    still_there = api_client.get(f"/api/v1/projects/{project_id}")
    assert still_there.status_code == 200
    assert still_there.json()["source"] is not None


def test_delete_export_with_traversal_id_variants_are_all_rejected(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "export traversal variants")[
        "project_id"
    ]
    for malicious_id in ("%2e%2e", "exp_%2e%2e", "not-a-real-export-id"):
        response = api_client.delete(f"/api/v1/projects/{project_id}/exports/{malicious_id}")
        assert response.status_code == 204  # idempotent no-op, never an error leak

    assert api_client.get(f"/api/v1/projects/{project_id}").status_code == 200


def test_legitimate_project_and_export_deletion_still_works(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """The fix must not turn into a false-positive that blocks real
    deletes."""
    project_id = create_project_with_video(api_client, "legit delete test")["project_id"]
    export_start = api_client.post(
        f"/api/v1/projects/{project_id}/exports",
        json={"frame_selection_policy": "all_frames", "acknowledge_all_frames_warning": True},
    )
    assert export_start.status_code == 202
    export_id = export_start.json()["export_id"]
    _wait_for_job(api_client, project_id, export_start.json()["job_id"])

    delete_export_response = api_client.delete(
        f"/api/v1/projects/{project_id}/exports/{export_id}"
    )
    assert delete_export_response.status_code == 204

    delete_project_response = api_client.delete(f"/api/v1/projects/{project_id}")
    assert delete_project_response.status_code == 204
    assert api_client.get(f"/api/v1/projects/{project_id}").status_code == 404
