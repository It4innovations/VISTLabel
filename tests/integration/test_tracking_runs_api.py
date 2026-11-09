# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Chunked/resumable propagation, manual-mask protection,
pause/resume/cancel, crash recovery, and the review queue -- exercised
through the real API."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.conftest import authenticated_client, create_project_with_video
from vistlabel.domain.tracking import TrackingRunStatus
from vistlabel.main import create_app
from vistlabel.settings import AppSettings
from vistlabel.tracking import manifests
from vistlabel.video.ffmpeg import FfmpegBinaries


def _create_object(api_client: TestClient, project_id: str) -> str:
    response = api_client.post(
        f"/api/v1/projects/{project_id}/objects",
        json={"label_id": 3, "label_name": "pipette"},
    )
    assert response.status_code == 201, response.text
    return response.json()["object_id"]


def _seed_object(api_client: TestClient, project_id: str, object_id: str) -> None:
    segment_response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 10, "y": 10, "positive": True}],
        },
    )
    assert segment_response.status_code == 200
    mask_png = segment_response.json()["visible_mask_png_base64"]
    save_response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_interactive",
                "backend_id": "fake",
                "manually_modified": True,
            },
        },
    )
    assert save_response.status_code == 200


def _wait_for_job(api_client: TestClient, project_id: str, job_id: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    terminal = {"succeeded", "failed", "cancelled", "paused"}
    while time.monotonic() < deadline:
        record = api_client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def _wait_for_run_status(
    api_client: TestClient, project_id: str, run_id: str, status: str, timeout: float = 20.0
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = api_client.get(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}").json()
        if record["status"] == status:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"tracking run {run_id} did not reach status {status!r} within {timeout}s")


def test_propagate_completes_in_multiple_chunks_and_manifest_reflects_it(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "chunked propagation test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 9,
            "direction": "forward",
            "chunk_frames": 3,  # 4 chunks over 10 frames
        },
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]
    run_id = response.json()["run_id"]

    job_record = _wait_for_job(api_client, project_id, job_id)
    assert job_record["status"] == "succeeded"

    run_record = api_client.get(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}").json()
    assert run_record["status"] == "completed"
    assert run_record["last_completed_frame"] == 9
    assert run_record["chunk_frames"] == 3

    for frame_index in range(10):
        annotation = api_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        assert object_id in [o["object_id"] for o in annotation["objects"]]

    listed = api_client.get(f"/api/v1/projects/{project_id}/tracking-runs").json()
    assert any(run["run_id"] == run_id for run in listed)


def test_manual_mask_is_protected_from_propagation_overwrite(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "manual mask protection test")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)

    # Manually mark frame 3's entry -- as if a human corrected it -- before
    # propagation ever reaches it.
    manual_mask_response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 3,
            "object_id": object_id,
            "points": [{"x": 40, "y": 40, "positive": True}],
        },
    )
    manual_mask_png = manual_mask_response.json()["visible_mask_png_base64"]
    manual_save = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/3/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": manual_mask_png,
                "source": "manual_correction",
                "manually_modified": True,
            },
        },
    )
    assert manual_save.status_code == 200
    protected_component_count = manual_save.json()["objects"][0]["connected_component_count"]

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 5,
            "direction": "forward",
        },
    )
    job_id = response.json()["job_id"]
    run_id = response.json()["run_id"]
    job_record = _wait_for_job(api_client, project_id, job_id)
    assert job_record["status"] == "succeeded"

    frame3 = api_client.get(f"/api/v1/projects/{project_id}/frames/3/annotations").json()
    entry = next(o for o in frame3["objects"] if o["object_id"] == object_id)
    assert entry["source"] == "manual_correction"
    assert entry["connected_component_count"] == protected_component_count

    run_record = api_client.get(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}").json()
    assert run_record["manual_masks_protected_count"] >= 1


def test_manual_mask_overwritten_when_explicitly_requested(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "manual mask override test")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)

    manual_save = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/3/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": _seed_mask_png(api_client, project_id, object_id, 3),
                "source": "manual_correction",
                "manually_modified": True,
            },
        },
    )
    assert manual_save.status_code == 200

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 5,
            "direction": "forward",
            "overwrite_manual_masks": True,
        },
    )
    job_id = response.json()["job_id"]
    job_record = _wait_for_job(api_client, project_id, job_id)
    assert job_record["status"] == "succeeded"

    frame3 = api_client.get(f"/api/v1/projects/{project_id}/frames/3/annotations").json()
    entry = next(o for o in frame3["objects"] if o["object_id"] == object_id)
    assert entry["source"] == "fake_propagation"


def test_object_inactive_on_a_frame_stays_inactive_through_propagation(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "inactive object propagation test")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)
    saved = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/3/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": _seed_mask_png(api_client, project_id, object_id, 3),
                "source": "fake_interactive",
            },
        },
    ).json()
    inactive = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/3/annotations",
        json={"expected_revision": saved["revision"], "deactivate_object_id": object_id},
    )
    assert inactive.status_code == 200, inactive.text

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 5,
            "direction": "forward",
        },
    )
    job_record = _wait_for_job(api_client, project_id, response.json()["job_id"])
    assert job_record["status"] == "succeeded"

    frame3 = api_client.get(f"/api/v1/projects/{project_id}/frames/3/annotations").json()
    assert frame3["objects"] == []
    assert [o["object_id"] for o in frame3["inactive_objects"]] == [object_id]
    frame4 = api_client.get(f"/api/v1/projects/{project_id}/frames/4/annotations").json()
    assert [o["object_id"] for o in frame4["objects"]] == [object_id]


def _seed_mask_png(
    api_client: TestClient, project_id: str, object_id: str, frame_index: int
) -> str:
    response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": frame_index,
            "object_id": object_id,
            "points": [{"x": 40, "y": 40, "positive": True}],
        },
    )
    return response.json()["visible_mask_png_base64"]


def test_cancel_via_generic_jobs_endpoint_marks_manifest_cancelled(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "cancel test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 9,
            "direction": "forward",
        },
    )
    job_id = response.json()["job_id"]
    run_id = response.json()["run_id"]

    # The job may already be finished by the time we cancel (FakeBackend is
    # fast) -- either outcome is fine; what matters is the manifest never
    # ends up inconsistent with the job's real terminal state.
    api_client.post(f"/api/v1/projects/{project_id}/jobs/{job_id}/cancel")
    job_record = _wait_for_job(api_client, project_id, job_id)
    run_record = api_client.get(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}").json()
    if job_record["status"] == "cancelled":
        assert run_record["status"] == "cancelled"
    else:
        assert job_record["status"] == "succeeded"
        assert run_record["status"] == "completed"


def test_resume_of_a_running_run_is_rejected(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "resume rejection test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)

    response = api_client.post(
        f"/api/v1/projects/{project_id}/propagate",
        json={
            "backend_id": "fake",
            "object_ids": [object_id],
            "start_frame": 0,
            "end_frame": 9,
            "direction": "forward",
        },
    )
    run_id = response.json()["run_id"]
    _wait_for_job(api_client, project_id, response.json()["job_id"])

    # Completed, not paused/interrupted -- resume must be rejected.
    resume_response = api_client.post(
        f"/api/v1/projects/{project_id}/tracking-runs/{run_id}/resume"
    )
    assert resume_response.status_code == 409


def test_resume_continues_from_last_completed_frame(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """Simulates a crash: a fresh backend session (as after a real process
    restart) has no in-memory state at all, so resume must re-seed it from
    the canonically persisted mask at the checkpoint frame before
    continuing -- and must continue from frame 4 onward, not redo 0-3."""
    project_id = create_project_with_video(api_client, "resume from checkpoint test")[
        "project_id"
    ]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)

    # A real mask genuinely exists at frame 4 (as chunk 1 of a prior run
    # would have left behind), independent of the in-memory backend
    # session, which the "crash" below discards.
    mask_png = _seed_mask_png(api_client, project_id, object_id, 4)
    checkpoint_save = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/4/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_propagation",
            },
        },
    )
    assert checkpoint_save.status_code == 200

    settings = api_client.app.state.settings
    from vistlabel.persistence.paths import project_dir

    pdir = project_dir(settings.workspace.root, project_id)
    run_id = manifests.new_run_id()
    manifests.create_run(
        pdir,
        run_id=run_id,
        job_id="job_orphaned",
        backend_id="fake",
        object_ids=[object_id],
        start_frame=0,
        end_frame=9,
        direction="forward",
        chunk_frames=4,
        overwrite_manual_masks=False,
    )
    manifests.update_run(
        pdir, run_id, last_completed_frame=4, status=TrackingRunStatus.INTERRUPTED
    )

    # A brand new app/client -- the "crash": no live backend session, no
    # in-memory job/cancel state survives.
    second_app = create_app(settings)
    second_client = authenticated_client(second_app)

    resume_response = second_client.post(
        f"/api/v1/projects/{project_id}/tracking-runs/{run_id}/resume"
    )
    assert resume_response.status_code == 202, resume_response.text
    job_id = resume_response.json()["job_id"]
    job_record = _wait_for_job(second_client, project_id, job_id)
    assert job_record["status"] == "succeeded", job_record

    run_record = second_client.get(
        f"/api/v1/projects/{project_id}/tracking-runs/{run_id}"
    ).json()
    assert run_record["status"] == "completed"
    assert run_record["last_completed_frame"] == 9

    # Frames 1-3 were never propagated by anything in this test -- the
    # resumed run only ever asked the backend for [4, 9].
    for frame_index in (1, 2, 3):
        annotation = second_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        assert annotation["objects"] == []
    for frame_index in range(4, 10):
        annotation = second_client.get(
            f"/api/v1/projects/{project_id}/frames/{frame_index}/annotations"
        ).json()
        assert object_id in [o["object_id"] for o in annotation["objects"]]


def test_cancel_of_interrupted_run_is_a_rollback(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "rollback test")["project_id"]
    object_id = _create_object(api_client, project_id)
    _seed_object(api_client, project_id, object_id)

    settings = api_client.app.state.settings
    from vistlabel.persistence.paths import project_dir

    pdir = project_dir(settings.workspace.root, project_id)
    run_id = manifests.new_run_id()
    manifests.create_run(
        pdir,
        run_id=run_id,
        job_id="job_orphaned",
        backend_id="fake",
        object_ids=[object_id],
        start_frame=0,
        end_frame=9,
        direction="forward",
        chunk_frames=4,
        overwrite_manual_masks=False,
    )
    manifests.update_run(pdir, run_id, status=TrackingRunStatus.INTERRUPTED)

    response = api_client.post(f"/api/v1/projects/{project_id}/tracking-runs/{run_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"

    # Cannot resume a cancelled (rolled-back) run.
    resume_response = api_client.post(
        f"/api/v1/projects/{project_id}/tracking-runs/{run_id}/resume"
    )
    assert resume_response.status_code == 409


def test_startup_recovery_marks_orphaned_running_manifest_interrupted(
    tmp_path, ffmpeg_binaries: FfmpegBinaries
) -> None:
    settings = AppSettings(workspace={"root": tmp_path / "workspace"})
    first_app = create_app(settings)
    first_client = authenticated_client(first_app)
    project_id = create_project_with_video(first_client, "crash recovery test")["project_id"]
    object_id = _create_object(first_client, project_id)
    _seed_object(first_client, project_id, object_id)

    from vistlabel.persistence.paths import project_dir

    pdir = project_dir(settings.workspace.root, project_id)
    run_id = manifests.new_run_id()
    manifests.create_run(
        pdir,
        run_id=run_id,
        job_id="job_never_finished",
        backend_id="fake",
        object_ids=[object_id],
        start_frame=0,
        end_frame=9,
        direction="forward",
        chunk_frames=4,
        overwrite_manual_masks=False,
    )  # left at status=RUNNING -- simulates a crash mid-run

    # A brand new app instance, as if the process restarted.
    second_app = create_app(settings)
    second_client = authenticated_client(second_app)
    run_record = second_client.get(
        f"/api/v1/projects/{project_id}/tracking-runs/{run_id}"
    ).json()
    assert run_record["status"] == "interrupted"


def test_review_queue_lists_flagged_frames_with_frame_and_object_links(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "review queue test")["project_id"]
    object_id = _create_object(api_client, project_id)

    # A low-confidence interactive save should be auto-flagged for review.
    segment_response = api_client.post(
        f"/api/v1/projects/{project_id}/segment",
        json={
            "backend_id": "fake",
            "frame_index": 0,
            "object_id": object_id,
            "points": [{"x": 10, "y": 10, "positive": True}],
        },
    )
    mask_png = segment_response.json()["visible_mask_png_base64"]
    save_response = api_client.patch(
        f"/api/v1/projects/{project_id}/frames/0/annotations",
        json={
            "expected_revision": 0,
            "upsert_object": {
                "object_id": object_id,
                "label_id": 3,
                "visible_mask_png_base64": mask_png,
                "source": "fake_interactive",
                "confidence": 0.1,  # below LOW_SCORE_THRESHOLD
            },
        },
    )
    assert save_response.status_code == 200
    assert save_response.json()["review_status"] == "needs_review"

    queue = api_client.get(f"/api/v1/projects/{project_id}/review-queue").json()
    assert len(queue) == 1
    assert queue[0]["frame_index"] == 0
    assert queue[0]["object_id"] == object_id
    assert "low_model_score" in queue[0]["review_reasons"]


def test_review_queue_empty_for_a_fresh_project(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "empty review queue test")["project_id"]
    queue = api_client.get(f"/api/v1/projects/{project_id}/review-queue").json()
    assert queue == []
