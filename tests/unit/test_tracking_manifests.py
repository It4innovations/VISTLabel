# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Persisted tracking-run manifests and checkpoints: create/get/update/list
plus the crash-recovery scan."""

from __future__ import annotations

from pathlib import Path

import pytest

from vistlabel.domain.tracking import TrackingRunStatus
from vistlabel.tracking import manifests


@pytest.fixture
def pdir(tmp_path: Path) -> Path:
    (tmp_path / "tracking" / "runs").mkdir(parents=True)
    return tmp_path


def test_create_and_get_round_trip(pdir: Path) -> None:
    run_id = manifests.new_run_id()
    created = manifests.create_run(
        pdir,
        run_id=run_id,
        job_id="job_1",
        backend_id="fake",
        object_ids=["obj_1"],
        start_frame=0,
        end_frame=9,
        direction="forward",
        chunk_frames=4,
        overwrite_manual_masks=False,
    )
    assert created.status == TrackingRunStatus.RUNNING
    assert created.last_completed_frame is None

    fetched = manifests.get_run(pdir, run_id)
    assert fetched == created


def test_update_run_changes_fields_and_bumps_updated_at(pdir: Path) -> None:
    run_id = manifests.new_run_id()
    created = manifests.create_run(
        pdir,
        run_id=run_id,
        job_id="job_1",
        backend_id="fake",
        object_ids=["obj_1"],
        start_frame=0,
        end_frame=9,
        direction="forward",
        chunk_frames=4,
        overwrite_manual_masks=False,
    )
    updated = manifests.update_run(pdir, run_id, last_completed_frame=3)
    assert updated.last_completed_frame == 3
    assert updated.updated_at >= created.updated_at
    assert updated.status == TrackingRunStatus.RUNNING  # untouched fields survive


def test_list_runs_returns_every_manifest(pdir: Path) -> None:
    for _ in range(3):
        run_id = manifests.new_run_id()
        manifests.create_run(
            pdir,
            run_id=run_id,
            job_id="job_x",
            backend_id="fake",
            object_ids=[],
            start_frame=0,
            end_frame=1,
            direction="forward",
            chunk_frames=4,
            overwrite_manual_masks=False,
        )
    assert len(manifests.list_runs(pdir)) == 3


def test_list_runs_empty_when_no_runs_dir(tmp_path: Path) -> None:
    assert manifests.list_runs(tmp_path) == []


def test_recover_interrupted_runs_marks_running_as_interrupted(pdir: Path) -> None:
    run_id = manifests.new_run_id()
    manifests.create_run(
        pdir,
        run_id=run_id,
        job_id="job_1",
        backend_id="fake",
        object_ids=["obj_1"],
        start_frame=0,
        end_frame=9,
        direction="forward",
        chunk_frames=4,
        overwrite_manual_masks=False,
    )  # status defaults to RUNNING -- simulates a manifest left behind by a crash

    recovered = manifests.recover_interrupted_runs(pdir)

    assert recovered == [run_id]
    assert manifests.get_run(pdir, run_id).status == TrackingRunStatus.INTERRUPTED


def test_recover_interrupted_runs_leaves_completed_runs_alone(pdir: Path) -> None:
    run_id = manifests.new_run_id()
    manifests.create_run(
        pdir,
        run_id=run_id,
        job_id="job_1",
        backend_id="fake",
        object_ids=["obj_1"],
        start_frame=0,
        end_frame=9,
        direction="forward",
        chunk_frames=4,
        overwrite_manual_masks=False,
    )
    manifests.update_run(pdir, run_id, status=TrackingRunStatus.COMPLETED)

    recovered = manifests.recover_interrupted_runs(pdir)

    assert recovered == []
    assert manifests.get_run(pdir, run_id).status == TrackingRunStatus.COMPLETED
