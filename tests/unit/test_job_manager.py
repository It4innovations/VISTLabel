# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from vistlabel.domain.jobs import JobRecord, JobStatus
from vistlabel.jobs.manager import JobContext, JobManager
from vistlabel.persistence.atomic import save_model
from vistlabel.persistence.paths import job_path, project_dir


def _wait_until_terminal(manager: JobManager, project_id: str, job_id: str, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    terminal = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.PAUSED}
    while time.monotonic() < deadline:
        record = manager.get(project_id, job_id)
        if record.status in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not reach a terminal state within {timeout}s")


def test_successful_job_persists_succeeded_status(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)

    def work(context: JobContext) -> None:
        pass

    job_id = manager.submit("proj-1", "demo", work)
    record = _wait_until_terminal(manager, "proj-1", job_id)

    assert record.status == JobStatus.SUCCEEDED
    assert record.progress_fraction == 1.0
    assert record.error is None


def test_failing_job_persists_error(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)

    def work(context: JobContext) -> None:
        raise ValueError("synthetic failure")

    job_id = manager.submit("proj-1", "demo", work)
    record = _wait_until_terminal(manager, "proj-1", job_id)

    assert record.status == JobStatus.FAILED
    assert "synthetic failure" in record.error


def test_cancelled_job_persists_cancelled_status(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    started = threading.Event()

    def work(context: JobContext) -> None:
        started.set()
        for _ in range(200):
            context.raise_if_cancelled()
            time.sleep(0.01)

    job_id = manager.submit("proj-1", "demo", work)
    assert started.wait(timeout=2.0)
    cancelled = manager.cancel(job_id)
    record = _wait_until_terminal(manager, "proj-1", job_id)

    assert cancelled is True
    assert record.status == JobStatus.CANCELLED


def test_cancel_of_unknown_job_returns_false(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)

    assert manager.cancel("job_doesnotexist") is False


def test_paused_job_persists_paused_status(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    started = threading.Event()

    def work(context: JobContext) -> None:
        started.set()
        for _ in range(200):
            context.raise_if_paused()
            time.sleep(0.01)

    job_id = manager.submit("proj-1", "demo", work)
    assert started.wait(timeout=2.0)
    paused = manager.pause(job_id)
    record = _wait_until_terminal(manager, "proj-1", job_id)

    assert paused is True
    assert record.status == JobStatus.PAUSED


def test_pause_of_unknown_job_returns_false(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)

    assert manager.pause("job_doesnotexist") is False


def test_pause_and_cancel_are_independent_signals(tmp_path: Path) -> None:
    """Pausing a job must not also look cancelled to it, and vice versa."""
    manager = JobManager(tmp_path)
    started = threading.Event()

    def work(context: JobContext) -> None:
        started.set()
        for _ in range(200):
            context.raise_if_paused()
            assert context.is_cancelled() is False
            time.sleep(0.01)

    job_id = manager.submit("proj-1", "demo", work)
    assert started.wait(timeout=2.0)
    manager.pause(job_id)
    record = _wait_until_terminal(manager, "proj-1", job_id)
    assert record.status == JobStatus.PAUSED


def test_report_progress_updates_persisted_fraction(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)
    ready = threading.Event()
    proceed = threading.Event()

    def work(context: JobContext) -> None:
        manager.report_progress("proj-1", context.job_id, 0.5)
        ready.set()
        proceed.wait(timeout=2.0)

    job_id = manager.submit("proj-1", "demo", work)
    assert ready.wait(timeout=2.0)
    record = manager.get("proj-1", job_id)
    assert record.progress_fraction == 0.5
    proceed.set()
    _wait_until_terminal(manager, "proj-1", job_id)


def test_get_unknown_job_raises(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)

    with pytest.raises(FileNotFoundError):
        manager.get("proj-1", "job_doesnotexist")


def test_job_survives_a_new_manager_instance_simulating_restart(tmp_path: Path) -> None:
    first = JobManager(tmp_path)

    def work(context: JobContext) -> None:
        pass

    job_id = first.submit("proj-1", "demo", work)
    _wait_until_terminal(first, "proj-1", job_id)

    second = JobManager(tmp_path)
    record = second.get("proj-1", job_id)
    assert record.status == JobStatus.SUCCEEDED


def test_recover_interrupted_jobs_marks_stuck_running_jobs_failed(tmp_path: Path) -> None:
    """Simulates a crash: a job record left in RUNNING status on disk, with
    no thread actually running it (this test never called .submit())."""
    from datetime import UTC, datetime

    pdir = project_dir(tmp_path, "proj-1")
    stuck = JobRecord(
        job_id="job_stuck",
        job_type="propagate",
        project_id="proj-1",
        status=JobStatus.RUNNING,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    save_model(job_path(pdir, "job_stuck"), stuck)

    manager = JobManager(tmp_path)
    recovered = manager.recover_interrupted_jobs()

    assert "job_stuck" in recovered
    record = manager.get("proj-1", "job_stuck")
    assert record.status == JobStatus.FAILED
    assert "interrupted" in record.error


def test_recover_interrupted_jobs_leaves_terminal_jobs_alone(tmp_path: Path) -> None:
    manager = JobManager(tmp_path)

    def work(context: JobContext) -> None:
        pass

    job_id = manager.submit("proj-1", "demo", work)
    _wait_until_terminal(manager, "proj-1", job_id)

    recovered = manager.recover_interrupted_jobs()

    assert job_id not in recovered
    assert manager.get("proj-1", job_id).status == JobStatus.SUCCEEDED
