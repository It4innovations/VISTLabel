# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Local GPU job manager: bounded background execution, cooperative
cancellation, progress reporting, and persisted job state that survives a
restart (a filesystem job record is not a database, but it must be
reconstructable).

Its job is keeping blocking work -- backend ``propagate`` calls and other
GPU inference -- off the FastAPI event loop and safely cancellable: GPU
inference and video decoding must never block the event loop.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from vistlabel.domain.jobs import JobRecord, JobStatus
from vistlabel.persistence.atomic import load_model, save_model
from vistlabel.persistence.paths import job_path, jobs_dir, project_dir, projects_root


class JobCancelled(Exception):
    pass


class JobPaused(Exception):
    """Distinct from :class:`JobCancelled`: work honoring a pause request
    has committed everything up to a clean checkpoint boundary and can be
    resumed later, unlike a cancelled job."""

    pass


@dataclass
class JobContext:
    job_id: str
    _cancel_event: threading.Event
    _pause_event: threading.Event

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled():
            raise JobCancelled(self.job_id)

    def is_pause_requested(self) -> bool:
        return self._pause_event.is_set()

    def raise_if_paused(self) -> None:
        if self.is_pause_requested():
            raise JobPaused(self.job_id)


class JobManager:
    """One instance per running application (``app.state.job_manager``),
    not per project -- ``max_workers`` is the actual cross-project
    concurrency limit, matching ``runtime.maximum_concurrent_gpu_jobs``."""

    #: A chunked propagation reports progress once per frame, far more
    #: often than the UI's 400ms poll needs -- besides being wasted disk
    #: I/O, writing the job record that often makes the Windows
    #: read/replace race (see ``persistence.atomic``) easy to hit in
    #: practice. Throttling to this interval keeps progress reporting
    #: informative without a write on every single frame.
    _PROGRESS_REPORT_MIN_INTERVAL_SECONDS = 0.5

    def __init__(self, workspace_root: Path, max_workers: int = 1) -> None:
        self._workspace_root = workspace_root
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._cancel_events: dict[str, threading.Event] = {}
        self._pause_events: dict[str, threading.Event] = {}
        self._last_progress_report: dict[str, float] = {}
        self._lock = threading.Lock()

    def submit(
        self, project_id: str, job_type: str, work: Callable[[JobContext], None]
    ) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        pdir = project_dir(self._workspace_root, project_id)
        now = datetime.now(UTC)
        record = JobRecord(
            job_id=job_id,
            job_type=job_type,
            project_id=project_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        save_model(job_path(pdir, job_id), record)

        cancel_event = threading.Event()
        pause_event = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = cancel_event
            self._pause_events[job_id] = pause_event
        context = JobContext(job_id=job_id, _cancel_event=cancel_event, _pause_event=pause_event)
        self._executor.submit(self._run, pdir, project_id, job_id, work, context)
        return job_id

    def _run(
        self,
        pdir: Path,
        project_id: str,
        job_id: str,
        work: Callable[[JobContext], None],
        context: JobContext,
    ) -> None:
        self._update(pdir, job_id, status=JobStatus.RUNNING)
        try:
            work(context)
        except JobCancelled:
            self._update(pdir, job_id, status=JobStatus.CANCELLED)
            return
        except JobPaused:
            self._update(pdir, job_id, status=JobStatus.PAUSED)
            return
        except Exception as exc:  # noqa: BLE001 - captured into the job record, not swallowed
            self._recover_from_possible_cuda_oom()
            self._update(pdir, job_id, status=JobStatus.FAILED, error=str(exc))
            return
        finally:
            with self._lock:
                self._cancel_events.pop(job_id, None)
                self._pause_events.pop(job_id, None)
                self._last_progress_report.pop(job_id, None)
        self._update(pdir, job_id, status=JobStatus.SUCCEEDED, progress_fraction=1.0)

    def _recover_from_possible_cuda_oom(self) -> None:
        """CUDA out-of-memory recovery: clearing the cache
        after *any* job failure is cheap and harmless on CPU-only or
        non-OOM failures, and gives the next GPU job a clean allocator
        state instead of inheriting a possibly-fragmented one."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _update(
        self,
        pdir: Path,
        job_id: str,
        *,
        status: JobStatus | None = None,
        progress_fraction: float | None = None,
        error: str | None = None,
    ) -> JobRecord:
        path = job_path(pdir, job_id)
        current = load_model(path, JobRecord)
        changes: dict[str, object] = {"updated_at": datetime.now(UTC)}
        if status is not None:
            changes["status"] = status
        if progress_fraction is not None:
            changes["progress_fraction"] = progress_fraction
        if error is not None:
            changes["error"] = error
        updated = current.model_copy(update=changes)
        save_model(path, updated)
        return updated

    def report_progress(self, project_id: str, job_id: str, fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        now = time.monotonic()
        last = self._last_progress_report.get(job_id)
        if (
            last is not None
            and fraction < 1.0
            and (now - last) < self._PROGRESS_REPORT_MIN_INTERVAL_SECONDS
        ):
            return
        self._last_progress_report[job_id] = now
        pdir = project_dir(self._workspace_root, project_id)
        self._update(pdir, job_id, progress_fraction=fraction)

    def get(self, project_id: str, job_id: str) -> JobRecord:
        pdir = project_dir(self._workspace_root, project_id)
        return load_model(job_path(pdir, job_id), JobRecord)

    def cancel(self, job_id: str) -> bool:
        """Requests cooperative cancellation. Returns False if the job is
        unknown or already finished -- the caller decides whether that's an
        error (job never existed) or fine (already done)."""
        with self._lock:
            event = self._cancel_events.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def pause(self, job_id: str) -> bool:
        """Requests a cooperative pause at the next checkpoint boundary.
        Returns False if the job is unknown or already finished, the same
        as :meth:`cancel`."""
        with self._lock:
            event = self._pause_events.get(job_id)
        if event is None:
            return False
        event.set()
        return True

    def recover_interrupted_jobs(self) -> list[str]:
        """Called once at application startup (a job state file must
        remain reconstructable after a restart). Nothing can genuinely
        still be ``pending``/``running`` immediately after the process just
        started -- no thread survives a restart -- so any job record found
        in one of those states was interrupted by a crash and is marked
        ``failed`` with an explicit reason, never left silently stuck."""
        root = projects_root(self._workspace_root)
        recovered: list[str] = []
        if not root.exists():
            return recovered
        for project_entry in root.iterdir():
            if not project_entry.is_dir():
                continue
            directory = jobs_dir(project_entry)
            if not directory.is_dir():
                continue
            for job_file in sorted(directory.glob("*.json")):
                record = load_model(job_file, JobRecord)
                if record.status in (JobStatus.PENDING, JobStatus.RUNNING):
                    self._update(
                        project_entry,
                        record.job_id,
                        status=JobStatus.FAILED,
                        error="interrupted by process restart",
                    )
                    recovered.append(record.job_id)
        return recovered
