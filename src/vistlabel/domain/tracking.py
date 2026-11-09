# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Tracking-run manifest schema for chunked, checkpointed propagation whose
results are persisted incrementally by chunk. A manifest is the
*resumable* record of one propagation
request -- independent of the generic :class:`~vistlabel.domain.jobs.JobRecord`,
whose ``job_id`` only identifies one execution *attempt*: pausing or
recovering from a crash starts a new job attempt against the same manifest.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class TrackingRunStatus(StrEnum):
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    #: Discovered at startup: the process died while this run's status was
    #: still "running". A restart identifies such incomplete work and offers
    #: safe resume or rollback -- it is never auto-resumed silently.
    INTERRUPTED = "interrupted"


class TrackingRunManifest(BaseModel):
    schema_version: int = 1
    run_id: str
    #: The JobRecord for the *current* execution attempt; changes across a
    #: pause/resume or crash/resume cycle, while run_id stays stable.
    job_id: str
    backend_id: str
    object_ids: list[str]
    start_frame: int
    end_frame: int
    direction: Literal["forward", "backward"]
    chunk_frames: int
    #: Highest frame index actually committed to disk so far, in the
    #: propagation's own direction of travel; None means no chunk has
    #: completed yet. Resuming continues from here, not from start_frame.
    last_completed_frame: int | None = None
    status: TrackingRunStatus = TrackingRunStatus.RUNNING
    #: object_ids skipped for at least one frame because an existing mask
    #: there was manually_modified (propagation never overwrites manually
    #: reviewed masks without an explicit policy) -- informational, not an
    #: error.
    manual_masks_protected_count: int = 0
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    #: True once the caller explicitly asked to overwrite manually-modified
    #: masks for this run (explicit opt-in -- the default always protects
    #: them).
    overwrite_manual_masks: bool = False
