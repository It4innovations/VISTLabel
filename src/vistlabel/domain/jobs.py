# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""GPU job state schema, populated by the job manager. Persisted job
records have a stable, versioned shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    #: Cooperatively paused at a chunk boundary -- unlike CANCELLED, a
    #: paused job's tracking-run manifest can be resumed with a new job
    #: attempt continuing from the last checkpoint.
    PAUSED = "paused"


class JobRecord(BaseModel):
    schema_version: int = 1
    job_id: str
    job_type: str
    project_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime
    updated_at: datetime
    progress_fraction: float | None = None
    error: str | None = None
