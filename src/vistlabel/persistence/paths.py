# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Project workspace layout: a single source of truth for every
path inside a project directory, so no other module hand-builds these
strings.
"""

from __future__ import annotations

from pathlib import Path

#: Directories created for every new project, relative to the project root.
#: `source/` starts empty -- upload finalizes `source/video.mp4` into it.
PROJECT_SUBDIRS: tuple[str, ...] = (
    "source",
    "cache/frames",
    "cache/thumbnails",
    "cache/model_sessions",
    "annotations/frames",
    "prompts/frames",
    "tracking/runs",
    "tracking/checkpoints",
    "jobs",
    "locks",
    "logs",
    "exports",
)

FRAME_SHARD_SIZE = 1000


def projects_root(workspace_root: Path) -> Path:
    return workspace_root / "projects"


def project_dir(workspace_root: Path, project_id: str) -> Path:
    return projects_root(workspace_root) / project_id


def project_json_path(project_dir: Path) -> Path:
    return project_dir / "project.json"


def labels_json_path(project_dir: Path) -> Path:
    return project_dir / "labels.json"


def objects_json_path(project_dir: Path) -> Path:
    return project_dir / "annotations" / "objects.json"


def deleted_objects_dir(project_dir: Path) -> Path:
    return project_dir / "annotations" / "deleted_objects"


def deleted_object_path(project_dir: Path, object_id: str) -> Path:
    return deleted_objects_dir(project_dir) / f"{object_id}.json"


def events_log_path(project_dir: Path) -> Path:
    return project_dir / "logs" / "events.jsonl"


def editor_lock_path(project_dir: Path) -> Path:
    return project_dir / "locks" / "editor.lock.json"


def source_video_path(project_dir: Path) -> Path:
    return project_dir / "source" / "video.mp4"


def frame_shard_dir(project_dir: Path, frame_index: int) -> Path:
    shard = f"{frame_index // FRAME_SHARD_SIZE:03d}"
    return project_dir / "annotations" / "frames" / shard


def frame_annotation_path(project_dir: Path, frame_index: int) -> Path:
    return frame_shard_dir(project_dir, frame_index) / f"{frame_index:06d}.json"


def job_path(project_dir: Path, job_id: str) -> Path:
    return project_dir / "jobs" / f"{job_id}.json"


def jobs_dir(project_dir: Path) -> Path:
    return project_dir / "jobs"


def tracking_run_path(project_dir: Path, run_id: str) -> Path:
    return project_dir / "tracking" / "runs" / f"{run_id}.json"


def tracking_runs_dir(project_dir: Path) -> Path:
    return project_dir / "tracking" / "runs"


def exports_dir(project_dir: Path) -> Path:
    return project_dir / "exports"


def export_dir(project_dir: Path, export_id: str) -> Path:
    return exports_dir(project_dir) / export_id


def export_manifest_path(project_dir: Path, export_id: str) -> Path:
    return export_dir(project_dir, export_id) / "export_manifest.json"


def validation_report_path(project_dir: Path, export_id: str) -> Path:
    return export_dir(project_dir, export_id) / "validation_report.json"


def export_zip_path(project_dir: Path, export_id: str) -> Path:
    return export_dir(project_dir, export_id) / "dataset.zip"


def annotated_frame_paths(project_dir: Path) -> list[Path]:
    """Every persisted per-frame annotation file, across all shards
    (the ``annotations/frames/<shard>/<frame_index>.json`` layout) -- a
    full scan, not an index; fine at this project's filesystem-only
    scale."""
    frames_dir = project_dir / "annotations" / "frames"
    if not frames_dir.is_dir():
        return []
    return sorted(frames_dir.glob("*/*.json"))
