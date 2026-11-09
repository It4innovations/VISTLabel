# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Per-frame annotation persistence with optimistic-revision autosave
(used by the brush/eraser commit flow).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from vistlabel.annotations.errors import RevisionConflictError
from vistlabel.domain.frames import FrameAnnotation, ReviewStatus
from vistlabel.persistence.atomic import load_model_migrated, save_model
from vistlabel.persistence.paths import annotated_frame_paths, frame_annotation_path


def get_frame_annotation(
    project_dir: Path, frame_index: int, timestamp_seconds: float
) -> FrameAnnotation:
    path = frame_annotation_path(project_dir, frame_index)
    if not path.exists():
        return FrameAnnotation(frame_index=frame_index, timestamp_seconds=timestamp_seconds)
    return load_model_migrated(path, FrameAnnotation, "frame_annotation")


def count_frame_progress(project_dir: Path) -> tuple[int, int]:
    """``(annotated, reviewed)`` frame counts from a full scan of the
    persisted frame files (see :attr:`FrameAnnotation.is_annotated`)."""
    annotated = reviewed = 0
    for path in annotated_frame_paths(project_dir):
        frame = load_model_migrated(path, FrameAnnotation, "frame_annotation")
        if frame.is_annotated:
            annotated += 1
        if frame.review_status is ReviewStatus.REVIEWED:
            reviewed += 1
    return annotated, reviewed


def _drop_superseded_inactive(frame: FrameAnnotation) -> FrameAnnotation:
    """An inactive entry is stale once the same object has a mask again (a
    new segment, or propagation told to overwrite manual edits): the new
    mask wins, and Activate would otherwise resurrect the old one over it."""
    active_ids = {obj.object_id for obj in frame.objects}
    kept = [obj for obj in frame.inactive_objects if obj.object_id not in active_ids]
    if len(kept) == len(frame.inactive_objects):
        return frame
    return frame.model_copy(update={"inactive_objects": kept})


def save_frame_annotation(
    project_dir: Path,
    frame_index: int,
    timestamp_seconds: float,
    expected_revision: int,
    mutate: Callable[[FrameAnnotation], FrameAnnotation],
) -> FrameAnnotation:
    current = get_frame_annotation(project_dir, frame_index, timestamp_seconds)
    if current.revision != expected_revision:
        raise RevisionConflictError(f"frame {frame_index}", expected_revision, current.revision)
    updated = _drop_superseded_inactive(mutate(current))
    updated = updated.model_copy(update={"revision": current.revision + 1})
    save_model(frame_annotation_path(project_dir, frame_index), updated)
    return updated


def apply_frame_annotation(
    project_dir: Path,
    frame_index: int,
    timestamp_seconds: float,
    mutate: Callable[[FrameAnnotation], FrameAnnotation],
) -> FrameAnnotation:
    """Unconditional variant of :func:`save_frame_annotation` for automated
    writers (the propagation job) that are the sole writer for the frames
    they touch and should not abort on a revision mismatch the way a human
    editing session must (the optimistic check is a human-concurrency
    guard, not meaningful for a job that owns the whole propagation range
    for its duration)."""
    current = get_frame_annotation(project_dir, frame_index, timestamp_seconds)
    updated = _drop_superseded_inactive(mutate(current))
    updated = updated.model_copy(update={"revision": current.revision + 1})
    save_model(frame_annotation_path(project_dir, frame_index), updated)
    return updated
