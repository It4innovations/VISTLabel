# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from vistlabel.annotations.errors import RevisionConflictError
from vistlabel.annotations.frames_service import get_frame_annotation, save_frame_annotation
from vistlabel.domain.frames import ReviewStatus


def test_get_frame_annotation_returns_default_when_no_file_exists(tmp_path: Path) -> None:
    annotation = get_frame_annotation(tmp_path, 5, timestamp_seconds=0.5)

    assert annotation.frame_index == 5
    assert annotation.timestamp_seconds == 0.5
    assert annotation.revision == 0
    assert annotation.objects == []


def test_save_frame_annotation_bumps_revision_and_persists(tmp_path: Path) -> None:
    updated = save_frame_annotation(
        tmp_path,
        0,
        0.0,
        expected_revision=0,
        mutate=lambda frame: frame.model_copy(update={"review_status": ReviewStatus.REVIEWED}),
    )

    assert updated.revision == 1
    assert updated.review_status == ReviewStatus.REVIEWED

    reloaded = get_frame_annotation(tmp_path, 0, timestamp_seconds=0.0)
    assert reloaded == updated


def test_save_frame_annotation_with_stale_revision_raises(tmp_path: Path) -> None:
    save_frame_annotation(tmp_path, 0, 0.0, expected_revision=0, mutate=lambda frame: frame)

    with pytest.raises(RevisionConflictError):
        save_frame_annotation(tmp_path, 0, 0.0, expected_revision=0, mutate=lambda frame: frame)
