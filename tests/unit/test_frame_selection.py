# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Frame inclusion policies, train/val split, negative sampling, and
adjacent-frame leakage detection."""

from __future__ import annotations

import pytest

from vistlabel.domain.exports import FrameSelectionPolicy
from vistlabel.domain.frames import FrameAnnotation, FrameObjectAnnotation, ReviewStatus, RleMask
from vistlabel.domain.project import FrameRangeAssignment, SplitLabel
from vistlabel.exports.frame_selection import (
    assign_splits,
    detect_adjacent_split_leakage,
    sample_negative_frames,
    select_positive_frames,
)

_RLE = RleMask(size=(10, 10), counts="fake")


def _frame(
    frame_index: int,
    *,
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED,
    has_object: bool = False,
) -> FrameAnnotation:
    objects = (
        [
            FrameObjectAnnotation(
                object_id="obj_1",
                label_id=0,
                visible_mask_rle=_RLE,
                connected_component_count=1,
                source="test",
            )
        ]
        if has_object
        else []
    )
    return FrameAnnotation(
        frame_index=frame_index, timestamp_seconds=frame_index / 10, review_status=review_status,
        objects=objects,
    )


def test_annotated_only_selects_frames_with_objects() -> None:
    frames = {0: _frame(0, has_object=True), 1: _frame(1, has_object=False)}
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.ANNOTATED_ONLY,
        every_nth=1,
        selected_intervals=[],
    )
    assert selected == [0]


def test_annotated_only_skips_frames_whose_only_mask_is_empty() -> None:
    # An erased mask leaves an object with zero components and no box --
    # the overview doesn't count that frame as annotated, so neither does
    # the export.
    erased = _frame(0, has_object=True)
    erased.objects[0].connected_component_count = 0
    frames = {0: erased, 1: _frame(1, has_object=True)}
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.ANNOTATED_ONLY,
        every_nth=1,
        selected_intervals=[],
    )
    assert selected == [1]


def test_manually_edited_only_selects_frames_with_manual_edits() -> None:
    manual_frame = _frame(0, has_object=True)
    manual_frame.objects[0].manually_modified = True
    frames = {0: manual_frame, 1: _frame(1, has_object=True)}
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.MANUALLY_EDITED_ONLY,
        every_nth=1,
        selected_intervals=[],
    )
    assert selected == [0]


def test_reviewed_only_selects_reviewed_frames() -> None:
    frames = {
        0: _frame(0, review_status=ReviewStatus.REVIEWED),
        1: _frame(1, review_status=ReviewStatus.NEEDS_REVIEW),
    }
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.REVIEWED_ONLY,
        every_nth=1,
        selected_intervals=[],
    )
    assert selected == [0]


def test_every_nth_reviewed_applies_stride() -> None:
    frames = {i: _frame(i, review_status=ReviewStatus.REVIEWED) for i in range(10)}
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.EVERY_NTH_REVIEWED,
        every_nth=3,
        selected_intervals=[],
    )
    assert selected == [0, 3, 6, 9]


def test_every_nth_annotated_strides_over_annotated_frames_only() -> None:
    frames = {i: _frame(i, has_object=i % 2 == 0) for i in range(12)}  # 0, 2, ..., 10
    frames[4] = frames[4].model_copy(update={"excluded": True})  # deleted frame
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.EVERY_NTH_ANNOTATED,
        every_nth=2,
        selected_intervals=[],
    )
    # Annotated: 0, 2, 6, 8, 10 (review status doesn't matter).
    assert selected == [0, 6, 10]


def test_selected_intervals_filters_by_range() -> None:
    frames = {i: _frame(i) for i in range(10)}
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.SELECTED_INTERVALS,
        every_nth=1,
        selected_intervals=[(2, 4), (8, 9)],
    )
    assert selected == [2, 3, 4, 8, 9]


def test_all_frames_selects_everything_present() -> None:
    frames = {0: _frame(0), 5: _frame(5)}
    selected = select_positive_frames(
        frame_annotations=frames,
        policy=FrameSelectionPolicy.ALL_FRAMES,
        every_nth=1,
        selected_intervals=[],
    )
    assert selected == [0, 5]


def test_sample_negative_frames_respects_fraction_and_excludes_positives() -> None:
    negatives = sample_negative_frames(
        total_frame_count=100,
        positive_frame_indices={0, 1, 2, 3, 4},
        excluded_frame_indices=set(),
        negative_frame_fraction=0.2,
    )
    assert len(negatives) == 1
    assert negatives[0] not in {0, 1, 2, 3, 4}


def test_sample_negative_frames_disabled_when_fraction_is_zero() -> None:
    assert (
        sample_negative_frames(
            total_frame_count=100,
            positive_frame_indices={0, 1},
            excluded_frame_indices=set(),
            negative_frame_fraction=0.0,
        )
        == []
    )


def test_assign_splits_without_assignments_falls_back_to_contiguous_block() -> None:
    split_by_frame, excluded = assign_splits(frame_indices=list(range(10)), split_assignments=[])
    assert excluded == []
    assert split_by_frame[0] == "train"
    assert split_by_frame[9] == "val"
    # A contiguous block, not a scattered random split.
    val_frames = sorted(idx for idx, split in split_by_frame.items() if split == "val")
    assert val_frames == list(range(min(val_frames), max(val_frames) + 1))


def test_assign_splits_with_explicit_assignments() -> None:
    assignments = [
        FrameRangeAssignment(start_frame=0, end_frame=4, label=SplitLabel.TRAIN),
        FrameRangeAssignment(start_frame=5, end_frame=7, label=SplitLabel.VAL),
        FrameRangeAssignment(start_frame=8, end_frame=9, label=SplitLabel.EXCLUDED),
    ]
    split_by_frame, excluded = assign_splits(
        frame_indices=list(range(10)), split_assignments=assignments
    )
    assert all(split_by_frame[i] == "train" for i in range(5))
    assert all(split_by_frame[i] == "val" for i in range(5, 8))
    assert set(excluded) == {8, 9}


def test_assign_splits_excludes_frames_with_no_covering_assignment() -> None:
    assignments = [FrameRangeAssignment(start_frame=0, end_frame=2, label=SplitLabel.TRAIN)]
    split_by_frame, excluded = assign_splits(frame_indices=[0, 1, 5], split_assignments=assignments)
    assert 5 in excluded
    assert 5 not in split_by_frame


def test_detect_adjacent_split_leakage_flags_neighboring_different_splits() -> None:
    split_by_frame = {0: "train", 1: "val", 2: "val"}
    risks = detect_adjacent_split_leakage(split_by_frame)
    assert len(risks) == 1
    assert "frame 0" in risks[0] and "frame 1" in risks[0]


def test_detect_adjacent_split_leakage_no_risk_when_splits_are_not_numerically_adjacent() -> None:
    split_by_frame = {0: "train", 1: "train", 10: "val", 11: "val"}
    assert detect_adjacent_split_leakage(split_by_frame) == []


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        select_positive_frames(
            frame_annotations={},
            policy="not-a-real-policy",  # type: ignore[arg-type]
            every_nth=1,
            selected_intervals=[],
        )
