# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Frame inclusion policies, train/validation/excluded split, and negative
-frame sampling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from vistlabel.domain.exports import FrameSelectionPolicy
from vistlabel.domain.frames import FrameAnnotation, ReviewStatus
from vistlabel.domain.project import FrameRangeAssignment, SplitLabel

#: Fallback when a single-video project has no explicit train/val/excluded
#: intervals: a contiguous temporal block at the end of the selected frames
#: becomes validation ("contiguous temporal blocks when only one video
#: exists" -- the least preferred splitting strategy, but the only one
#: possible with exactly one source video per project).
_DEFAULT_VAL_FRACTION = 0.2


@dataclass(frozen=True)
class SelectedFrame:
    frame_index: int
    is_negative: bool
    split: Literal["train", "val"]


def _split_label_for_frame(
    frame_index: int, assignments: list[FrameRangeAssignment]
) -> SplitLabel | None:
    for assignment in assignments:
        if assignment.start_frame <= frame_index <= assignment.end_frame:
            return assignment.label
    return None


def select_positive_frames(
    *,
    frame_annotations: dict[int, FrameAnnotation],
    policy: FrameSelectionPolicy,
    every_nth: int,
    selected_intervals: list[tuple[int, int]],
) -> list[int]:
    """Frames with real annotations, chosen per the selection policy --
    "positive" as opposed to the empty-label negative frames sampled
    separately by :func:`sample_negative_frames`."""
    if policy is FrameSelectionPolicy.ALL_FRAMES:
        return sorted(frame_annotations)

    if policy is FrameSelectionPolicy.ANNOTATED_ONLY:
        # Same rule as the overview's "annotated" count, so the two agree.
        return sorted(idx for idx, frame in frame_annotations.items() if frame.is_annotated)

    if policy is FrameSelectionPolicy.MANUALLY_EDITED_ONLY:
        return sorted(
            idx
            for idx, frame in frame_annotations.items()
            if any(obj.manually_modified for obj in frame.objects)
        )

    if policy is FrameSelectionPolicy.REVIEWED_ONLY:
        return sorted(
            idx
            for idx, frame in frame_annotations.items()
            if frame.review_status is ReviewStatus.REVIEWED
        )

    if policy is FrameSelectionPolicy.EVERY_NTH_REVIEWED:
        reviewed = sorted(
            idx
            for idx, frame in frame_annotations.items()
            if frame.review_status is ReviewStatus.REVIEWED
        )
        stride = max(1, every_nth)
        return reviewed[::stride]

    if policy is FrameSelectionPolicy.EVERY_NTH_ANNOTATED:
        annotated = sorted(idx for idx, frame in frame_annotations.items() if frame.is_annotated)
        return annotated[:: max(1, every_nth)]

    if policy is FrameSelectionPolicy.SELECTED_INTERVALS:
        return sorted(
            idx
            for idx in frame_annotations
            if any(start <= idx <= end for start, end in selected_intervals)
        )

    raise ValueError(f"unknown frame selection policy: {policy!r}")  # pragma: no cover


def sample_negative_frames(
    *,
    total_frame_count: int,
    positive_frame_indices: set[int],
    excluded_frame_indices: set[int],
    negative_frame_fraction: float,
) -> list[int]:
    """A controlled fraction of negative frames with empty TXT files.
    Sampled with a fixed stride across the whole candidate pool rather
    than taking a contiguous block, so negatives aren't thousands of
    nearly identical neighboring frames."""
    if negative_frame_fraction <= 0 or not positive_frame_indices:
        return []
    candidates = [
        idx
        for idx in range(total_frame_count)
        if idx not in positive_frame_indices and idx not in excluded_frame_indices
    ]
    if not candidates:
        return []
    target_count = max(1, round(len(positive_frame_indices) * negative_frame_fraction))
    if target_count >= len(candidates):
        return candidates
    stride = len(candidates) / target_count
    indices = [candidates[int(i * stride)] for i in range(target_count)]
    return sorted(set(indices))


def assign_splits(
    *,
    frame_indices: list[int],
    split_assignments: list[FrameRangeAssignment],
) -> tuple[dict[int, Literal["train", "val"]], list[int]]:
    """Returns ``(split_by_frame, excluded_frame_indices)``. When the
    project has no explicit split assignments at all, falls back to a
    contiguous temporal block rather than a random per-frame split, which
    would leak near-duplicate neighboring frames across train/val."""
    if not split_assignments:
        ordered = sorted(frame_indices)
        val_count = max(1, round(len(ordered) * _DEFAULT_VAL_FRACTION)) if ordered else 0
        val_set = set(ordered[len(ordered) - val_count :]) if val_count else set()
        return {idx: ("val" if idx in val_set else "train") for idx in ordered}, []

    split_by_frame: dict[int, Literal["train", "val"]] = {}
    excluded: list[int] = []
    for idx in frame_indices:
        label = _split_label_for_frame(idx, split_assignments)
        if label is None or label is SplitLabel.EXCLUDED:
            excluded.append(idx)
        elif label is SplitLabel.TRAIN:
            split_by_frame[idx] = "train"
        else:
            split_by_frame[idx] = "val"
    return split_by_frame, excluded


def detect_adjacent_split_leakage(split_by_frame: Mapping[int, str]) -> list[str]:
    """Nearly identical adjacent frames must not create misleading
    validation results. Flags any pair of frame indices one
    apart that landed in different splits -- a real, checkable proxy for
    "these are probably near-duplicates split across train/val"."""
    risks: list[str] = []
    for idx, split in split_by_frame.items():
        neighbor_split = split_by_frame.get(idx + 1)
        if neighbor_split is not None and neighbor_split != split:
            risks.append(
                f"frame {idx} ({split}) and frame {idx + 1} ({neighbor_split}) are adjacent "
                "but assigned to different splits"
            )
    return risks
