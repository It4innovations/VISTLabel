# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Occlusion-state and review-reason derivation. Pure functions over
already-computed mask facts (component count, score, the mask array
itself) plus optional prior-frame context -- no model-specific state,
since model-specific state must never be embedded in canonical
annotations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vistlabel.domain.frames import OcclusionState, ReviewReason

LOW_SCORE_THRESHOLD = 0.5
LARGE_AREA_CHANGE_FRACTION = 0.5
#: A centroid moving more than this fraction of the frame's diagonal in a
#: single frame step is "abrupt" -- relative to frame size, not a fixed
#: pixel count, so it scales across different video resolutions.
ABRUPT_CENTROID_DISPLACEMENT_FRACTION = 0.1


def compute_occlusion_state(component_count: int) -> OcclusionState:
    """A 2-component mask means ``occlusion_state: partially_occluded`` --
    this is that rule made explicit and applied everywhere a mask is saved,
    rather than left for each caller to set (inconsistently) by hand."""
    if component_count == 0:
        return OcclusionState.FULLY_OCCLUDED
    if component_count >= 2:
        return OcclusionState.PARTIALLY_OCCLUDED
    return OcclusionState.VISIBLE


def mask_touches_border(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def frame_local_review_reasons(mask: np.ndarray, score: float | None) -> list[ReviewReason]:
    """Checks that need only the current mask/score -- usable for a single
    interactive edit, not only during propagation."""
    reasons: list[ReviewReason] = []
    if score is not None and score < LOW_SCORE_THRESHOLD:
        reasons.append(ReviewReason.LOW_MODEL_SCORE)
    if mask_touches_border(mask):
        reasons.append(ReviewReason.MASK_TOUCHES_FRAME_BORDER)
    return reasons


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


@dataclass(frozen=True)
class PreviousFrameContext:
    component_count: int
    area_pixels: int
    centroid: tuple[float, float] | None = None


def previous_frame_context(mask: np.ndarray, component_count: int) -> PreviousFrameContext:
    return PreviousFrameContext(
        component_count=component_count, area_pixels=int(mask.sum()), centroid=_centroid(mask)
    )


def temporal_review_reasons(
    *,
    mask: np.ndarray,
    component_count: int,
    previous: PreviousFrameContext,
) -> list[ReviewReason]:
    """Checks that compare against the previous frame in a propagation
    sequence -- the disappearance/reappearance/component-count-change
    reasons only make sense across a temporal sequence, not for a
    single manual edit landing on one frame in isolation."""
    reasons: list[ReviewReason] = []
    area = int(mask.sum())

    if previous.component_count > 0 and component_count == 0:
        reasons.append(ReviewReason.DISAPPEARANCE)
        reasons.append(ReviewReason.COMPLETE_OCCLUSION)
    elif previous.component_count == 0 and component_count > 0:
        reasons.append(ReviewReason.REAPPEARANCE)

    if component_count != previous.component_count:
        reasons.append(ReviewReason.COMPONENT_COUNT_CHANGE)

    if previous.area_pixels > 0:
        change_fraction = abs(area - previous.area_pixels) / previous.area_pixels
        if change_fraction >= LARGE_AREA_CHANGE_FRACTION:
            reasons.append(ReviewReason.LARGE_AREA_CHANGE)

    current_centroid = _centroid(mask)
    if previous.centroid is not None and current_centroid is not None:
        frame_diagonal = float(np.hypot(*mask.shape))
        displacement = float(np.hypot(*(np.subtract(current_centroid, previous.centroid))))
        if frame_diagonal > 0 and displacement / frame_diagonal >= (
            ABRUPT_CENTROID_DISPLACEMENT_FRACTION
        ):
            reasons.append(ReviewReason.ABRUPT_CENTROID_DISPLACEMENT)

    return reasons


def review_sampling_reasons(frame_index: int, stride: int | None) -> list[ReviewReason]:
    """Manual review sampling: flag every ``stride``-th frame for review
    regardless of any other signal, so long propagation runs still get
    some guaranteed human QA coverage even when nothing else trips.
    ``stride`` is administrator-configured
    (``tracking.review_sampling_stride``); ``None``/``0`` disables it."""
    if stride and stride > 0 and frame_index % stride == 0:
        return [ReviewReason.MANUAL_REVIEW_SAMPLING]
    return []
