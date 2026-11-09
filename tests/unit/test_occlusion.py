# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from vistlabel.annotations.occlusion import (
    PreviousFrameContext,
    compute_occlusion_state,
    frame_local_review_reasons,
    mask_touches_border,
    previous_frame_context,
    review_sampling_reasons,
    temporal_review_reasons,
)
from vistlabel.domain.frames import OcclusionState, ReviewReason


def _mask(height: int = 20, width: int = 20) -> np.ndarray:
    return np.zeros((height, width), dtype=bool)


def test_compute_occlusion_state_zero_components_is_fully_occluded() -> None:
    assert compute_occlusion_state(0) == OcclusionState.FULLY_OCCLUDED


def test_compute_occlusion_state_one_component_is_visible() -> None:
    assert compute_occlusion_state(1) == OcclusionState.VISIBLE


def test_compute_occlusion_state_two_or_more_components_is_partially_occluded() -> None:
    assert compute_occlusion_state(2) == OcclusionState.PARTIALLY_OCCLUDED
    assert compute_occlusion_state(3) == OcclusionState.PARTIALLY_OCCLUDED


def test_mask_touches_border_true_on_edge_pixel() -> None:
    mask = _mask()
    mask[0, 5] = True
    assert mask_touches_border(mask) is True


def test_mask_touches_border_false_when_interior_only() -> None:
    mask = _mask()
    mask[5:10, 5:10] = True
    assert mask_touches_border(mask) is False


def test_frame_local_review_reasons_flags_low_score() -> None:
    mask = _mask()
    mask[5:10, 5:10] = True
    reasons = frame_local_review_reasons(mask, score=0.2)
    assert ReviewReason.LOW_MODEL_SCORE in reasons


def test_frame_local_review_reasons_no_flags_for_confident_interior_mask() -> None:
    mask = _mask()
    mask[5:10, 5:10] = True
    reasons = frame_local_review_reasons(mask, score=0.95)
    assert reasons == []


def test_previous_frame_context_records_component_count_and_area() -> None:
    mask = _mask()
    mask[0:5, 0:5] = True  # area 25, centroid at (2.0, 2.0)

    context = previous_frame_context(mask, component_count=1)

    assert context == PreviousFrameContext(
        component_count=1, area_pixels=25, centroid=(2.0, 2.0)
    )


def test_temporal_review_reasons_flags_disappearance_and_complete_occlusion() -> None:
    previous = PreviousFrameContext(component_count=1, area_pixels=100)
    empty_mask = _mask()

    reasons = temporal_review_reasons(mask=empty_mask, component_count=0, previous=previous)

    assert ReviewReason.DISAPPEARANCE in reasons
    assert ReviewReason.COMPLETE_OCCLUSION in reasons


def test_temporal_review_reasons_flags_reappearance() -> None:
    previous = PreviousFrameContext(component_count=0, area_pixels=0)
    mask = _mask()
    mask[5:10, 5:10] = True

    reasons = temporal_review_reasons(mask=mask, component_count=1, previous=previous)

    assert ReviewReason.REAPPEARANCE in reasons


def test_temporal_review_reasons_flags_component_count_change() -> None:
    previous = PreviousFrameContext(component_count=1, area_pixels=25)
    mask = _mask()
    mask[0:5, 0:5] = True
    mask[15:20, 15:20] = True

    reasons = temporal_review_reasons(mask=mask, component_count=2, previous=previous)

    assert ReviewReason.COMPONENT_COUNT_CHANGE in reasons


def test_temporal_review_reasons_flags_large_area_change() -> None:
    previous = PreviousFrameContext(component_count=1, area_pixels=1000)
    mask = _mask()
    mask[0:2, 0:2] = True  # area 4, tiny compared to previous 1000

    reasons = temporal_review_reasons(mask=mask, component_count=1, previous=previous)

    assert ReviewReason.LARGE_AREA_CHANGE in reasons


def test_temporal_review_reasons_no_flags_for_stable_object() -> None:
    mask = _mask()
    mask[5:10, 5:10] = True  # area 25
    previous = previous_frame_context(mask, component_count=1)

    reasons = temporal_review_reasons(mask=mask, component_count=1, previous=previous)

    assert reasons == []


def test_temporal_review_reasons_flags_abrupt_centroid_displacement() -> None:
    height, width = 20, 20
    previous_mask = _mask(height, width)
    previous_mask[0:2, 0:2] = True  # centroid near (1, 1)
    previous = previous_frame_context(previous_mask, component_count=1)

    current_mask = _mask(height, width)
    current_mask[17:19, 17:19] = True  # centroid near (18, 18) -- far corner

    reasons = temporal_review_reasons(mask=current_mask, component_count=1, previous=previous)

    assert ReviewReason.ABRUPT_CENTROID_DISPLACEMENT in reasons


def test_temporal_review_reasons_no_flag_for_small_centroid_shift() -> None:
    height, width = 100, 100
    previous_mask = _mask(height, width)
    previous_mask[50:52, 50:52] = True
    previous = previous_frame_context(previous_mask, component_count=1)

    current_mask = _mask(height, width)
    current_mask[51:53, 51:53] = True  # shifted by one pixel

    reasons = temporal_review_reasons(mask=current_mask, component_count=1, previous=previous)

    assert ReviewReason.ABRUPT_CENTROID_DISPLACEMENT not in reasons


def test_review_sampling_reasons_flags_every_stride_th_frame() -> None:
    assert review_sampling_reasons(0, stride=5) == [ReviewReason.MANUAL_REVIEW_SAMPLING]
    assert review_sampling_reasons(5, stride=5) == [ReviewReason.MANUAL_REVIEW_SAMPLING]
    assert review_sampling_reasons(3, stride=5) == []


def test_review_sampling_reasons_disabled_when_stride_is_none_or_zero() -> None:
    assert review_sampling_reasons(0, stride=None) == []
    assert review_sampling_reasons(0, stride=0) == []
