# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Shared backend contract assertions.

Every ``VideoSegmentationBackend`` implementation -- FakeBackend,
Sam21Backend, Sam3Backend, YoloSamBackend -- must satisfy this contract. Real
GPU-backed backends run the same functions from a separately marked
``@pytest.mark.gpu`` test module, passing a real, decodable ``video_path``
(the fake backend tolerates a nonexistent placeholder; a real backend does
not); this module itself contains no GPU dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vistlabel.segmentation.base import (
    BackendCapability,
    BackendCapabilityError,
    PointPrompt,
    VideoSegmentationBackend,
)

_PLACEHOLDER_VIDEO_PATH = Path("unused.mp4")


def assert_descriptor_and_capabilities(backend: VideoSegmentationBackend) -> None:
    descriptor = backend.descriptor()
    assert descriptor.backend_id
    assert descriptor.display_name
    assert isinstance(descriptor.capabilities, frozenset)
    assert descriptor.available is True or descriptor.unavailable_reason


def assert_open_close_session(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    session_id = backend.open_video("contract-project", video_path)
    assert isinstance(session_id, str) and session_id
    backend.close_video(session_id)
    # closing twice must not raise
    backend.close_video(session_id)


def assert_point_prompt_produces_visible_mask(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    session_id = backend.open_video("contract-project", video_path)
    try:
        prediction = backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="obj-1",
            points=[PointPrompt(x=8, y=8, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        assert prediction.object_id == "obj-1"
        assert prediction.frame_index == 0
        assert prediction.visible_mask.dtype == np.bool_
        assert prediction.visible_mask.any()
        assert prediction.component_count >= 1
    finally:
        backend.close_video(session_id)


def assert_multipart_visible_mask_is_preserved(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    descriptor = backend.descriptor()
    if BackendCapability.MULTIPART_VISIBLE_MASK not in descriptor.capabilities:
        pytest.skip(f"{descriptor.backend_id} does not declare MULTIPART_VISIBLE_MASK")

    session_id = backend.open_video("contract-project", video_path)
    try:
        prediction = backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="pipette-1",
            points=[
                PointPrompt(x=4, y=4, positive=True),
                PointPrompt(x=40, y=40, positive=True),
            ],
            box_xyxy=None,
            previous_mask=None,
        )
        # Two widely separated positive points must survive as one logical
        # object with a visible mask spanning >= 2 disconnected components
        # -- never split into two objects.
        assert prediction.object_id == "pipette-1"
        assert prediction.component_count >= 2
    finally:
        backend.close_video(session_id)


def assert_propagation_where_supported(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    descriptor = backend.descriptor()
    if BackendCapability.VIDEO_PROPAGATION not in descriptor.capabilities:
        pytest.skip(f"{descriptor.backend_id} does not declare VIDEO_PROPAGATION")

    session_id = backend.open_video("contract-project", video_path)
    try:
        backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="obj-1",
            points=[PointPrompt(x=10, y=10, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        predictions = list(
            backend.propagate(
                session_id,
                object_ids=["obj-1"],
                start_frame=0,
                end_frame=3,
                direction="forward",
            )
        )
        assert [prediction.frame_index for prediction in predictions] == [0, 1, 2, 3]
        assert all(prediction.object_id == "obj-1" for prediction in predictions)
    finally:
        backend.close_video(session_id)


def assert_forgotten_object_is_seeded_by_the_replacement_mask(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    """A correction made on a frame an earlier run already tracked through
    has to reach the next run: re-tracking that frame instead would throw
    the corrected mask away."""
    descriptor = backend.descriptor()
    if BackendCapability.VIDEO_PROPAGATION not in descriptor.capabilities:
        pytest.skip(f"{descriptor.backend_id} does not declare VIDEO_PROPAGATION")

    session_id = backend.open_video("contract-project", video_path)
    try:
        first = backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="obj-1",
            points=[PointPrompt(x=10, y=10, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        list(backend.propagate(session_id, ["obj-1"], 0, 3, "forward"))

        height, width = first.visible_mask.shape
        corrected = np.zeros((height, width), dtype=bool)
        corrected[height // 4 : height // 2, width // 4 : width // 2] = True
        backend.forget_object(session_id, "obj-1")
        backend.replace_object_mask(session_id, 2, "obj-1", corrected)

        seed = next(iter(backend.propagate(session_id, ["obj-1"], 2, 3, "forward")))
        overlap = (seed.visible_mask & corrected).sum() / (seed.visible_mask | corrected).sum()
        assert seed.frame_index == 2
        assert overlap > 0.8
    finally:
        backend.close_video(session_id)


def assert_cancellation_by_partial_consumption(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    """Cancellation is caller-side: abandon the generator and close the session.

    The Protocol has no explicit cancellation token; real job-level
    cancellation is implemented by the GPU job manager wrapping this
    generator, not by the backend itself.
    """
    descriptor = backend.descriptor()
    if BackendCapability.VIDEO_PROPAGATION not in descriptor.capabilities:
        pytest.skip(f"{descriptor.backend_id} does not declare VIDEO_PROPAGATION")

    session_id = backend.open_video("contract-project", video_path)
    try:
        backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="obj-1",
            points=[PointPrompt(x=10, y=10, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        generator = backend.propagate(
            session_id,
            object_ids=["obj-1"],
            start_frame=0,
            end_frame=50,
            direction="forward",
        )
        next(iter(generator))  # consume exactly one, then abandon
    finally:
        backend.close_video(session_id)  # must not raise despite the abandoned generator


def assert_unsupported_capability_raises_typed_error(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    descriptor = backend.descriptor()
    if BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION in descriptor.capabilities:
        pytest.skip(f"{descriptor.backend_id} declares CLASS_AWARE_AUTOMATIC_PREANNOTATION")

    session_id = backend.open_video("contract-project", video_path)
    try:
        with pytest.raises(BackendCapabilityError):
            list(backend.auto_preannotate(session_id, frame_indices=[0], requested_class_ids=[0]))
    finally:
        backend.close_video(session_id)


def run_full_contract(
    backend: VideoSegmentationBackend, video_path: Path = _PLACEHOLDER_VIDEO_PATH
) -> None:
    assert_descriptor_and_capabilities(backend)
    assert_open_close_session(backend, video_path)
    assert_point_prompt_produces_visible_mask(backend, video_path)
    assert_multipart_visible_mask_is_preserved(backend, video_path)
    assert_propagation_where_supported(backend, video_path)
    assert_forgotten_object_is_seeded_by_the_replacement_mask(backend, video_path)
    assert_cancellation_by_partial_consumption(backend, video_path)
    assert_unsupported_capability_raises_typed_error(backend, video_path)
