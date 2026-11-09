# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np

from vistlabel.persistence.rle import count_connected_components
from vistlabel.segmentation.base import PointPrompt
from vistlabel.segmentation.fake import FakeBackend


def test_count_connected_components_on_known_shapes() -> None:
    empty = np.zeros((10, 10), dtype=bool)
    assert count_connected_components(empty) == 0

    single_block = np.zeros((10, 10), dtype=bool)
    single_block[2:5, 2:5] = True
    assert count_connected_components(single_block) == 1

    two_blocks = np.zeros((10, 10), dtype=bool)
    two_blocks[0:2, 0:2] = True
    two_blocks[7:9, 7:9] = True
    assert count_connected_components(two_blocks) == 2


def test_negative_point_removes_pixels_from_positive_region() -> None:
    backend = FakeBackend(frame_height=32, frame_width=32, frame_count=8)
    session_id = backend.open_video("proj", Path("unused.mp4"))
    try:
        without_negative = backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="obj-1",
            points=[PointPrompt(x=16, y=16, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        with_negative = backend.segment_interactive(
            session_id,
            frame_index=1,
            object_id="obj-1",
            points=[
                PointPrompt(x=16, y=16, positive=True),
                PointPrompt(x=16, y=16, positive=False),
            ],
            box_xyxy=None,
            previous_mask=None,
        )
        assert without_negative.visible_mask.sum() > 0
        assert with_negative.visible_mask.sum() == 0
    finally:
        backend.close_video(session_id)


def test_replace_object_mask_rejects_wrong_shape() -> None:
    backend = FakeBackend(frame_height=32, frame_width=32, frame_count=8)
    session_id = backend.open_video("proj", Path("unused.mp4"))
    try:
        wrong_shape_mask = np.zeros((10, 10), dtype=bool)
        try:
            backend.replace_object_mask(session_id, 0, "obj-1", wrong_shape_mask)
            raised = False
        except ValueError:
            raised = True
        assert raised
    finally:
        backend.close_video(session_id)
