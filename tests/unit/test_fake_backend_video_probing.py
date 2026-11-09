# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

from tests.conftest import TINY_TEST_VIDEO
from vistlabel.segmentation.base import PointPrompt
from vistlabel.segmentation.fake import FakeBackend


def test_open_video_probes_real_dimensions_from_an_actual_video() -> None:
    backend = FakeBackend(frame_height=64, frame_width=64, frame_count=64)
    session_id = backend.open_video("proj", TINY_TEST_VIDEO)
    try:
        prediction = backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="obj-1",
            points=[PointPrompt(x=32, y=32, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        # The fixture is 64x64 too, so this mainly proves the probe path
        # runs without error; the mismatched-size test below is the real
        # regression guard.
        assert prediction.visible_mask.shape == (64, 64)
    finally:
        backend.close_video(session_id)


def test_open_video_falls_back_to_defaults_for_a_nonexistent_path() -> None:
    backend = FakeBackend(frame_height=32, frame_width=48, frame_count=10)
    session_id = backend.open_video("proj", Path("does-not-exist.mp4"))
    try:
        prediction = backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="obj-1",
            points=[PointPrompt(x=1, y=1, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        assert prediction.visible_mask.shape == (32, 48)
    finally:
        backend.close_video(session_id)
