# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import TINY_TEST_VIDEO
from vistlabel.video.ffmpeg import FfmpegBinaries
from vistlabel.video.validation import VideoValidationError, validate_video


def test_validate_video_extracts_expected_metadata(ffmpeg_binaries: FfmpegBinaries) -> None:
    info = validate_video(ffmpeg_binaries, TINY_TEST_VIDEO)

    assert info.width == 64
    assert info.height == 64
    assert info.video_codec == "h264"
    assert info.pixel_format == "yuv420p"
    assert info.duration_seconds == pytest.approx(1.0, abs=0.1)
    assert info.nominal_fps == pytest.approx(10.0, abs=0.1)
    assert info.frame_count_hint == 10
    assert info.variable_frame_rate is False
    assert info.rotation_degrees == 0


def test_validate_video_rejects_a_non_video_file(
    ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    fake_video = tmp_path / "not_a_video.mp4"
    fake_video.write_bytes(b"this is not an mp4 container at all")

    with pytest.raises(VideoValidationError):
        validate_video(ffmpeg_binaries, fake_video)


def test_validate_video_rejects_a_missing_file(
    ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    with pytest.raises(VideoValidationError):
        validate_video(ffmpeg_binaries, tmp_path / "does-not-exist.mp4")
