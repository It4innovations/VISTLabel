# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import TINY_TEST_VIDEO
from vistlabel.video.frames import (
    build_frame_index,
    decode_frame,
    detect_variable_frame_rate,
    get_or_decode_frame_jpeg,
    get_or_decode_thumbnail_jpeg,
    load_frame_timestamps,
    save_frame_timestamps,
)


def test_build_frame_index_returns_ten_increasing_timestamps() -> None:
    timestamps = build_frame_index(TINY_TEST_VIDEO)

    assert len(timestamps) == 10
    assert timestamps == sorted(timestamps)
    assert timestamps[0] == pytest.approx(0.0, abs=1e-6)


def test_detect_variable_frame_rate_is_false_for_constant_rate_fixture() -> None:
    timestamps = build_frame_index(TINY_TEST_VIDEO)

    assert detect_variable_frame_rate(timestamps) is False


def test_detect_variable_frame_rate_is_true_for_irregular_deltas() -> None:
    irregular = [0.0, 0.1, 0.2, 0.5, 0.6]

    assert detect_variable_frame_rate(irregular) is True


def test_save_and_load_frame_timestamps_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "frame_timestamps.json"
    timestamps = [0.0, 0.1, 0.2]

    save_frame_timestamps(path, timestamps)

    assert load_frame_timestamps(path) == timestamps


def test_decode_frame_returns_an_image_of_the_expected_size() -> None:
    timestamps = build_frame_index(TINY_TEST_VIDEO)

    image = decode_frame(TINY_TEST_VIDEO, 0, timestamps)

    assert image.size == (64, 64)


def test_decode_frame_out_of_range_raises_index_error() -> None:
    timestamps = build_frame_index(TINY_TEST_VIDEO)

    with pytest.raises(IndexError):
        decode_frame(TINY_TEST_VIDEO, len(timestamps), timestamps)


def test_get_or_decode_frame_jpeg_caches_to_disk(tmp_path: Path) -> None:
    timestamps = build_frame_index(TINY_TEST_VIDEO)
    cache_path = tmp_path / "000000.jpg"
    assert not cache_path.exists()

    data = get_or_decode_frame_jpeg(cache_path, TINY_TEST_VIDEO, 0, timestamps)

    assert cache_path.exists()
    assert data[:2] == b"\xff\xd8"
    assert cache_path.read_bytes() == data


def test_get_or_decode_thumbnail_jpeg_caches_to_disk(tmp_path: Path) -> None:
    timestamps = build_frame_index(TINY_TEST_VIDEO)
    cache_path = tmp_path / "000000_thumb.jpg"

    data = get_or_decode_thumbnail_jpeg(cache_path, TINY_TEST_VIDEO, 0, timestamps)

    assert cache_path.exists()
    assert data[:2] == b"\xff\xd8"
