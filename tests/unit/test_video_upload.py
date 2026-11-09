# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import ValidationError

from tests.conftest import TINY_TEST_VIDEO
from vistlabel.domain.project import VideoImportOptions
from vistlabel.persistence.paths import events_log_path, source_video_path
from vistlabel.video.ffmpeg import FfmpegBinaries
from vistlabel.video.frames import load_frame_timestamps
from vistlabel.video.upload import (
    TranscodePlan,
    UploadRejectedError,
    finalize_video,
    normalize_variable_frame_rate,
    plan_transcode,
    save_upload_streaming,
)
from vistlabel.video.validation import ProbedVideoInfo, VideoValidationError, validate_video

_T = TypeVar("_T")


def _run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion from a plain (non-async) test.

    Deliberately runs ``asyncio.run`` in a fresh worker thread rather than
    directly: in the full test session (unit + browser together),
    pytest-playwright's sync API leaves this thread's asyncio state such
    that a later bare ``asyncio.run()`` call raises "cannot be called from
    a running event loop" even though nothing in this test is actually
    running one. A brand-new thread has no such leaked state.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


class _FakeAsyncFile:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    async def read(self, size: int) -> bytes:
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def test_save_upload_streaming_writes_file_and_computes_sha256(tmp_path: Path) -> None:
    data = TINY_TEST_VIDEO.read_bytes()
    part_path = tmp_path / "source" / "video.mp4.part"

    size_bytes, sha256_hex = _run_async(
        save_upload_streaming(
            _FakeAsyncFile(data),
            part_path,
            chunk_size_bytes=64,
            max_size_bytes=10 * 1024 * 1024,
            min_free_disk_bytes=0,
        )
    )

    assert size_bytes == len(data)
    assert sha256_hex == hashlib.sha256(data).hexdigest()
    assert part_path.read_bytes() == data


def test_save_upload_streaming_rejects_oversized_upload_and_cleans_up(tmp_path: Path) -> None:
    data = TINY_TEST_VIDEO.read_bytes()
    part_path = tmp_path / "source" / "video.mp4.part"

    with pytest.raises(UploadRejectedError):
        _run_async(
            save_upload_streaming(
                _FakeAsyncFile(data),
                part_path,
                chunk_size_bytes=64,
                max_size_bytes=10,  # smaller than the fixture
                min_free_disk_bytes=0,
            )
        )

    assert not part_path.exists()


class _FakeDiskUsage:
    def __init__(self, free: int) -> None:
        self.free = free
        self.total = free
        self.used = 0


def test_save_upload_streaming_rejects_insufficient_disk_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part_path = tmp_path / "source" / "video.mp4.part"
    monkeypatch.setattr(shutil, "disk_usage", lambda _path: _FakeDiskUsage(free=0))

    with pytest.raises(UploadRejectedError):
        _run_async(
            save_upload_streaming(
                _FakeAsyncFile(b"irrelevant"),
                part_path,
                chunk_size_bytes=64,
                max_size_bytes=10 * 1024 * 1024,
                min_free_disk_bytes=10**12,
            )
        )


def test_finalize_video_builds_index_and_moves_file_into_place(
    ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    project_dir = tmp_path / "project"
    (project_dir / "source").mkdir(parents=True)
    (project_dir / "cache").mkdir(parents=True)
    (project_dir / "logs").mkdir(parents=True)
    part_path = project_dir / "source" / "video.mp4.part"
    shutil.copyfile(TINY_TEST_VIDEO, part_path)
    data = TINY_TEST_VIDEO.read_bytes()

    finalized = finalize_video(
        project_dir,
        part_path,
        ffmpeg_binaries,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )

    assert finalized.frame_count == 10
    assert finalized.probed.width == 64
    assert not part_path.exists()
    assert source_video_path(project_dir).is_file()

    timestamps = load_frame_timestamps(project_dir / "cache" / "frame_timestamps.json")
    assert len(timestamps) == 10
    assert events_log_path(project_dir).exists()


def test_normalize_variable_frame_rate_produces_a_cfr_video(
    ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    part_path = tmp_path / "video.mp4.part"
    subprocess.run(
        [
            str(ffmpeg_binaries.ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-t",
            "1",
            "-i",
            "testsrc=size=64x64:rate=30",
            "-f",
            "lavfi",
            "-t",
            "1",
            "-i",
            "testsrc2=size=64x64:rate=15",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-f",
            "mp4",
            str(part_path),
        ],
        check=True,
    )
    assert validate_video(ffmpeg_binaries, part_path).variable_frame_rate is True

    normalize_variable_frame_rate(part_path, ffmpeg_binaries, 30)

    normalized = validate_video(ffmpeg_binaries, part_path)
    assert normalized.variable_frame_rate is False
    assert normalized.nominal_fps == 30.0


def _probed(
    width: int = 3840,
    height: int = 2160,
    fps: float = 15.0,
    *,
    variable_frame_rate: bool = False,
    rotation_degrees: int = 0,
) -> ProbedVideoInfo:
    return ProbedVideoInfo(
        video_codec="h264",
        pixel_format="yuv420p",
        width=width,
        height=height,
        nominal_fps=fps,
        duration_seconds=78.3,
        frame_count_hint=1175,
        variable_frame_rate=variable_frame_rate,
        rotation_degrees=rotation_degrees,
    )


def _plan(probed: ProbedVideoInfo, **options: Any) -> TranscodePlan | None:
    return plan_transcode(
        probed, VideoImportOptions(**options), normalize_vfr=True, normalized_frame_rate=30
    )


def test_plan_is_none_for_native_cfr_video() -> None:
    assert _plan(_probed()) is None


def test_plan_fits_within_box_keeping_aspect_ratio() -> None:
    plan = _plan(_probed(), max_width=1920, max_height=1080)
    assert plan is not None
    assert (plan.width, plan.height) == (1920, 1080)
    assert plan.output_fps is None  # timing untouched

    plan = _plan(_probed(), max_width=648, max_height=384)
    assert plan is not None
    # 3840x2160 * 0.16875 = 648x364.5 -> floored to even.
    assert (plan.width, plan.height) == (648, 364)


def test_plan_never_upscales() -> None:
    assert _plan(_probed(1280, 720), max_width=1920, max_height=1080) is None


def test_plan_fits_the_displayed_frame_of_a_rotated_video() -> None:
    plan = _plan(_probed(3840, 2160, rotation_degrees=90), max_width=1920, max_height=1080)
    assert plan is not None
    assert (plan.width, plan.height) == (606, 1080)


def test_plan_keep_every_nth_divides_the_frame_rate() -> None:
    plan = _plan(_probed(fps=15.0), keep_every_nth=3)
    assert plan is not None
    assert plan.fps is None
    assert plan.keep_every_nth == 3
    assert plan.output_fps == Fraction(5)


def test_plan_target_fps_above_source_is_rejected() -> None:
    with pytest.raises(VideoValidationError, match="duplicate frames"):
        _plan(_probed(fps=15.0), target_fps=30)


def test_plan_target_fps_equal_to_source_is_a_no_op() -> None:
    assert _plan(_probed(fps=15.0), target_fps=15) is None


def test_plan_vfr_normalization_never_exceeds_the_average_rate() -> None:
    plan = _plan(_probed(fps=14.8, variable_frame_rate=True))
    assert plan is not None
    assert plan.fps == Fraction(15)
    assert plan.output_fps == Fraction(15)


def test_plan_target_fps_replaces_vfr_normalization_rate() -> None:
    plan = _plan(_probed(fps=29.5, variable_frame_rate=True), target_fps=10)
    assert plan is not None
    assert plan.fps == Fraction(10)


@pytest.mark.parametrize(
    "options",
    [
        {"max_width": 1920},
        {"target_fps": 10, "keep_every_nth": 2},
        {"keep_every_nth": 0},
        {"max_width": 8, "max_height": 8},
    ],
)
def test_import_options_reject_invalid_combinations(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        VideoImportOptions(**options)


def _finalize_with(
    tmp_path: Path, binaries: FfmpegBinaries, options: VideoImportOptions
) -> tuple[Path, Any]:
    project_dir = tmp_path / "project"
    for subdir in ("source", "cache", "logs"):
        (project_dir / subdir).mkdir(parents=True)
    part_path = project_dir / "source" / "video.mp4.part"
    shutil.copyfile(TINY_TEST_VIDEO, part_path)
    data = TINY_TEST_VIDEO.read_bytes()
    finalized = finalize_video(
        project_dir,
        part_path,
        binaries,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        import_options=options,
    )
    return project_dir, finalized


def test_finalize_video_downscales_and_keeps_every_nth_frame(
    ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    # Fixture: 64x64, 10 FPS, 10 frames.
    project_dir, finalized = _finalize_with(
        tmp_path, ffmpeg_binaries, VideoImportOptions(max_width=32, max_height=32, keep_every_nth=2)
    )

    assert (finalized.probed.width, finalized.probed.height) == (32, 32)
    assert finalized.frame_count == 5
    assert finalized.probed.nominal_fps == pytest.approx(5.0)
    assert finalized.variable_frame_rate is False
    assert finalized.original is not None
    assert (finalized.original.width, finalized.original.nominal_fps) == (64, 10.0)
    assert finalized.sha256 == _sha256_of(source_video_path(project_dir))
    assert '"video_transcoded"' in events_log_path(project_dir).read_text(encoding="utf-8")


def test_finalize_video_resamples_to_target_fps(
    ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    _project_dir, finalized = _finalize_with(
        tmp_path, ffmpeg_binaries, VideoImportOptions(target_fps=5)
    )

    assert finalized.frame_count == 5
    assert finalized.probed.nominal_fps == pytest.approx(5.0)
    assert finalized.probed.width == 64


def test_finalize_video_without_options_stores_the_upload_unchanged(
    ffmpeg_binaries: FfmpegBinaries, tmp_path: Path
) -> None:
    project_dir, finalized = _finalize_with(tmp_path, ffmpeg_binaries, VideoImportOptions())

    assert finalized.original is None
    assert source_video_path(project_dir).read_bytes() == TINY_TEST_VIDEO.read_bytes()


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
