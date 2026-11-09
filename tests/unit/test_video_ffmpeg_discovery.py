# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from vistlabel.video.ffmpeg import (
    FfmpegNotFoundError,
    find_ffmpeg_binaries,
    try_find_ffmpeg_binaries,
)


def test_find_ffmpeg_binaries_in_explicit_directory(tmp_path: Path) -> None:
    (tmp_path / "ffmpeg.exe").touch()
    (tmp_path / "ffprobe.exe").touch()

    binaries = find_ffmpeg_binaries(tmp_path)

    assert binaries.ffmpeg == tmp_path / "ffmpeg.exe"
    assert binaries.ffprobe == tmp_path / "ffprobe.exe"


def test_find_ffmpeg_binaries_raises_when_nothing_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)

    with pytest.raises(FfmpegNotFoundError):
        find_ffmpeg_binaries(tmp_path / "empty")


def test_try_find_ffmpeg_binaries_returns_none_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)

    assert try_find_ffmpeg_binaries(Path("also-empty")) is None
