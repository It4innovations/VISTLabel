# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""FFmpeg/ffprobe binary discovery and startup diagnostics.

Checks ``runtime.ffmpeg_dir`` (if configured) before falling back to
``PATH``, so a workstation that has ffmpeg installed somewhere not on PATH
does not require a system-wide environment change.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

_FFMPEG_EXE_NAMES = ("ffmpeg.exe", "ffmpeg")
_FFPROBE_EXE_NAMES = ("ffprobe.exe", "ffprobe")


@dataclass(frozen=True)
class FfmpegBinaries:
    ffmpeg: Path
    ffprobe: Path


class FfmpegNotFoundError(Exception):
    def __init__(self, ffmpeg_dir: Path | None) -> None:
        self.ffmpeg_dir = ffmpeg_dir
        hint = f" (checked {ffmpeg_dir} and PATH)" if ffmpeg_dir else " (checked PATH)"
        super().__init__(f"ffmpeg/ffprobe not found{hint}")


def _find_in_dir(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def find_ffmpeg_binaries(ffmpeg_dir: Path | None = None) -> FfmpegBinaries:
    if ffmpeg_dir is not None:
        ffmpeg = _find_in_dir(ffmpeg_dir, _FFMPEG_EXE_NAMES)
        ffprobe = _find_in_dir(ffmpeg_dir, _FFPROBE_EXE_NAMES)
        if ffmpeg and ffprobe:
            return FfmpegBinaries(ffmpeg=ffmpeg, ffprobe=ffprobe)

    ffmpeg_on_path = shutil.which("ffmpeg")
    ffprobe_on_path = shutil.which("ffprobe")
    if ffmpeg_on_path and ffprobe_on_path:
        return FfmpegBinaries(ffmpeg=Path(ffmpeg_on_path), ffprobe=Path(ffprobe_on_path))

    raise FfmpegNotFoundError(ffmpeg_dir)


def try_find_ffmpeg_binaries(ffmpeg_dir: Path | None = None) -> FfmpegBinaries | None:
    try:
        return find_ffmpeg_binaries(ffmpeg_dir)
    except FfmpegNotFoundError:
        return None
