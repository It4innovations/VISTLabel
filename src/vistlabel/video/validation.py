# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""ffprobe-based video validation: never trust the filename,
extension or client-supplied MIME type -- inspect the actual container and
streams.
"""

from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path

from pydantic import BaseModel

from vistlabel.video.ffmpeg import FfmpegBinaries


class VideoValidationError(Exception):
    pass


class ProbedVideoInfo(BaseModel):
    video_codec: str
    pixel_format: str
    width: int
    height: int
    nominal_fps: float
    duration_seconds: float
    frame_count_hint: int
    variable_frame_rate: bool
    rotation_degrees: int


def _run_ffprobe(ffprobe_path: Path, video_path: Path) -> dict:
    try:
        result = subprocess.run(
            [
                str(ffprobe_path),
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoValidationError(f"failed to run ffprobe: {exc}") from exc
    if result.returncode != 0:
        raise VideoValidationError(
            f"ffprobe rejected the file (exit {result.returncode}): {result.stderr.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoValidationError(f"could not parse ffprobe output: {exc}") from exc


def _parse_frame_rate(raw: str) -> float:
    try:
        return float(Fraction(raw))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _rotation_degrees(stream: dict) -> int:
    tags_rotate = stream.get("tags", {}).get("rotate")
    if tags_rotate is not None:
        try:
            return int(tags_rotate) % 360
        except ValueError:
            return 0
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            try:
                return int(side_data["rotation"]) % 360
            except (TypeError, ValueError):
                continue
    return 0


def validate_video(binaries: FfmpegBinaries, video_path: Path) -> ProbedVideoInfo:
    probe = _run_ffprobe(binaries.ffprobe, video_path)

    video_streams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
    if not video_streams:
        raise VideoValidationError("no decodable video stream found")
    stream = video_streams[0]

    width = stream.get("width")
    height = stream.get("height")
    if not width or not height:
        raise VideoValidationError("video stream is missing width/height")

    pixel_format = stream.get("pix_fmt")
    if not pixel_format:
        raise VideoValidationError("video stream is missing pixel format")

    format_info = probe.get("format", {})
    duration_raw = stream.get("duration") or format_info.get("duration")
    if not duration_raw:
        raise VideoValidationError("could not determine video duration")
    duration_seconds = float(duration_raw)
    if duration_seconds <= 0:
        raise VideoValidationError(f"invalid duration: {duration_seconds}")

    r_frame_rate = _parse_frame_rate(stream.get("r_frame_rate", "0/0"))
    avg_frame_rate = _parse_frame_rate(stream.get("avg_frame_rate", "0/0"))
    nominal_fps = avg_frame_rate or r_frame_rate
    if nominal_fps <= 0:
        raise VideoValidationError("could not determine a usable frame rate")

    nb_frames_raw = stream.get("nb_frames")
    frame_count_hint = (
        int(nb_frames_raw) if nb_frames_raw else round(duration_seconds * nominal_fps)
    )

    # Heuristic only -- an authoritative check happens once during the
    # frame-index build, which walks real decoded timestamps.
    variable_frame_rate = (
        r_frame_rate > 0 and avg_frame_rate > 0 and abs(r_frame_rate - avg_frame_rate) > 0.01
    )

    return ProbedVideoInfo(
        video_codec=stream.get("codec_name", "unknown"),
        pixel_format=pixel_format,
        width=int(width),
        height=int(height),
        nominal_fps=nominal_fps,
        duration_seconds=duration_seconds,
        frame_count_hint=frame_count_hint,
        variable_frame_rate=variable_frame_rate,
        rotation_degrees=_rotation_degrees(stream),
    )
