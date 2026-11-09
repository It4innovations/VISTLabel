# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Streaming MP4 upload, validation and atomic finalize.

The full pipeline: browser upload -> `source/video.mp4.part` (written
incrementally, never loaded fully into RAM) -> ffprobe validation -> an
optional single re-encode (VFR normalization, operator-chosen resolution and
frame rate) -> a full decode pass building the deterministic frame index ->
atomic rename to
`source/video.mp4` -> project state update. Any failure removes the partial
upload and (by the caller) the just-created project directory -- there is
no "retry upload on an existing project" path yet, matching the single
combined create+upload workflow.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

from vistlabel.domain.project import OriginalVideoInfo, VideoImportOptions
from vistlabel.persistence.jsonl import append_jsonl
from vistlabel.persistence.paths import events_log_path, source_video_path
from vistlabel.video.ffmpeg import FfmpegBinaries
from vistlabel.video.frames import (
    build_frame_index,
    detect_variable_frame_rate,
    save_frame_timestamps,
)
from vistlabel.video.validation import ProbedVideoInfo, VideoValidationError, validate_video


class UploadRejectedError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class AsyncReadable(Protocol):
    async def read(self, size: int) -> bytes: ...


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class TranscodePlan:
    """One ffmpeg re-encode covering everything an upload needs: VFR
    normalization, the operator's frame-rate choice and downscaling."""

    #: Output frame size; ``None`` keeps the source size.
    width: int | None = None
    height: int | None = None
    #: Resample to this constant rate (``fps`` filter) before frame selection.
    fps: Fraction | None = None
    #: Keep frames 0, n, 2n, ... of the (resampled) stream.
    keep_every_nth: int = 1
    #: Constant rate of the output; ``None`` keeps the source timing as-is.
    output_fps: Fraction | None = None

    def describe(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": str(self.fps) if self.fps is not None else None,
            "keep_every_nth": self.keep_every_nth,
            "output_fps": str(self.output_fps) if self.output_fps is not None else None,
        }


def _as_fraction(rate: float) -> Fraction:
    # 1001 keeps NTSC rates exact (29.97 -> 30000/1001).
    return Fraction(rate).limit_denominator(1001)


def _fit_within(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int] | None:
    """Largest even-sized frame with the source aspect ratio that fits in
    ``max_width`` x ``max_height``; ``None`` if the source already fits
    (never upscales)."""
    scale = min(max_width / width, max_height / height)
    if scale >= 1:
        return None
    # Floor to even (H.264 4:2:0 needs it) so the result never exceeds the box.
    out_width = max(2, int((width * scale + 1e-6) // 2) * 2)
    out_height = max(2, int((height * scale + 1e-6) // 2) * 2)
    return out_width, out_height


def plan_transcode(
    probed: ProbedVideoInfo,
    options: VideoImportOptions,
    *,
    normalize_vfr: bool,
    normalized_frame_rate: int,
) -> TranscodePlan | None:
    """Work out the re-encode an upload needs, or ``None`` if the file can
    be stored as uploaded. Raises :class:`VideoValidationError` for options
    that do not make sense for this video."""
    source_fps = _as_fraction(probed.nominal_fps)

    size: tuple[int, int] | None = None
    if options.max_width is not None and options.max_height is not None:
        # ffmpeg auto-rotates, so fit the *displayed* frame, not the coded one.
        width, height = probed.width, probed.height
        if probed.rotation_degrees in (90, 270):
            width, height = height, width
        size = _fit_within(width, height, options.max_width, options.max_height)

    fps: Fraction | None = None
    if options.target_fps is not None:
        if options.target_fps > probed.nominal_fps + 0.01:
            raise VideoValidationError(
                f"target frame rate {options.target_fps:g} FPS is above the video's "
                f"{probed.nominal_fps:g} FPS -- that would only duplicate frames"
            )
        if probed.variable_frame_rate or abs(options.target_fps - probed.nominal_fps) > 0.01:
            fps = _as_fraction(options.target_fps)
    elif normalize_vfr and probed.variable_frame_rate:
        # Never resample above the recording's average rate: a 15 FPS phone
        # clip normalized to 30 would just be every frame twice.
        fps = Fraction(max(1, min(normalized_frame_rate, round(probed.nominal_fps))))

    keep_every_nth = options.keep_every_nth or 1
    if size is None and fps is None and keep_every_nth == 1:
        return None

    output_fps: Fraction | None = None
    if fps is not None or keep_every_nth > 1:
        output_fps = (fps or source_fps) / keep_every_nth
    return TranscodePlan(
        width=size[0] if size else None,
        height=size[1] if size else None,
        fps=fps,
        keep_every_nth=keep_every_nth,
        output_fps=output_fps,
    )


def transcode_video(video_path: Path, binaries: FfmpegBinaries, plan: TranscodePlan) -> None:
    """Replace ``video_path`` with an H.264 re-encode following ``plan``."""
    filters: list[str] = []
    if plan.fps is not None:
        filters.append(f"fps={plan.fps}")
    if plan.keep_every_nth > 1:
        assert plan.output_fps is not None
        filters.append(f"select='not(mod(n,{plan.keep_every_nth}))'")
        # Re-time the kept frames onto an even grid at the reduced rate.
        filters.append(f"setpts=N/({plan.output_fps})/TB")
    if plan.width is not None and plan.height is not None:
        # Lanczos keeps fine detail (small codes, edges) sharper than bicubic.
        filters.append(f"scale={plan.width}:{plan.height}:flags=lanczos")
        filters.append("setsar=1")

    timing = (
        ["-fps_mode", "cfr", "-r", str(plan.output_fps)]
        if plan.output_fps is not None
        else ["-fps_mode", "passthrough"]
    )
    transcoded_path = video_path.with_suffix(".transcoded.mp4")
    try:
        result = subprocess.run(
            [
                str(binaries.ffmpeg),
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                *(["-vf", ",".join(filters)] if filters else []),
                *timing,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-preset",
                "medium",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(transcoded_path),
            ],
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        transcoded_path.unlink(missing_ok=True)
        raise VideoValidationError(f"failed to re-encode the video: {exc}") from exc
    if result.returncode != 0:
        transcoded_path.unlink(missing_ok=True)
        raise VideoValidationError(f"FFmpeg could not re-encode the video: {result.stderr.strip()}")
    transcoded_path.replace(video_path)


def normalize_variable_frame_rate(
    video_path: Path, binaries: FfmpegBinaries, frame_rate: int
) -> None:
    """Replace a staged VFR upload with an H.264 constant-frame-rate copy."""
    rate = Fraction(frame_rate)
    transcode_video(video_path, binaries, TranscodePlan(fps=rate, output_fps=rate))


@dataclass(frozen=True)
class FinalizedVideo:
    probed: ProbedVideoInfo
    frame_count: int
    variable_frame_rate: bool
    size_bytes: int
    sha256: str
    stored_path: str
    #: The upload as received, when the stored video is a re-encode of it.
    original: OriginalVideoInfo | None = None


async def save_upload_streaming(
    upload_file: AsyncReadable,
    part_path: Path,
    *,
    chunk_size_bytes: int,
    max_size_bytes: int,
    min_free_disk_bytes: int,
) -> tuple[int, str]:
    """Stream to ``part_path`` incrementally. Returns (size_bytes, sha256).
    Raises :class:`UploadRejectedError` and removes the partial file if the
    size limit is exceeded or free disk space is insufficient."""
    part_path.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(part_path.parent).free
    if free_bytes < min_free_disk_bytes:
        raise UploadRejectedError(
            f"insufficient free disk space: {free_bytes} bytes available, "
            f"{min_free_disk_bytes} required"
        )

    hasher = hashlib.sha256()
    total_bytes = 0
    try:
        with open(part_path, "wb") as handle:
            while True:
                chunk = await upload_file.read(chunk_size_bytes)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_size_bytes:
                    raise UploadRejectedError(
                        f"file exceeds the maximum allowed size of {max_size_bytes} bytes"
                    )
                hasher.update(chunk)
                handle.write(chunk)
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise
    return total_bytes, hasher.hexdigest()


def finalize_video(
    project_dir: Path,
    part_path: Path,
    binaries: FfmpegBinaries,
    *,
    size_bytes: int,
    sha256: str,
    normalize_vfr: bool = True,
    normalized_frame_rate: int = 30,
    import_options: VideoImportOptions | None = None,
) -> FinalizedVideo:
    """Validate, re-encode if the import options or a variable frame rate
    call for it, build the frame index, and atomically rename `.part` into
    place. Raises on validation failure; the caller is responsible for
    removing `part_path` (and the whole project) in that
    case -- this function only cleans up after itself on the success path.
    """
    probed = validate_video(binaries, part_path)
    plan = plan_transcode(
        probed,
        import_options or VideoImportOptions(),
        normalize_vfr=normalize_vfr,
        normalized_frame_rate=normalized_frame_rate,
    )
    original: OriginalVideoInfo | None = None
    if plan is not None:
        original = OriginalVideoInfo(
            width=probed.width,
            height=probed.height,
            nominal_fps=probed.nominal_fps,
            frame_count_hint=probed.frame_count_hint,
            sha256=sha256,
            size_bytes=size_bytes,
        )
        transcode_video(part_path, binaries, plan)
        probed = validate_video(binaries, part_path)
        size_bytes = part_path.stat().st_size
        sha256 = _sha256_file(part_path)
        append_jsonl(
            events_log_path(project_dir),
            {"event": "video_transcoded", "original": original.model_dump(), **plan.describe()},
        )
    timestamps = build_frame_index(part_path)
    variable_frame_rate = probed.variable_frame_rate or detect_variable_frame_rate(timestamps)

    final_path = source_video_path(project_dir)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.replace(final_path)

    save_frame_timestamps(project_dir / "cache" / "frame_timestamps.json", timestamps)

    append_jsonl(
        events_log_path(project_dir),
        {
            "event": "video_finalized",
            "size_bytes": size_bytes,
            "sha256": sha256,
            "frame_count": len(timestamps),
            "duration_seconds": probed.duration_seconds,
        },
    )

    return FinalizedVideo(
        probed=probed,
        frame_count=len(timestamps),
        variable_frame_rate=variable_frame_rate,
        size_bytes=size_bytes,
        sha256=sha256,
        stored_path="source/video.mp4",
        original=original,
    )
