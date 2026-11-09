# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Deterministic frame access: a frame index is built once by
decoding the whole stream in presentation order (correct even with B-frames,
since PyAV's decoder already reorders for us), then individual frames are
decoded on demand by seeking to the nearest preceding keyframe and decoding
forward -- never by re-decoding the whole file per request.

Building the frame index still means one full decode pass over the video at
upload-finalize time. For typical recordings (minutes long) this is fast;
it does not scale to very large (tens of GB) uploads without further work.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from pathlib import Path

import av
from PIL import Image

THUMBNAIL_MAX_DIMENSION = 160


class VideoFrameError(Exception):
    pass


def build_frame_index(video_path: Path) -> list[float]:
    """Decode the whole video once, in presentation order, and return each
    frame's timestamp in seconds. ``len(result)`` is the authoritative frame
    count -- more reliable than ffprobe's ``nb_frames`` hint."""
    timestamps: list[float] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        if time_base is None:
            raise VideoFrameError("video stream has no time_base")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamps.append(float(frame.pts * time_base))
    if not timestamps:
        raise VideoFrameError("no frames could be decoded from this video")
    return timestamps


def iter_decoded_frames(video_path: Path) -> Iterator[Image.Image]:
    """Decode the whole video once, in presentation order, yielding PIL
    images one at a time.

    Used by real segmentation backends that feed the video
    into a model session in bounded chunks: only the frames a caller is
    currently holding stay in memory, so a long or high-resolution video
    never materializes as one giant frame list (a 4K/2700-frame recording
    is ~63 GB as PIL images). Same full-decode-pass tradeoff as
    :func:`build_frame_index`, not an additional scaling cost.
    """
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            yield frame.to_image()


def decode_all_frames(video_path: Path) -> list[Image.Image]:
    """Decode the whole video once, in presentation order, as PIL images.

    Only safe for short/low-resolution fixtures; production backends chunk
    via :func:`iter_decoded_frames` instead of holding every frame at once.
    """
    images = list(iter_decoded_frames(video_path))
    if not images:
        raise VideoFrameError("no frames could be decoded from this video")
    return images


def detect_variable_frame_rate(timestamps: list[float], tolerance_fraction: float = 0.05) -> bool:
    """Authoritative VFR check from real decoded timestamps -- more precise
    than the ffprobe r_frame_rate/avg_frame_rate heuristic used as a fast
    pre-flight signal in ``validation.py``."""
    if len(timestamps) < 3:
        return False
    deltas = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    mean_delta = sum(deltas) / len(deltas)
    if mean_delta <= 0:
        return False
    return any(abs(delta - mean_delta) > tolerance_fraction * mean_delta for delta in deltas)


def save_frame_timestamps(path: Path, timestamps: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timestamps), encoding="utf-8")


def load_frame_timestamps(path: Path) -> list[float]:
    return list(json.loads(path.read_text(encoding="utf-8")))


def decode_frame(video_path: Path, frame_index: int, timestamps: list[float]) -> Image.Image:
    if not 0 <= frame_index < len(timestamps):
        raise IndexError(f"frame_index {frame_index} out of range [0, {len(timestamps)})")
    target_seconds = timestamps[frame_index]

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base
        if time_base is None:
            raise VideoFrameError("video stream has no time_base")
        target_pts = int(target_seconds / time_base)
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if float(frame.pts * time_base) >= target_seconds - 1e-6:
                return frame.to_image()
    raise VideoFrameError(f"could not decode frame {frame_index} ({target_seconds:.3f}s)")


def _encode_jpeg(image: Image.Image, *, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def get_or_decode_frame_jpeg(
    cache_path: Path,
    video_path: Path,
    frame_index: int,
    timestamps: list[float],
    *,
    jpeg_quality: int = 95,
) -> bytes:
    """Cache is disposable: a plain, non-atomic write is
    sufficient -- a crash mid-write just means the next request re-decodes."""
    if cache_path.exists():
        return cache_path.read_bytes()
    image = decode_frame(video_path, frame_index, timestamps)
    data = _encode_jpeg(image, quality=jpeg_quality)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data


def get_or_decode_thumbnail_jpeg(
    cache_path: Path,
    video_path: Path,
    frame_index: int,
    timestamps: list[float],
    *,
    max_dimension: int = THUMBNAIL_MAX_DIMENSION,
    jpeg_quality: int = 85,
) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    image = decode_frame(video_path, frame_index, timestamps)
    image.thumbnail((max_dimension, max_dimension))
    data = _encode_jpeg(image, quality=jpeg_quality)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return data
