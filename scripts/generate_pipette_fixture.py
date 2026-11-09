# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Generates tests/fixtures/pipette_occlusion_video.mp4: a
synthetic clip where a thin horizontal "pipette" bar is temporarily split
into two visible segments by a dark "finger" block sweeping across its
middle, then reunifies. Deterministic and fully synthetic (drawn with PIL,
encoded with PyAV) -- not a real recording, safe to commit, reproducible
by re-running this script.

Usage::

    python scripts/generate_pipette_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

WIDTH = 128
HEIGHT = 64
FRAME_COUNT = 20
FPS = 10

PIPETTE_Y0, PIPETTE_Y1 = 28, 36
PIPETTE_X0, PIPETTE_X1 = 10, 118
FINGER_WIDTH = 18
# Finger sweeps across the pipette's midsection between these frames.
FINGER_ENTER_FRAME = 7
FINGER_EXIT_FRAME = 13


def _finger_center_x(frame_index: int) -> float | None:
    if not (FINGER_ENTER_FRAME <= frame_index <= FINGER_EXIT_FRAME):
        return None
    span = FINGER_EXIT_FRAME - FINGER_ENTER_FRAME
    progress = (frame_index - FINGER_ENTER_FRAME) / span
    start_x = (PIPETTE_X0 + PIPETTE_X1) / 2 - 20
    end_x = (PIPETTE_X0 + PIPETTE_X1) / 2 + 20
    return start_x + (end_x - start_x) * progress


def _draw_frame(frame_index: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), color=(20, 20, 20))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [PIPETTE_X0, PIPETTE_Y0, PIPETTE_X1, PIPETTE_Y1], fill=(200, 200, 200)
    )
    center_x = _finger_center_x(frame_index)
    if center_x is not None:
        left = int(center_x - FINGER_WIDTH / 2)
        right = int(center_x + FINGER_WIDTH / 2)
        draw.rectangle([left, 0, right, HEIGHT - 1], fill=(60, 40, 30))
    return image


def generate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "0"}  # lossless -- keep the synthetic edges exact
        for frame_index in range(FRAME_COUNT):
            image = _draw_frame(frame_index)
            frame = av.VideoFrame.from_ndarray(np.array(image), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def main() -> int:
    target = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / (
        "pipette_occlusion_video.mp4"
    )
    generate(target)
    print(f"wrote {target}")
    print(f"finger occludes the pipette on frames {FINGER_ENTER_FRAME}-{FINGER_EXIT_FRAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
