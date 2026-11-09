# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Sam21Backend benchmark: resolution, frame count, throughput, peak VRAM,
and model variant for a given video.

Usage::

    python scripts/benchmark_sam21.py --video tests/fixtures/tiny_test_video.mp4
    python scripts/benchmark_sam21.py --video my_recording.mp4 \
        --checkpoint models/sam2.1-hiera-large
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vistlabel.segmentation.base import PointPrompt  # noqa: E402
from vistlabel.segmentation.sam21 import Sam21Backend  # noqa: E402
from vistlabel.settings import RuntimeSettings, Sam21ModelSettings  # noqa: E402
from vistlabel.video.frames import build_frame_index  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/sam2.1-hiera-base-plus"))
    parser.add_argument("--variant", default="sam2.1_hiera_base_plus")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    import torch

    settings = Sam21ModelSettings(enabled=True, checkpoint=args.checkpoint, variant=args.variant)
    runtime_settings = RuntimeSettings(cuda_device=args.device, default_dtype=args.dtype)
    backend = Sam21Backend(settings, runtime_settings)

    descriptor = backend.descriptor()
    if not descriptor.available:
        print(f"backend unavailable: {descriptor.unavailable_reason}", file=sys.stderr)
        return 1

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    load_start = time.monotonic()
    session_id = backend.open_video("benchmark", args.video)
    open_duration = time.monotonic() - load_start

    timestamps = build_frame_index(args.video)
    frame_count = len(timestamps)

    segment_start = time.monotonic()
    first = backend.segment_interactive(
        session_id,
        frame_index=0,
        object_id="benchmark-object",
        points=[PointPrompt(x=1, y=1, positive=True)],
        box_xyxy=None,
        previous_mask=None,
    )
    segment_duration = time.monotonic() - segment_start

    propagate_start = time.monotonic()
    predictions = list(
        backend.propagate(
            session_id,
            object_ids=["benchmark-object"],
            start_frame=0,
            end_frame=frame_count - 1,
            direction="forward",
        )
    )
    propagate_duration = time.monotonic() - propagate_start
    backend.close_video(session_id)

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
    )

    report = {
        "video": str(args.video),
        "variant": args.variant,
        "checkpoint_sha256": descriptor.checkpoint_sha256,
        "code_revision": descriptor.code_revision,
        "resolution": [first.visible_mask.shape[1], first.visible_mask.shape[0]],
        "frame_count": frame_count,
        "open_video_seconds": round(open_duration, 4),
        "first_segment_seconds": round(segment_duration, 4),
        "propagate_seconds": round(propagate_duration, 4),
        "propagate_frames_per_second": (
            round(len(predictions) / propagate_duration, 2) if propagate_duration > 0 else None
        ),
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
