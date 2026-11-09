# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Small device/dtype resolution helpers shared by real (torch-backed)
backend adapters. Imports torch lazily (inside the functions) so this
module -- and anything that imports it just for type-checking purposes --
never requires the optional ``gpu`` extra to be installed.
"""

from __future__ import annotations

import logging
import os
from itertools import islice
from typing import TYPE_CHECKING, Any

from vistlabel.video.frames import iter_decoded_frames

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import torch

logger = logging.getLogger(__name__)

#: Frames preprocessed per video-processor call. Bounds the transient
#: full-resolution tensor a chunk becomes on the processing device (a 4K
#: chunk of 16 is ~1.6 GB in float32) while keeping per-call overhead
#: amortized; only the resized model-input frames outlive the chunk.
PREPROCESS_CHUNK_FRAMES = 16


def iter_preprocessed_video_chunks(
    video_processor: Any, video_path: Path, device: str
) -> Iterator[tuple[Any, int, int]]:
    """Decode ``video_path`` and run it through a transformers video
    processor in chunks of :data:`PREPROCESS_CHUNK_FRAMES`, yielding
    ``(pixel_values, original_height, original_width)`` per chunk, where
    ``pixel_values`` is a ``(chunk_len, 3, H', W')`` tensor at the model's
    input resolution.

    This replaces ``processor.init_video_session(video=...)``'s
    preprocessing, which stacks *every* full-resolution frame into a single
    device tensor before resizing -- hundreds of GB for long 4K videos.
    Here at most one chunk exists at full resolution at a time.
    """
    frames = iter_decoded_frames(video_path)
    while True:
        chunk = list(islice(frames, PREPROCESS_CHUNK_FRAMES))
        if not chunk:
            return
        processed = video_processor(videos=chunk, device=device, return_tensors="pt")
        height, width = (int(value) for value in processed.original_sizes[0])
        yield processed.pixel_values_videos[0], height, width


def clear_object_memory(inference_session: Any, obj_id: int) -> None:
    """Drop everything a transformers SAM video session holds for ``obj_id``
    while keeping the object registered and the other objects untouched.

    Needed before a corrected mask seeds propagation, for two reasons.
    transformers only makes an input a conditioning frame on a frame the
    object was never tracked through; elsewhere the next propagation just
    recomputes that frame and the mask has no effect. And older outputs
    outvote a single new mask: with them left in place, a real SAM 2.1 and
    SAM 3 run both grew a corrected mask straight back on the next frame."""
    obj_idx = inference_session.obj_id_to_idx(obj_id)
    inference_session.point_inputs_per_obj[obj_idx].clear()
    inference_session.mask_inputs_per_obj[obj_idx].clear()
    inference_session.frames_tracked_per_obj[obj_idx].clear()
    for outputs in inference_session.output_dict_per_obj[obj_idx].values():
        outputs.clear()


def resolve_device(cuda_device: str) -> str:
    """``"auto"`` picks CUDA if available, else CPU; anything else (e.g.
    ``"cuda:0"``, ``"cpu"``) is passed through verbatim.

    Falling back to CPU is logged as a warning: it is orders of magnitude
    slower and usually means CUDA_VISIBLE_DEVICES names a GPU this machine
    doesn't have (e.g. the launcher's GPU 1 on a single-GPU laptop) or a
    CPU-only torch build."""
    if cuda_device != "auto":
        return cuda_device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    logger.warning(
        "cuda_device=auto: no CUDA device visible, running models on CPU "
        "(torch %s, CUDA build %s, CUDA_VISIBLE_DEVICES=%r)",
        torch.__version__,
        torch.version.cuda,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    return "cpu"


_DTYPE_NAMES = ("float16", "bfloat16", "float32")


def resolve_dtype(name: str) -> Any:
    """Returns a ``torch.dtype``. Typed ``Any`` rather than ``torch.dtype``
    so this signature itself doesn't require torch's types to be resolvable
    without the ``gpu`` extra installed."""
    import torch

    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {_DTYPE_NAMES}")
    return mapping[name]
