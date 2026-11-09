# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Canonical COCO-compatible compressed RLE mask codec.

Backed by ``pycocotools`` (ships prebuilt wheels for Windows and Linux, no
C-compiler toolchain required) rather than a hand-rolled implementation, so
the exact byte-for-byte ``counts`` format matches what downstream COCO
tooling (COCO export, `pycocotools`-based training/eval code) expects.
Round-tripping through this module is lossless and preserves disconnected
components and holes -- there is no approximation step here, unlike the
YOLO polygon compatibility conversions in ``exports/polygon.py``.
"""

from __future__ import annotations

import numpy as np
from pycocotools import mask as _mask_utils

from vistlabel.domain.frames import RleMask


def encode_mask(mask: np.ndarray) -> RleMask:
    """Encode a 2D boolean/uint8 visible mask as canonical COCO RLE."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    encoded = _mask_utils.encode(fortran_mask)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    height, width = encoded["size"]
    return RleMask(size=(height, width), counts=counts)


def decode_mask(rle: RleMask) -> np.ndarray:
    """Decode canonical COCO RLE back into a 2D boolean visible mask."""
    coco_rle = {"size": list(rle.size), "counts": rle.counts.encode("ascii")}
    decoded = _mask_utils.decode(coco_rle)  # type: ignore[arg-type]
    return decoded.astype(bool)


def count_connected_components(mask: np.ndarray) -> int:
    """Count 4-connected components of a (typically larger) real mask.

    ``pycocotools``' RLE primitives expose no connected-component
    labelling, so this is a dependency-free union-find over horizontal
    *runs*: run extraction is vectorized numpy, and the Python-level work
    scales with the number of runs (thousands for a real mask), not the
    number of pixels (millions for a 4K frame). Propagation calls this
    once per object per frame, so a per-pixel Python flood fill (the
    previous implementation) dominated whole tracking runs.
    """
    if mask.size == 0:
        return 0
    padded = np.zeros((mask.shape[0], mask.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = mask.astype(bool)
    deltas = np.diff(padded, axis=1)
    start_rows, start_cols = np.nonzero(deltas == 1)
    _end_rows, end_cols = np.nonzero(deltas == -1)  # same row order as starts
    run_count = len(start_rows)
    if run_count == 0:
        return 0

    parent = list(range(run_count))

    def find(run: int) -> int:
        root = run
        while parent[root] != root:
            root = parent[root]
        while parent[run] != root:
            parent[run], run = root, parent[run]
        return root

    # Runs are emitted row by row, left to right. Two runs in adjacent rows
    # are 4-connected iff their column ranges [start, end) overlap; a
    # two-pointer sweep over each adjacent row pair visits every run once.
    row_boundaries = np.searchsorted(start_rows, np.arange(mask.shape[0] + 1))
    for row in range(1, mask.shape[0]):
        above = row_boundaries[row - 1]
        above_end = row_boundaries[row]
        current = row_boundaries[row]
        current_end = row_boundaries[row + 1]
        while above < above_end and current < current_end:
            if start_cols[current] < end_cols[above] and start_cols[above] < end_cols[current]:
                root_a, root_b = find(above), find(current)
                if root_a != root_b:
                    parent[root_b] = root_a
            # Advance whichever run ends first; the other may still overlap
            # the next run on the opposite row.
            if end_cols[above] < end_cols[current]:
                above += 1
            else:
                current += 1
    return sum(1 for run in range(run_count) if find(run) == run)
