# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Chunked propagation processing: pure boundary-arithmetic tests for
the chunk-range generator propagation.py uses to turn one big propagation
request into per-chunk checkpoints.

Consecutive chunks deliberately *share* their boundary frame (chunk N+1
starts exactly where chunk N ended) -- ``backend.propagate`` requires its
``start_frame`` to already have a seeded mask in the backend's session, and
the only frame guaranteed to have one for chunk N+1 is chunk N's own last
output.
"""

from __future__ import annotations

from vistlabel.api.propagation import _chunk_ranges


def test_single_chunk_covers_whole_range_when_chunk_frames_is_large() -> None:
    assert list(_chunk_ranges(0, 9, "forward", chunk_frames=200)) == [(0, 9)]


def test_forward_chunks_overlap_at_boundaries() -> None:
    assert list(_chunk_ranges(0, 9, "forward", chunk_frames=3)) == [
        (0, 3),
        (3, 6),
        (6, 9),
    ]


def test_backward_chunks_overlap_at_boundaries() -> None:
    assert list(_chunk_ranges(9, 0, "backward", chunk_frames=3)) == [
        (9, 6),
        (6, 3),
        (3, 0),
    ]


def test_single_frame_range_yields_one_no_op_chunk() -> None:
    assert list(_chunk_ranges(5, 5, "forward", chunk_frames=200)) == [(5, 5)]


def test_chunk_frames_of_one_advances_by_one_frame_per_chunk() -> None:
    assert list(_chunk_ranges(0, 2, "forward", chunk_frames=1)) == [(0, 1), (1, 2)]


def test_uneven_range_makes_a_smaller_final_chunk() -> None:
    assert list(_chunk_ranges(0, 10, "forward", chunk_frames=3)) == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 10),
    ]


def test_every_frame_in_the_range_is_covered_by_at_least_one_chunk() -> None:
    chunks = list(_chunk_ranges(0, 20, "forward", chunk_frames=7))
    covered: set[int] = set()
    for start, end in chunks:
        covered.update(range(start, end + 1))
    assert covered == set(range(21))
