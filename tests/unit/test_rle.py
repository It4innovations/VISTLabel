# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np

from vistlabel.persistence.rle import count_connected_components, decode_mask, encode_mask


def test_round_trip_preserves_disconnected_components() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:6, 2:6] = True
    mask[30:35, 30:36] = True  # a second, disconnected component

    rle = encode_mask(mask)
    decoded = decode_mask(rle)

    assert np.array_equal(decoded, mask)
    assert count_connected_components(decoded) == 2


def test_round_trip_preserves_holes() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 2:18] = True
    mask[8:12, 8:12] = False  # a hole in the middle

    rle = encode_mask(mask)
    decoded = decode_mask(rle)

    assert np.array_equal(decoded, mask)


def test_encoded_size_matches_mask_shape() -> None:
    mask = np.zeros((17, 33), dtype=bool)
    mask[1:3, 1:3] = True

    rle = encode_mask(mask)

    assert rle.size == (17, 33)


def test_empty_mask_round_trips_to_zero_components() -> None:
    mask = np.zeros((10, 10), dtype=bool)

    rle = encode_mask(mask)
    decoded = decode_mask(rle)

    assert not decoded.any()
    assert count_connected_components(decoded) == 0
