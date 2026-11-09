# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from vistlabel.annotations.png_mask import (
    MaskDecodeError,
    decode_mask_png_base64,
    encode_mask_png_base64,
)


def test_round_trip_preserves_mask() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[2:5, 2:5] = True
    mask[15:18, 25:28] = True

    encoded = encode_mask_png_base64(mask)
    decoded = decode_mask_png_base64(encoded, expected_shape=(20, 30))

    assert np.array_equal(decoded, mask)


def test_decode_rejects_wrong_shape() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    encoded = encode_mask_png_base64(mask)

    with pytest.raises(MaskDecodeError):
        decode_mask_png_base64(encoded, expected_shape=(20, 20))


def test_decode_rejects_garbage_input() -> None:
    with pytest.raises(MaskDecodeError):
        decode_mask_png_base64("not-valid-base64!!", expected_shape=(10, 10))
