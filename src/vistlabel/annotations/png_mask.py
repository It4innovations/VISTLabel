# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Lossless PNG mask exchange with the browser. The browser may exchange
temporary lossless PNG mask blobs for simplicity; the backend converts PNG
to canonical RLE before persistence. Base64 wraps the PNG bytes for JSON
transport.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


class MaskDecodeError(Exception):
    pass


def encode_mask_png_bytes(mask: np.ndarray) -> bytes:
    image = Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def encode_mask_png_base64(mask: np.ndarray) -> str:
    return base64.b64encode(encode_mask_png_bytes(mask)).decode("ascii")


def decode_mask_png_base64(data: str, *, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(data, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any malformed input is a client error
        raise MaskDecodeError(f"could not decode mask PNG: {exc}") from exc

    array = np.array(image.convert("L"))
    if array.shape != expected_shape:
        raise MaskDecodeError(
            f"mask shape {array.shape} does not match expected frame shape {expected_shape}"
        )
    unique_values = set(np.unique(array).tolist())
    if not unique_values <= {0, 255}:
        raise MaskDecodeError(
            f"mask must be strictly binary (0/255 grayscale), got values {sorted(unique_values)}"
        )
    return array > 0
