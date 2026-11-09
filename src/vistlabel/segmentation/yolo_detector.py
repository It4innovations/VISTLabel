# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Custom-YOLO-checkpoint detector adapter: run YOLO on a frame and use its
boxes/optional masks as SAM prompts.

:class:`YoloDetector` is a narrow ``Protocol`` so :class:`YoloSamBackend`'s
orchestration logic (association across keyframes, duplicate/conflict
detection, reconciliation with SAM propagation) is testable on the CPU
without Ultralytics or a trained checkpoint -- the same reasoning that
gives every other real backend a ``fake``-style test double.
:class:`UltralyticsYoloDetector` is the real adapter, used only once a
pinned custom checkpoint is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class YoloDetection:
    box_xyxy: tuple[float, float, float, float]
    yolo_class_id: int
    score: float
    #: Coarse instance mask at the frame's resolution, if the checkpoint is
    #: a segmentation (not detection-only) model.
    coarse_mask: np.ndarray | None


class YoloDetector(Protocol):
    def class_names(self) -> dict[int, str]: ...

    def detect(self, frame: np.ndarray) -> list[YoloDetection]: ...


class UltralyticsYoloDetector:
    """Wraps a pinned, custom-trained Ultralytics checkpoint. Never loads a
    generic pretrained model as a substitute -- the caller is
    responsible for only ever constructing this with an administrator
    -configured custom checkpoint path."""

    def __init__(self, checkpoint: Path, confidence_threshold: float = 0.25) -> None:
        from ultralytics import YOLO

        self._model: Any = YOLO(str(checkpoint))
        self._confidence_threshold = confidence_threshold

    def class_names(self) -> dict[int, str]:
        return {int(k): str(v) for k, v in self._model.names.items()}

    def detect(self, frame: np.ndarray) -> list[YoloDetection]:
        results = self._model.predict(frame, verbose=False, conf=self._confidence_threshold)[0]
        detections: list[YoloDetection] = []
        boxes = results.boxes
        if boxes is None:
            return detections
        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        scores = boxes.conf.detach().cpu().numpy()
        masks = None
        if results.masks is not None:
            masks = results.masks.data.detach().cpu().numpy()

        height, width = frame.shape[0], frame.shape[1]
        for index in range(len(xyxy)):
            coarse_mask = None
            if masks is not None:
                raw_mask = masks[index] > 0.5
                coarse_mask = _resize_mask_nearest(raw_mask, height, width)
            detections.append(
                YoloDetection(
                    box_xyxy=tuple(float(v) for v in xyxy[index]),  # type: ignore[arg-type]
                    yolo_class_id=int(classes[index]),
                    score=float(scores[index]),
                    coarse_mask=coarse_mask,
                )
            )
        return detections


def _resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Ultralytics segmentation masks come back at the model's internal
    resolution, not the source frame's -- nearest-neighbour index resampling
    to line the coarse mask up with the frame SAM will refine against."""
    if mask.shape == (height, width):
        return mask
    row_indices = (np.arange(height) * mask.shape[0] / height).astype(int)
    row_indices = row_indices.clip(0, mask.shape[0] - 1)
    col_indices = (np.arange(width) * mask.shape[1] / width).astype(int)
    col_indices = col_indices.clip(0, mask.shape[1] - 1)
    return mask[row_indices][:, col_indices]
