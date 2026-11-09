# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Real Ultralytics adapter parsing ("run YOLO, use boxes and optional
masks as SAM prompts"), against a real ``ultralytics.YOLO``
model -- not the scripted double in
``tests/unit/test_yolo_sam_orchestration.py``, which covers the
orchestration logic but never touches actual Ultralytics tensors.

Deliberately does *not* use a downloaded pretrained checkpoint (a generic
pretrained model must never be presented as identifying a project's
own classes, so this suite never claims meaningful detections).
Instead it builds a real ``yolov8n-seg`` model from Ultralytics' own
architecture YAML (ships inside the package, no network access) with
random, untrained weights, and forces ``confidence_threshold=0.0`` so
:meth:`UltralyticsYoloDetector.detect` returns *something* to parse --
this is purely a tensor-shape/parsing smoke test, in the same "minimal
image inference" spirit as the startup diagnostics, not a claim about
detection quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def untrained_checkpoint(tmp_path_factory) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        pytest.skip("'gpu' extra not installed (pip install -e '.[gpu]')")

    model = YOLO("yolov8n-seg.yaml")
    checkpoint = tmp_path_factory.mktemp("yolo") / "untrained_seg.pt"
    model.save(str(checkpoint))
    return checkpoint


def test_detector_parses_real_ultralytics_boxes_and_masks(untrained_checkpoint: Path) -> None:
    from vistlabel.segmentation.yolo_detector import UltralyticsYoloDetector

    detector = UltralyticsYoloDetector(untrained_checkpoint, confidence_threshold=0.0)
    names = detector.class_names()
    assert len(names) == 80  # yolov8n-seg.yaml's default (COCO-shaped) head

    frame = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    detections = detector.detect(frame)
    assert len(detections) > 0

    for detection in detections:
        assert detection.yolo_class_id in names
        x0, y0, x1, y1 = detection.box_xyxy
        assert x1 >= x0 and y1 >= y0
        assert 0.0 <= detection.score <= 1.0
        if detection.coarse_mask is not None:
            assert detection.coarse_mask.shape == (64, 64)
            assert detection.coarse_mask.dtype == np.bool_


def test_detector_returns_empty_list_at_default_confidence(untrained_checkpoint: Path) -> None:
    """Real ultralytics default confidence threshold correctly discards
    every detection from an untrained model -- the empty-list code path
    the orchestration logic also has to handle correctly."""
    from vistlabel.segmentation.yolo_detector import UltralyticsYoloDetector

    detector = UltralyticsYoloDetector(untrained_checkpoint)  # default threshold
    frame = (np.random.default_rng(1).random((64, 64, 3)) * 255).astype(np.uint8)
    assert detector.detect(frame) == []


def test_yolo_sam_backend_validates_class_mapping_against_real_checkpoint_classes(
    untrained_checkpoint: Path, tmp_path: Path
) -> None:
    """A class_mapping.json referencing a class id the
    checkpoint doesn't have must fail loudly, not silently misassign."""
    from vistlabel.segmentation.base import BackendUnavailableError
    from vistlabel.segmentation.fake import FakeBackend
    from vistlabel.segmentation.yolo_sam import YoloSamBackend
    from vistlabel.settings import YoloSamModelSettings

    class_mapping = tmp_path / "mapping.json"
    class_mapping.write_text(
        '{"schema_version": 1, "model_classes": {"999": "not_a_real_class"}}', encoding="utf-8"
    )
    settings = YoloSamModelSettings(
        enabled=True, yolo_checkpoint=untrained_checkpoint, class_mapping=class_mapping
    )
    backend = YoloSamBackend(settings, FakeBackend())
    assert backend.descriptor().available is True  # descriptor() doesn't load the model

    with pytest.raises(BackendUnavailableError, match="999"):
        backend.open_video("contract-project", Path("unused.mp4"))
