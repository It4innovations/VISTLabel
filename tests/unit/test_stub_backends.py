# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from vistlabel.segmentation.base import BackendCapability, BackendUnavailableError
from vistlabel.segmentation.fake import FakeBackend
from vistlabel.segmentation.yolo_sam import YoloSamBackend
from vistlabel.settings import YoloSamModelSettings


# No pure "always raises unavailable" stub backends remain: Sam21Backend
# and Sam3Backend (text/concept prompts, SAM 3) are both real backends.
# Their config-driven availability tests live in
# tests/unit/test_sam21_backend_config.py and test_sam3_backend_config.py;
# the "available" paths, which need the gpu extra and real checkpoints, in
# tests/model_contract/.
def test_yolo_sam_disabled_by_default() -> None:
    descriptor = YoloSamBackend(YoloSamModelSettings(), FakeBackend()).descriptor()
    assert descriptor.available is False
    assert "disabled" in descriptor.unavailable_reason
    assert BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION in descriptor.capabilities


def test_yolo_sam_never_falls_back_to_a_generic_checkpoint(tmp_path: Path) -> None:
    """No code path may substitute a generic pretrained model."""
    checkpoint = tmp_path / "generic_pretrained_yolo.pt"
    checkpoint.write_bytes(b"pretend generic weights")
    backend = YoloSamBackend(
        YoloSamModelSettings(enabled=True, yolo_checkpoint=checkpoint), FakeBackend()
    )

    descriptor = backend.descriptor()

    assert descriptor.available is False
    assert "class mapping" in descriptor.unavailable_reason

    with pytest.raises(BackendUnavailableError):
        list(backend.auto_preannotate("session", [0], [0]))


def test_yolo_sam_rejects_invalid_class_mapping_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "custom.pt"
    checkpoint.write_bytes(b"custom weights")
    class_mapping = tmp_path / "mapping.json"
    class_mapping.write_text("not valid json{", encoding="utf-8")
    backend = YoloSamBackend(
        YoloSamModelSettings(enabled=True, yolo_checkpoint=checkpoint, class_mapping=class_mapping),
        FakeBackend(),
    )

    descriptor = backend.descriptor()

    assert descriptor.available is False
    assert "not valid JSON" in descriptor.unavailable_reason
