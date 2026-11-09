# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Config-driven availability logic for Sam21Backend.

CPU-only: exercises `descriptor()`'s unavailable-reason computation, which
only checks configuration and file presence -- it never loads the model.
The "actually available and loadable" path needs the `gpu` extra and a
real checkpoint; see tests/model_contract/test_sam21_backend_contract.py.
"""

from __future__ import annotations

from pathlib import Path

from vistlabel.segmentation.base import BackendCapability
from vistlabel.segmentation.sam21 import Sam21Backend
from vistlabel.settings import RuntimeSettings, Sam21ModelSettings

_RUNTIME = RuntimeSettings()


def test_disabled_by_default() -> None:
    backend = Sam21Backend(Sam21ModelSettings(), _RUNTIME)

    descriptor = backend.descriptor()

    assert descriptor.backend_id == "sam21"
    assert descriptor.available is False
    assert "disabled" in descriptor.unavailable_reason


def test_enabled_without_checkpoint_reports_missing_checkpoint() -> None:
    backend = Sam21Backend(Sam21ModelSettings(enabled=True), _RUNTIME)

    descriptor = backend.descriptor()

    assert descriptor.available is False
    assert "checkpoint" in descriptor.unavailable_reason


def test_enabled_with_nonexistent_checkpoint_dir(tmp_path: Path) -> None:
    backend = Sam21Backend(
        Sam21ModelSettings(enabled=True, checkpoint=tmp_path / "missing-dir"), _RUNTIME
    )

    descriptor = backend.descriptor()

    assert descriptor.available is False
    assert "missing" in descriptor.unavailable_reason
    assert descriptor.checkpoint_sha256 is None


def test_enabled_with_incomplete_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "incomplete"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    # model.safetensors deliberately absent

    backend = Sam21Backend(Sam21ModelSettings(enabled=True, checkpoint=checkpoint_dir), _RUNTIME)

    descriptor = backend.descriptor()

    assert descriptor.available is False
    assert "missing" in descriptor.unavailable_reason


def test_declares_expected_capabilities() -> None:
    backend = Sam21Backend(Sam21ModelSettings(), _RUNTIME)

    descriptor = backend.descriptor()

    assert descriptor.capabilities == frozenset(
        {
            BackendCapability.POINT_PROMPTS,
            BackendCapability.BOX_PROMPTS,
            BackendCapability.MASK_PROMPTS,
            BackendCapability.VIDEO_PROPAGATION,
            BackendCapability.MULTI_OBJECT,
            BackendCapability.MULTIPART_VISIBLE_MASK,
        }
    )
