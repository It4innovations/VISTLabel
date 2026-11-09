# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Config-driven availability and capability honesty for Sam3Backend.

CPU-only: exercises ``descriptor()``'s unavailable-reason computation and
the refusal paths, neither of which loads a model. The "actually available
and loadable" path needs the ``gpu`` extra and the gated ``facebook/sam3``
checkpoint.

The backend id is ``sam3`` but the model behind it is SAM 3 -- SAM 3.1
ships without a transformers integration (see segmentation/sam3.py) -- so
these tests also pin that the descriptor says so rather than claiming 3.1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vistlabel.segmentation.base import (
    BackendCapability,
    BackendCapabilityError,
    BackendUnavailableError,
)
from vistlabel.segmentation.sam3 import Sam3Backend
from vistlabel.settings import RuntimeSettings, Sam3ModelSettings

_RUNTIME = RuntimeSettings()


def _backend(**kwargs: object) -> Sam3Backend:
    return Sam3Backend(Sam3ModelSettings(**kwargs), _RUNTIME)  # type: ignore[arg-type]


def test_disabled_by_default() -> None:
    descriptor = _backend().descriptor()

    assert descriptor.backend_id == "sam3"
    assert descriptor.available is False
    assert "disabled" in descriptor.unavailable_reason


def test_reports_sam3_not_sam3() -> None:
    """Claiming "SAM 3.1" would misreport which weights produced a mask,
    and the checkpoint hash is recorded per inference event."""
    descriptor = _backend().descriptor()

    assert descriptor.display_name == "SAM 3"
    assert descriptor.model_name == "sam3"


def test_enabled_without_checkpoint_reports_missing_checkpoint() -> None:
    descriptor = _backend(enabled=True).descriptor()

    assert descriptor.available is False
    assert "checkpoint" in descriptor.unavailable_reason


def test_enabled_with_nonexistent_checkpoint_dir(tmp_path: Path) -> None:
    descriptor = _backend(enabled=True, checkpoint=tmp_path / "missing-dir").descriptor()

    assert descriptor.available is False
    assert "not found" in descriptor.unavailable_reason
    assert "download_models.py" in descriptor.unavailable_reason
    assert descriptor.checkpoint_sha256 is None


def test_enabled_with_incomplete_checkpoint_dir(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "incomplete"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
    # model.safetensors deliberately absent

    descriptor = _backend(enabled=True, checkpoint=checkpoint_dir).descriptor()

    assert descriptor.available is False
    assert "missing" in descriptor.unavailable_reason
    assert "model.safetensors" in descriptor.unavailable_reason


def test_declares_capabilities_it_can_honour() -> None:
    """SAM 3 has separate text and tracker model paths in transformers."""
    descriptor = _backend().descriptor()

    assert descriptor.capabilities == frozenset(
        {
            BackendCapability.POINT_PROMPTS,
            BackendCapability.BOX_PROMPTS,
            BackendCapability.MASK_PROMPTS,
            BackendCapability.TEXT_PROMPTS,
            BackendCapability.VIDEO_PROPAGATION,
            BackendCapability.MULTI_OBJECT,
            BackendCapability.MULTIPART_VISIBLE_MASK,
        }
    )
    for absent in (
        BackendCapability.EXEMPLAR_PROMPTS,
        BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION,
    ):
        assert absent not in descriptor.capabilities


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (
            lambda b: list(b.auto_preannotate("s", [0], [0])),
            BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION,
        ),
    ],
)
def test_unsupported_operations_raise_capability_error(operation, expected) -> None:
    """A typed capability error, never a silent no-op and never a generic
    exception the API layer could not distinguish from a real failure."""
    backend = _backend()

    with pytest.raises(BackendCapabilityError) as excinfo:
        operation(backend)

    assert excinfo.value.capability is expected


def test_concept_search_on_disabled_backend_reports_unavailable() -> None:
    backend = _backend()

    with pytest.raises(BackendUnavailableError):
        list(backend.find_matching_objects("s", 0, "pipette", None))


def test_open_video_on_disabled_backend_reports_unavailable() -> None:
    with pytest.raises(BackendUnavailableError):
        _backend().open_video("proj", Path("unused.mp4"))
