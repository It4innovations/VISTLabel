# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Real GPU contract test for Sam21Backend.

Requires the `gpu` extra installed and a downloaded checkpoint
(`python scripts/download_models.py --model sam21-base-plus`). Skips
-- does not fail -- when either is unavailable, since that's an
environment fact, not a code correctness question (same pattern as the
`ffmpeg_binaries` fixture).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import TINY_TEST_VIDEO
from tests.model_contract.backend_contract import run_full_contract

pytestmark = pytest.mark.gpu

CHECKPOINT_DIR = Path("models/sam2.1-hiera-base-plus")


@pytest.fixture(scope="module")
def sam21_backend():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("'gpu' extra not installed (pip install -e '.[gpu]')")

    if not (CHECKPOINT_DIR / "model.safetensors").is_file():
        pytest.skip(
            f"checkpoint not downloaded at {CHECKPOINT_DIR} "
            "(python scripts/download_models.py --model sam21-base-plus)"
        )

    from vistlabel.segmentation.sam21 import Sam21Backend
    from vistlabel.settings import RuntimeSettings, Sam21ModelSettings

    settings = Sam21ModelSettings(enabled=True, checkpoint=CHECKPOINT_DIR)
    runtime_settings = RuntimeSettings(cuda_device="auto", default_dtype="float16")
    return Sam21Backend(settings, runtime_settings)


def test_sam21_backend_is_available_with_a_real_checkpoint(sam21_backend) -> None:
    descriptor = sam21_backend.descriptor()
    assert descriptor.available is True
    assert descriptor.unavailable_reason is None
    assert descriptor.checkpoint_sha256 is not None
    assert len(descriptor.checkpoint_sha256) == 64


def test_sam21_backend_satisfies_the_common_contract(sam21_backend) -> None:
    run_full_contract(sam21_backend, video_path=TINY_TEST_VIDEO)


def test_auto_preannotate_raises_capability_error_once_available(sam21_backend) -> None:
    from vistlabel.segmentation.base import BackendCapabilityError

    with pytest.raises(BackendCapabilityError):
        list(sam21_backend.auto_preannotate("session", frame_indices=[0], requested_class_ids=[0]))


def test_sam21_backend_tracks_a_point_across_the_whole_fixture_video(sam21_backend) -> None:
    from vistlabel.segmentation.base import PointPrompt

    session_id = sam21_backend.open_video("contract-project", TINY_TEST_VIDEO)
    try:
        first = sam21_backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="tracked-obj",
            points=[PointPrompt(x=32, y=32, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        assert first.visible_mask.any()

        predictions = list(
            sam21_backend.propagate(
                session_id,
                object_ids=["tracked-obj"],
                start_frame=0,
                end_frame=9,
                direction="forward",
            )
        )
        assert [p.frame_index for p in predictions] == list(range(10))
        assert all(p.visible_mask.any() for p in predictions)
    finally:
        sam21_backend.close_video(session_id)
