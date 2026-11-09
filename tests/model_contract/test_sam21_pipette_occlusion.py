# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Does the *real* SAM 2.1 model handle the pipette-occlusion scenario
correctly, not just our own bookkeeping around it? See
tests/integration/test_pipette_occlusion_fixture.py for the
(model-independent) ground-truth version of the same acceptance criteria.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import FIXTURES_DIR
from vistlabel.segmentation.base import PointPrompt

pytestmark = pytest.mark.gpu

CHECKPOINT_DIR = Path("models/sam2.1-hiera-base-plus")
PIPETTE_FIXTURE = FIXTURES_DIR / "pipette_occlusion_video.mp4"
# Must match scripts/generate_pipette_fixture.py.
FINGER_ENTER_FRAME = 7
FINGER_EXIT_FRAME = 13


@pytest.fixture(scope="module")
def sam21_backend():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("'gpu' extra not installed (pip install -e '.[gpu]')")
    if not (CHECKPOINT_DIR / "model.safetensors").is_file():
        pytest.skip(f"checkpoint not downloaded at {CHECKPOINT_DIR}")

    from vistlabel.segmentation.sam21 import Sam21Backend
    from vistlabel.settings import RuntimeSettings, Sam21ModelSettings

    settings = Sam21ModelSettings(enabled=True, checkpoint=CHECKPOINT_DIR)
    runtime_settings = RuntimeSettings(cuda_device="auto", default_dtype="float16")
    return Sam21Backend(settings, runtime_settings)


def test_single_point_loses_the_object_through_full_occlusion(sam21_backend) -> None:
    """Documents *why* "Add Visible Component to Selected Object" exists,
    using the real model rather than assuming it: seeding with a single
    point that ends up under the finger, SAM 2.1 tracks only that local
    region, which shrinks to nothing once occluded and does not reacquire
    the far segment on its own once the finger has passed."""
    session_id = sam21_backend.open_video("pipette-single-point", PIPETTE_FIXTURE)
    try:
        sam21_backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="the-pipette",
            points=[PointPrompt(x=64, y=32, positive=True)],
            box_xyxy=None,
            previous_mask=None,
        )
        predictions = list(
            sam21_backend.propagate(
                session_id,
                object_ids=["the-pipette"],
                start_frame=0,
                end_frame=19,
                direction="forward",
            )
        )
        by_frame = {p.frame_index: p for p in predictions}
        assert not by_frame[19].visible_mask.any(), (
            "if this starts failing, single-point tracking got better at "
            "reacquiring the far segment -- the Add Visible Component workaround "
            "may be less load-bearing than this test assumes; revisit"
        )
    finally:
        sam21_backend.close_video(session_id)


def test_hand_correction_on_a_tracked_frame_steers_the_next_run(sam21_backend) -> None:
    """The next frame must follow the corrected mask. Left in memory, the
    earlier frames outvote one correction and the cut-off half grows back."""
    session_id = sam21_backend.open_video("pipette-correction", PIPETTE_FIXTURE)
    try:
        sam21_backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="the-pipette",
            points=[
                PointPrompt(x=25, y=32, positive=True),
                PointPrompt(x=100, y=32, positive=True),
            ],
            box_xyxy=None,
            previous_mask=None,
        )
        tracked = list(sam21_backend.propagate(session_id, ["the-pipette"], 0, 6, "forward"))
        full_mask = tracked[3].visible_mask
        columns = np.where(full_mask.any(axis=0))[0]
        cut = (columns[0] + columns[-1]) // 2
        corrected = full_mask.copy()
        corrected[:, cut:] = False

        sam21_backend.forget_object(session_id, "the-pipette")
        sam21_backend.replace_object_mask(session_id, 3, "the-pipette", corrected)
        next_frames = list(sam21_backend.propagate(session_id, ["the-pipette"], 3, 4, "forward"))

        assert [p.frame_index for p in next_frames] == [3, 4]
        assert next_frames[1].visible_mask.any()
        assert not next_frames[1].visible_mask[:, cut + 2 :].any()
    finally:
        sam21_backend.close_video(session_id)


def test_two_seed_points_preserve_one_object_through_the_occlusion(sam21_backend) -> None:
    """The actual acceptance criteria, seeded the recommended way: collect
    positive prompts on every visible pipette section *before*
    propagating -- i.e. exactly what Shift+click ("Add Visible Component
    to Selected Object") produces in the UI. With that seeding, the real
    model's component counts match the synthetic ground truth in
    tests/integration/test_pipette_occlusion_fixture.py exactly, not just
    approximately."""
    session_id = sam21_backend.open_video("pipette-two-points", PIPETTE_FIXTURE)
    try:
        first = sam21_backend.segment_interactive(
            session_id,
            frame_index=0,
            object_id="the-pipette",
            points=[
                PointPrompt(x=25, y=32, positive=True),
                PointPrompt(x=100, y=32, positive=True),
            ],
            box_xyxy=None,
            previous_mask=None,
        )
        assert first.component_count == 1

        predictions = list(
            sam21_backend.propagate(
                session_id,
                object_ids=["the-pipette"],
                start_frame=0,
                end_frame=19,
                direction="forward",
            )
        )
        # One logical object_id the whole way through -- the Protocol only
        # ever lets us ask about the object_id we passed in, so this is
        # really asserting the backend never silently drops/renames it.
        assert len(predictions) == 20
        assert all(p.object_id == "the-pipette" for p in predictions)
        assert [p.frame_index for p in predictions] == list(range(20))

        by_frame = {p.frame_index: p for p in predictions}
        for frame_index, prediction in by_frame.items():
            if FINGER_ENTER_FRAME <= frame_index <= FINGER_EXIT_FRAME:
                assert prediction.component_count == 2, frame_index
            else:
                assert prediction.component_count == 1, frame_index
            assert prediction.visible_mask.any(), frame_index

        # Reappearance preserves the original identity: the
        # last frame is fully visible again under the same object_id, not
        # a fresh one.
        assert by_frame[19].object_id == "the-pipette"
    finally:
        sam21_backend.close_video(session_id)
