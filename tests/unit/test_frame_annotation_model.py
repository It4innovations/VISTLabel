# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from vistlabel.domain.frames import FrameAnnotation

_ENTRY = {
    "object_id": "obj_1",
    "label_id": 0,
    "visible_mask_rle": {"size": [10, 10], "counts": "fake"},
    "connected_component_count": 1,
    "source": "test",
}


def test_inactive_objects_reads_the_earlier_hidden_objects_key() -> None:
    # Frames saved while the field was still called hidden_objects must not
    # silently lose those masks (unknown keys are otherwise ignored).
    frame = FrameAnnotation.model_validate(
        {"frame_index": 0, "timestamp_seconds": 0.0, "hidden_objects": [_ENTRY]}
    )
    assert [o.object_id for o in frame.inactive_objects] == ["obj_1"]

    saved = json.loads(frame.model_dump_json())
    assert [o["object_id"] for o in saved["inactive_objects"]] == ["obj_1"]
    assert "hidden_objects" not in saved


def test_inactive_objects_round_trips_under_its_own_name() -> None:
    frame = FrameAnnotation.model_validate(
        {"frame_index": 0, "timestamp_seconds": 0.0, "inactive_objects": [_ENTRY]}
    )
    assert FrameAnnotation.model_validate_json(frame.model_dump_json()) == frame
