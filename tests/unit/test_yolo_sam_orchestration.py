# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""YOLO+SAM orchestration logic (associate detections across keyframes,
initialize logical identities, reconcile against SAM propagation, detect
duplicates), exercised deterministically on the CPU: :class:`FakeBackend` as
the SAM refiner and a scripted fake YOLO detector, so none of this depends
on a trained checkpoint or GPU. Real-checkpoint parsing
(:class:`~vistlabel.segmentation.yolo_detector.UltralyticsYoloDetector`) has
its own GPU-marked coverage in ``tests/model_contract``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import FIXTURES_DIR
from vistlabel.segmentation.fake import FakeBackend
from vistlabel.segmentation.yolo_class_mapping import YoloClassMapping
from vistlabel.segmentation.yolo_detector import YoloDetection, YoloDetector
from vistlabel.segmentation.yolo_sam import YoloSamBackend
from vistlabel.settings import YoloSamModelSettings

VIDEO_PATH = FIXTURES_DIR / "tiny_test_video.mp4"


class ScriptedYoloDetector(YoloDetector):
    """Returns one pre-scripted detection list per call, in call order --
    deterministic and independent of actual frame pixel content, since the
    thing under test is the orchestration logic around detections, not
    detection itself."""

    def __init__(self, names: dict[int, str], script: list[list[YoloDetection]]) -> None:
        self._names = names
        self._script = list(script)
        self.calls = 0

    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def detect(self, frame) -> list[YoloDetection]:  # noqa: ANN001 - matches Protocol
        detections = self._script[self.calls]
        self.calls += 1
        return detections


def _det(box, class_id, score=0.9) -> YoloDetection:
    return YoloDetection(box_xyxy=box, yolo_class_id=class_id, score=score, coarse_mask=None)


@pytest.fixture
def yolo_backend(tmp_path: Path) -> tuple[YoloSamBackend, ScriptedYoloDetector]:
    checkpoint = tmp_path / "lab_yolo.pt"
    checkpoint.write_bytes(b"not a real checkpoint, just bytes for hashing")
    class_mapping_path = tmp_path / "yolo_class_mapping.json"
    class_mapping_path.write_text(
        '{"schema_version": 1, "model_classes": '
        '{"0": "pipette", "1": "glove", "2": "tube"}}',
        encoding="utf-8",
    )
    settings = YoloSamModelSettings(
        enabled=True, yolo_checkpoint=checkpoint, class_mapping=class_mapping_path
    )
    backend = YoloSamBackend(settings, FakeBackend())

    # Keyframe 0: one new "pipette" detection.
    # Keyframe 4: the pipette re-detected exactly where SAM propagated it
    # (association), a new "glove" object, an unmapped class (99, must be
    # rejected) and a mapped-but-not-requested class (2 "tube", must be
    # rejected since requested_class_ids excludes it).
    # Keyframe 8: two overlapping "pipette" detections (duplicate/conflict);
    # no fresh "glove" detection (carried over via propagation only).
    script = [
        [_det((4, 4, 12, 12), class_id=0)],
        [
            _det((12, 4, 20, 12), class_id=0),
            _det((40, 40, 48, 48), class_id=1),
            _det((1, 1, 5, 5), class_id=99),
            _det((30, 30, 34, 34), class_id=2),
        ],
        [
            _det((20, 4, 28, 12), class_id=0),
            _det((21, 4, 29, 12), class_id=0),
        ],
    ]
    detector = ScriptedYoloDetector(names={0: "pipette", 1: "glove", 2: "tube"}, script=script)
    backend._detector = detector  # bypass real ultralytics load; see module docstring
    backend._class_mapping = YoloClassMapping(
        schema_version=1, model_classes={0: "pipette", 1: "glove", 2: "tube"}
    )
    return backend, detector


def test_auto_preannotate_associates_detects_and_reconciles(
    yolo_backend: tuple[YoloSamBackend, ScriptedYoloDetector],
) -> None:
    backend, detector = yolo_backend
    session_id = backend.open_video("proj-1", VIDEO_PATH)
    try:
        predictions = list(
            backend.auto_preannotate(
                session_id, frame_indices=[0, 4, 8], requested_class_ids=[0, 1]
            )
        )
    finally:
        backend.close_video(session_id)

    assert detector.calls == 3  # exactly one detect() call per keyframe

    by_frame_and_object: dict[tuple[int, str], list[dict]] = {}
    for prediction in predictions:
        key = (prediction.frame_index, prediction.object_id)
        by_frame_and_object.setdefault(key, []).append(prediction.backend_metadata)

    # Frame 0: exactly one new pipette object.
    frame0 = [k for k in by_frame_and_object if k[0] == 0]
    assert len(frame0) == 1
    pipette_object_id = frame0[0][1]
    assert by_frame_and_object[frame0[0]][0]["new_object"] is True
    assert by_frame_and_object[frame0[0]][0]["label_name"] == "pipette"

    # Frame 4: the same pipette object_id is reused (association), plus one
    # new glove object; the unmapped (99) and unrequested (2) detections
    # never appear anywhere in the output.
    frame4_object_ids = {k[1] for k in by_frame_and_object if k[0] == 4}
    assert pipette_object_id in frame4_object_ids
    assert by_frame_and_object[(4, pipette_object_id)][0]["new_object"] is False
    glove_candidates = frame4_object_ids - {pipette_object_id}
    assert len(glove_candidates) == 1
    glove_object_id = next(iter(glove_candidates))
    assert by_frame_and_object[(4, glove_object_id)][0]["label_name"] == "glove"
    assert all(
        metadata_list[0]["yolo_class_id"] in (0, 1)
        for (frame, _obj), metadata_list in by_frame_and_object.items()
        if frame == 4
    )

    # Between-keyframe frames (1-3, 5-7) got real coverage too.
    between_keyframe_frames = {
        prediction.frame_index
        for prediction in predictions
        if prediction.backend_metadata.get("between_keyframes")
    }
    assert between_keyframe_frames == {1, 2, 3, 5, 6, 7}

    # Frame 8: the pipette object is reconfirmed AND a duplicate detection
    # of the same class overlapping it heavily is flagged as a conflict on
    # both sides -- not silently accepted as two independent objects.
    frame8_pipette_class_object_ids = {
        obj_id
        for (frame, obj_id), metadata_list in by_frame_and_object.items()
        if frame == 8 and metadata_list[0]["yolo_class_id"] == 0
    }
    assert len(frame8_pipette_class_object_ids) == 2  # the original + the new duplicate
    conflict_flagged = [
        obj_id
        for obj_id in frame8_pipette_class_object_ids
        if "object_overlap_conflict" in by_frame_and_object[(8, obj_id)][-1]["review_reasons"]
    ]
    assert set(conflict_flagged) == frame8_pipette_class_object_ids

    # Frame 8: glove got no fresh detection, but is still carried over via
    # SAM's own propagation from frame 4.
    assert (8, glove_object_id) in by_frame_and_object
    glove_frame8_metadata = by_frame_and_object[(8, glove_object_id)][0]
    assert glove_frame8_metadata["new_object"] is False
    assert glove_frame8_metadata["review_reasons"] == []


def test_auto_preannotate_with_no_keyframes_yields_nothing(
    yolo_backend: tuple[YoloSamBackend, ScriptedYoloDetector],
) -> None:
    backend, detector = yolo_backend
    session_id = backend.open_video("proj-1", VIDEO_PATH)
    try:
        predictions = list(
            backend.auto_preannotate(session_id, frame_indices=[], requested_class_ids=[0])
        )
    finally:
        backend.close_video(session_id)
    assert predictions == []
    assert detector.calls == 0
