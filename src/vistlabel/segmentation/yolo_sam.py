# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""YOLO+SAM backend adapter: an orchestrator, not a single model. A
pinned custom YOLO checkpoint proposes class-aware boxes (and optional
coarse masks); the configured SAM refiner (``sam21`` by default,
optionally ``sam3``) turns each box into a precise mask; detections are
associated into logical objects across keyframes and reconciled against
SAM's own between-keyframe propagation.

A *generic* pretrained YOLO checkpoint must never be accepted as a
substitute -- this adapter has no code path that would even
attempt to load one; it only reports available once a real custom
checkpoint, a valid class mapping, and an available refiner backend are all
configured.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vistlabel.persistence.rle import count_connected_components
from vistlabel.segmentation.base import (
    BackendCapability,
    BackendDescriptor,
    BackendUnavailableError,
    ConceptProposal,
    ExemplarPrompt,
    MaskPrediction,
    PointPrompt,
    VideoSegmentationBackend,
)
from vistlabel.segmentation.checkpoints import hash_checkpoint_file
from vistlabel.segmentation.yolo_class_mapping import (
    ClassMappingError,
    YoloClassMapping,
    load_class_mapping,
)
from vistlabel.segmentation.yolo_detector import (
    UltralyticsYoloDetector,
    YoloDetection,
    YoloDetector,
)
from vistlabel.settings import YoloSamModelSettings
from vistlabel.video.frames import build_frame_index, decode_frame

_CAPABILITIES = frozenset(
    {
        BackendCapability.BOX_PROMPTS,
        BackendCapability.VIDEO_PROPAGATION,
        BackendCapability.MULTI_OBJECT,
        BackendCapability.MULTIPART_VISIBLE_MASK,
        BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION,
    }
)

#: A fresh keyframe detection reuses a still-active object's identity when
#: their boxes overlap this much -- this is how candidates are associated
#: across keyframes into logical object identities.
_ASSOCIATION_IOU_THRESHOLD = 0.3
#: Two accepted detections of the *same* class overlapping this much at the
#: same keyframe are a duplicate/conflict, not two real objects.
_DUPLICATE_IOU_THRESHOLD = 0.7
#: SAM's own between-keyframe propagation arriving at a keyframe disagreeing
#: with YOLO's fresh detection there below this mask IoU is flagged for
#: review, not silently overwritten.
_DISAGREEMENT_IOU_THRESHOLD = 0.5


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


@dataclass
class _ActiveObject:
    object_id: str
    yolo_class_id: int
    label_name: str
    last_mask: np.ndarray
    last_frame_index: int


@dataclass
class _Session:
    session_id: str
    refiner_session_id: str
    #: YOLO only ever looks at the requested keyframes, so frames are
    #: seek-decoded on demand (``decode_frame``) instead of holding every
    #: decoded frame in memory for the session's lifetime -- a 4K
    #: recording's full frame list is tens of GB.
    video_path: Path
    timestamps: list[float]
    height: int
    width: int
    active_objects: dict[str, _ActiveObject] = field(default_factory=dict)


class YoloSamBackend:
    backend_id = "yolo_sam"

    def __init__(self, settings: YoloSamModelSettings, refiner: VideoSegmentationBackend) -> None:
        self._settings = settings
        self._refiner = refiner
        self._detector: YoloDetector | None = None
        self._class_mapping: YoloClassMapping | None = None
        self._sessions: dict[str, _Session] = {}

    # -- availability -----------------------------------------------------

    def _unavailable_reason(self) -> str | None:
        if not self._settings.enabled:
            return "disabled in configuration (models.yolo_sam.enabled: false)"
        if self._settings.yolo_checkpoint is None:
            return "no custom YOLO checkpoint configured (models.yolo_sam.yolo_checkpoint)"
        if not self._settings.yolo_checkpoint.is_file():
            return f"YOLO checkpoint file not found: {self._settings.yolo_checkpoint}"
        if self._settings.class_mapping is None:
            return "no class mapping configured (models.yolo_sam.class_mapping)"
        try:
            load_class_mapping(self._settings.class_mapping)
        except ClassMappingError as exc:
            return str(exc)
        refiner_descriptor = self._refiner.descriptor()
        if not refiner_descriptor.available:
            return (
                f"configured refiner backend {self._settings.refiner_backend!r} is "
                f"unavailable: {refiner_descriptor.unavailable_reason}"
            )
        # ultralytics is only needed to load the detector; once one is in
        # place (or injected, as the CPU orchestration tests do), its absence
        # doesn't make the backend unusable.
        if self._detector is None:
            try:
                import ultralytics  # noqa: F401
            except ImportError as exc:
                return f"'gpu' extra not installed ({exc}); pip install -e '.[gpu]'"
        return None

    def descriptor(self) -> BackendDescriptor:
        reason = self._unavailable_reason()
        return BackendDescriptor(
            backend_id=self.backend_id,
            display_name="YOLO + SAM",
            capabilities=_CAPABILITIES,
            available=reason is None,
            unavailable_reason=reason,
            model_name=f"yolo+{self._settings.refiner_backend}",
            checkpoint_sha256=hash_checkpoint_file(self._settings.yolo_checkpoint),
            code_revision=None,
        )

    def _require_available(self) -> None:
        reason = self._unavailable_reason()
        if reason is not None:
            raise BackendUnavailableError(self.backend_id, reason)

    def _ensure_loaded(self) -> None:
        self._require_available()
        if self._detector is not None and self._class_mapping is not None:
            return
        assert self._settings.yolo_checkpoint is not None
        assert self._settings.class_mapping is not None
        class_mapping = load_class_mapping(self._settings.class_mapping)
        detector: YoloDetector = UltralyticsYoloDetector(self._settings.yolo_checkpoint)

        # Validate the mapping against the checkpoint's own
        # classes, not just against itself -- a mapping referencing a class
        # id the checkpoint doesn't have would silently misassign labels.
        detector_class_ids = set(detector.class_names())
        unknown = set(class_mapping.model_classes) - detector_class_ids
        if unknown:
            raise BackendUnavailableError(
                self.backend_id,
                f"class_mapping references YOLO class id(s) {sorted(unknown)} not present "
                f"in the checkpoint (checkpoint has classes {sorted(detector_class_ids)})",
            )
        self._class_mapping = class_mapping
        self._detector = detector

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown or already-closed session: {session_id}") from exc

    # -- protocol: session lifecycle and pass-through to the refiner --------

    def open_video(self, project_id: str, video_path: Path) -> str:
        self._ensure_loaded()
        refiner_session_id = self._refiner.open_video(project_id, video_path)
        timestamps = build_frame_index(video_path)
        first_frame = decode_frame(video_path, 0, timestamps)
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(
            session_id=session_id,
            refiner_session_id=refiner_session_id,
            video_path=video_path,
            timestamps=timestamps,
            height=first_frame.height,
            width=first_frame.width,
        )
        return session_id

    def close_video(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        self._refiner.close_video(session.refiner_session_id)

    def segment_interactive(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        points: list[PointPrompt],
        box_xyxy: tuple[float, float, float, float] | None,
        previous_mask: np.ndarray | None,
    ) -> MaskPrediction:
        session = self._get_session(session_id)
        return self._refiner.segment_interactive(
            session.refiner_session_id, frame_index, object_id, points, box_xyxy, previous_mask
        )

    def propagate(
        self,
        session_id: str,
        object_ids: list[str],
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> Iterable[MaskPrediction]:
        session = self._get_session(session_id)
        return self._refiner.propagate(
            session.refiner_session_id, object_ids, start_frame, end_frame, direction
        )

    def replace_object_mask(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        visible_mask: np.ndarray,
    ) -> MaskPrediction:
        session = self._get_session(session_id)
        return self._refiner.replace_object_mask(
            session.refiner_session_id, frame_index, object_id, visible_mask
        )

    def forget_object(self, session_id: str, object_id: str) -> None:
        session = self._get_session(session_id)
        self._refiner.forget_object(session.refiner_session_id, object_id)

    def find_matching_objects(
        self,
        session_id: str,
        frame_index: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        session = self._get_session(session_id)
        return self._refiner.find_matching_objects(
            session.refiner_session_id, frame_index, text, exemplar
        )

    def find_and_track_matching_objects(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        session = self._get_session(session_id)
        return self._refiner.find_and_track_matching_objects(
            session.refiner_session_id, start_frame, end_frame, text, exemplar
        )

    # -- protocol: real orchestration ---------------------------------------

    def _detect_accepted(
        self, session: _Session, frame_index: int, requested_class_ids: set[int]
    ) -> list[YoloDetection]:
        assert self._detector is not None
        assert self._class_mapping is not None
        frame = np.array(decode_frame(session.video_path, frame_index, session.timestamps))
        detections = self._detector.detect(frame)
        return [
            detection
            for detection in detections
            if detection.yolo_class_id in self._class_mapping.model_classes
            and detection.yolo_class_id in requested_class_ids
        ]

    def auto_preannotate(
        self,
        session_id: str,
        frame_indices: list[int],
        requested_class_ids: list[int],
    ) -> Iterator[MaskPrediction]:
        self._require_available()
        assert self._class_mapping is not None
        session = self._get_session(session_id)
        keyframes = sorted(set(frame_indices))
        if not keyframes:
            return
        requested = set(requested_class_ids) if requested_class_ids else set(
            self._class_mapping.model_classes
        )

        for index, frame_index in enumerate(keyframes):
            # 1. propagate every currently-active object forward to this
            #    keyframe first, so fresh detections can be matched against
            #    where SAM already expects them.
            propagated_by_object: dict[str, np.ndarray] = {}
            if index > 0 and session.active_objects:
                previous_frame_index = keyframes[index - 1]
                object_ids = list(session.active_objects)
                for prediction in self._refiner.propagate(
                    session.refiner_session_id,
                    object_ids,
                    previous_frame_index,
                    frame_index,
                    "forward",
                ):
                    if prediction.frame_index == frame_index:
                        propagated_by_object[prediction.object_id] = prediction.visible_mask
                    elif prediction.frame_index != previous_frame_index:
                        # Between-keyframe frame -- not a keyframe result
                        # itself, but real coverage the caller should
                        # persist too.
                        active = session.active_objects[prediction.object_id]
                        yield MaskPrediction(
                            object_id=prediction.object_id,
                            frame_index=prediction.frame_index,
                            visible_mask=prediction.visible_mask,
                            score=prediction.score,
                            component_count=prediction.component_count,
                            backend_metadata={
                                "backend": self.backend_id,
                                "yolo_class_id": active.yolo_class_id,
                                "label_name": active.label_name,
                                "between_keyframes": True,
                                "review_reasons": [],
                            },
                        )

            # 2. run YOLO fresh at this keyframe, filtered to mapped +
            #    requested classes.
            detections = self._detect_accepted(session, frame_index, requested)

            # 3. associate each accepted detection with a still-active
            #    object of the same class (greedy best-IoU match against
            #    where SAM propagated it to), else mint a new identity.
            matched_object_ids: set[str] = set()
            frame_results: list[tuple[YoloDetection, str, bool]] = []
            for detection in detections:
                best_object_id: str | None = None
                best_iou = 0.0
                for object_id, active in session.active_objects.items():
                    if active.yolo_class_id != detection.yolo_class_id:
                        continue
                    if object_id in matched_object_ids:
                        continue
                    propagated_mask = propagated_by_object.get(object_id)
                    if propagated_mask is None:
                        continue
                    propagated_box = _mask_bbox(propagated_mask)
                    if propagated_box is None:
                        continue
                    iou = _box_iou(propagated_box, detection.box_xyxy)
                    if iou > best_iou:
                        best_iou, best_object_id = iou, object_id
                if best_object_id is not None and best_iou >= _ASSOCIATION_IOU_THRESHOLD:
                    matched_object_ids.add(best_object_id)
                    frame_results.append((detection, best_object_id, True))
                else:
                    frame_results.append((detection, f"yolo_{uuid.uuid4().hex[:12]}", False))

            # 4. refine each accepted detection's box into a precise mask,
            #    reconciling matched ones against where SAM expected them.
            refined_by_object: dict[str, np.ndarray] = {}
            for detection, object_id, was_matched in frame_results:
                previous_mask = detection.coarse_mask
                prediction = self._refiner.segment_interactive(
                    session.refiner_session_id,
                    frame_index,
                    object_id,
                    points=[],
                    box_xyxy=detection.box_xyxy,
                    previous_mask=previous_mask,
                )
                refined_by_object[object_id] = prediction.visible_mask
                label_name = self._class_mapping.model_classes[detection.yolo_class_id]

                review_reasons: list[str] = []
                if was_matched:
                    agreement = _mask_iou(propagated_by_object[object_id], prediction.visible_mask)
                    if agreement < _DISAGREEMENT_IOU_THRESHOLD:
                        review_reasons.append("backend_disagreement")

                session.active_objects[object_id] = _ActiveObject(
                    object_id=object_id,
                    yolo_class_id=detection.yolo_class_id,
                    label_name=label_name,
                    last_mask=prediction.visible_mask,
                    last_frame_index=frame_index,
                )
                yield MaskPrediction(
                    object_id=object_id,
                    frame_index=frame_index,
                    visible_mask=prediction.visible_mask,
                    score=detection.score,
                    component_count=prediction.component_count,
                    backend_metadata={
                        "backend": self.backend_id,
                        "yolo_class_id": detection.yolo_class_id,
                        "label_name": label_name,
                        "new_object": not was_matched,
                        "review_reasons": review_reasons,
                    },
                )

            # 5. duplicate/conflict detection: two accepted objects of the
            #    same class overlapping heavily at this keyframe -- yield
            #    a correction for both with the flag added,
            #    rather than silently keeping two near-identical objects.
            object_ids_this_frame = [object_id for _, object_id, _ in frame_results]
            for i, object_id_a in enumerate(object_ids_this_frame):
                for object_id_b in object_ids_this_frame[i + 1 :]:
                    active_a = session.active_objects[object_id_a]
                    active_b = session.active_objects[object_id_b]
                    if active_a.yolo_class_id != active_b.yolo_class_id:
                        continue
                    if _mask_iou(refined_by_object[object_id_a], refined_by_object[object_id_b]) < (
                        _DUPLICATE_IOU_THRESHOLD
                    ):
                        continue
                    for object_id in (object_id_a, object_id_b):
                        active = session.active_objects[object_id]
                        yield MaskPrediction(
                            object_id=object_id,
                            frame_index=frame_index,
                            visible_mask=refined_by_object[object_id],
                            score=None,
                            component_count=count_connected_components(refined_by_object[object_id]),
                            backend_metadata={
                                "backend": self.backend_id,
                                "yolo_class_id": active.yolo_class_id,
                                "label_name": active.label_name,
                                "new_object": False,
                                "review_reasons": ["object_overlap_conflict"],
                            },
                        )

            # 6. active objects that propagated into this keyframe but got
            #    no fresh detection here -- still real coverage, carried
            #    over without a fresh YOLO confirmation.
            for object_id, propagated_mask in propagated_by_object.items():
                if object_id in matched_object_ids:
                    continue
                active = session.active_objects[object_id]
                active.last_mask = propagated_mask
                active.last_frame_index = frame_index
                yield MaskPrediction(
                    object_id=object_id,
                    frame_index=frame_index,
                    visible_mask=propagated_mask,
                    score=None,
                    component_count=count_connected_components(propagated_mask),
                    backend_metadata={
                        "backend": self.backend_id,
                        "yolo_class_id": active.yolo_class_id,
                        "label_name": active.label_name,
                        "new_object": False,
                        "review_reasons": [],
                    },
                )
