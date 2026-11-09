# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Deterministic, CPU-only test double implementing the full backend contract.

Used to exercise the complete UI-to-backend path and the common backend
contract test suite without any GPU or model checkpoint. It never claims
class-aware capabilities it cannot provide -- ``auto_preannotate`` raises
:class:`BackendCapabilityError`, matching the requirement that no backend
silently substitutes unrelated results.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np

from vistlabel import __version__
from vistlabel.persistence.rle import count_connected_components
from vistlabel.segmentation.base import (
    BackendCapability,
    BackendCapabilityError,
    BackendDescriptor,
    ConceptProposal,
    ExemplarPrompt,
    MaskPrediction,
    PointPrompt,
    VideoSegmentationBackend,
)

_POINT_RADIUS = 4


@dataclass
class _Session:
    session_id: str
    project_id: str
    video_path: Path
    height: int
    width: int
    frame_count: int
    object_masks: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)


def _disk_mask(height: int, width: int, cx: float, cy: float, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2


def _box_mask(height: int, width: int, box: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = box
    mask = np.zeros((height, width), dtype=bool)
    xi0, yi0 = max(0, int(round(min(x0, x1)))), max(0, int(round(min(y0, y1))))
    xi1, yi1 = min(width, int(round(max(x0, x1)))), min(height, int(round(max(y0, y1))))
    mask[yi0:yi1, xi0:xi1] = True
    return mask


class FakeBackend:
    """In-memory, deterministic stand-in for a real segmentation backend."""

    backend_id = "fake"

    def __init__(
        self, frame_height: int = 64, frame_width: int = 64, frame_count: int = 64
    ) -> None:
        self._height = frame_height
        self._width = frame_width
        self._frame_count = frame_count
        self._sessions: dict[str, _Session] = {}

    def descriptor(self) -> BackendDescriptor:
        return BackendDescriptor(
            backend_id=self.backend_id,
            display_name="Fake Backend (deterministic, CPU-only)",
            capabilities=frozenset(
                {
                    BackendCapability.POINT_PROMPTS,
                    BackendCapability.BOX_PROMPTS,
                    BackendCapability.MASK_PROMPTS,
                    BackendCapability.VIDEO_PROPAGATION,
                    BackendCapability.MULTI_OBJECT,
                    BackendCapability.MULTIPART_VISIBLE_MASK,
                }
            ),
            available=True,
            unavailable_reason=None,
            model_name="vistlabel-fake",
            checkpoint_sha256=None,
            code_revision=__version__,
        )

    def _probe_dimensions(self, video_path: Path) -> tuple[int, int, int]:
        """Best-effort real width/height/frame_count so masks line up with
        the actual uploaded video -- falls back to the fixed
        synthetic defaults for diagnostics/contract-test callers that pass a
        throwaway/nonexistent path, matching prior behaviour."""
        try:
            with av.open(str(video_path)) as container:
                stream = container.streams.video[0]
                height, width = stream.height, stream.width
                frame_count = stream.frames or self._frame_count
        except Exception:  # noqa: BLE001 - best-effort probe, any failure falls back
            return self._height, self._width, self._frame_count
        return height, width, frame_count

    def open_video(self, project_id: str, video_path: Path) -> str:
        height, width, frame_count = self._probe_dimensions(video_path)
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(
            session_id=session_id,
            project_id=project_id,
            video_path=video_path,
            height=height,
            width=width,
            frame_count=frame_count,
        )
        return session_id

    def close_video(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _get_session(self, session_id: str) -> _Session:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown or already-closed session: {session_id}") from exc

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
        mask = np.zeros((session.height, session.width), dtype=bool)
        if previous_mask is not None:
            mask |= previous_mask.astype(bool)
        if box_xyxy is not None:
            mask |= _box_mask(session.height, session.width, box_xyxy)
        for point in points:
            disk = _disk_mask(session.height, session.width, point.x, point.y, _POINT_RADIUS)
            if point.positive:
                mask |= disk
            else:
                mask &= ~disk

        session.object_masks.setdefault(object_id, {})[frame_index] = mask
        return MaskPrediction(
            object_id=object_id,
            frame_index=frame_index,
            visible_mask=mask,
            score=1.0,
            component_count=count_connected_components(mask),
            backend_metadata={"synthetic": True},
        )

    def propagate(
        self,
        session_id: str,
        object_ids: list[str],
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> Iterator[MaskPrediction]:
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
        session = self._get_session(session_id)
        step = 1 if direction == "forward" else -1
        frame_range = range(start_frame, end_frame + step, step)

        for object_id in object_ids:
            frames_for_object = session.object_masks.setdefault(object_id, {})
            if start_frame not in frames_for_object:
                raise BackendCapabilityError(
                    self.backend_id,
                    BackendCapability.VIDEO_PROPAGATION,
                    f"object {object_id!r} has no mask on start_frame={start_frame}; "
                    "segment_interactive must be called first",
                )
            current_mask = frames_for_object[start_frame]
            distance = 0
            for frame_index in frame_range:
                if frame_index != start_frame:
                    current_mask = np.roll(current_mask, shift=step * 2, axis=1)
                    distance += 1
                    frames_for_object[frame_index] = current_mask
                score = max(0.5, 1.0 - 0.01 * distance)
                yield MaskPrediction(
                    object_id=object_id,
                    frame_index=frame_index,
                    visible_mask=current_mask,
                    score=score,
                    component_count=count_connected_components(current_mask),
                    backend_metadata={"synthetic": True, "propagated_from": start_frame},
                )

    def replace_object_mask(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        visible_mask: np.ndarray,
    ) -> MaskPrediction:
        session = self._get_session(session_id)
        if visible_mask.shape != (session.height, session.width):
            raise ValueError(
                f"mask shape {visible_mask.shape} does not match session frame shape "
                f"{(session.height, session.width)}"
            )
        mask = visible_mask.astype(bool)
        session.object_masks.setdefault(object_id, {})[frame_index] = mask
        return MaskPrediction(
            object_id=object_id,
            frame_index=frame_index,
            visible_mask=mask,
            score=None,
            component_count=count_connected_components(mask),
            backend_metadata={"source": "manual_replace"},
        )

    def forget_object(self, session_id: str, object_id: str) -> None:
        self._get_session(session_id).object_masks.pop(object_id, None)

    def auto_preannotate(
        self,
        session_id: str,
        frame_indices: list[int],
        requested_class_ids: list[int],
    ) -> Iterator[MaskPrediction]:
        raise BackendCapabilityError(
            self.backend_id,
            BackendCapability.CLASS_AWARE_AUTOMATIC_PREANNOTATION,
            "the fake backend has no notion of object classes; "
            "use YoloSamBackend with a configured custom checkpoint",
        )

    def find_matching_objects(
        self,
        session_id: str,
        frame_index: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterator[ConceptProposal]:
        raise BackendCapabilityError(
            self.backend_id,
            BackendCapability.TEXT_PROMPTS,
            "the fake backend has no concept/text-search model; use Sam3Backend",
        )

    def find_and_track_matching_objects(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterator[ConceptProposal]:
        raise BackendCapabilityError(
            self.backend_id,
            BackendCapability.TEXT_PROMPTS,
            "the fake backend has no concept/text-search model; use Sam3Backend",
        )


# Explicit static check that FakeBackend satisfies the Protocol shape.
_conforms: type[VideoSegmentationBackend] = FakeBackend
