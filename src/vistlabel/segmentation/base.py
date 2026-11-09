# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Backend-neutral segmentation protocol.

Model-specific state must never leak into canonical annotations: callers
only ever see :class:`BackendDescriptor`, :class:`MaskPrediction` and the
prompt dataclasses defined here, regardless of which concrete backend
(``Sam21Backend``, ``Sam3Backend``, ``YoloSamBackend``, ``FakeBackend``, ...)
produced them.

A backend that does not support a given operation must raise
:class:`BackendCapabilityError`, never silently no-op.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np


class BackendCapability(StrEnum):
    POINT_PROMPTS = "point_prompts"
    BOX_PROMPTS = "box_prompts"
    MASK_PROMPTS = "mask_prompts"
    TEXT_PROMPTS = "text_prompts"
    EXEMPLAR_PROMPTS = "exemplar_prompts"
    VIDEO_PROPAGATION = "video_propagation"
    MULTI_OBJECT = "multi_object"
    CLASS_AWARE_AUTOMATIC_PREANNOTATION = "class_aware_automatic_preannotation"
    MULTIPART_VISIBLE_MASK = "multipart_visible_mask"


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    display_name: str
    capabilities: frozenset[BackendCapability]
    available: bool
    unavailable_reason: str | None
    model_name: str
    checkpoint_sha256: str | None
    code_revision: str | None


@dataclass(frozen=True)
class PointPrompt:
    x: float
    y: float
    positive: bool


@dataclass
class MaskPrediction:
    object_id: str
    frame_index: int
    visible_mask: np.ndarray
    score: float | None
    component_count: int
    backend_metadata: dict[str, object]


@dataclass(frozen=True)
class ExemplarPrompt:
    """A single already-visible instance the user boxes as an example of
    "more like this" (an exemplar prompt), distinct from a
    :class:`PointPrompt`/box used to segment *that specific* object: an
    exemplar is a search seed, never itself upserted as an annotation."""

    frame_index: int
    box_xyxy: tuple[float, float, float, float]


@dataclass
class ConceptProposal:
    """One candidate detection from a text/exemplar concept search.
    Deliberately *not* a :class:`MaskPrediction` / canonical object: the
    user must confirm or change the class before a proposal is accepted,
    so ``suggested_label_name`` is only ever a hint for the UI's label
    picker, never trusted as a resolved ``label_id``. No model-specific
    structure leaks into canonical annotations -- a proposal only becomes
    canonical by going through the normal object/frame-annotation upsert
    path once accepted.
    ``proposal_id`` groups the same detected instance across frames when it
    came from :meth:`VideoSegmentationBackend.find_and_track_matching_objects`;
    for a single-frame :meth:`~VideoSegmentationBackend.find_matching_objects`
    call it merely distinguishes proposals within that one response."""

    proposal_id: str
    frame_index: int
    visible_mask: np.ndarray
    score: float
    suggested_label_name: str
    box_xyxy: tuple[float, float, float, float]


class BackendCapabilityError(Exception):
    """Raised when a backend does not support a requested operation.

    Callers (the API layer, the job manager) must translate this into a
    typed error response -- never a silent no-op and never a generic
    exception that would be indistinguishable from a real failure.
    """

    def __init__(self, backend_id: str, capability: BackendCapability, reason: str) -> None:
        self.backend_id = backend_id
        self.capability = capability
        self.reason = reason
        super().__init__(f"{backend_id}: unsupported capability {capability.value}: {reason}")


class BackendUnavailableError(Exception):
    """Raised by every operation on a backend whose ``descriptor().available``
    is ``False`` -- never silently substitute a different model.
    Distinct from :class:`BackendCapabilityError`, which is about one
    unsupported operation on an otherwise-usable backend."""

    def __init__(self, backend_id: str, reason: str) -> None:
        self.backend_id = backend_id
        self.reason = reason
        super().__init__(f"{backend_id}: unavailable: {reason}")


class VideoSegmentationBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...

    def open_video(self, project_id: str, video_path: Path) -> str: ...

    def close_video(self, session_id: str) -> None: ...

    def segment_interactive(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        points: list[PointPrompt],
        box_xyxy: tuple[float, float, float, float] | None,
        previous_mask: np.ndarray | None,
    ) -> MaskPrediction: ...

    def propagate(
        self,
        session_id: str,
        object_ids: list[str],
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> Iterable[MaskPrediction]: ...

    def replace_object_mask(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        visible_mask: np.ndarray,
    ) -> MaskPrediction: ...

    def forget_object(self, session_id: str, object_id: str) -> None:
        """Drop everything the session remembers about ``object_id``, so the
        next :meth:`replace_object_mask` is its only seed. Other objects are
        untouched."""
        ...

    def auto_preannotate(
        self,
        session_id: str,
        frame_indices: list[int],
        requested_class_ids: list[int],
    ) -> Iterable[MaskPrediction]: ...

    def find_matching_objects(
        self,
        session_id: str,
        frame_index: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        """Find all matching objects in the current frame: open-vocabulary/
        exemplar search on a single frame. Raise
        :class:`BackendCapabilityError` (``TEXT_PROMPTS`` or
        ``EXEMPLAR_PROMPTS``) if unsupported -- never silently return an
        empty list for an unsupported prompt kind."""
        ...

    def find_and_track_matching_objects(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        """Find and track matching objects in a selected interval: the same
        search, propagated across a frame range, with ``proposal_id``
        stable per detected instance across frames so the
        UI can offer "accept this whole track" as one object."""
        ...
