# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Shared base for real backends (Sam21Backend, Sam3Backend,
YoloSamBackend) in their unavailable state: the adapter, capability
declaration and configuration schema exist even when the model code or
checkpoint does not.

Every operation raises :class:`BackendUnavailableError` with the same
reason `descriptor()` reports -- there is no partial functionality to
expose, and a backend must never silently substitute a different model.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from vistlabel.segmentation.base import (
    BackendUnavailableError,
    ConceptProposal,
    ExemplarPrompt,
    MaskPrediction,
    PointPrompt,
)


class UnavailableBackendMixin:
    backend_id: str

    def _unavailable_reason(self) -> str:
        raise NotImplementedError

    def _raise(self) -> None:
        raise BackendUnavailableError(self.backend_id, self._unavailable_reason())

    def open_video(self, project_id: str, video_path: Path) -> str:
        self._raise()
        raise AssertionError("unreachable")

    def close_video(self, session_id: str) -> None:
        self._raise()

    def segment_interactive(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        points: list[PointPrompt],
        box_xyxy: tuple[float, float, float, float] | None,
        previous_mask: np.ndarray | None,
    ) -> MaskPrediction:
        self._raise()
        raise AssertionError("unreachable")

    def propagate(
        self,
        session_id: str,
        object_ids: list[str],
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> Iterable[MaskPrediction]:
        self._raise()
        raise AssertionError("unreachable")

    def replace_object_mask(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        visible_mask: np.ndarray,
    ) -> MaskPrediction:
        self._raise()
        raise AssertionError("unreachable")

    def forget_object(self, session_id: str, object_id: str) -> None:
        self._raise()

    def auto_preannotate(
        self,
        session_id: str,
        frame_indices: list[int],
        requested_class_ids: list[int],
    ) -> Iterable[MaskPrediction]:
        self._raise()
        raise AssertionError("unreachable")

    def find_matching_objects(
        self,
        session_id: str,
        frame_index: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        self._raise()
        raise AssertionError("unreachable")

    def find_and_track_matching_objects(
        self,
        session_id: str,
        start_frame: int,
        end_frame: int,
        text: str | None,
        exemplar: ExemplarPrompt | None,
    ) -> Iterable[ConceptProposal]:
        self._raise()
        raise AssertionError("unreachable")
