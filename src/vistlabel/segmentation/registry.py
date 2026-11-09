# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Backend discovery and availability status.

``fake`` is always available. ``sam21``/``sam3``/``yolo_sam`` are always
*registered*, with their adapters and capability declarations, even when
they cannot run: a backend whose model, dependencies or configuration are
missing reports itself unavailable rather than being omitted or silently
swapped for a different model.
"""

from __future__ import annotations

from vistlabel.segmentation.base import BackendDescriptor, VideoSegmentationBackend
from vistlabel.segmentation.fake import FakeBackend
from vistlabel.segmentation.sam3 import Sam3Backend
from vistlabel.segmentation.sam21 import Sam21Backend
from vistlabel.segmentation.yolo_sam import YoloSamBackend
from vistlabel.settings import AppSettings


class UnknownBackendError(KeyError):
    def __init__(self, backend_id: str, known_backend_ids: list[str]) -> None:
        self.backend_id = backend_id
        self.known_backend_ids = known_backend_ids
        super().__init__(
            f"unknown backend_id {backend_id!r}; known backends: {known_backend_ids}"
        )


class BackendRegistry:
    def __init__(self, backends: dict[str, VideoSegmentationBackend]) -> None:
        self._backends = dict(backends)

    def get(self, backend_id: str) -> VideoSegmentationBackend:
        try:
            return self._backends[backend_id]
        except KeyError:
            raise UnknownBackendError(backend_id, sorted(self._backends)) from None

    def list_descriptors(self) -> list[BackendDescriptor]:
        return [backend.descriptor() for backend in self._backends.values()]

    def __contains__(self, backend_id: str) -> bool:
        return backend_id in self._backends


def build_registry(settings: AppSettings) -> BackendRegistry:
    """Build the registry available to the running application.

    ``fake`` is always present regardless of configuration -- it has no
    external dependency and backs the annotation editor and the backend
    contract test suite. ``yolo_sam`` is an orchestrator: it wraps
    whichever of ``sam21``/``sam3`` is configured as its refiner rather
    than owning model code of its own.
    """
    sam21 = Sam21Backend(settings.models.sam21, settings.runtime)
    sam3 = Sam3Backend(settings.models.sam3, settings.runtime)
    refiners: dict[str, VideoSegmentationBackend] = {"sam21": sam21, "sam3": sam3}
    refiner = refiners[settings.models.yolo_sam.refiner_backend]
    # Insertion order is the order the UI offers backends in, so the real
    # models lead and ``fake`` -- a development/testing stub, never a
    # production choice -- sits last.
    backends: dict[str, VideoSegmentationBackend] = {
        "sam21": sam21,
        "sam3": sam3,
        "yolo_sam": YoloSamBackend(settings.models.yolo_sam, refiner),
        "fake": FakeBackend(),
    }
    return BackendRegistry(backends)
