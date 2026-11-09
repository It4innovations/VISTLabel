# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Backend session cache: one open video session per (project, backend),
reused across interactive segmentation calls instead of opening a new
session per request.

Explicit lifecycle: sessions for a project are closed via
:meth:`close_project_sessions` when that project is deleted, releasing
whatever the backend was holding (for the fake backend, nothing; for a real
GPU backend, model/session memory) instead of leaking until process exit.
Bounded LRU eviction additionally closes the least-recently-used
session once more than ``max_sessions`` are open at once, across every
project/backend combination -- a long-running server touching many
projects must not accumulate GPU session state without bound.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

from vistlabel.segmentation.base import VideoSegmentationBackend
from vistlabel.segmentation.registry import BackendRegistry, UnknownBackendError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SESSIONS = 4


class BackendSessionManager:
    def __init__(self, max_sessions: int = _DEFAULT_MAX_SESSIONS) -> None:
        self._max_sessions = max_sessions
        # Insertion/access order == recency order; the first item is
        # always the least-recently-used one (OrderedDict.move_to_end on
        # every access keeps this true).
        self._sessions: OrderedDict[tuple[str, str], tuple[str, VideoSegmentationBackend]] = (
            OrderedDict()
        )

    def get_or_open(
        self, backend: VideoSegmentationBackend, project_id: str, video_path: Path
    ) -> str:
        key = (project_id, backend.descriptor().backend_id)
        entry = self._sessions.get(key)
        if entry is not None:
            self._sessions.move_to_end(key)
            return entry[0]

        session_id = backend.open_video(project_id, video_path)
        self._sessions[key] = (session_id, backend)
        self._sessions.move_to_end(key)
        self._evict_over_capacity(keep=key)
        return session_id

    def _evict_over_capacity(self, *, keep: tuple[str, str]) -> None:
        while len(self._sessions) > self._max_sessions:
            lru_key = next(iter(self._sessions))
            if lru_key == keep:
                # Only possible if max_sessions < 1; nothing sensible to
                # evict instead, so stop rather than closing what was just
                # opened.
                break
            session_id, backend = self._sessions.pop(lru_key)
            try:
                backend.close_video(session_id)
            except Exception:  # noqa: BLE001 - eviction must never crash the caller
                logger.warning("failed to close evicted session %s", lru_key, exc_info=True)

    def close_project_sessions(self, project_id: str, registry: BackendRegistry) -> None:
        for (proj, backend_id), (session_id, _backend) in list(self._sessions.items()):
            if proj != project_id:
                continue
            try:
                registry.get(backend_id).close_video(session_id)
            except UnknownBackendError:
                pass
            del self._sessions[(proj, backend_id)]
