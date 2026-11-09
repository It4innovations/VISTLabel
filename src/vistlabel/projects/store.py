# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Project service: creation, listing, opening, deletion, locking and
optimistic-revision updates.

Video upload/finalization (``vistlabel.video.upload``) and frame-level
annotation persistence (``vistlabel.annotations``) build on top of this and
live in their own modules.
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from vistlabel.annotations.frames_service import count_frame_progress
from vistlabel.domain.labels import LabelDefinition, LabelsFile, default_labels_file
from vistlabel.domain.objects import ObjectsFile
from vistlabel.domain.project import ProjectState, ProjectStatus
from vistlabel.persistence.atomic import load_model_migrated, save_model
from vistlabel.persistence.locks import DEFAULT_STALE_AFTER, ProjectLock
from vistlabel.persistence.paths import (
    PROJECT_SUBDIRS,
    editor_lock_path,
    labels_json_path,
    objects_json_path,
    project_dir,
    project_json_path,
    projects_root,
)
from vistlabel.persistence.safety import is_safe_child_path
from vistlabel.projects.errors import ProjectNotFoundError, RevisionConflictError
from vistlabel.projects.ids import is_valid_project_id, new_project_id
from vistlabel.projects.sanitize import sanitize_project_name

logger = logging.getLogger(__name__)


def _with_frame_progress(directory: Path, state: ProjectState) -> ProjectState:
    """Fill in the annotated/reviewed counts from the frame files. They are
    derived on read, not maintained in project.json: every frame write path
    (autosave, propagation, auto pre-annotation, review) would otherwise
    have to update them, and the stored values were never written at all."""
    try:
        annotated, reviewed = count_frame_progress(directory)
    except (ValidationError, ValueError, OSError) as exc:
        logger.warning("could not count frame progress in %s: %s", directory.name, exc)
        return state
    progress = state.progress.model_copy(
        update={"annotated_frame_count": annotated, "reviewed_frame_count": reviewed}
    )
    return state.model_copy(update={"progress": progress})

_TMP_PREFIX = ".tmp-"


class ProjectStore:
    def __init__(self, workspace_root: Path, lock_stale_after: timedelta = DEFAULT_STALE_AFTER):
        self._workspace_root = workspace_root
        self._lock_stale_after = lock_stale_after

    # -- creation -----------------------------------------------------

    def create(self, name: str) -> ProjectState:
        display_name = sanitize_project_name(name)
        project_id = new_project_id()
        root = projects_root(self._workspace_root)
        tmp_dir = root / f"{_TMP_PREFIX}{uuid.uuid4().hex}"
        final_dir = project_dir(self._workspace_root, project_id)

        for subdir in PROJECT_SUBDIRS:
            (tmp_dir / subdir).mkdir(parents=True, exist_ok=True)

        now = datetime.now(UTC)
        state = ProjectState(
            project_id=project_id,
            name=display_name,
            status=ProjectStatus.PENDING,
            revision=0,
            created_at=now,
            modified_at=now,
        )
        save_model(project_json_path(tmp_dir), state)
        save_model(labels_json_path(tmp_dir), default_labels_file())
        save_model(objects_json_path(tmp_dir), ObjectsFile(objects=[]))

        # Single atomic promotion: before this line only a `.tmp-*` directory
        # exists; a crash before it leaves no half-initialized project_id
        # directory behind.
        os.replace(tmp_dir, final_dir)
        return state

    def recover_incomplete_projects(self) -> list[str]:
        """Remove leftover `.tmp-*` directories from a crash during
        :meth:`create`. Safe to call at every application startup."""
        root = projects_root(self._workspace_root)
        removed: list[str] = []
        if not root.exists():
            return removed
        for entry in root.iterdir():
            if entry.is_dir() and entry.name.startswith(_TMP_PREFIX):
                shutil.rmtree(entry, ignore_errors=True)
                removed.append(entry.name)
                logger.warning("removed incomplete project directory %s", entry.name)
        return removed

    def recover_interrupted_tracking_runs(self) -> dict[str, list[str]]:
        """Lets a restart identify incomplete work and offer safe resume or
        rollback: marks any tracking-run manifest still
        ``running`` (impossible right after a fresh process start) as
        ``interrupted`` across every project. Never auto-resumes; resuming
        is a distinct, explicit API call the operator/user makes."""
        from vistlabel.tracking.manifests import recover_interrupted_runs

        recovered: dict[str, list[str]] = {}
        for state in self.list_projects():
            run_ids = recover_interrupted_runs(project_dir(self._workspace_root, state.project_id))
            if run_ids:
                recovered[state.project_id] = run_ids
        return recovered

    # -- read -----------------------------------------------------------

    def list_projects(self) -> list[ProjectState]:
        root = projects_root(self._workspace_root)
        if not root.exists():
            return []
        results: list[ProjectState] = []
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not entry.is_dir() or entry.name.startswith("."):
                continue
            json_path = project_json_path(entry)
            if not json_path.exists():
                logger.warning("skipping project directory without project.json: %s", entry.name)
                continue
            try:
                state = load_model_migrated(json_path, ProjectState, "project")
            except (ValidationError, ValueError, OSError) as exc:
                logger.warning("skipping unreadable project.json in %s: %s", entry.name, exc)
                continue
            results.append(_with_frame_progress(entry, state))
        return results

    def get(self, project_id: str) -> ProjectState:
        json_path = project_json_path(project_dir(self._workspace_root, project_id))
        if not json_path.exists():
            raise ProjectNotFoundError(project_id)
        return load_model_migrated(json_path, ProjectState, "project")

    # -- mutation ---------------------------------------------------------

    def update(
        self,
        project_id: str,
        expected_revision: int,
        mutate: Callable[[ProjectState], ProjectState],
    ) -> ProjectState:
        json_path = project_json_path(project_dir(self._workspace_root, project_id))
        if not json_path.exists():
            raise ProjectNotFoundError(project_id)
        current = load_model_migrated(json_path, ProjectState, "project")
        if current.revision != expected_revision:
            raise RevisionConflictError(project_id, expected_revision, current.revision)
        updated = mutate(current)
        updated = updated.model_copy(
            update={
                "revision": current.revision + 1,
                "modified_at": datetime.now(UTC),
            }
        )
        save_model(json_path, updated)
        return updated

    def delete(self, project_id: str) -> None:
        # Validate the id's shape and confirm the resolved
        # target is a real, contained project directory *before* ever
        # calling rmtree -- Path.exists() alone is not enough, since a
        # crafted id like ".." resolves to projects_root's own parent,
        # which "exists" just as much as a real project does.
        if not is_valid_project_id(project_id):
            raise ProjectNotFoundError(project_id)
        target = project_dir(self._workspace_root, project_id)
        if not is_safe_child_path(projects_root(self._workspace_root), target):
            raise ProjectNotFoundError(project_id)
        if not project_json_path(target).is_file():
            raise ProjectNotFoundError(project_id)
        shutil.rmtree(target)

    # -- labels -----------------------------------------------------------

    def get_labels(self, project_id: str) -> LabelsFile:
        pdir = project_dir(self._workspace_root, project_id)
        if not project_json_path(pdir).exists():
            raise ProjectNotFoundError(project_id)
        return load_model_migrated(labels_json_path(pdir), LabelsFile, "labels")

    def set_labels(self, project_id: str, labels: list[LabelDefinition]) -> LabelsFile:
        pdir = project_dir(self._workspace_root, project_id)
        if not project_json_path(pdir).exists():
            raise ProjectNotFoundError(project_id)
        labels_file = LabelsFile(labels=labels)  # re-runs uniqueness validation
        save_model(labels_json_path(pdir), labels_file)
        return labels_file

    # -- editing lock -------------------------------------------------

    def lock_for(self, project_id: str) -> ProjectLock:
        pdir = project_dir(self._workspace_root, project_id)
        if not project_json_path(pdir).exists():
            raise ProjectNotFoundError(project_id)
        return ProjectLock(editor_lock_path(pdir), stale_after=self._lock_stale_after)


__all__ = ["ProjectStore"]
