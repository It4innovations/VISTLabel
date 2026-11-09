# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Logical-object management endpoints (creation, relabeling,
activation/deactivation, deletion of objects and restoring them for
Undo), and the whole-class versions the Class order list uses: rename,
merge, delete."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from vistlabel.annotations.errors import (
    ObjectExistsError,
    ObjectLockedError,
    ObjectNotFoundError,
    RevisionConflictError,
)
from vistlabel.annotations.objects_service import (
    create_object,
    delete_class_objects,
    delete_object,
    get_objects,
    move_class_objects,
    read_deleted_objects,
    rename_class_objects,
    restore_objects,
    update_object,
)
from vistlabel.api.deps import require_project_with_video
from vistlabel.domain.labels import LabelDefinition, LabelsFile
from vistlabel.domain.objects import DeletedObject, GeometryHint, ObjectRecord, ObjectsFile

router = APIRouter(prefix="/api/v1/projects", tags=["objects"])


class CreateObjectRequest(BaseModel):
    label_id: int
    label_name: str


class UpdateObjectRequest(BaseModel):
    expected_revision: int
    label_id: int | None = None
    label_name: str | None = None
    active: bool | None = None
    #: Locked objects can't be deleted until unlocked.
    locked: bool | None = None
    #: Optional elongated-axis hint, e.g. for a
    #: pipette -- association/review metadata only, never part of the mask.
    geometry_hint: GeometryHint | None = None
    clear_geometry_hint: bool = False


class RenameLabelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class MergeLabelRequest(BaseModel):
    into_label_id: int


class ClassChangeResult(BaseModel):
    labels: LabelsFile
    #: Objects renamed, moved to the other class, or deleted.
    object_count: int
    frame_count: int
    #: The objects deleted, for Undo (``POST .../objects/restore``).
    deleted_object_ids: list[str] = Field(default_factory=list)


class RestoreObjectsRequest(BaseModel):
    object_ids: list[str] = Field(min_length=1, max_length=10_000)


class RestoreObjectsResult(BaseModel):
    objects: list[ObjectRecord]
    labels: LabelsFile
    frame_count: int


@router.get("/{project_id}/objects")
def list_objects(project_id: str, request: Request) -> ObjectsFile:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    return get_objects(pdir)


@router.post("/{project_id}/objects", status_code=201)
def post_object(project_id: str, payload: CreateObjectRequest, request: Request) -> ObjectRecord:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    return create_object(pdir, payload.label_id, payload.label_name)


@router.patch("/{project_id}/objects/{object_id}")
def patch_object(
    project_id: str, object_id: str, payload: UpdateObjectRequest, request: Request
) -> ObjectRecord:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    previous_label_id = _label_id_of(pdir, object_id)
    try:
        updated = update_object(
            pdir,
            object_id,
            payload.expected_revision,
            label_id=payload.label_id,
            label_name=payload.label_name,
            active=payload.active,
            locked=payload.locked,
            geometry_hint=payload.geometry_hint,
            clear_geometry_hint=payload.clear_geometry_hint,
        )
    except RevisionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if previous_label_id is not None and updated.label_id != previous_label_id:
        _drop_label_if_unused(request, project_id, pdir, previous_label_id)
    return updated


@router.delete("/{project_id}/objects/{object_id}", status_code=204)
def delete_object_endpoint(
    project_id: str, object_id: str, expected_revision: int, request: Request
) -> Response:
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    label_id = _label_id_of(pdir, object_id)
    labels = request.app.state.project_store.get_labels(project_id).labels
    try:
        delete_object(pdir, object_id, expected_revision, labels=labels)
    except (RevisionConflictError, ObjectLockedError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if label_id is not None:
        _drop_label_if_unused(request, project_id, pdir, label_id)
    return Response(status_code=204)


@router.post("/{project_id}/objects/restore")
def restore_objects_endpoint(
    project_id: str, payload: RestoreObjectsRequest, request: Request
) -> RestoreObjectsResult:
    """Undo a delete: put these deleted objects back, with their masks on
    every frame and their class (re-added at its old place in the class
    order if it left with them)."""
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    try:
        deleted = read_deleted_objects(pdir, payload.object_ids)
        labels = _labels_for_restore(request, project_id, deleted)
        restored, frames = restore_objects(pdir, deleted, labels)
    except ObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"nothing to undo: {exc}") from exc
    except ObjectExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return RestoreObjectsResult(
        objects=restored,
        labels=request.app.state.project_store.get_labels(project_id),
        frame_count=frames,
    )


def _labels_for_restore(
    request: Request, project_id: str, deleted: list[DeletedObject]
) -> dict[str, LabelDefinition]:
    """Each object goes back to the class with its name, wherever that is in
    the order now. If that class left with its last object, it is re-added
    where it was, with its old ``label_id`` -- unless a class made since has
    taken that id, then with a new one."""
    store = request.app.state.project_store
    labels = list(store.get_labels(project_id).labels)
    chosen: dict[str, LabelDefinition] = {}
    added = False
    for item in deleted:
        name = item.record.label_name
        label = next((existing for existing in labels if existing.name == name), None)
        if label is None:
            label_id = item.record.label_id
            if any(existing.label_id == label_id for existing in labels):
                label_id = max(existing.label_id for existing in labels) + 1
            label = (item.label or LabelDefinition(label_id=label_id, name=name)).model_copy(
                update={"label_id": label_id, "name": name}
            )
            position = item.label_position
            labels.insert(len(labels) if position is None else min(position, len(labels)), label)
            added = True
        chosen[item.record.object_id] = label
    if added:
        store.set_labels(project_id, labels)
    return chosen


@router.patch("/{project_id}/labels/{label_id}")
def rename_label(
    project_id: str, label_id: int, payload: RenameLabelRequest, request: Request
) -> ClassChangeResult:
    """Rename a class. Renaming it to another class's name would leave two
    classes with one name, so that is refused -- merge into it instead."""
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    labels = _labels_with(request, project_id, label_id)
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="the class name is empty")
    if any(label.name == name and label.label_id != label_id for label in labels):
        raise HTTPException(
            status_code=409, detail=f'a class named "{name}" already exists; merge into it instead'
        )
    renamed = rename_class_objects(pdir, label_id, name)
    updated = request.app.state.project_store.set_labels(
        project_id,
        [
            label.model_copy(update={"name": name}) if label.label_id == label_id else label
            for label in labels
        ],
    )
    return ClassChangeResult(labels=updated, object_count=renamed, frame_count=0)


@router.post("/{project_id}/labels/{label_id}/merge")
def merge_label(
    project_id: str, label_id: int, payload: MergeLabelRequest, request: Request
) -> ClassChangeResult:
    """Move every object of this class, on every frame, into another class,
    then drop this one. The other class keeps its place in the order."""
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    labels = _labels_with(request, project_id, label_id)
    into = next((label for label in labels if label.label_id == payload.into_label_id), None)
    if into is None:
        raise HTTPException(status_code=404, detail=f"no class {payload.into_label_id}")
    if into.label_id == label_id:
        raise HTTPException(status_code=422, detail="a class can't be merged into itself")
    moved, frames = move_class_objects(pdir, label_id, into.label_id, into.name)
    updated = _drop_label(request, project_id, labels, label_id)
    return ClassChangeResult(labels=updated, object_count=moved, frame_count=frames)


@router.delete("/{project_id}/labels/{label_id}")
def delete_label(project_id: str, label_id: int, request: Request) -> ClassChangeResult:
    """Delete a class and every object of it, with their masks on every
    frame. Refused while any of them is locked."""
    _project, pdir, _timestamps = require_project_with_video(request, project_id)
    labels = _labels_with(request, project_id, label_id)
    try:
        deleted, frames = delete_class_objects(pdir, label_id, labels=labels)
    except ObjectLockedError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"{len(exc.object_ids)} of its objects are locked; unlock them to delete it",
        ) from exc
    updated = _drop_label(request, project_id, labels, label_id)
    return ClassChangeResult(
        labels=updated, object_count=len(deleted), frame_count=frames, deleted_object_ids=deleted
    )


def _labels_with(request: Request, project_id: str, label_id: int) -> list[LabelDefinition]:
    labels = request.app.state.project_store.get_labels(project_id).labels
    if not any(label.label_id == label_id for label in labels):
        raise HTTPException(status_code=404, detail=f"no class {label_id}")
    return labels


def _drop_label(
    request: Request, project_id: str, labels: list[LabelDefinition], label_id: int
) -> LabelsFile:
    return request.app.state.project_store.set_labels(
        project_id, [label for label in labels if label.label_id != label_id]
    )


def _label_id_of(pdir: Path, object_id: str) -> int | None:
    return next((o.label_id for o in get_objects(pdir).objects if o.object_id == object_id), None)


def _drop_label_if_unused(request: Request, project_id: str, pdir: Path, label_id: int) -> None:
    """A class is listed (and exported) only while some object has it: once
    its last object is deleted or relabeled, it leaves labels.json and the
    classes after it move up one number. Other unused labels, e.g. names an
    imported dataset declared but never used, are left alone."""
    if any(o.label_id == label_id for o in get_objects(pdir).objects):
        return
    store = request.app.state.project_store
    labels = store.get_labels(project_id).labels
    kept = [label for label in labels if label.label_id != label_id]
    if len(kept) != len(labels):
        store.set_labels(project_id, kept)
