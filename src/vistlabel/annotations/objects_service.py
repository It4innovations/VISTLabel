# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Logical-object management: create, relabel, activate/deactivate, delete
and restore (Undo of a delete), and the same for every object of one
class at once. ``annotations/objects.json`` is small enough that the
whole file is read/written on every change, revision-checked the same way
as ``project.json``.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from vistlabel.annotations.errors import (
    ObjectExistsError,
    ObjectLockedError,
    ObjectNotFoundError,
    RevisionConflictError,
)
from vistlabel.annotations.frames_service import apply_frame_annotation
from vistlabel.domain.frames import FrameAnnotation, FrameObjectAnnotation
from vistlabel.domain.labels import LabelDefinition
from vistlabel.domain.objects import (
    AnnotationMode,
    DeletedObject,
    DeletedObjectFrame,
    GeometryHint,
    ObjectRecord,
    ObjectsFile,
)
from vistlabel.persistence.atomic import load_model, load_model_migrated, save_model
from vistlabel.persistence.paths import (
    annotated_frame_paths,
    deleted_object_path,
    deleted_objects_dir,
    objects_json_path,
)

#: Kept deleted objects older than this are pruned on the next delete. Undo
#: only reaches deletes made since the page was opened, so this is plenty.
DELETED_OBJECT_RETENTION_SECONDS = 7 * 24 * 3600

#: Object ids are generated here (``obj_<hex>``); anything else can't name a
#: kept deleted object, and must not reach a file path.
_OBJECT_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def get_objects(project_dir: Path) -> ObjectsFile:
    return load_model_migrated(objects_json_path(project_dir), ObjectsFile, "objects")


def create_object(project_dir: Path, label_id: int, label_name: str) -> ObjectRecord:
    objects_file = get_objects(project_dir)
    record = ObjectRecord(
        object_id=f"obj_{uuid.uuid4().hex[:12]}",
        label_id=label_id,
        label_name=label_name,
        annotation_mode=AnnotationMode.MODAL_VISIBLE,
    )
    updated = objects_file.model_copy(
        update={
            "objects": [*objects_file.objects, record],
            "revision": objects_file.revision + 1,
        }
    )
    save_model(objects_json_path(project_dir), updated)
    return record


def update_object(
    project_dir: Path,
    object_id: str,
    expected_revision: int,
    *,
    label_id: int | None = None,
    label_name: str | None = None,
    active: bool | None = None,
    locked: bool | None = None,
    geometry_hint: GeometryHint | None = None,
    clear_geometry_hint: bool = False,
) -> ObjectRecord:
    """Frames store the object's ``label_id`` on each of its masks, and
    exports read it from there, so a new class is written to every frame
    too -- otherwise the object would keep exporting as its old class.
    Frames go first, as in :func:`delete_object`: if this stops partway,
    sending the same change again finishes the job."""
    objects_file = get_objects(project_dir)
    if objects_file.revision != expected_revision:
        raise RevisionConflictError("objects.json", expected_revision, objects_file.revision)
    if not any(record.object_id == object_id for record in objects_file.objects):
        raise ObjectNotFoundError(object_id)

    if label_id is not None:
        _relabel_frames(project_dir, {object_id}, label_id)

    updated_record: ObjectRecord | None = None
    new_objects: list[ObjectRecord] = []
    for record in objects_file.objects:
        if record.object_id != object_id:
            new_objects.append(record)
            continue
        changes: dict[str, object] = {}
        if label_id is not None:
            changes["label_id"] = label_id
        if label_name is not None:
            changes["label_name"] = label_name
        if active is not None:
            changes["active"] = active
        if locked is not None:
            changes["locked"] = locked
        if clear_geometry_hint:
            changes["geometry_hint"] = None
        elif geometry_hint is not None:
            changes["geometry_hint"] = geometry_hint
        updated_record = record.model_copy(update=changes)
        new_objects.append(updated_record)

    if updated_record is None:
        raise ObjectNotFoundError(object_id)

    updated_file = objects_file.model_copy(
        update={"objects": new_objects, "revision": objects_file.revision + 1}
    )
    save_model(objects_json_path(project_dir), updated_file)
    return updated_record


def _relabel_frames(project_dir: Path, object_ids: set[str], label_id: int) -> int:
    """Set ``label_id`` on these objects' masks (active and inactive) in
    every frame where they have a different one. Returns how many frames
    changed."""

    def stale(entries: list[FrameObjectAnnotation]) -> bool:
        return any(o.object_id in object_ids and o.label_id != label_id for o in entries)

    def relabel(entries: list[FrameObjectAnnotation]) -> list[FrameObjectAnnotation]:
        return [
            o.model_copy(update={"label_id": label_id}) if o.object_id in object_ids else o
            for o in entries
        ]

    frames_changed = 0
    for path in annotated_frame_paths(project_dir):
        frame = load_model_migrated(path, FrameAnnotation, "frame_annotation")
        if not (stale(frame.objects) or stale(frame.inactive_objects)):
            continue
        apply_frame_annotation(
            project_dir,
            frame.frame_index,
            frame.timestamp_seconds,
            lambda current: current.model_copy(
                update={
                    "objects": relabel(current.objects),
                    "inactive_objects": relabel(current.inactive_objects),
                }
            ),
        )
        frames_changed += 1
    return frames_changed


def _references(frame: FrameAnnotation, object_ids: set[str]) -> bool:
    return (
        any(o.object_id in object_ids for o in frame.objects)
        or any(o.object_id in object_ids for o in frame.inactive_objects)
        or any(
            e.occluder_object_id in object_ids or e.occluded_object_id in object_ids
            for e in frame.occlusion_edges
        )
    )


def _strip_objects(frame: FrameAnnotation, object_ids: set[str]) -> FrameAnnotation:
    return frame.model_copy(
        update={
            "objects": [o for o in frame.objects if o.object_id not in object_ids],
            "inactive_objects": [
                o for o in frame.inactive_objects if o.object_id not in object_ids
            ],
            "occlusion_edges": [
                e
                for e in frame.occlusion_edges
                if e.occluder_object_id not in object_ids and e.occluded_object_id not in object_ids
            ],
        }
    )


def _strip_frames(project_dir: Path, object_ids: set[str], frames: dict[int, float]) -> int:
    """Remove these objects' masks, inactive masks and occlusion edges from
    these frames (``frame_index -> timestamp``, as :func:`_keep_deleted`
    found them). Returns how many frames changed."""
    for frame_index, timestamp_seconds in sorted(frames.items()):
        apply_frame_annotation(
            project_dir,
            frame_index,
            timestamp_seconds,
            lambda current: _strip_objects(current, object_ids),
        )
    return len(frames)


def _part_of(frame: FrameAnnotation, object_id: str) -> DeletedObjectFrame | None:
    objects = [o for o in frame.objects if o.object_id == object_id]
    inactive = [o for o in frame.inactive_objects if o.object_id == object_id]
    edges = [
        e
        for e in frame.occlusion_edges
        if object_id in (e.occluder_object_id, e.occluded_object_id)
    ]
    if not (objects or inactive or edges):
        return None
    return DeletedObjectFrame(
        frame_index=frame.frame_index,
        timestamp_seconds=frame.timestamp_seconds,
        objects=objects,
        inactive_objects=inactive,
        occlusion_edges=edges,
    )


def _keep_deleted(
    project_dir: Path, records: list[ObjectRecord], labels: Sequence[LabelDefinition]
) -> dict[int, float]:
    """Save what deleting these objects will remove, before any of it is
    removed, so Undo can restore it (see :func:`restore_objects`). If an
    earlier delete of one stopped partway, the frames it already stripped
    are only in its saved copy, so those are carried over. Returns the
    frames that still have something to strip."""
    object_ids = {record.object_id for record in records}
    parts: dict[str, list[DeletedObjectFrame]] = {object_id: [] for object_id in object_ids}
    frames: dict[int, float] = {}
    for path in annotated_frame_paths(project_dir):
        frame = load_model_migrated(path, FrameAnnotation, "frame_annotation")
        if not _references(frame, object_ids):
            continue
        frames[frame.frame_index] = frame.timestamp_seconds
        for object_id in object_ids:
            part = _part_of(frame, object_id)
            if part is not None:
                parts[object_id].append(part)

    _prune_deleted(project_dir)
    for record in records:
        path = deleted_object_path(project_dir, record.object_id)
        kept = parts[record.object_id]
        if path.is_file():
            current = {part.frame_index for part in kept}
            earlier = load_model(path, DeletedObject).frames
            kept = [*kept, *(part for part in earlier if part.frame_index not in current)]
        position = next(
            (i for i, label in enumerate(labels) if label.label_id == record.label_id), None
        )
        save_model(
            path,
            DeletedObject(
                record=record,
                label=None if position is None else labels[position],
                label_position=position,
                frames=sorted(kept, key=lambda part: part.frame_index),
            ),
        )
    return frames


def _prune_deleted(project_dir: Path) -> None:
    folder = deleted_objects_dir(project_dir)
    if not folder.is_dir():
        return
    cutoff = time.time() - DELETED_OBJECT_RETENTION_SECONDS
    for path in folder.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass  # pruning is housekeeping; the next delete tries again


def delete_object(
    project_dir: Path,
    object_id: str,
    expected_revision: int,
    *,
    labels: Sequence[LabelDefinition] = (),
) -> int:
    """Delete the object from the project *and* from every frame (its masks,
    inactive masks and occlusion edges) -- otherwise its masks stay in the
    frame files and keep being exported. All of that is kept first, with
    its class from ``labels``, so Undo can restore it. Frames go next: if
    this stops partway, the object still exists and deleting it again
    finishes the job, rather than leaving masks behind for an object that's
    gone. Returns how many frames were changed. A locked object is
    refused."""
    objects_file = get_objects(project_dir)
    if objects_file.revision != expected_revision:
        raise RevisionConflictError("objects.json", expected_revision, objects_file.revision)

    record = next((o for o in objects_file.objects if o.object_id == object_id), None)
    if record is None:
        raise ObjectNotFoundError(object_id)
    if record.locked:
        raise ObjectLockedError([object_id])
    remaining = [o for o in objects_file.objects if o.object_id != object_id]

    frames = _keep_deleted(project_dir, [record], labels)
    frames_changed = _strip_frames(project_dir, {object_id}, frames)

    updated_file = objects_file.model_copy(
        update={"objects": remaining, "revision": objects_file.revision + 1}
    )
    save_model(objects_json_path(project_dir), updated_file)
    return frames_changed


# -- whole classes ------------------------------------------------------------
# One pass over the frames for all of a class's objects, not one per object.
# Frames go first here too, so repeating an interrupted call finishes it.


def _save_objects(
    project_dir: Path, objects_file: ObjectsFile, objects: list[ObjectRecord]
) -> None:
    save_model(
        objects_json_path(project_dir),
        objects_file.model_copy(update={"objects": objects, "revision": objects_file.revision + 1}),
    )


def delete_class_objects(
    project_dir: Path, label_id: int, *, labels: Sequence[LabelDefinition] = ()
) -> tuple[list[str], int]:
    """Delete every object of this class, with its masks on every frame,
    kept for Undo as in :func:`delete_object`. Returns ``(ids of the objects
    deleted, frames changed)``. If any of them is locked, nothing is
    deleted."""
    objects_file = get_objects(project_dir)
    records = [o for o in objects_file.objects if o.label_id == label_id]
    doomed = {o.object_id for o in records}
    if not doomed:
        return [], 0
    locked = [o.object_id for o in records if o.locked]
    if locked:
        raise ObjectLockedError(locked)
    frames = _keep_deleted(project_dir, records, labels)
    frames_changed = _strip_frames(project_dir, doomed, frames)
    _save_objects(
        project_dir,
        objects_file,
        [o for o in objects_file.objects if o.object_id not in doomed],
    )
    return [o.object_id for o in records], frames_changed


# -- undo a delete ------------------------------------------------------------


def read_deleted_objects(project_dir: Path, object_ids: Sequence[str]) -> list[DeletedObject]:
    """The kept copies of these deleted objects. Raises
    :class:`ObjectNotFoundError` if one has none (never deleted, already
    restored, or pruned) and :class:`ObjectExistsError` if one is in the
    project again."""
    existing = {o.object_id for o in get_objects(project_dir).objects}
    found: list[DeletedObject] = []
    for object_id in dict.fromkeys(object_ids):
        if object_id in existing:
            raise ObjectExistsError(object_id)
        if not _OBJECT_ID.fullmatch(object_id):
            raise ObjectNotFoundError(object_id)
        path = deleted_object_path(project_dir, object_id)
        if not path.is_file():
            raise ObjectNotFoundError(object_id)
        found.append(load_model(path, DeletedObject))
    return found


def restore_objects(
    project_dir: Path, deleted: list[DeletedObject], labels: dict[str, LabelDefinition]
) -> tuple[list[ObjectRecord], int]:
    """Put deleted objects back as they were: their masks, inactive masks
    and occlusion edges on every frame (an edge only if the object at its
    other end exists), then the objects themselves, under the class
    ``labels`` gives each (the API picks it -- the class may be gone or
    renumbered by now). Frames go first, as in :func:`delete_object`: if
    this stops partway, restoring again finishes the job. Returns
    ``(restored objects, frames changed)``."""
    objects_file = get_objects(project_dir)
    existing = {o.object_id for o in objects_file.objects}
    for item in deleted:
        if item.record.object_id in existing:
            raise ObjectExistsError(item.record.object_id)
    known = existing | {item.record.object_id for item in deleted}
    label_ids = {object_id: label.label_id for object_id, label in labels.items()}

    def relabeled(entries: list[FrameObjectAnnotation]) -> list[FrameObjectAnnotation]:
        return [e.model_copy(update={"label_id": label_ids[e.object_id]}) for e in entries]

    by_frame: dict[int, list[DeletedObjectFrame]] = defaultdict(list)
    for item in deleted:
        for part in item.frames:
            by_frame[part.frame_index].append(part)

    def put_back(parts: list[DeletedObjectFrame]) -> Callable[[FrameAnnotation], FrameAnnotation]:
        return lambda current: _put_back(current, parts)

    def _put_back(current: FrameAnnotation, parts: list[DeletedObjectFrame]) -> FrameAnnotation:
        # Already there: put back by an earlier restore that stopped partway.
        present = {o.object_id for o in (*current.objects, *current.inactive_objects)}
        objects, inactive = list(current.objects), list(current.inactive_objects)
        edges = list(current.occlusion_edges)
        for part in parts:
            objects += [o for o in relabeled(part.objects) if o.object_id not in present]
            inactive += [o for o in relabeled(part.inactive_objects) if o.object_id not in present]
            for edge in part.occlusion_edges:
                ends = {edge.occluder_object_id, edge.occluded_object_id}
                if ends <= known and edge not in edges:
                    edges.append(edge)
        return current.model_copy(
            update={"objects": objects, "inactive_objects": inactive, "occlusion_edges": edges}
        )

    for frame_index, parts in sorted(by_frame.items()):
        apply_frame_annotation(
            project_dir,
            frame_index,
            parts[0].timestamp_seconds,
            put_back(parts),
        )

    restored = [
        item.record.model_copy(
            update={
                "label_id": labels[item.record.object_id].label_id,
                "label_name": labels[item.record.object_id].name,
            }
        )
        for item in deleted
    ]
    _save_objects(project_dir, objects_file, [*objects_file.objects, *restored])
    for item in deleted:
        deleted_object_path(project_dir, item.record.object_id).unlink(missing_ok=True)
    return restored, len(by_frame)


def move_class_objects(
    project_dir: Path, label_id: int, into_label_id: int, into_label_name: str
) -> tuple[int, int]:
    """Give every object of class ``label_id`` the class ``into_label_id``,
    on every frame too (exports read the class from there). Returns
    ``(objects moved, frames changed)``."""
    objects_file = get_objects(project_dir)
    moved = {o.object_id for o in objects_file.objects if o.label_id == label_id}
    if not moved:
        return 0, 0
    frames_changed = _relabel_frames(project_dir, moved, into_label_id)
    _save_objects(
        project_dir,
        objects_file,
        [
            o.model_copy(update={"label_id": into_label_id, "label_name": into_label_name})
            if o.object_id in moved
            else o
            for o in objects_file.objects
        ],
    )
    return len(moved), frames_changed


def rename_class_objects(project_dir: Path, label_id: int, name: str) -> int:
    """Objects keep a copy of their class name; frames store only the
    ``label_id``, so they need no change. Returns how many objects changed."""
    objects_file = get_objects(project_dir)
    renamed = [o for o in objects_file.objects if o.label_id == label_id and o.label_name != name]
    if renamed:
        _save_objects(
            project_dir,
            objects_file,
            [
                o.model_copy(update={"label_name": name}) if o.label_id == label_id else o
                for o in objects_file.objects
            ],
        )
    return len(renamed)
