# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Export orchestration: frame selection -> train/val split
-> COCO + YOLO writers -> manifest/validation report -> ZIP. Runs as a
background job (the :class:`~vistlabel.jobs.manager.JobManager` pattern)
since a full-project export can touch every frame.

Both writers can emit segmentation or detection boxes (``annotation_type``),
into either the Ultralytics or the RF-DETR/Roboflow COCO folder layout
(``dataset_layout``).
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from vistlabel.annotations.objects_service import get_objects
from vistlabel.domain.exports import (
    AnnotationType,
    DatasetLayout,
    ExportManifest,
    ExportStatus,
    FrameSelectionPolicy,
    ValidationReport,
)
from vistlabel.domain.frames import FrameAnnotation
from vistlabel.domain.labels import LabelDefinition, LabelsFile
from vistlabel.domain.project import FrameRangeAssignment
from vistlabel.exports.coco_writer import build_coco_dataset
from vistlabel.exports.frame_selection import (
    assign_splits,
    detect_adjacent_split_leakage,
    sample_negative_frames,
    select_positive_frames,
)
from vistlabel.exports.yolo_writer import (
    YoloMultipartPolicy,
    build_dataset_yaml,
    build_yolo_bbox_lines,
    build_yolo_label_lines,
)
from vistlabel.jobs.manager import JobContext
from vistlabel.persistence.atomic import load_model_migrated, save_model
from vistlabel.persistence.paths import (
    annotated_frame_paths,
    export_dir,
    export_manifest_path,
    export_zip_path,
    exports_dir,
    validation_report_path,
)
from vistlabel.persistence.rle import decode_mask
from vistlabel.persistence.safety import is_safe_child_path
from vistlabel.validation.engine import (
    find_duplicate_annotations,
    find_missing_files,
    is_tiny_mask,
    validate_bbox_coordinates,
    validate_polygon_coordinates,
)
from vistlabel.video.frames import decode_frame


@dataclass
class ExportRequest:
    export_id: str
    project_id: str
    frame_selection_policy: FrameSelectionPolicy
    every_nth: int
    selected_intervals: list[tuple[int, int]]
    negative_frame_fraction: float
    yolo_multipart_policy: YoloMultipartPolicy
    always_include_coco_exact_annotations: bool
    image_format: Literal["jpg", "png"]
    jpeg_quality: int
    annotation_type: AnnotationType
    dataset_layout: DatasetLayout
    #: From the project's own probed source video info -- known up front,
    #: not inferred from decoding a frame, so an export selecting zero
    #: frames (e.g. nothing reviewed yet) still succeeds with an empty but
    #: well-formed dataset instead of failing on "no frames decoded."
    image_width: int
    image_height: int


#: The COCO layout's folder name for each split -- RF-DETR looks for exactly
#: ``train``/``valid`` (and an optional ``test``), not ``val``.
_COCO_SPLIT_DIRS: dict[str, str] = {"train": "train", "val": "valid"}
_COCO_ANNOTATIONS_FILENAME = "_annotations.coco.json"


def _image_dir(export_root: Path, layout: DatasetLayout, split: str) -> Path:
    if layout == "coco":
        return export_root / _COCO_SPLIT_DIRS[split]
    return export_root / "images" / split


@dataclass
class _WrittenImage:
    frame_index: int
    filename: str
    split: Literal["train", "val"]


def _load_all_frame_annotations(project_dir: Path) -> dict[int, FrameAnnotation]:
    result: dict[int, FrameAnnotation] = {}
    for path in annotated_frame_paths(project_dir):
        frame = load_model_migrated(path, FrameAnnotation, "frame_annotation")
        result[frame.frame_index] = frame
    return result


def run_export(
    *,
    context: JobContext,
    project_dir: Path,
    video_path: Path,
    timestamps: list[float],
    request: ExportRequest,
    split_assignments: list[FrameRangeAssignment],
    labels: LabelsFile,
    source_video_sha256: str | None,
    report_progress: Callable[[float], None],
) -> None:
    export_root = export_dir(project_dir, request.export_id)
    now = datetime.now(UTC)
    manifest = ExportManifest(
        export_id=request.export_id,
        project_id=request.project_id,
        source_video_sha256=source_video_sha256,
        created_at=now,
        status=ExportStatus.RUNNING,
        job_id=context.job_id,
        frame_selection_policy=request.frame_selection_policy,
        yolo_multipart_policy=request.yolo_multipart_policy,
        always_include_coco_exact_annotations=request.always_include_coco_exact_annotations,
        negative_frame_fraction=request.negative_frame_fraction,
        image_format=request.image_format,
        annotation_type=request.annotation_type,
        dataset_layout=request.dataset_layout,
    )
    save_model(export_manifest_path(project_dir, request.export_id), manifest)

    report = ValidationReport(
        export_id=request.export_id,
        generated_at=now,
        source_project_id=request.project_id,
        source_video_sha256=source_video_sha256,
    )

    try:
        manifest = _run_export_unsafe(
            context=context,
            project_dir=project_dir,
            video_path=video_path,
            timestamps=timestamps,
            request=request,
            split_assignments=split_assignments,
            labels=labels,
            export_root=export_root,
            manifest=manifest,
            report=report,
            report_progress=report_progress,
        )
    except Exception as exc:
        manifest = manifest.model_copy(update={"status": ExportStatus.FAILED, "error": str(exc)})
        save_model(export_manifest_path(project_dir, request.export_id), manifest)
        raise

    manifest = manifest.model_copy(update={"status": ExportStatus.SUCCEEDED})
    save_model(export_manifest_path(project_dir, request.export_id), manifest)


def _run_export_unsafe(
    *,
    context: JobContext,
    project_dir: Path,
    video_path: Path,
    timestamps: list[float],
    request: ExportRequest,
    split_assignments: list[FrameRangeAssignment],
    labels: LabelsFile,
    export_root: Path,
    manifest: ExportManifest,
    report: ValidationReport,
    report_progress: Callable[[float], None],
) -> ExportManifest:
    frame_annotations = _load_all_frame_annotations(project_dir)
    # "Deleted" frames: out of the positives here and out of the negative
    # pool below, whatever the selection policy or split assignments say.
    deleted_frames = {idx for idx, frame in frame_annotations.items() if frame.excluded}
    frame_annotations = {
        idx: frame for idx, frame in frame_annotations.items() if idx not in deleted_frames
    }
    # Deleting an object strips its masks from every frame, but a mask can
    # still outlive it (e.g. a propagation run still writing for an object
    # deleted mid-run) -- never export a box for an object that's gone.
    known_objects = {record.object_id for record in get_objects(project_dir).objects}
    orphaned_masks = 0
    for idx, annotation in list(frame_annotations.items()):
        kept = [obj for obj in annotation.objects if obj.object_id in known_objects]
        if len(kept) != len(annotation.objects):
            orphaned_masks += len(annotation.objects) - len(kept)
            frame_annotations[idx] = annotation.model_copy(update={"objects": kept})
    labels, frame_annotations = _number_classes_by_position(labels, frame_annotations)
    label_name_by_id = {label.label_id: label.name for label in labels.labels}

    positive_frames = select_positive_frames(
        frame_annotations=frame_annotations,
        policy=request.frame_selection_policy,
        every_nth=request.every_nth,
        selected_intervals=request.selected_intervals,
    )
    split_by_frame, excluded_from_positive = assign_splits(
        frame_indices=positive_frames, split_assignments=split_assignments
    )
    # Frames select_positive_frames chose but assign_splits excluded (no
    # covering assignment, or explicitly marked excluded) never get
    # written -- only frames with a real split entry are "positive" from
    # here on.
    positive_frame_set = set(positive_frames) - set(excluded_from_positive)
    negative_frames = sample_negative_frames(
        total_frame_count=len(timestamps),
        positive_frame_indices=set(positive_frames),
        excluded_frame_indices=set(excluded_from_positive) | deleted_frames,
        negative_frame_fraction=request.negative_frame_fraction,
    )
    negative_split_by_frame, _ = assign_splits(
        frame_indices=negative_frames, split_assignments=split_assignments
    )
    split_by_frame.update(negative_split_by_frame)

    report.frame_leakage_risks = detect_adjacent_split_leakage(split_by_frame)
    if orphaned_masks:
        report.warnings.append(
            f"{orphaned_masks} mask(s) of deleted objects were left out of the export"
        )
    if deleted_frames:
        report.warnings.append(
            f"{len(deleted_frames)} frame(s) deleted in the editor were left out of the export"
        )
    if excluded_from_positive:
        report.warnings.append(
            f"{len(excluded_from_positive)} selected frame(s) were excluded from the export "
            "(explicitly marked excluded, or not covered by any train/val/excluded assignment)"
        )

    layout = request.dataset_layout
    write_yolo_labels = layout == "ultralytics"
    labels_dir = export_root / "labels"
    for split in ("train", "val"):
        _image_dir(export_root, layout, split).mkdir(parents=True, exist_ok=True)
        if write_yolo_labels:
            (labels_dir / split).mkdir(parents=True, exist_ok=True)

    image_width, image_height = request.image_width, request.image_height
    all_frames = sorted(positive_frame_set | set(negative_frames))
    total = max(len(all_frames), 1)
    written: list[_WrittenImage] = []
    duplicate_check_entries: list[tuple[int, str, str]] = []

    for completed, frame_index in enumerate(all_frames, start=1):
        context.raise_if_cancelled()
        split = split_by_frame[frame_index]
        image = decode_frame(video_path, frame_index, timestamps)
        stem = f"{request.project_id}_{frame_index:06d}"
        filename = f"{stem}.{request.image_format}"
        image_path = _image_dir(export_root, layout, split) / filename
        save_kwargs = {"quality": request.jpeg_quality} if request.image_format == "jpg" else {}
        image.convert("RGB").save(
            image_path, format="JPEG" if request.image_format == "jpg" else "PNG", **save_kwargs
        )
        written.append(_WrittenImage(frame_index=frame_index, filename=filename, split=split))

        frame = frame_annotations.get(frame_index)
        label_lines: list[str] = []
        if frame is not None and frame_index in positive_frame_set:
            for obj in frame.objects:
                duplicate_check_entries.append(
                    (frame_index, obj.object_id, obj.visible_mask_rle.counts)
                )
                mask = decode_mask(obj.visible_mask_rle)
                if is_tiny_mask(mask):
                    report.tiny_masks.append(
                        f"frame {frame_index} object {obj.object_id}: area {int(mask.sum())}px"
                    )
                label_name = label_name_by_id.get(obj.label_id, f"label_{obj.label_id}")
                report.object_count_by_class[label_name] = (
                    report.object_count_by_class.get(label_name, 0) + 1
                )
                if obj.backend_provenance is not None:
                    backend_id = obj.backend_provenance.backend_id
                    report.model_provenance_summary[backend_id] = (
                        report.model_provenance_summary.get(backend_id, 0) + 1
                    )

            if write_yolo_labels:
                label_lines = _yolo_lines_for_frame(
                    frame=frame,
                    frame_index=frame_index,
                    request=request,
                    label_name_by_id=label_name_by_id,
                    report=report,
                )
        else:
            report.negative_frame_count += 1

        if write_yolo_labels:
            (labels_dir / split / f"{stem}.txt").write_text(
                "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
            )
        report.image_count = completed
        report_progress(completed / total)

    report.warnings.extend(
        f"duplicate annotation: {message}"
        for message in find_duplicate_annotations(duplicate_check_entries)
    )

    train_frames = sorted(frame for frame, split in split_by_frame.items() if split == "train")
    val_frames = sorted(frame for frame, split in split_by_frame.items() if split == "val")
    report.train_frame_count = len(train_frames)
    report.val_frame_count = len(val_frames)

    image_filename_by_frame = {item.frame_index: item.filename for item in written}

    def write_coco(split_frames: list[int], destination: Path) -> None:
        coco = build_coco_dataset(
            frame_indices=split_frames,
            frame_annotations=frame_annotations,
            labels=labels,
            image_filename_by_frame=image_filename_by_frame,
            image_width=image_width,
            image_height=image_height,
            annotation_type=request.annotation_type,
        )
        destination.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    expected_files = [
        _image_dir(export_root, layout, item.split) / item.filename for item in written
    ]
    if layout == "coco":
        # The COCO file is the dataset itself here, so it is written for
        # both splits regardless of always_include_coco_exact_annotations.
        for split, frames in (("train", train_frames), ("val", val_frames)):
            destination = _image_dir(export_root, layout, split) / _COCO_ANNOTATIONS_FILENAME
            write_coco(frames, destination)
            expected_files.append(destination)
        if not val_frames:
            report.warnings.append(
                "the valid/ split is empty -- RF-DETR evaluates on it during training, so "
                "assign a val interval (or export more frames) before training"
            )
    else:
        if request.always_include_coco_exact_annotations:
            annotations_dir = export_root / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            for split, frames in (("train", train_frames), ("val", val_frames)):
                write_coco(frames, annotations_dir / f"instances_{split}.json")

        (export_root / "dataset.yaml").write_text(
            build_dataset_yaml(label_name_by_id=label_name_by_id, has_val_split=bool(val_frames)),
            encoding="utf-8",
        )
        expected_files += [
            labels_dir / item.split / f"{item.filename.rsplit('.', 1)[0]}.txt" for item in written
        ]
    report.missing_or_malformed_labels.extend(find_missing_files(expected_files))

    manifest = manifest.model_copy(
        update={
            "frame_count": len(all_frames),
            "train_frame_count": len(train_frames),
            "val_frame_count": len(val_frames),
            "negative_frame_count": report.negative_frame_count,
        }
    )

    save_model(validation_report_path(project_dir, request.export_id), report)

    zip_path = export_zip_path(project_dir, request.export_id)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(export_root.rglob("*")):
            if path.is_file() and path != zip_path:
                archive.write(path, arcname=str(path.relative_to(export_root)))

    return manifest


def _number_classes_by_position(
    labels: LabelsFile, frame_annotations: dict[int, FrameAnnotation]
) -> tuple[LabelsFile, dict[int, FrameAnnotation]]:
    """Swap every ``label_id`` for the exported class number: the label's
    position in ``labels.json``, which is the order the operator arranges
    in the Objects tab. The writers then see dense ids 0..N-1 in that order.
    A label_id no longer in ``labels.json`` gets a number after all of
    them rather than colliding with one."""
    numbered = [
        label.model_copy(update={"label_id": index}) for index, label in enumerate(labels.labels)
    ]
    class_by_label_id = {label.label_id: index for index, label in enumerate(labels.labels)}
    for annotation in frame_annotations.values():
        for obj in annotation.objects:
            if obj.label_id not in class_by_label_id:
                class_by_label_id[obj.label_id] = len(numbered)
                numbered.append(
                    LabelDefinition(label_id=len(numbered), name=f"label_{obj.label_id}")
                )
    renumbered = {
        idx: annotation.model_copy(
            update={
                "objects": [
                    obj.model_copy(update={"label_id": class_by_label_id[obj.label_id]})
                    for obj in annotation.objects
                ]
            }
        )
        for idx, annotation in frame_annotations.items()
    }
    return LabelsFile(labels=numbered), renumbered


def _yolo_lines_for_frame(
    *,
    frame: FrameAnnotation,
    frame_index: int,
    request: ExportRequest,
    label_name_by_id: dict[int, str],
    report: ValidationReport,
) -> list[str]:
    validate: Callable[[list[float]], str | None]
    if request.annotation_type == "bbox":
        lines, invalid = build_yolo_bbox_lines(
            frame=frame, width=request.image_width, height=request.image_height
        )
        report.invalid_polygons.extend(invalid)
        validate = validate_bbox_coordinates
    else:
        lines, conversions, dropped, invalid = build_yolo_label_lines(
            frame=frame,
            width=request.image_width,
            height=request.image_height,
            multipart_policy=request.yolo_multipart_policy,
            label_name_by_id=label_name_by_id,
        )
        report.multipart_instances.extend(conversions)
        if request.yolo_multipart_policy == "minimal_bridge":
            report.yolo_bridge_conversions.extend(conversions)
        elif request.yolo_multipart_policy == "split_components":
            report.split_component_conversions.extend(conversions)
        report.dropped_instances.extend(dropped)
        report.invalid_polygons.extend(invalid)
        validate = validate_polygon_coordinates
    for line in lines:
        error = validate([float(value) for value in line.split(" ")[1:]])
        if error is not None:
            report.invalid_polygons.append(f"frame {frame_index}: {error}")
    return lines


#: Matches exactly what start_export() generates in api/exports.py --
#: "exp_" + 12 lowercase hex chars. Validate the shape before
#: an export_id from a request path parameter is ever used to build a
#: filesystem path.
EXPORT_ID_PATTERN = re.compile(r"^exp_[0-9a-f]{12}$")


def is_valid_export_id(export_id: str) -> bool:
    return bool(EXPORT_ID_PATTERN.match(export_id))


def delete_export(project_dir: Path, export_id: str) -> None:
    """Idempotent: deleting an export that doesn't exist (or was never
    real to begin with -- a malformed/traversal-attempt id) is a silent
    no-op, matching ordinary DELETE-on-missing-resource semantics, not an
    error that would distinguish "wrong id" from "attempted traversal" for
    a caller probing the endpoint."""
    if not is_valid_export_id(export_id):
        return
    target = export_dir(project_dir, export_id)
    if not is_safe_child_path(exports_dir(project_dir), target):
        return
    if not export_manifest_path(project_dir, export_id).is_file():
        return
    shutil.rmtree(target)
