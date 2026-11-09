# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Dataset import: a COCO or YOLO dataset ZIP becomes a new project.

The project model is one video per project, so the images
are stitched into one: natural filename order, one image per frame, at a
constant operator-chosen frame rate. Images smaller than the largest are
padded at the right/bottom rather than scaled, so every annotation keeps
its pixel coordinates.

Every annotation becomes a canonical RLE mask: polygons and
RLE are rasterized exactly as pycocotools does; a box-only annotation
becomes a filled rectangle with ``source="import_bbox"``, so it can be told
apart and refined with SAM later. Identity across frames comes from the
dataset's track ids where it has them, otherwise from greedy same-class
IoU matching between consecutive frames.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from pydantic import BaseModel

from vistlabel.annotations.occlusion import compute_occlusion_state
from vistlabel.domain.frames import FrameAnnotation, FrameObjectAnnotation, RleMask
from vistlabel.domain.labels import LabelDefinition, LabelsFile
from vistlabel.domain.objects import ObjectRecord, ObjectsFile
from vistlabel.domain.project import FrameRangeAssignment, SplitLabel
from vistlabel.imports.archive import DatasetArchive, DatasetImportError, frame_order_key
from vistlabel.imports.coco_reader import read_coco
from vistlabel.imports.dataset import (
    DatasetAnnotation,
    DatasetFormat,
    DatasetImage,
    ImportWarnings,
    ParsedDataset,
)
from vistlabel.imports.yolo_reader import read_yolo
from vistlabel.persistence.atomic import save_model
from vistlabel.persistence.jsonl import append_jsonl
from vistlabel.persistence.paths import (
    events_log_path,
    frame_annotation_path,
    labels_json_path,
    objects_json_path,
)
from vistlabel.persistence.rle import count_connected_components, encode_mask
from vistlabel.video.ffmpeg import FfmpegBinaries
from vistlabel.video.frames import load_frame_timestamps
from vistlabel.video.upload import FinalizedVideo, finalize_video

#: Same-class masks on consecutive frames overlapping at least this much
#: are taken to be one object.
IOU_LINK_THRESHOLD = 0.5

#: H.264's own ceiling on either frame dimension.
MAX_CANVAS_DIMENSION = 16384

#: 4:4:4 keeps thin colored edges that 4:2:0 would smear; CRF 12 is
#: visually lossless. Short GOPs keep seeking to any one image cheap.
_ENCODER_ARGS = (
    "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv444p", "-g", "30",
)  # fmt: skip

RequestedFormat = Literal["auto", "coco", "yolo"]
_SPLIT_LABELS = {"train": SplitLabel.TRAIN, "val": SplitLabel.VAL}


class DatasetImportReport(BaseModel):
    dataset_format: DatasetFormat
    #: Images imported as frames; ``source_image_count`` is how many the
    #: dataset had before ``keep_every_nth`` thinned them out.
    image_count: int
    source_image_count: int
    annotation_count: int
    bbox_annotation_count: int
    object_count: int
    label_count: int
    width: int
    height: int
    warnings: list[str]


@dataclass
class DatasetImportResult:
    finalized: FinalizedVideo
    report: DatasetImportReport
    split_assignments: list[FrameRangeAssignment]


@dataclass
class _Instance:
    label_id: int
    track_id: int | str | None
    rle: RleMask
    component_count: int
    bbox_only: bool


# -- parsing --------------------------------------------------------------


def parse_dataset(archive: DatasetArchive, requested: RequestedFormat) -> ParsedDataset:
    if requested == "yolo":
        return read_yolo(archive)
    coco = read_coco(archive)
    if coco is not None:
        return coco
    if requested == "coco":
        raise DatasetImportError(
            "found no COCO annotation file (a .json with 'images' and 'annotations' lists)"
        )
    return read_yolo(archive)


def _unique_label_names(labels: dict[int, str]) -> list[LabelDefinition]:
    used: set[str] = set()
    definitions = []
    for label_id in sorted(labels):
        name = labels[label_id] or f"class_{label_id}"
        if name in used:
            name = f"{name} ({label_id})"
        used.add(name)
        definitions.append(LabelDefinition(label_id=label_id, name=name))
    return definitions


# -- masks ------------------------------------------------------------------


def _box_mask(box: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    # Rounded, not floored/ceiled: a VISTLabel box export (integer pixels) and
    # its YOLO normalization (six decimals) both land on the original pixels.
    x, y, box_width, box_height = box
    x0, x1 = max(0, round(x)), min(width, round(x + box_width))
    y0, y1 = max(0, round(y)), min(height, round(y + box_height))
    mask = np.zeros((height, width), dtype=bool)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def rasterize(annotation: DatasetAnnotation, width: int, height: int) -> np.ndarray | None:
    """The annotation as a ``height x width`` boolean mask, or ``None`` if its
    RLE was made for a different image size."""
    if annotation.rle is not None:
        size = annotation.rle["size"]
        if list(size) != [height, width]:
            return None
        counts = annotation.rle["counts"]
        if isinstance(counts, list):
            rle = mask_utils.frPyObjects(annotation.rle, height, width)  # type: ignore[call-overload]
        else:
            rle = {"size": [height, width], "counts": str(counts).encode("ascii")}
        return mask_utils.decode(rle).astype(bool)
    if annotation.polygons is not None:
        rles = mask_utils.frPyObjects(annotation.polygons, height, width)  # type: ignore[arg-type]
        return mask_utils.decode(mask_utils.merge(rles)).astype(bool)
    assert annotation.bbox is not None
    return _box_mask(annotation.bbox, width, height)


def _build_instances(
    archive: DatasetArchive,
    images: Sequence[DatasetImage],
    canvas: tuple[int, int],
    warnings: ImportWarnings,
) -> list[list[_Instance]]:
    canvas_width, canvas_height = canvas
    frames = []
    for image in images:
        width, height = archive.image_size(image.member)
        instances = []
        for annotation in image.annotations:
            mask = rasterize(annotation, width, height)
            if mask is None:
                warnings.add(
                    "RLE masks whose size differs from their image were skipped", image.member
                )
                continue
            if not mask.any():
                warnings.add("annotations that cover no pixel were skipped", image.member)
                continue
            if mask.shape != (canvas_height, canvas_width):
                padded = np.zeros((canvas_height, canvas_width), dtype=bool)
                padded[:height, :width] = mask
                mask = padded
            instances.append(
                _Instance(
                    label_id=annotation.label_id,
                    track_id=annotation.track_id,
                    rle=encode_mask(mask),
                    component_count=count_connected_components(mask),
                    bbox_only=annotation.is_bbox_only,
                )
            )
        frames.append(instances)
    return frames


# -- identity -----------------------------------------------------------------


def _coco_rle(rle: RleMask) -> dict:
    return {"size": list(rle.size), "counts": rle.counts.encode("ascii")}


def link_instances(
    frames: Sequence[Sequence[_Instance]], warnings: ImportWarnings
) -> list[list[int]]:
    """An object number for every instance. A dataset track id (per class)
    always wins; instances without one are matched greedily, best IoU first,
    to the previous frame's untracked instances of the same class."""
    next_object = 0
    tracked: dict[tuple[int, int | str], int] = {}
    previous: list[tuple[_Instance, int]] = []
    result = []

    for instances in frames:
        objects = [-1] * len(instances)
        for index, instance in enumerate(instances):
            if instance.track_id is None:
                continue
            key = (instance.label_id, instance.track_id)
            if key not in tracked:
                tracked[key] = next_object
                next_object += 1
            if tracked[key] in objects:
                warnings.add(
                    "a track id appearing twice on one image was split off as a new object",
                    str(instance.track_id),
                )
                objects[index] = next_object
                next_object += 1
            else:
                objects[index] = tracked[key]

        untracked = [index for index, instance in enumerate(instances) if instance.track_id is None]
        candidates: list[tuple[float, int, int]] = []
        for label_id in {instances[index].label_id for index in untracked}:
            current = [index for index in untracked if instances[index].label_id == label_id]
            before = [p for p, (inst, _) in enumerate(previous) if inst.label_id == label_id]
            if not before:
                continue
            ious = np.asarray(
                mask_utils.iou(
                    [_coco_rle(instances[index].rle) for index in current],  # type: ignore[arg-type]
                    [_coco_rle(previous[p][0].rle) for p in before],  # type: ignore[arg-type]
                    [0] * len(before),
                )
            )
            for row, index in enumerate(current):
                for column, p in enumerate(before):
                    if ious[row, column] >= IOU_LINK_THRESHOLD:
                        candidates.append((float(ious[row, column]), index, p))
        taken: set[int] = set()
        for _iou, index, p in sorted(candidates, reverse=True):
            if objects[index] == -1 and p not in taken:
                objects[index] = previous[p][1]
                taken.add(p)
        for index in untracked:
            if objects[index] == -1:
                objects[index] = next_object
                next_object += 1

        previous = [(instances[index], objects[index]) for index in untracked]
        result.append(objects)
    return result


# -- video ------------------------------------------------------------------


def _encode_video(
    archive: DatasetArchive,
    images: Sequence[DatasetImage],
    canvas: tuple[int, int],
    frame_rate: Fraction,
    binaries: FfmpegBinaries,
    output_path: Path,
) -> None:
    canvas_width, canvas_height = canvas
    command = [
        str(binaries.ffmpeg), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{canvas_width}x{canvas_height}",
        "-framerate", str(frame_rate), "-i", "-",
        *_ENCODER_ARGS, "-f", "mp4", str(output_path),
    ]  # fmt: skip
    # stderr goes to a file: a pipe nobody drains while frames are written
    # can fill up and deadlock ffmpeg.
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr
        )
        assert process.stdin is not None
        try:
            for image in images:
                frame = archive.open_image(image.member)
                if frame.size != canvas:
                    padded = Image.new("RGB", canvas)
                    padded.paste(frame, (0, 0))
                    frame = padded
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            process.wait(timeout=3600)
        except BrokenPipeError:
            process.wait(timeout=60)
        except BaseException:
            process.kill()
            process.wait()
            raise
        if process.returncode != 0:
            stderr.seek(0)
            message = stderr.read().decode("utf-8", "replace").strip()
            raise DatasetImportError(f"FFmpeg could not encode the images: {message}")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# -- splits -------------------------------------------------------------------


def split_assignments_for(images: Sequence[DatasetImage]) -> list[FrameRangeAssignment]:
    """The dataset's own train/val split as frame ranges, so a re-export
    keeps it. Only when it has both: a train-only dataset is better served by
    the export's own temporal split than by an empty val set. Other images
    (``test``, or none) stay unassigned -- which export leaves out."""
    splits = [image.split for image in images]
    if not {"train", "val"} <= set(splits):
        return []
    assignments: list[FrameRangeAssignment] = []
    for frame_index, split in enumerate(splits):
        label = _SPLIT_LABELS.get(split or "")
        if label is None:
            continue
        last = assignments[-1] if assignments else None
        if last is not None and last.label is label and last.end_frame == frame_index - 1:
            last.end_frame = frame_index
        else:
            assignments.append(
                FrameRangeAssignment(start_frame=frame_index, end_frame=frame_index, label=label)
            )
    return assignments


# -- orchestration ----------------------------------------------------------


def import_dataset(
    project_dir: Path,
    archive_path: Path,
    binaries: FfmpegBinaries,
    *,
    requested_format: RequestedFormat,
    frame_rate: float,
    keep_every_nth: int = 1,
) -> DatasetImportResult:
    """Parse, rasterize and link everything first -- a bad dataset fails
    before the slow encode -- then stitch the video, finalize it like an
    upload and write labels, objects and frames. ``keep_every_nth`` keeps
    images 0, n, 2n, ... in frame order, with their annotations; the rest
    are left out entirely. The caller deletes the project if this raises."""
    with DatasetArchive(archive_path) as archive:
        dataset = parse_dataset(archive, requested_format)
        warnings = dataset.warnings
        ordered = sorted(dataset.images, key=lambda image: frame_order_key(image.member))
        images = ordered[:: max(1, keep_every_nth)]

        sizes = {archive.image_size(image.member) for image in images}
        canvas = (max(w for w, _ in sizes), max(h for _, h in sizes))
        if max(canvas) > MAX_CANVAS_DIMENSION:
            raise DatasetImportError(
                f"images up to {canvas[0]}x{canvas[1]} exceed the {MAX_CANVAS_DIMENSION} px "
                "video limit"
            )
        if len(sizes) > 1:
            warnings.add(
                f"images have {len(sizes)} different sizes -- smaller ones are padded with black "
                f"at the right/bottom to {canvas[0]}x{canvas[1]}"
            )

        frames = _build_instances(archive, images, canvas, warnings)
        object_numbers = link_instances(frames, warnings)

        part_path = project_dir / "source" / "video.mp4.part"
        part_path.parent.mkdir(parents=True, exist_ok=True)
        _encode_video(
            archive,
            images,
            canvas,
            Fraction(frame_rate).limit_denominator(1001),
            binaries,
            part_path,
        )

    finalized = finalize_video(
        project_dir,
        part_path,
        binaries,
        size_bytes=part_path.stat().st_size,
        sha256=_sha256_file(part_path),
        normalize_vfr=False,
    )
    if finalized.frame_count != len(images):
        raise DatasetImportError(
            f"the stitched video has {finalized.frame_count} frames for {len(images)} images"
        )
    timestamps = load_frame_timestamps(project_dir / "cache" / "frame_timestamps.json")

    labels = _unique_label_names(dataset.labels)
    label_names = {label.label_id: label.name for label in labels}
    save_model(labels_json_path(project_dir), LabelsFile(labels=labels))

    object_ids: dict[int, str] = {}
    records: list[ObjectRecord] = []
    for instances, numbers in zip(frames, object_numbers, strict=True):
        for instance, number in zip(instances, numbers, strict=True):
            if number not in object_ids:
                object_ids[number] = f"obj_{uuid.uuid4().hex[:12]}"
                records.append(
                    ObjectRecord(
                        object_id=object_ids[number],
                        label_id=instance.label_id,
                        label_name=label_names[instance.label_id],
                    )
                )
    save_model(objects_json_path(project_dir), ObjectsFile(objects=records))

    annotation_count = bbox_count = 0
    for frame_index, (instances, numbers) in enumerate(zip(frames, object_numbers, strict=True)):
        if not instances:
            continue
        objects = [
            FrameObjectAnnotation(
                object_id=object_ids[number],
                label_id=instance.label_id,
                visible_mask_rle=instance.rle,
                connected_component_count=instance.component_count,
                occlusion_state=compute_occlusion_state(instance.component_count),
                source="import_bbox" if instance.bbox_only else "import",
            )
            for instance, number in zip(instances, numbers, strict=True)
        ]
        annotation_count += len(objects)
        bbox_count += sum(1 for instance in instances if instance.bbox_only)
        save_model(
            frame_annotation_path(project_dir, frame_index),
            FrameAnnotation(
                frame_index=frame_index,
                timestamp_seconds=timestamps[frame_index],
                objects=objects,
            ),
        )

    report = DatasetImportReport(
        dataset_format=dataset.format,
        image_count=len(images),
        source_image_count=len(ordered),
        annotation_count=annotation_count,
        bbox_annotation_count=bbox_count,
        object_count=len(records),
        label_count=len(labels),
        width=canvas[0],
        height=canvas[1],
        warnings=warnings.as_list(),
    )
    append_jsonl(
        events_log_path(project_dir),
        {
            "event": "dataset_imported",
            **report.model_dump(),
            "images": [image.member for image in images],
        },
    )
    return DatasetImportResult(
        finalized=finalized, report=report, split_assignments=split_assignments_for(images)
    )
