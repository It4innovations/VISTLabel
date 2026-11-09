# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Format-neutral shape both readers (COCO, YOLO) parse a dataset into.
Geometry stays in absolute pixel coordinates of its own image; turning it
into canonical masks is the import service's job."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DatasetFormat = Literal["coco", "yolo"]


@dataclass
class DatasetAnnotation:
    label_id: int
    #: Exactly one geometry is set. ``rle`` is a COCO RLE dict (compressed
    #: string or uncompressed list ``counts``); ``polygons`` are flat
    #: ``[x1, y1, x2, y2, ...]`` lists; ``bbox`` is ``(x, y, width, height)``.
    polygons: list[list[float]] | None = None
    rle: dict | None = None
    bbox: tuple[float, float, float, float] | None = None
    #: COCO-video / CVAT style identity across images, when the dataset has one.
    track_id: int | str | None = None

    @property
    def is_bbox_only(self) -> bool:
        return self.polygons is None and self.rle is None


@dataclass
class DatasetImage:
    member: str
    split: str | None
    annotations: list[DatasetAnnotation] = field(default_factory=list)


class ImportWarnings:
    """Collapses repeats -- a dataset can have thousands of identical
    problems -- into one line each, with a count and the first example."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}
        self._examples: dict[str, str] = {}

    def add(self, message: str, example: str | None = None) -> None:
        self._counts[message] = self._counts.get(message, 0) + 1
        if example is not None:
            self._examples.setdefault(message, example)

    def as_list(self) -> list[str]:
        lines = []
        for message, count in self._counts.items():
            line = message if count == 1 else f"{message} ({count}x)"
            if message in self._examples:
                line += f", e.g. {self._examples[message]}"
            lines.append(line)
        return lines


@dataclass
class ParsedDataset:
    format: DatasetFormat
    #: ``label_id -> name``; ids are the dataset's own category ids / class
    #: indices where possible, so a re-export numbers classes the same way.
    labels: dict[int, str]
    images: list[DatasetImage]
    warnings: ImportWarnings


class LabelRegistry:
    """Merges category lists (several COCO files, one per split) by name,
    keeping each name's first-seen id unless another name already has it."""

    def __init__(self) -> None:
        self.names: dict[int, str] = {}
        self._ids: dict[str, int] = {}

    def add(self, name: str, preferred_id: int | None) -> int:
        if name in self._ids:
            return self._ids[name]
        if preferred_id is None or preferred_id < 0 or preferred_id in self.names:
            preferred_id = max(self.names, default=-1) + 1
        self.names[preferred_id] = name
        self._ids[name] = preferred_id
        return preferred_id
