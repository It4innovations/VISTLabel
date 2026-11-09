# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Read-only access to an uploaded dataset ZIP.

Members are read straight out of the archive -- nothing is ever extracted
-- so a crafted member name (``../../x``, an absolute path) never reaches
the filesystem: names are only dictionary keys here.
"""

from __future__ import annotations

import io
import posixpath
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

#: An annotation file (COCO JSON, YAML, YOLO TXT) larger than this is not
#: read -- a guard against a ZIP bomb exhausting memory, far above any real
#: annotation file.
MAX_TEXT_MEMBER_BYTES = 1024**3

#: EXIF orientations that swap width and height.
_TRANSPOSING_ORIENTATIONS = frozenset({5, 6, 7, 8})
_EXIF_ORIENTATION_TAG = 0x0112

#: Everything PIL or the ZIP can raise for a broken or oversized image.
_IMAGE_ERRORS = (UnidentifiedImageError, Image.DecompressionBombError, OSError, zipfile.BadZipFile)

_SPLIT_PATTERN = re.compile(r"(?<![a-z])(validation|valid|val|train|test)(?![a-z])")
_SPLIT_NAMES = {"validation": "val", "valid": "val", "val": "val", "train": "train", "test": "test"}


class DatasetImportError(Exception):
    """The upload is not an importable dataset; the message says why and is
    shown to the operator as-is."""


def split_from_path(parts: list[str]) -> str | None:
    """``train``/``val``/``test`` named by the deepest path part that names
    one (``valid/``, ``images/train/``, ``instances_val2017``), else ``None``."""
    for part in reversed(parts):
        matches = _SPLIT_PATTERN.findall(part.lower())
        if matches:
            return _SPLIT_NAMES[matches[-1]]
    return None


def _natural_key(text: str) -> list[object]:
    return [int(chunk) if chunk.isdigit() else chunk for chunk in re.split(r"(\d+)", text.lower())]


def frame_order_key(member: str) -> tuple[list[object], list[object]]:
    """Dataset images become frames in natural filename order, across all
    split folders -- so frames cut from one video (``x_000009.jpg`` in
    ``train/``, ``x_000010.jpg`` in ``valid/``) come back in video order."""
    return _natural_key(posixpath.basename(member)), _natural_key(member)


def _is_hidden(name: str) -> bool:
    # __MACOSX/ and ._resource-forks would otherwise pass as broken images.
    return any(part.startswith(".") or part == "__MACOSX" for part in name.split("/"))


class DatasetArchive:
    def __init__(self, path: Path) -> None:
        try:
            self._zip = zipfile.ZipFile(path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise DatasetImportError("the uploaded file is not a valid ZIP archive") from exc
        self._infos: dict[str, zipfile.ZipInfo] = {}
        for info in self._zip.infolist():
            name = posixpath.normpath(info.filename.replace("\\", "/")).lstrip("/")
            if info.is_dir() or _is_hidden(name) or name.startswith(".."):
                continue
            self._infos[name] = info
        self.names: list[str] = sorted(self._infos)
        self.image_names: list[str] = [
            name for name in self.names if posixpath.splitext(name)[1].lower() in IMAGE_SUFFIXES
        ]
        self._images_by_basename: dict[str, list[str]] = defaultdict(list)
        for name in self.image_names:
            self._images_by_basename[posixpath.basename(name)].append(name)
        self._sizes: dict[str, tuple[int, int]] = {}

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> DatasetArchive:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def has(self, name: str) -> bool:
        return name in self._infos

    def read_text(self, name: str) -> str:
        info = self._infos[name]
        if info.file_size > MAX_TEXT_MEMBER_BYTES:
            raise DatasetImportError(f"{name} is too large to be an annotation file")
        try:
            return self._zip.read(info).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DatasetImportError(f"{name} is not UTF-8 text") from exc
        except (zipfile.BadZipFile, OSError) as exc:
            raise DatasetImportError(f"could not read {name} from the archive: {exc}") from exc

    def image_size(self, name: str) -> tuple[int, int]:
        """``(width, height)`` as displayed, i.e. after EXIF orientation --
        what annotation tools show and label. Reads only the header."""
        if name not in self._sizes:
            try:
                with self._zip.open(self._infos[name]) as handle, Image.open(handle) as image:
                    width, height = image.size
                    if image.getexif().get(_EXIF_ORIENTATION_TAG) in _TRANSPOSING_ORIENTATIONS:
                        width, height = height, width
            except _IMAGE_ERRORS as exc:
                raise DatasetImportError(f"{name} is not a readable image: {exc}") from exc
            self._sizes[name] = (width, height)
        return self._sizes[name]

    def open_image(self, name: str) -> Image.Image:
        """The decoded image, EXIF-oriented, as RGB."""
        try:
            with Image.open(io.BytesIO(self._zip.read(self._infos[name]))) as image:
                return ImageOps.exif_transpose(image).convert("RGB")
        except _IMAGE_ERRORS as exc:
            raise DatasetImportError(f"{name} is not a readable image: {exc}") from exc

    def resolve_image(self, file_name: str, base_dir: str) -> str | None:
        """The archive member a COCO ``file_name`` refers to: relative to the
        annotation file's folder, relative to the archive root, or -- since
        layouts differ (``images/train/`` next to ``annotations/``, Roboflow's
        flat ``train/``) -- the one image whose path ends the same way."""
        file_name = file_name.replace("\\", "/")
        for candidate in (posixpath.join(base_dir, file_name), file_name):
            candidate = posixpath.normpath(candidate)
            if candidate in self._infos:
                return candidate
        matches = self._images_by_basename.get(posixpath.basename(file_name), [])
        suffix = "/" + posixpath.normpath(file_name).lstrip("./")
        for narrowed in (
            matches,
            [match for match in matches if ("/" + match).endswith(suffix)],
            [match for match in matches if base_dir and match.startswith(base_dir + "/")],
        ):
            if len(narrowed) == 1:
                return narrowed[0]
        return None
