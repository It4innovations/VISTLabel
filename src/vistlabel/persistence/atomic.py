# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Atomic JSON writes and Pydantic (de)serialization helpers.

Write sequence: serialize to a temporary file in the *same* directory,
flush, optionally fsync, then atomically replace the destination
(``os.replace`` -- unlike ``os.rename`` this is well-defined cross-platform
even when the destination already exists). A crash or exception at any
point before the final replace leaves the previous file completely
untouched -- there is no partially written authoritative JSON file.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from vistlabel.persistence.migrations import migrate

T = TypeVar("T", bound=BaseModel)

#: Windows can transiently refuse to open a file for reading while another
#: thread's ``os.replace`` for the *same path* is in flight -- unlike
#: POSIX rename, it does not guarantee a reader never observes this
#: transitional state (frequent job/manifest polling during a chunked
#: propagation run makes this race realistic, not just theoretical). A
#: short bounded retry absorbs it; a real, persistent permission problem
#: still surfaces as an error after these attempts.
_READ_RETRY_ATTEMPTS = 5
_READ_RETRY_DELAY_SECONDS = 0.01

#: The same race also bites the other way around: ``os.replace`` itself can
#: raise ``PermissionError`` (WinError 5) when a reader has the destination
#: path open at that instant, since Windows rename semantics are stricter
#: than POSIX's. Retry the replace too, not just the read.
_REPLACE_RETRY_ATTEMPTS = 5
_REPLACE_RETRY_DELAY_SECONDS = 0.01


def _read_text_with_retry(path: Path) -> str:
    for attempt in range(_READ_RETRY_ATTEMPTS):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt == _READ_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_READ_RETRY_DELAY_SECONDS)
    raise AssertionError("unreachable")  # pragma: no cover


def _replace_with_retry(tmp_name: str, path: Path) -> None:
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(tmp_name, path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS)


def atomic_write_text(path: Path, text: str, *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        _replace_with_retry(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_json(path: Path, data: dict, *, fsync: bool = True) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False), fsync=fsync)


def save_model(path: Path, model: BaseModel, *, fsync: bool = True) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2), fsync=fsync)


def load_model(path: Path, model_cls: type[T]) -> T:
    return model_cls.model_validate_json(_read_text_with_retry(path))


def load_model_migrated(path: Path, model_cls: type[T], document_kind: str) -> T:
    """Like :func:`load_model`, but runs the document through the schema
    migration interface first (``persistence.migrations``)."""
    raw = json.loads(_read_text_with_retry(path))
    migrated = migrate(document_kind, raw)
    return model_cls.model_validate(migrated)
