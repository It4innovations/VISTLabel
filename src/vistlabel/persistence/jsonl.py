# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Append-only JSONL event logs: annotation events, backend inference
provenance, audit events, warnings, export events. Each line is
independently parseable and appended, never rewritten in place.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path


def append_jsonl(path: Path, record: dict, *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=False, default=str)
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)
