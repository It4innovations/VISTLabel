# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

from vistlabel.persistence.jsonl import append_jsonl, iter_jsonl


def test_append_and_iter_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    append_jsonl(path, {"event": "created", "n": 1})
    append_jsonl(path, {"event": "updated", "n": 2})

    records = list(iter_jsonl(path))

    assert records == [{"event": "created", "n": 1}, {"event": "updated", "n": 2}]


def test_iter_jsonl_on_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert list(iter_jsonl(tmp_path / "does-not-exist.jsonl")) == []


def test_each_line_is_independently_parseable(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"b": 2})

    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
