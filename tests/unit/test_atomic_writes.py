# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from vistlabel.persistence.atomic import atomic_write_json, load_model, save_model


class _Sample(BaseModel):
    value: int
    label: str


def test_atomic_write_json_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "doc.json"

    atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})

    assert target.exists()
    assert not any(p.name.startswith(".doc.json.") for p in tmp_path.iterdir())


def test_atomic_write_leaves_no_temp_file_behind(tmp_path: Path) -> None:
    target = tmp_path / "doc.json"

    atomic_write_json(target, {"a": 1})

    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_failed_write_never_touches_the_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "doc.json"
    atomic_write_json(target, {"a": 1})
    original_content = target.read_text(encoding="utf-8")

    import os as os_module

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(os_module, "replace", _boom)

    with pytest.raises(OSError):
        atomic_write_json(target, {"a": 2, "b": "this must never land"})

    # The original authoritative file is completely untouched.
    assert target.read_text(encoding="utf-8") == original_content
    # And no stray temp file survives the failed attempt.
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_save_and_load_model_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "sample.json"
    model = _Sample(value=42, label="hello")

    save_model(target, model)
    loaded = load_model(target, _Sample)

    assert loaded == model
