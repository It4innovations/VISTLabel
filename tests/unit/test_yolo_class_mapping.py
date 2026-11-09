# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""The yolo_class_mapping.json schema and rejection of unknown or unmapped
classes -- validated at load time here."""

from __future__ import annotations

from pathlib import Path

import pytest

from vistlabel.segmentation.yolo_class_mapping import ClassMappingError, load_class_mapping


def test_loads_the_checked_in_example_file() -> None:
    mapping = load_class_mapping(Path("config/yolo_class_mapping.example.json"))
    assert mapping.schema_version == 1
    assert mapping.model_classes[3] == "pipette"
    assert len(mapping.model_classes) == 7


def test_missing_file_raises_class_mapping_error(tmp_path: Path) -> None:
    with pytest.raises(ClassMappingError):
        load_class_mapping(tmp_path / "missing.json")


def test_not_json_raises_class_mapping_error(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text("not json{{{", encoding="utf-8")
    with pytest.raises(ClassMappingError):
        load_class_mapping(path)


def test_wrong_schema_version_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text('{"schema_version": 2, "model_classes": {"0": "pipette"}}', encoding="utf-8")
    with pytest.raises(ClassMappingError, match="schema_version"):
        load_class_mapping(path)


def test_empty_model_classes_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text('{"schema_version": 1, "model_classes": {}}', encoding="utf-8")
    with pytest.raises(ClassMappingError, match="model_classes"):
        load_class_mapping(path)


def test_non_integer_class_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(
        '{"schema_version": 1, "model_classes": {"not-a-number": "pipette"}}', encoding="utf-8"
    )
    with pytest.raises(ClassMappingError, match="not-a-number"):
        load_class_mapping(path)


def test_duplicate_label_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "mapping.json"
    path.write_text(
        '{"schema_version": 1, "model_classes": {"0": "pipette", "1": "pipette"}}',
        encoding="utf-8",
    )
    with pytest.raises(ClassMappingError, match="unique"):
        load_class_mapping(path)
