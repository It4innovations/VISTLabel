# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Path traversal prevention -- unit-level coverage for the building
blocks (`is_safe_child_path`, id-format validators) that keep request
input from escaping the workspace."""

from __future__ import annotations

from pathlib import Path

from vistlabel.exports.service import is_valid_export_id
from vistlabel.persistence.safety import is_safe_child_path
from vistlabel.projects.ids import is_valid_project_id, new_project_id


def test_is_safe_child_path_true_for_a_real_child(tmp_path: Path) -> None:
    base = tmp_path / "projects"
    base.mkdir()
    child = base / "prj_example"
    child.mkdir()
    assert is_safe_child_path(base, child) is True


def test_is_safe_child_path_false_for_the_base_itself(tmp_path: Path) -> None:
    base = tmp_path / "projects"
    base.mkdir()
    assert is_safe_child_path(base, base) is False


def test_is_safe_child_path_false_for_a_traversal_to_the_parent(tmp_path: Path) -> None:
    base = tmp_path / "projects"
    base.mkdir()
    traversal_target = base / ".."  # what "project_id=.." resolves to
    assert is_safe_child_path(base, traversal_target) is False


def test_is_safe_child_path_false_for_a_sibling(tmp_path: Path) -> None:
    base = tmp_path / "projects"
    base.mkdir()
    sibling = tmp_path / "other"
    sibling.mkdir()
    assert is_safe_child_path(base, sibling) is False


def test_is_valid_project_id_accepts_real_generated_ids() -> None:
    for _ in range(20):
        assert is_valid_project_id(new_project_id()) is True


def test_is_valid_project_id_rejects_traversal_and_malformed_ids() -> None:
    for candidate in ("..", "prj_..", "prj_", "", "not_a_project_id", "prj_" + "x" * 26):
        assert is_valid_project_id(candidate) is False


def test_is_valid_export_id_accepts_the_real_generated_shape() -> None:
    assert is_valid_export_id("exp_0123456789ab") is True


def test_is_valid_export_id_rejects_traversal_and_malformed_ids() -> None:
    for candidate in ("..", "exp_..", "exp_", "", "not_an_export_id", "exp_" + "g" * 12):
        assert is_valid_export_id(candidate) is False
