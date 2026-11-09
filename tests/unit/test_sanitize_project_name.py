# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from vistlabel.projects.sanitize import MAX_NAME_LENGTH, sanitize_project_name


def test_trims_whitespace() -> None:
    assert sanitize_project_name("  isolator run 1  ") == "isolator run 1"


def test_strips_control_characters() -> None:
    assert sanitize_project_name("isolator\x00run\x07") == "isolatorrun"


def test_replaces_path_separators() -> None:
    assert sanitize_project_name("../../etc/passwd") == ".._.._etc_passwd"


def test_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        sanitize_project_name("   ")


def test_truncates_overly_long_names() -> None:
    long_name = "x" * (MAX_NAME_LENGTH + 50)

    result = sanitize_project_name(long_name)

    assert len(result) == MAX_NAME_LENGTH
