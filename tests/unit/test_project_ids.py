# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from vistlabel.projects.ids import _encode_crockford, new_project_id, new_ulid


def test_new_project_id_has_expected_shape() -> None:
    project_id = new_project_id()

    assert project_id.startswith("prj_")
    assert len(project_id) == len("prj_") + 26


def test_ulids_are_unique() -> None:
    ulids = {new_ulid() for _ in range(1000)}

    assert len(ulids) == 1000


def test_ulids_use_only_crockford_base32_characters() -> None:
    allowed = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")

    ulid = new_ulid()

    assert set(ulid) <= allowed


def test_crockford_encoding_preserves_numeric_ordering() -> None:
    # The timestamp component's sortability (the actual property new_ulid()
    # relies on) depends on this holding for any two timestamps, not just
    # ones observed a millisecond apart in a live test run.
    earlier = _encode_crockford(1_700_000_000_000, 10)
    later = _encode_crockford(1_700_000_000_001, 10)

    assert earlier < later
