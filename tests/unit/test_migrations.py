# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from vistlabel.persistence import migrations


def test_document_at_current_version_passes_through_unchanged() -> None:
    data = {"schema_version": 1, "project_id": "prj_x"}

    result = migrations.migrate("project", data)

    assert result == data


def test_unknown_document_kind_raises() -> None:
    with pytest.raises(KeyError):
        migrations.migrate("not_a_real_kind", {"schema_version": 1})


def test_future_schema_version_is_rejected() -> None:
    with pytest.raises(migrations.UnsupportedSchemaVersionError):
        migrations.migrate("project", {"schema_version": 999})


def test_registered_migration_upgrades_old_documents() -> None:
    migrations.CURRENT_SCHEMA_VERSIONS["_test_doc"] = 2

    def _v1_to_v2(data: dict) -> dict:
        data = dict(data)
        data["schema_version"] = 2
        data["new_field"] = "default"
        return data

    migrations.register_migration("_test_doc", 1, _v1_to_v2)
    try:
        result = migrations.migrate("_test_doc", {"schema_version": 1, "old": True})
    finally:
        del migrations.CURRENT_SCHEMA_VERSIONS["_test_doc"]
        del migrations._migrations["_test_doc"]

    assert result == {"schema_version": 2, "old": True, "new_field": "default"}


def test_missing_migration_path_raises() -> None:
    migrations.CURRENT_SCHEMA_VERSIONS["_test_doc_2"] = 2
    try:
        with pytest.raises(migrations.UnsupportedSchemaVersionError):
            migrations.migrate("_test_doc_2", {"schema_version": 1})
    finally:
        del migrations.CURRENT_SCHEMA_VERSIONS["_test_doc_2"]
