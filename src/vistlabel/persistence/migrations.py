# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Schema versioning and migration interface.

Every canonical document kind has a `schema_version` field and a current
target version registered here. No migrations are registered yet --
everything is at version 1 -- but the interface exists now so a
future schema change has a place to plug in a migration function instead of
becoming an ad hoc `if "old_field" in data` check scattered across the
codebase.
"""

from __future__ import annotations

from collections.abc import Callable

MigrationFn = Callable[[dict], dict]

CURRENT_SCHEMA_VERSIONS: dict[str, int] = {
    "project": 1,
    "labels": 1,
    "objects": 1,
    "frame_annotation": 1,
    "job": 1,
    "export_manifest": 1,
    "validation_report": 1,
}

_migrations: dict[str, dict[int, MigrationFn]] = {}


class UnsupportedSchemaVersionError(Exception):
    pass


def register_migration(document_kind: str, from_version: int, fn: MigrationFn) -> None:
    """Register a function upgrading ``document_kind`` from ``from_version``
    to ``from_version + 1``. ``fn`` must return data with
    ``schema_version == from_version + 1`` set."""
    _migrations.setdefault(document_kind, {})[from_version] = fn


def migrate(document_kind: str, data: dict) -> dict:
    if document_kind not in CURRENT_SCHEMA_VERSIONS:
        raise KeyError(f"unknown document kind: {document_kind!r}")
    current_version = CURRENT_SCHEMA_VERSIONS[document_kind]
    version = data.get("schema_version", 1)
    if version > current_version:
        raise UnsupportedSchemaVersionError(
            f"{document_kind}: file schema_version {version} is newer than "
            f"the supported version {current_version}; refusing to guess"
        )
    migrations_for_kind = _migrations.get(document_kind, {})
    while version < current_version:
        migration_fn = migrations_for_kind.get(version)
        if migration_fn is None:
            raise UnsupportedSchemaVersionError(
                f"{document_kind}: no migration registered from schema_version {version} "
                f"to reach {current_version}"
            )
        data = migration_fn(data)
        new_version = data.get("schema_version", version + 1)
        if new_version <= version:
            raise UnsupportedSchemaVersionError(
                f"{document_kind}: migration from {version} did not advance schema_version"
            )
        version = new_version
    return data
