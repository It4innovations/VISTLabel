# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations


class RevisionConflictError(Exception):
    """Optimistic-concurrency conflict on a frame annotation or the
    project's objects file."""

    def __init__(self, resource: str, expected: int, actual: int) -> None:
        self.resource = resource
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{resource}: revision conflict (expected {expected}, current is {actual}); "
            "reload and retry"
        )


class ObjectNotFoundError(Exception):
    def __init__(self, object_id: str) -> None:
        self.object_id = object_id
        super().__init__(f"no such object: {object_id!r}")


class ObjectLockedError(Exception):
    def __init__(self, object_ids: list[str]) -> None:
        self.object_ids = object_ids
        names = ", ".join(repr(object_id) for object_id in object_ids)
        super().__init__(f"locked, unlock to delete: {names}")


class ObjectExistsError(Exception):
    def __init__(self, object_id: str) -> None:
        self.object_id = object_id
        super().__init__(f"object {object_id!r} is in the project already")
