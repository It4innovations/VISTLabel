# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations


class ProjectNotFoundError(Exception):
    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"no such project: {project_id!r}")


class RevisionConflictError(Exception):
    def __init__(self, project_id: str, expected: int, actual: int) -> None:
        self.project_id = project_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"{project_id}: revision conflict (expected {expected}, current is {actual}); "
            "reload the project and retry"
        )
