# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest

from vistlabel.domain.labels import LabelDefinition
from vistlabel.domain.project import ProjectStatus
from vistlabel.persistence.paths import project_dir, projects_root
from vistlabel.projects.errors import ProjectNotFoundError, RevisionConflictError
from vistlabel.projects.store import ProjectStore


def test_create_lists_gets_and_deletes_a_project(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)

    created = store.create("isolator_experiment_01")
    assert created.status == ProjectStatus.PENDING
    assert created.revision == 0

    listed = store.list_projects()
    assert [p.project_id for p in listed] == [created.project_id]

    fetched = store.get(created.project_id)
    assert fetched == created

    store.delete(created.project_id)
    assert store.list_projects() == []
    with pytest.raises(ProjectNotFoundError):
        store.get(created.project_id)


def test_get_of_unknown_project_raises() -> None:
    store = ProjectStore(Path("does-not-matter"))
    with pytest.raises(ProjectNotFoundError):
        store.get("prj_doesnotexist")


def test_project_creates_full_workspace_layout(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    created = store.create("layout test")

    pdir = project_dir(tmp_path, created.project_id)
    for expected in (
        "source",
        "cache/frames",
        "cache/thumbnails",
        "annotations/frames",
        "tracking/runs",
        "jobs",
        "locks",
        "logs",
        "exports",
    ):
        assert (pdir / expected).is_dir()
    assert (pdir / "project.json").is_file()
    assert (pdir / "labels.json").is_file()
    assert (pdir / "annotations" / "objects.json").is_file()


def test_new_projects_start_with_an_empty_ontology(tmp_path: Path) -> None:
    # The operator names objects freely in the annotation UI; there is no
    # built-in enumeration capping the vocabulary.
    store = ProjectStore(tmp_path)
    created = store.create("labels test")

    labels = store.get_labels(created.project_id)

    assert labels.labels == []


def test_set_labels_rejects_duplicate_ids(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    created = store.create("labels test")

    with pytest.raises(ValueError):
        store.set_labels(
            created.project_id,
            [
                LabelDefinition(label_id=0, name="a"),
                LabelDefinition(label_id=0, name="b"),
            ],
        )


def test_update_bumps_revision_and_persists_mutation(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    created = store.create("revision test")

    updated = store.update(
        created.project_id,
        expected_revision=0,
        mutate=lambda state: state.model_copy(
            update={"ui_state": state.ui_state.model_copy(update={"last_opened_frame": 42})}
        ),
    )

    assert updated.revision == 1
    assert updated.ui_state.last_opened_frame == 42
    assert store.get(created.project_id).ui_state.last_opened_frame == 42


def test_update_with_stale_revision_raises_conflict(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    created = store.create("conflict test")

    store.update(created.project_id, expected_revision=0, mutate=lambda state: state)

    with pytest.raises(RevisionConflictError):
        store.update(created.project_id, expected_revision=0, mutate=lambda state: state)


def test_project_survives_a_new_store_instance_simulating_restart(tmp_path: Path) -> None:
    first_process_store = ProjectStore(tmp_path)
    created = first_process_store.create("restart test")
    del first_process_store

    second_process_store = ProjectStore(tmp_path)
    reopened = second_process_store.get(created.project_id)

    assert reopened == created


def test_recover_incomplete_projects_removes_stale_tmp_dirs(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    leftover = projects_root(tmp_path) / ".tmp-crashed-during-create"
    leftover.mkdir(parents=True)
    (leftover / "project.json").write_text("{}", encoding="utf-8")

    removed = store.recover_incomplete_projects()

    assert removed == [".tmp-crashed-during-create"]
    assert not leftover.exists()
    # A real project created afterwards must not be affected.
    created = store.create("after recovery")
    assert store.list_projects() == [created]


def test_list_projects_skips_corrupted_project_json(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    good = store.create("good project")
    bad_dir = projects_root(tmp_path) / "prj_corrupted"
    bad_dir.mkdir(parents=True)
    (bad_dir / "project.json").write_text("{not valid json", encoding="utf-8")

    listed = store.list_projects()

    assert [p.project_id for p in listed] == [good.project_id]
