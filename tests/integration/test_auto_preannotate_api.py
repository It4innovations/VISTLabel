# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""The ``POST .../auto-preannotate`` endpoint, exercised through the real
API/job pipeline: label-name resolution against *this* project's own
``labels.json``, object creation, and persistence -- the part of the
YOLO+SAM pre-annotation that lives outside :class:`YoloSamBackend` (which has
its own CPU-only orchestration coverage in
``tests/unit/test_yolo_sam_orchestration.py``).
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from tests.conftest import create_project_with_video
from vistlabel.main import create_app
from vistlabel.segmentation.fake import FakeBackend
from vistlabel.segmentation.registry import BackendRegistry
from vistlabel.segmentation.yolo_class_mapping import YoloClassMapping
from vistlabel.segmentation.yolo_detector import YoloDetection, YoloDetector
from vistlabel.segmentation.yolo_sam import YoloSamBackend
from vistlabel.settings import AppSettings, YoloSamModelSettings
from vistlabel.video.ffmpeg import FfmpegBinaries


class _ScriptedDetector(YoloDetector):
    def __init__(self, names: dict[int, str], script: list[list[YoloDetection]]) -> None:
        self._names = names
        self._script = list(script)
        self._calls = 0

    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def detect(self, frame) -> list[YoloDetection]:  # noqa: ANN001
        detections = self._script[self._calls]
        self._calls += 1
        return detections


def _client_with_working_yolo_sam(
    tmp_path, script: list[list[YoloDetection]], names: dict[int, str]
) -> TestClient:
    settings = AppSettings(workspace={"root": tmp_path / "workspace"})
    app = create_app(settings)

    checkpoint = tmp_path / "custom.pt"
    checkpoint.write_bytes(b"stand-in checkpoint bytes")
    class_mapping_path = tmp_path / "mapping.json"
    model_classes = ", ".join(f'"{cid}": "{name}"' for cid, name in names.items())
    class_mapping_path.write_text(
        f'{{"schema_version": 1, "model_classes": {{{model_classes}}}}}', encoding="utf-8"
    )
    yolo_settings = YoloSamModelSettings(
        enabled=True, yolo_checkpoint=checkpoint, class_mapping=class_mapping_path
    )
    fake_refiner = FakeBackend()
    yolo_backend = YoloSamBackend(yolo_settings, fake_refiner)
    yolo_backend._detector = _ScriptedDetector(names, script)  # bypass real ultralytics load
    yolo_backend._class_mapping = YoloClassMapping(schema_version=1, model_classes=names)

    app.state.registry = BackendRegistry({"fake": fake_refiner, "yolo_sam": yolo_backend})
    return TestClient(app)


def _wait_for_job(api_client: TestClient, project_id: str, job_id: str, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    terminal = {"succeeded", "failed", "cancelled"}
    while time.monotonic() < deadline:
        record = api_client.get(f"/api/v1/projects/{project_id}/jobs/{job_id}").json()
        if record["status"] in terminal:
            return record
        time.sleep(0.02)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_auto_preannotate_creates_objects_and_persists_masks(
    tmp_path, ffmpeg_binaries: FfmpegBinaries
) -> None:
    detection = YoloDetection(
        box_xyxy=(4, 4, 12, 12), yolo_class_id=0, score=0.95, coarse_mask=None
    )
    script = [[detection]]
    api_client = _client_with_working_yolo_sam(tmp_path, script, names={0: "pipette"})
    project_id = create_project_with_video(api_client, "auto-preannotate test")["project_id"]

    # Projects start with an empty ontology, so the label the detector will
    # report has to exist in *this* project before it can be matched.
    api_client.put(
        f"/api/v1/projects/{project_id}/labels",
        json={"labels": [{"label_id": 3, "name": "pipette"}]},
    )

    response = api_client.post(
        f"/api/v1/projects/{project_id}/auto-preannotate",
        json={"backend_id": "yolo_sam", "frame_indices": [0], "requested_class_ids": [0]},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    record = _wait_for_job(api_client, project_id, job_id)
    assert record["status"] == "succeeded", record

    objects = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["objects"]
    assert len(objects) == 1
    assert objects[0]["label_id"] == 3
    assert objects[0]["label_name"] == "pipette"

    annotation = api_client.get(f"/api/v1/projects/{project_id}/frames/0/annotations").json()
    assert len(annotation["objects"]) == 1
    assert annotation["objects"][0]["object_id"] == objects[0]["object_id"]
    assert annotation["objects"][0]["connected_component_count"] == 1


def test_auto_preannotate_rejects_class_not_in_project_ontology(
    tmp_path, ffmpeg_binaries: FfmpegBinaries
) -> None:
    """A class the mapping file knows about but this project's
    labels.json does not is silently skipped, never guessed."""
    script = [
        [
            YoloDetection(
                box_xyxy=(4, 4, 12, 12), yolo_class_id=0, score=0.95, coarse_mask=None
            )
        ]
    ]
    api_client = _client_with_working_yolo_sam(
        tmp_path, script, names={0: "a_class_no_project_has"}
    )
    project_id = create_project_with_video(api_client, "auto-preannotate reject test")[
        "project_id"
    ]

    # Seed an unrelated label so the skip below demonstrates "this class is
    # not in the ontology" rather than the weaker "the ontology is empty".
    api_client.put(
        f"/api/v1/projects/{project_id}/labels",
        json={"labels": [{"label_id": 0, "name": "some_other_class"}]},
    )

    response = api_client.post(
        f"/api/v1/projects/{project_id}/auto-preannotate",
        json={"backend_id": "yolo_sam", "frame_indices": [0], "requested_class_ids": [0]},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    record = _wait_for_job(api_client, project_id, job_id)
    assert record["status"] == "succeeded", record

    objects = api_client.get(f"/api/v1/projects/{project_id}/objects").json()["objects"]
    assert objects == []


def test_auto_preannotate_with_unavailable_backend_returns_409(
    api_client: TestClient, ffmpeg_binaries: FfmpegBinaries
) -> None:
    project_id = create_project_with_video(api_client, "unavailable backend test")["project_id"]
    response = api_client.post(
        f"/api/v1/projects/{project_id}/auto-preannotate",
        json={"backend_id": "yolo_sam", "frame_indices": [0], "requested_class_ids": []},
    )
    assert response.status_code == 409
