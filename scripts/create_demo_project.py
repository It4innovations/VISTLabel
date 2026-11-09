# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Create a deterministic demo project for manual QA of the annotation
workspace.

Uses the internal service layer directly (not HTTP) so it can run against
a workspace without a live server: creates a project from the checked-in
synthetic fixture video, adds one object, and accepts a fake-backend
segmentation on the first frame -- enough to open the annotation page and
immediately see a real object with a real mask.

Usage::

    python scripts/create_demo_project.py [--workspace ./workspace]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vistlabel.annotations.frames_service import save_frame_annotation  # noqa: E402
from vistlabel.annotations.objects_service import create_object  # noqa: E402
from vistlabel.annotations.png_mask import encode_mask_png_base64  # noqa: E402
from vistlabel.domain.frames import FrameObjectAnnotation  # noqa: E402
from vistlabel.domain.labels import LabelDefinition  # noqa: E402
from vistlabel.domain.project import ProjectStatus, SourceVideoInfo  # noqa: E402
from vistlabel.persistence.paths import project_dir, source_video_path  # noqa: E402
from vistlabel.persistence.rle import count_connected_components, encode_mask  # noqa: E402
from vistlabel.projects.store import ProjectStore  # noqa: E402
from vistlabel.segmentation.base import PointPrompt  # noqa: E402
from vistlabel.segmentation.fake import FakeBackend  # noqa: E402
from vistlabel.video.ffmpeg import find_ffmpeg_binaries  # noqa: E402
from vistlabel.video.frames import build_frame_index, save_frame_timestamps  # noqa: E402
from vistlabel.video.validation import validate_video  # noqa: E402

DEMO_VIDEO = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tiny_test_video.mp4"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("./workspace"))
    parser.add_argument("--video", type=Path, default=DEMO_VIDEO)
    parser.add_argument("--ffmpeg-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if not args.video.is_file():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1

    store = ProjectStore(args.workspace)
    project = store.create("demo_isolator")
    pdir = project_dir(args.workspace, project.project_id)
    part_path = pdir / "source" / "video.mp4.part"
    part_path.write_bytes(args.video.read_bytes())

    binaries = find_ffmpeg_binaries(args.ffmpeg_dir)
    probed = validate_video(binaries, part_path)
    timestamps = build_frame_index(part_path)
    part_path.replace(source_video_path(pdir))
    save_frame_timestamps(pdir / "cache" / "frame_timestamps.json", timestamps)

    source_info = SourceVideoInfo(
        original_filename=args.video.name,
        stored_path="source/video.mp4",
        sha256="demo",
        size_bytes=args.video.stat().st_size,
        video_codec=probed.video_codec,
        pixel_format=probed.pixel_format,
        width=probed.width,
        height=probed.height,
        nominal_fps=probed.nominal_fps,
        duration_seconds=probed.duration_seconds,
        frame_count=len(timestamps),
        variable_frame_rate=probed.variable_frame_rate,
    )
    project = store.update(
        project.project_id,
        expected_revision=project.revision,
        mutate=lambda state: state.model_copy(
            update={"status": ProjectStatus.IN_PROGRESS, "source": source_info}
        ),
    )

    # Projects start with an empty ontology (names are operator-defined), so
    # register the demo label before attaching an object to it -- otherwise
    # labels.json and objects.json would disagree at export time.
    store.set_labels(project.project_id, [LabelDefinition(label_id=0, name="pipette")])
    demo_object = create_object(pdir, label_id=0, label_name="pipette")

    backend = FakeBackend()
    session_id = backend.open_video(project.project_id, source_video_path(pdir))
    prediction = backend.segment_interactive(
        session_id,
        frame_index=0,
        object_id=demo_object.object_id,
        points=[PointPrompt(x=probed.width / 2, y=probed.height / 2, positive=True)],
        box_xyxy=None,
        previous_mask=None,
    )
    backend.close_video(session_id)

    save_frame_annotation(
        pdir,
        0,
        timestamps[0],
        expected_revision=0,
        mutate=lambda frame: frame.model_copy(
            update={
                "objects": [
                    FrameObjectAnnotation(
                        object_id=demo_object.object_id,
                        label_id=demo_object.label_id,
                        visible_mask_rle=encode_mask(prediction.visible_mask),
                        connected_component_count=count_connected_components(
                            prediction.visible_mask
                        ),
                        source="fake_interactive",
                        manually_modified=False,
                    )
                ]
            }
        ),
    )
    # Sanity check that the mask round-trips through the same PNG path the
    # browser uses, without actually needing the value here.
    encode_mask_png_base64(prediction.visible_mask)

    print(f"Created demo project {project.project_id!r} in {args.workspace}")
    print(f"Open it at: /projects/{project.project_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
