# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Model backend diagnostic command.

Usage::

    python -m vistlabel.diagnostics.models --backend fake
    python -m vistlabel.diagnostics.models --backend sam3   # once implemented

For a real backend this imports the configured model code, loads the
configured checkpoint, runs a minimal image inference and a minimal video
propagation test, and reports device, dtype, peak VRAM, checkpoint hash and
code revision. The fake backend runs the same sequence against its
synthetic in-memory session so the diagnostic harness itself is exercised
without any real model installed.

A failing diagnostic must not crash the whole application: this command
reports a structured failure and exits non-zero so the caller (or the
backend registry at startup) can mark the backend unavailable with an exact
reason.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from pydantic import BaseModel

from vistlabel.segmentation.base import PointPrompt
from vistlabel.segmentation.registry import UnknownBackendError, build_registry
from vistlabel.settings import AppSettings, load_settings


class ModelDiagnosticReport(BaseModel):
    backend_id: str
    ok: bool
    available: bool
    model_name: str | None = None
    code_revision: str | None = None
    checkpoint_sha256: str | None = None
    device: str = "cpu"
    dtype: str | None = None
    peak_vram_mb: float | None = None
    duration_seconds: float | None = None
    error: str | None = None


def _write_synthetic_diagnostic_video(path: Path) -> None:
    """A handful of tiny solid-color frames, muxed with PyAV -- a real,
    decodable video for backends (Sam21Backend onward) that need one,
    without depending on system ffmpeg or a checked-in test fixture."""
    import av
    import numpy as np

    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("libx264", rate=5)
        stream.width = 32
        stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(5):
            array = np.full((32, 32, 3), fill_value=(i * 40) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def run_diagnostic(settings: AppSettings, backend_id: str) -> ModelDiagnosticReport:
    registry = build_registry(settings)
    try:
        backend = registry.get(backend_id)
    except UnknownBackendError as exc:
        return ModelDiagnosticReport(
            backend_id=backend_id,
            ok=False,
            available=False,
            error=str(exc),
        )

    descriptor = backend.descriptor()
    if not descriptor.available:
        return ModelDiagnosticReport(
            backend_id=backend_id,
            ok=False,
            available=False,
            model_name=descriptor.model_name,
            code_revision=descriptor.code_revision,
            error=descriptor.unavailable_reason or "backend reports itself unavailable",
        )

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        cuda_available = False

    started = time.monotonic()
    session_id = None
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "diagnostic.mp4"
            _write_synthetic_diagnostic_video(video_path)
            session_id = backend.open_video("diagnostic", video_path)

            # 1. minimal image inference
            backend.segment_interactive(
                session_id,
                frame_index=0,
                object_id="diagnostic-object",
                points=[PointPrompt(x=5, y=5, positive=True)],
                box_xyxy=None,
                previous_mask=None,
            )

            # 2. minimal video propagation test
            list(
                backend.propagate(
                    session_id,
                    object_ids=["diagnostic-object"],
                    start_frame=0,
                    end_frame=2,
                    direction="forward",
                )
            )
    except Exception as exc:  # noqa: BLE001 - a failing diagnostic must never crash the caller
        return ModelDiagnosticReport(
            backend_id=backend_id,
            ok=False,
            available=descriptor.available,
            model_name=descriptor.model_name,
            code_revision=descriptor.code_revision,
            checkpoint_sha256=descriptor.checkpoint_sha256,
            device=getattr(backend, "device", None) or "cpu",
            dtype=str(getattr(backend, "dtype", None) or "") or None,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if session_id is not None:
            backend.close_video(session_id)

    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6 if cuda_available else None
    return ModelDiagnosticReport(
        backend_id=backend_id,
        ok=True,
        available=True,
        model_name=descriptor.model_name,
        code_revision=descriptor.code_revision,
        checkpoint_sha256=descriptor.checkpoint_sha256,
        device=getattr(backend, "device", None) or "cpu",
        dtype=str(getattr(backend, "dtype", None) or "") or None,
        peak_vram_mb=round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        duration_seconds=round(time.monotonic() - started, 4),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vistlabel.diagnostics.models",
        description="Run a minimal image + video inference smoke test against a backend.",
    )
    parser.add_argument(
        "--backend", required=True, help="backend_id, e.g. fake, sam21, sam3, yolo_sam"
    )
    parser.add_argument("--config", type=Path, default=None, help="path to config/app.yaml")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    report = run_diagnostic(settings, args.backend)
    print(report.model_dump_json(indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
