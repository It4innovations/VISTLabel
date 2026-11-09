# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Application-level startup diagnostics used by ``GET /api/v1/health``.

Reports filesystem, FFmpeg, Python, package and (optional) CUDA status.
CUDA/PyTorch are an optional extra (``pip install vistlabel[gpu]``) -- the
application must start and report health without any model checkpoint or
GPU stack installed, so a missing ``torch`` is reported, not raised.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from vistlabel import __version__
from vistlabel.settings import AppSettings
from vistlabel.video.ffmpeg import FfmpegNotFoundError, find_ffmpeg_binaries

SUPPORTED_PYTHON_RANGE = "3.11.x - 3.13.x (target for the GPU backends)"

Status = Literal["ok", "degraded", "missing", "error"]


class CheckResult(BaseModel):
    status: Status
    detail: str


class SystemDiagnosticsReport(BaseModel):
    status: Status
    package_version: str
    filesystem: CheckResult
    ffmpeg: CheckResult
    python: CheckResult
    cuda: CheckResult


def check_filesystem(workspace_root: Path) -> CheckResult:
    try:
        workspace_root.mkdir(parents=True, exist_ok=True)
        probe = workspace_root / ".vistlabel_write_check"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckResult(status="error", detail=f"workspace root not writable: {exc}")
    return CheckResult(status="ok", detail=f"workspace root writable: {workspace_root.resolve()}")


def check_ffmpeg(ffmpeg_dir: Path | None = None) -> CheckResult:
    try:
        binaries = find_ffmpeg_binaries(ffmpeg_dir)
    except FfmpegNotFoundError as exc:
        return CheckResult(status="missing", detail=str(exc))
    try:
        result = subprocess.run(
            [str(binaries.ffprobe), "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        first_line = result.stdout.splitlines()[0] if result.stdout else "unknown version"
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(status="error", detail=f"ffprobe failed to run: {exc}")
    return CheckResult(status="ok", detail=f"{first_line} ({binaries.ffprobe})")


def check_python() -> CheckResult:
    major, minor = sys.version_info[:2]
    detail = f"{sys.version.split()[0]} ({sys.executable})"
    if (major, minor) in ((3, 11), (3, 12), (3, 13)):
        return CheckResult(status="ok", detail=detail)
    return CheckResult(
        status="degraded",
        detail=(
            f"{detail} -- outside the target range for the GPU backends "
            f"({SUPPORTED_PYTHON_RANGE}); fine for CPU-only use"
        ),
    )


def check_cuda() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            status="missing",
            detail="torch not installed (install the 'gpu' extra)",
        )
    if not torch.cuda.is_available():
        return CheckResult(status="missing", detail="torch installed but no CUDA device visible")
    device_name = torch.cuda.get_device_name(0)
    return CheckResult(status="ok", detail=f"{device_name} (CUDA {torch.version.cuda})")


def _overall_status(results: list[CheckResult]) -> Status:
    if any(result.status == "error" for result in results):
        return "error"
    if any(result.status in ("missing", "degraded") for result in results):
        return "degraded"
    return "ok"


def run_system_diagnostics(settings: AppSettings) -> SystemDiagnosticsReport:
    filesystem = check_filesystem(settings.workspace.root)
    ffmpeg = check_ffmpeg(settings.runtime.ffmpeg_dir)
    python = check_python()
    cuda = check_cuda()
    return SystemDiagnosticsReport(
        status=_overall_status([filesystem, ffmpeg, python, cuda]),
        package_version=__version__,
        filesystem=filesystem,
        ffmpeg=ffmpeg,
        python=python,
        cuda=cuda,
    )


if __name__ == "__main__":  # pragma: no cover - manual convenience entry point
    report = run_system_diagnostics(AppSettings())
    print(report.model_dump_json(indent=2))
    sys.exit(0 if report.status != "error" else 1)
