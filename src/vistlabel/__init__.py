# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""VISTLabel: Video Instance Segmentation, Tracking and Labeling."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vistlabel")
except PackageNotFoundError:  # pragma: no cover - only happens if not installed
    __version__ = "0.0.0+unknown"
