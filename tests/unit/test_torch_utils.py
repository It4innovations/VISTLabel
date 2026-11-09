# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging

import pytest

from vistlabel.segmentation.torch_utils import resolve_device

torch = pytest.importorskip("torch")


def test_explicit_device_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda:1") == "cuda:1"


def test_auto_picks_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with caplog.at_level(logging.WARNING):
        assert resolve_device("auto") == "cuda"
    assert not caplog.records


def test_auto_cpu_fallback_warns_with_visible_devices(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The single-GPU-laptop case: the launcher pinned GPU 1, which hides
    # the only card.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with caplog.at_level(logging.WARNING):
        assert resolve_device("auto") == "cpu"
    assert "running models on CPU" in caplog.text
    assert "CUDA_VISIBLE_DEVICES='1'" in caplog.text
