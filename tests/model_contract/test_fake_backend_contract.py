# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

from tests.model_contract.backend_contract import run_full_contract
from vistlabel.segmentation.fake import FakeBackend


def test_fake_backend_satisfies_the_common_contract() -> None:
    run_full_contract(FakeBackend())
