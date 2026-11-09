# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
from pathlib import Path

from vistlabel.segmentation.checkpoints import hash_checkpoint_file


def test_hash_of_missing_path_is_none() -> None:
    assert hash_checkpoint_file(None) is None


def test_hash_of_nonexistent_file_is_none(tmp_path: Path) -> None:
    assert hash_checkpoint_file(tmp_path / "does-not-exist.pt") is None


def test_hash_matches_hashlib(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    data = b"some deterministic checkpoint bytes" * 1000
    checkpoint.write_bytes(data)

    assert hash_checkpoint_file(checkpoint) == hashlib.sha256(data).hexdigest()


def test_hash_is_memoized_but_follows_content_changes(tmp_path: Path) -> None:
    """descriptor() runs on every /models and /metrics call, so the digest is
    cached -- but it must still track what is actually on disk."""
    from vistlabel.segmentation.checkpoints import clear_checkpoint_hash_cache

    clear_checkpoint_hash_cache()
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"first revision")

    first = hash_checkpoint_file(checkpoint)
    assert first == hashlib.sha256(b"first revision").hexdigest()
    assert hash_checkpoint_file(checkpoint) == first

    # Rewriting with a different length changes the stat key, so the stale
    # digest is not served.
    checkpoint.write_bytes(b"a considerably longer second revision")
    assert hash_checkpoint_file(checkpoint) == (
        hashlib.sha256(b"a considerably longer second revision").hexdigest()
    )


def test_hash_of_missing_or_unset_checkpoint_is_none(tmp_path: Path) -> None:
    assert hash_checkpoint_file(None) is None
    assert hash_checkpoint_file(tmp_path / "nope.safetensors") is None
    assert hash_checkpoint_file(tmp_path) is None  # a directory, not a file
