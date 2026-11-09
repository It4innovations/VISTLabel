# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Model checkpoint hashing (every inference event records a checkpoint
hash). A shared helper so every real backend adapter computes it the same
way instead of reimplementing SHA-256 streaming.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024

#: (resolved path, size, mtime_ns) -> digest. ``descriptor()`` is called on
#: every ``GET /api/v1/models`` and ``GET /api/v1/metrics``, and hashing a
#: multi-hundred-megabyte checkpoint each time made those endpoints take
#: seconds. Keying on the stat triple keeps the provenance guarantee that the
#: reported digest belongs to the bytes actually on disk: replace or rewrite
#: the file and the key changes, so it is re-hashed.
_cache: dict[tuple[str, int, int], str] = {}
_cache_lock = threading.Lock()


def hash_checkpoint_file(path: Path | None) -> str | None:
    """SHA-256 of a checkpoint file, or ``None`` if not configured/missing.
    Never raises -- an absent checkpoint is an ordinary "unavailable"
    condition the caller's descriptor() reports, not an error here."""
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None

    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached

    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while chunk := handle.read(_CHUNK_SIZE):
                hasher.update(chunk)
    except OSError:
        return None
    digest = hasher.hexdigest()

    with _cache_lock:
        _cache[key] = digest
    return digest


def clear_checkpoint_hash_cache() -> None:
    """Drop every memoized digest. Only needed by tests that rewrite a
    checkpoint within the same mtime granularity."""
    with _cache_lock:
        _cache.clear()
