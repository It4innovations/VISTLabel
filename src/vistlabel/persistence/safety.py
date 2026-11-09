# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Path-containment safety check that prevents path traversal.

Every place in this codebase that deletes a directory built from a
client-supplied identifier (``project_id``, ``export_id``, ...) must
confirm the resolved target actually lives inside the expected parent
before calling ``shutil.rmtree`` -- checking only ``Path.exists()`` is not
enough, since a crafted id like ``".."`` (delivered percent-encoded so
client-side URL normalization doesn't collapse it first) resolves to the
*parent* directory, which very much "exists" too.
"""

from __future__ import annotations

from pathlib import Path


def is_safe_child_path(base: Path, target: Path) -> bool:
    """True iff ``target``, once resolved, is a real, strict descendant of
    ``base`` -- never a traversal out to a sibling, ancestor, or ``base``
    itself (deleting ``base`` in place of one of its children is exactly
    as dangerous as escaping it entirely)."""
    try:
        resolved_target = target.resolve()
        resolved_base = base.resolve()
    except OSError:
        return False
    return resolved_base in resolved_target.parents
