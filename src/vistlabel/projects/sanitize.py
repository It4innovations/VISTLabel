# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Project display-name sanitization. The project directory name is always the server-generated
project_id (see ``ids.py``), never the user-supplied name -- this function
only guards against control characters, path separators (defense in depth,
in case the name is ever used in a path or shown unescaped) and unbounded
length.
"""

from __future__ import annotations

import re

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
MAX_NAME_LENGTH = 200


def sanitize_project_name(raw: str) -> str:
    cleaned = _CONTROL_CHARS.sub("", raw).strip()
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    if not cleaned:
        raise ValueError("project name must not be empty")
    return cleaned[:MAX_NAME_LENGTH]
