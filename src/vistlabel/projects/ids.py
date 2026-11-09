# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Server-side project ID generation: project IDs are always generated
server-side, never derived from client input.

IDs are ULIDs (48-bit millisecond timestamp + 80 bits of randomness,
Crockford base32) prefixed ``prj_``, giving `prj_01J4ABCD1234`-style
identifiers. Being time-sortable is a genuine benefit
here: there is no database index, so a lexicographic directory listing of
`workspace/projects/` is already roughly creation-ordered.
"""

from __future__ import annotations

import re
import secrets
import time

_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
#: Matches exactly what new_project_id() produces -- "prj_" + a 26-char
#: Crockford-base32 ULID. A project_id arriving from a request path
#: parameter must be validated against this shape before it is ever used
#: to build a filesystem path, not merely checked for existence afterward
#: -- this is what prevents path traversal through the project id.
PROJECT_ID_PATTERN = re.compile(r"^prj_[0-9A-HJKMNP-TV-Z]{26}$")


def _encode_crockford(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        value, remainder = divmod(value, 32)
        chars.append(_CROCKFORD_ALPHABET[remainder])
    return "".join(reversed(chars))


def new_ulid() -> str:
    timestamp_ms = int(time.time() * 1000)
    randomness = secrets.randbits(80)
    return _encode_crockford(timestamp_ms, 10) + _encode_crockford(randomness, 16)


def new_project_id() -> str:
    return f"prj_{new_ulid()}"


def is_valid_project_id(project_id: str) -> bool:
    return bool(PROJECT_ID_PATTERN.match(project_id))
