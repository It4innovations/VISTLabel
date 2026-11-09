# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""The keyboard-shortcut registry (``static/help/shortcuts.json``).

One JSON file feeds both the table on the ``/help`` index and the table in
``docs/user-guide.md``, which ``scripts/generate_help_reference.py``
regenerates between the markers below. ``tests/unit/test_help.py`` fails
when the guide's copy is stale.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

SHORTCUTS_FILE = Path(__file__).resolve().parent.parent / "static" / "help" / "shortcuts.json"
SHORTCUT_MARKERS = ("<!-- shortcuts:begin -->", "<!-- shortcuts:end -->")


class Shortcut(TypedDict):
    id: str
    keys: str
    scope: str
    description: str


@lru_cache(maxsize=1)
def load_shortcuts() -> tuple[Shortcut, ...]:
    data = json.loads(SHORTCUTS_FILE.read_text(encoding="utf-8"))
    return tuple(data["shortcuts"])


def render_shortcut_table() -> str:
    """The registry as a Markdown pipe table."""
    lines = ["| Keys | Where | Action |", "| --- | --- | --- |"]
    for shortcut in load_shortcuts():
        lines.append(
            f"| `{shortcut['keys']}` | {shortcut['scope']} | {shortcut['description']} |"
        )
    return "\n".join(lines)


def merge_shortcut_table(document: str) -> str:
    """``document`` with the text between the shortcut markers replaced by
    the current table."""
    begin, end = SHORTCUT_MARKERS
    if begin not in document or end not in document:
        raise ValueError(f"document must contain {begin} and {end}")
    head, rest = document.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{render_shortcut_table()}\n{end}{tail}"
