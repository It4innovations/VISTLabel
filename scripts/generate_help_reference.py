# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Write the keyboard-shortcut table into ``docs/user-guide.md`` from the
shortcut registry (``src/vistlabel/static/help/shortcuts.json``).

Run after editing the registry::

    python scripts/generate_help_reference.py          # rewrite the table
    python scripts/generate_help_reference.py --check  # exit 1 when stale

``tests/unit/test_help.py`` fails while the table is out of date.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vistlabel.help.shortcuts import merge_shortcut_table  # noqa: E402

GUIDE = REPO / "docs" / "user-guide.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 when the table is stale")
    args = parser.parse_args(argv)
    # newline="" keeps the file's own line endings, so the rewrite (and the
    # --check comparison) never trips over CRLF vs LF.
    with GUIDE.open(encoding="utf-8", newline="") as handle:
        current = handle.read()
    newline = "\r\n" if "\r\n" in current else "\n"
    merged = merge_shortcut_table(current.replace("\r\n", "\n")).replace("\n", newline)
    if args.check:
        if merged != current:
            print(f"stale: {GUIDE.relative_to(REPO)} -- run scripts/generate_help_reference.py")
            return 1
        return 0
    with GUIDE.open("w", encoding="utf-8", newline="") as handle:
        handle.write(merged)
    print(f"wrote the shortcut table into {GUIDE.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
