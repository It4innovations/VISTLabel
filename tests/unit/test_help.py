# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Help subsystem: the Markdown renderer, the shortcut registry and the
generated shortcut table in docs/user-guide.md."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

from vistlabel.api.help import HELP_INDEX
from vistlabel.help.markdown import render_markdown, rewrite_link, slugify
from vistlabel.help.shortcuts import load_shortcuts, merge_shortcut_table

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs"
SERVED = {page.name for page in HELP_INDEX}


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_help_reference", REPO / "scripts" / "generate_help_reference.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------- markdown


def test_markdown_renders_headings_lists_code_tables_and_local_links() -> None:
    source = """# Title `code`

Intro with **bold**, *italic*, `inline` and a [link](operator-guide.md#logs)
and an [external](https://example.org/x).

## Section one

- first
- second `x`
  continued

1. one
2. two

| A | B |
| --- | --- |
| 1 | `2` |

```text
raw <b>not html</b>
```

> quoted

## Section one
"""
    rendered = render_markdown(source)
    assert rendered.title == "Title code"
    assert '<h1 id="title-code">' in rendered.html
    assert [h.anchor for h in rendered.headings] == ["title-code", "section-one", "section-one-1"]
    assert '<a href="/help/operator-guide#logs">link</a>' in rendered.html
    assert 'href="https://example.org/x" rel="noopener" target="_blank"' in rendered.html
    assert "<strong>bold</strong>" in rendered.html and "<em>italic</em>" in rendered.html
    assert "<ul><li>first</li><li>second <code>x</code> continued</li></ul>" in rendered.html
    assert "<ol><li>one</li><li>two</li></ol>" in rendered.html
    assert "<table>" in rendered.html and "<td><code>2</code></td>" in rendered.html
    assert "<pre><code>raw &lt;b&gt;not html&lt;/b&gt;</code></pre>" in rendered.html
    assert "<blockquote><p>quoted</p></blockquote>" in rendered.html


def test_markdown_escapes_html_and_rejects_script_links() -> None:
    rendered = render_markdown("<script>alert(1)</script> [x](javascript:alert(1))")
    assert "<script>" not in rendered.html
    assert "javascript:" not in rendered.html


def test_markdown_keeps_kbd_but_nothing_else() -> None:
    rendered = render_markdown('Press <kbd>Ctrl</kbd> or <kbd onclick="x">Z</kbd> <b>now</b>')
    assert "<kbd>Ctrl</kbd>" in rendered.html
    # Only the bare tag is restored; one with attributes stays escaped text.
    assert "<kbd onclick" not in rendered.html
    assert "&lt;kbd onclick" in rendered.html
    assert "<b>" not in rendered.html


def test_code_spans_are_not_read_as_emphasis_or_links() -> None:
    rendered = render_markdown("See `docs/*.md` and `[x](y.md)` and `__init__`.")
    assert "<code>docs/*.md</code>" in rendered.html
    assert "<code>[x](y.md)</code>" in rendered.html
    assert "<code>__init__</code>" in rendered.html
    assert "<em>" not in rendered.html and "<a " not in rendered.html


def test_links_to_unserved_repository_files_keep_only_their_label() -> None:
    rendered = render_markdown(
        "See [`README.md`](../README.md), [ADRs](docs/adr/) and [guide](user-guide.md).",
        served=SERVED,
    )
    assert "<code>README.md</code>" in rendered.html
    assert "../README.md" not in rendered.html and "docs/adr/" not in rendered.html
    assert '<a href="/help/user-guide">guide</a>' in rendered.html


def test_byte_order_mark_and_marker_comments_are_dropped() -> None:
    rendered = render_markdown("﻿# Title\n\n<!-- shortcuts:begin -->\ntext\n")
    assert rendered.title == "Title"
    assert "shortcuts:begin" not in rendered.html
    assert "<p>text</p>" in rendered.html


def test_indented_fence_inside_a_list_is_dedented() -> None:
    rendered = render_markdown("- step\n\n  ```bash\n  vistlabel serve\n  ```\n")
    assert "<pre><code>vistlabel serve</code></pre>" in rendered.html


def test_slugify_and_link_rewrite() -> None:
    assert slugify("Frames and playback") == "frames-and-playback"
    assert slugify("Security hardening checklist (Section 31)") == (
        "security-hardening-checklist-section-31"
    )
    assert rewrite_link("docs/user-guide.md#export") == "/help/user-guide#export"
    assert rewrite_link("./known-limitations.md") == "/help/known-limitations"
    assert rewrite_link("../README.md", SERVED) is None
    assert rewrite_link("https://x.y/z.md") == "https://x.y/z.md"
    assert rewrite_link("#export") == "#export"


@pytest.mark.parametrize("name", sorted(SERVED))
def test_every_served_doc_exists_and_renders(name: str) -> None:
    path = DOCS / f"{name}.md"
    assert path.is_file(), f"{name} is in HELP_INDEX but docs/{name}.md is missing"
    rendered = render_markdown(path.read_text(encoding="utf-8"), served=SERVED)
    assert rendered.title
    assert not re.search(r"<li>\s*</li>", rendered.html), name
    # No Markdown syntax left behind by an unsupported construct.
    assert "](" not in re.sub(r"<pre>.*?</pre>|<code>.*?</code>", "", rendered.html, flags=re.S)


def test_in_page_links_of_the_user_guide_resolve() -> None:
    rendered = render_markdown((DOCS / "user-guide.md").read_text(encoding="utf-8"), SERVED)
    anchors = {h.anchor for h in rendered.headings}
    for target in re.findall(r'href="#([^"]+)"', rendered.html):
        assert target in anchors, f"user-guide links to missing #{target}"


# ------------------------------------------------------------ shortcuts


def test_shortcut_registry_is_well_formed() -> None:
    shortcuts = load_shortcuts()
    ids = [s["id"] for s in shortcuts]
    assert len(ids) == len(set(ids))
    for shortcut in shortcuts:
        assert shortcut["keys"] and shortcut["description"]
        assert shortcut["scope"] in {"workspace", "canvas", "text field", "dialogs"}


def test_shortcut_table_in_user_guide_is_generated() -> None:
    current = (DOCS / "user-guide.md").read_text(encoding="utf-8")
    assert merge_shortcut_table(current) == current, (
        "docs/user-guide.md shortcut table is stale; run scripts/generate_help_reference.py"
    )
    assert _load_generator().main(["--check"]) == 0


def test_merge_requires_both_markers() -> None:
    with pytest.raises(ValueError):
        merge_shortcut_table("# no markers here")


def test_registry_lists_the_workspace_keyboard_handler_keys() -> None:
    """Every single-letter shortcut the workspace's keydown handler acts on
    is documented, so the Help can't silently fall behind the code."""
    source = (REPO / "src" / "vistlabel" / "static" / "js" / "annotation-canvas.js").read_text(
        encoding="utf-8"
    )
    handled = {letter.upper() for letter in re.findall(r'case "([a-z])":', source)}
    documented = {
        part.strip() for s in load_shortcuts() for part in s["keys"].split("/")
    }
    assert handled, "keyboard handler not found"
    assert handled <= documented, f"undocumented shortcuts: {sorted(handled - documented)}"
