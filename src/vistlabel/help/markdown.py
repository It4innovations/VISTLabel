# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Minimal, dependency-free Markdown to HTML converter for the Help pages.

It covers what the project documentation uses: ATX headings (with
GitHub-style anchor ids), paragraphs, fenced code blocks, inline code,
bold/italic, links, images (reduced to their alt text), ordered and
unordered lists (nested by indentation), pipe tables, block quotes and
horizontal rules. Whole-line HTML comments (the generated-section markers in
``docs/user-guide.md``) are dropped, and ``<kbd>`` is the one inline tag
kept.

Everything is HTML-escaped first, so documentation can never inject markup.
A link to another ``*.md`` document becomes the local ``/help/<name>`` route
when that document is one of the served Help pages; a link to any other
repository file (``../README.md``, ``../models/README.md``) is not reachable
from the browser, so only its label is kept.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass, field

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*```")
_UL = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OL = re.compile(r"^(\s*)\d+[.)]\s+(.*)$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_COMMENT = re.compile(r"^\s*<!--.*-->\s*$")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_CODE = re.compile(r"`([^`]+)`")
_KBD = re.compile(r"&lt;kbd&gt;(.+?)&lt;/kbd&gt;")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(
    r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])|(?<![\w_])_(?!\s)(.+?)(?<!\s)_(?![\w_])"
)
_MD_LINK = re.compile(r"^(?:\.{1,2}/)*(?:docs/)?([A-Za-z0-9_\-]+)\.md(#[A-Za-z0-9_\-]+)?$")
_PLACEHOLDER = re.compile("\x00(\\d+)\x00")
_SAFE_SCHEMES = ("/", "#", "http://", "https://", "mailto:")


def slugify(text: str) -> str:
    """GitHub-style heading id: lowercase, punctuation removed, each
    whitespace character becomes ``-`` (no collapsing, as GitHub does)."""
    plain = re.sub(r"`|\*|_", "", text)
    plain = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", plain)
    slug = re.sub(r"[^\w\s-]", "", plain.strip().lower())
    slug = re.sub(r"\s", "-", slug)
    return slug or "section"


def rewrite_link(target: str, served: Collection[str] | None = None) -> str | None:
    """Map a link target to an ``href``, or ``None`` when it points at a
    repository file the browser cannot reach.

    ``docs/x.md#anchor``, ``./x.md`` and ``x.md`` become ``/help/x#anchor``
    when ``x`` is served (``served=None`` accepts any name). External,
    absolute and in-page links pass through unchanged."""
    if target.startswith(_SAFE_SCHEMES):
        return target
    match = _MD_LINK.match(target)
    if match and (served is None or match.group(1) in served):
        return f"/help/{match.group(1)}{match.group(2) or ''}"
    return None


def render_inline(text: str, served: Collection[str] | None = None) -> str:
    # Code spans are taken out first and put back last, so nothing inside
    # them is read as emphasis or a link (`docs/*.md`, `[x](y)`).
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    escaped = html.escape(text, quote=False)
    escaped = _CODE.sub(lambda m: keep(f"<code>{m.group(1)}</code>"), escaped)
    # <kbd> is the one HTML tag the docs may use. Its content is already
    # escaped, so restoring just the tag pair cannot let markup through.
    escaped = _KBD.sub(lambda m: keep(f"<kbd>{m.group(1)}</kbd>"), escaped)
    escaped = _IMAGE.sub(lambda m: m.group(1), escaped)

    def link(match: re.Match[str]) -> str:
        label, target = match.group(1), html.unescape(match.group(2))
        href = rewrite_link(target, served)
        if href is None:
            return label
        if not href.startswith(_SAFE_SCHEMES):
            href = "#"
        external = ' rel="noopener" target="_blank"' if href.startswith("http") else ""
        return keep(f'<a href="{html.escape(href, quote=True)}"{external}>') + label + keep("</a>")

    escaped = _LINK.sub(link, escaped)
    escaped = _BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", escaped)
    escaped = _ITALIC.sub(lambda m: f"<em>{m.group(1) or m.group(2)}</em>", escaped)
    return _PLACEHOLDER.sub(lambda m: stash[int(m.group(1))], escaped)


@dataclass
class Heading:
    level: int
    text: str
    anchor: str


@dataclass
class Rendered:
    html: str
    title: str
    headings: list[Heading] = field(default_factory=list)


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_markdown(source: str, served: Collection[str] | None = None) -> Rendered:
    """Render ``source``; ``served`` names the Help pages links may target
    (``None`` treats every ``*.md`` link as served)."""
    lines = source.lstrip("﻿").splitlines()
    out: list[str] = []
    headings: list[Heading] = []
    used_anchors: dict[str, int] = {}
    title = ""
    paragraph: list[str] = []
    list_stack: list[tuple[str, int]] = []  # (tag, indent)
    in_code = False
    fence_indent = 0
    code: list[str] = []
    index = 0
    inline: Callable[[str], str] = lambda text: render_inline(text, served)  # noqa: E731

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph = []

    def close_lists(to_indent: int = -1) -> None:
        while list_stack and list_stack[-1][1] > to_indent:
            tag, _ = list_stack.pop()
            out[-1] += f"</li></{tag}>"

    def unique_anchor(text: str) -> str:
        base = slugify(text)
        count = used_anchors.get(base, 0)
        used_anchors[base] = count + 1
        return base if count == 0 else f"{base}-{count}"

    def starts_block(line: str) -> bool:
        return bool(
            _UL.match(line) or _OL.match(line) or _HEADING.match(line) or _FENCE.match(line)
        )

    while index < len(lines):
        line = lines[index]
        if in_code:
            if _FENCE.match(line):
                out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
                code = []
                in_code = False
            else:
                # A fence indented inside a list item indents its body too.
                code.append(line[fence_indent:] if line[:fence_indent].isspace() else line)
            index += 1
            continue
        if _FENCE.match(line):
            flush_paragraph()
            close_lists()
            in_code = True
            fence_indent = len(line) - len(line.lstrip())
            index += 1
            continue
        stripped = line.strip()
        if not stripped or _COMMENT.match(line):
            flush_paragraph()
            # A blank line inside a list keeps the list open when the next
            # line continues it (an indented item or paragraph).
            nxt = lines[index + 1] if index + 1 < len(lines) else ""
            if not (list_stack and nxt.startswith(" ") and nxt.strip()):
                close_lists()
            index += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            close_lists()
            level = len(heading.group(1))
            text = heading.group(2)
            anchor = unique_anchor(text)
            plain = re.sub(r"`", "", text)
            headings.append(Heading(level=level, text=plain, anchor=anchor))
            if not title and level == 1:
                title = plain
            out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            index += 1
            continue
        if stripped in ("---", "***", "___"):
            flush_paragraph()
            close_lists()
            out.append("<hr />")
            index += 1
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            close_lists()
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue
        if "|" in stripped and index + 1 < len(lines) and _TABLE_SEP.match(lines[index + 1]):
            flush_paragraph()
            close_lists()
            header = _split_row(stripped)
            rows: list[list[str]] = []
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(_split_row(lines[index]))
                index += 1
            out.append('<div class="lv-help-table"><table>')
            out.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header))
            out.append("</tr></thead><tbody>")
            for row in rows:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
            out.append("</tbody></table></div>")
            continue
        item = _UL.match(line) or _OL.match(line)
        if item:
            flush_paragraph()
            indent = len(item.group(1).replace("\t", "    "))
            tag = "ul" if _UL.match(line) else "ol"
            content = inline(item.group(2))
            if list_stack and list_stack[-1][1] == indent:
                out[-1] += f"</li><li>{content}"
            elif not list_stack or indent > list_stack[-1][1]:
                # A deeper indent opens a nested list inside the open <li>.
                list_stack.append((tag, indent))
                if len(list_stack) > 1:
                    out[-1] += f"<{tag}><li>{content}"
                else:
                    out.append(f"<{tag}><li>{content}")
            else:
                close_lists(indent)
                if list_stack and list_stack[-1][1] == indent:
                    out[-1] += f"</li><li>{content}"
                else:
                    list_stack.append((tag, indent))
                    out.append(f"<{tag}><li>{content}")
            index += 1
            # Continuation lines of the same item: indented, not a new block.
            while (
                index < len(lines)
                and lines[index].strip()
                and lines[index].startswith(" ")
                and not starts_block(lines[index])
            ):
                out[-1] += " " + inline(lines[index].strip())
                index += 1
            continue
        if list_stack:
            if line.startswith(" "):
                # An indented paragraph after a blank line inside a list item.
                para = [stripped]
                index += 1
                while (
                    index < len(lines)
                    and lines[index].strip()
                    and lines[index].startswith(" ")
                    and not starts_block(lines[index])
                ):
                    para.append(lines[index].strip())
                    index += 1
                out[-1] += f"<p>{inline(' '.join(para))}</p>"
                continue
            close_lists()
        paragraph.append(stripped)
        index += 1
    if in_code:
        out.append(f"<pre><code>{html.escape(chr(10).join(code))}</code></pre>")
    flush_paragraph()
    close_lists()
    return Rendered(html="\n".join(out), title=title, headings=headings)
