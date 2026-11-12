# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""The local Help pages: /help, /help/<name> and the links into them."""

from __future__ import annotations

import html
import re

import pytest
from fastapi.testclient import TestClient
from markupsafe import escape

from tests.conftest import create_project_with_video
from vistlabel.api.help import HELP_INDEX
from vistlabel.help.shortcuts import load_shortcuts


def test_help_index_lists_every_page_and_links_the_shortcuts(api_client: TestClient) -> None:
    response = api_client.get("/help")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    for page in HELP_INDEX:
        assert f'href="/help/{page.name}"' in body, page.name
    # The shortcut table lives only in the user guide; the overview links to it.
    assert 'href="/help/user-guide#keyboard-shortcuts"' in body
    for shortcut in load_shortcuts():
        assert str(escape(shortcut["description"])) not in body

    guide = html.unescape(api_client.get("/help/user-guide").text)
    assert guide.count('id="keyboard-shortcuts"') == 1
    for shortcut in load_shortcuts():
        assert shortcut["description"] in guide


@pytest.mark.parametrize("name", [page.name for page in HELP_INDEX])
def test_every_listed_page_renders(api_client: TestClient, name: str) -> None:
    response = api_client.get(f"/help/{name}")

    assert response.status_code == 200
    assert '<article class="lv-help-article" id="help-article">' in response.text
    assert "<h1 " in response.text
    # The page being shown is the highlighted one in the sidebar.
    active = re.findall(r'<li class="active">\s*<a href="([^"]+)"', response.text)
    assert active == [f"/help/{name}"]


def test_user_guide_has_an_outline_and_the_generated_shortcut_table(
    api_client: TestClient,
) -> None:
    body = api_client.get("/help/user-guide").text

    assert 'id="help-toc"' in body
    assert '<a href="#export">Export</a>' in body
    assert "<kbd>Ctrl</kbd>" in body
    assert "shortcuts:begin" not in body
    assert "Mark the current frame reviewed" in body


@pytest.mark.parametrize(
    "name",
    [
        "does-not-exist",
        "README",  # a repository document, but not a Help page
        "known-limitations",  # a former Help page, no longer listed
    ],
)
def test_unknown_or_unlisted_pages_are_404(api_client: TestClient, name: str) -> None:
    response = api_client.get(f"/help/{name}")

    assert response.status_code == 404
    assert "Page not found" in response.text


def test_path_traversal_never_reads_outside_the_help_pages(api_client: TestClient) -> None:
    for path in (
        "/help/../pyproject.toml",
        "/help/%2e%2e/pyproject",
        "/help/..%2Fpyproject",
        "/help/..%5Cpyproject",
    ):
        response = api_client.get(path)
        assert "[build-system]" not in response.text


def test_help_images_are_served_from_the_docs_images_folder(api_client: TestClient) -> None:
    response = api_client.get("/help/images/01-sign-in.png")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.content[1:4] == b"PNG"

    svg = api_client.get("/help/images/architecture.svg")
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    assert "sandbox" in svg.headers["content-security-policy"]


@pytest.mark.parametrize(
    "path",
    [
        "/help/images/does-not-exist.png",
        "/help/images/user-guide.md",  # not an image type
        "/help/images/..%2Fuser-guide.md",
        "/help/images/..%5C..%5Cpyproject.toml",
        "/help/images/%2E%2E",
    ],
)
def test_help_images_refuse_anything_else(api_client: TestClient, path: str) -> None:
    response = api_client.get(path)

    assert response.status_code == 404
    assert "[build-system]" not in response.text


def test_help_images_need_a_session(anonymous_client: TestClient) -> None:
    response = anonymous_client.get("/help/images/01-sign-in.png", follow_redirects=False)

    assert response.status_code in (303, 401)


def test_the_user_guide_shows_its_screenshots(api_client: TestClient) -> None:
    body = api_client.get("/help/user-guide").text

    assert '<img src="/help/images/01-sign-in.png" alt="Sign-in page"' in body
    assert "<figcaption>Sign-in page</figcaption>" in body


def test_project_list_links_to_help(api_client: TestClient) -> None:
    body = api_client.get("/").text

    assert '<a href="/help" id="open-help-link">Help</a>' in body


def test_annotation_workspace_opens_the_user_guide_in_a_new_tab(
    api_client: TestClient, ffmpeg_binaries: object
) -> None:
    project = create_project_with_video(api_client, "help link")

    body = api_client.get(f"/projects/{project['project_id']}").text

    assert 'id="open-help-link"' in body
    assert 'href="/help/user-guide"' in body
    assert 'target="_blank"' in body
