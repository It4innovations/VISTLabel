# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from argon2 import PasswordHasher
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vistlabel.auth.account import Account, AccountStore, utc_now_iso
from vistlabel.main import create_app
from vistlabel.persistence.atomic import save_model
from vistlabel.settings import AppSettings
from vistlabel.video.ffmpeg import FfmpegBinaries, try_find_ffmpeg_binaries

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_TEST_VIDEO = FIXTURES_DIR / "tiny_test_video.mp4"

TEST_USERNAME = "tester"
TEST_PASSWORD = "correct-horse-battery"

#: Every state-changing API call must carry the same-origin marker header.
CSRF_HEADERS = {"X-Requested-With": "VISTLabel"}

#: The production Argon2id parameters cost ~0.1 s per hash and per verify.
#: Verification reads the parameters from the hash itself, so an account
#: written with a cheap hash signs in instantly -- hundreds of tests each
#: create an account and sign in. tests/unit/test_auth.py covers the real
#: parameters.
_CHEAP_TEST_HASH = PasswordHasher(time_cost=1, memory_cost=8, parallelism=1).hash(TEST_PASSWORD)


def write_test_account(workspace_root: Path) -> AccountStore:
    store = AccountStore(workspace_root)
    now = utc_now_iso()
    save_model(
        store.path,
        Account(
            username=TEST_USERNAME,
            password_hash=_CHEAP_TEST_HASH,
            created_at=now,
            updated_at=now,
        ),
        fsync=False,
    )
    return store


def login(client: TestClient | httpx.Client) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": TEST_USERNAME, "password": TEST_PASSWORD},
        headers=CSRF_HEADERS,
    )
    assert response.status_code == 200, response.text


def authenticated_client(app: FastAPI) -> TestClient:
    """A TestClient signed in to ``app`` (creating the test account in its
    workspace if needed), with the CSRF header pre-set."""
    accounts: AccountStore = app.state.accounts
    if not accounts.is_configured():
        write_test_account(app.state.settings.workspace.root)
    client = TestClient(app)
    client.headers.update(CSRF_HEADERS)
    login(client)
    return client


@pytest.fixture(autouse=True)
def _no_free_disk_space_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The upload's free-disk-space floor (20 GiB by default) is a property
    of the machine, not of the code: tests upload a few-KB fixture video,
    yet failed wholesale on a test machine with ~13 GB free. Every AppSettings
    a test builds -- here, in the browser live server, or inline -- reads
    this from the environment. The check itself is covered by
    tests/unit/test_video_upload.py, which passes its own threshold."""
    monkeypatch.setenv("VISTLABEL_UPLOAD__MINIMUM_FREE_DISK_SPACE_GB", "0")


@pytest.fixture(scope="session")
def ffmpeg_binaries() -> FfmpegBinaries:
    """Resolves ffmpeg/ffprobe via PATH, or via VISTLABEL_RUNTIME__FFMPEG_DIR
    for local development where ffmpeg is installed but not on PATH (see
    README.md). Skips (not fails) dependent tests when unavailable, since
    whether ffmpeg is installed is an environment fact, not a code
    correctness question."""
    ffmpeg_dir_env = os.environ.get("VISTLABEL_RUNTIME__FFMPEG_DIR")
    ffmpeg_dir = Path(ffmpeg_dir_env) if ffmpeg_dir_env else None
    binaries = try_find_ffmpeg_binaries(ffmpeg_dir)
    if binaries is None:
        pytest.skip("ffmpeg/ffprobe not available (set VISTLABEL_RUNTIME__FFMPEG_DIR)")
    return binaries


@pytest.fixture
def app_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(workspace={"root": tmp_path / "workspace"})


@pytest.fixture
def configured_account(app_settings: AppSettings) -> AccountStore:
    return write_test_account(app_settings.workspace.root)


@pytest.fixture
def anonymous_client(app_settings: AppSettings, configured_account: AccountStore) -> TestClient:
    """Client against an app with an account but no session."""
    client = TestClient(create_app(app_settings))
    client.headers.update(CSRF_HEADERS)
    return client


@pytest.fixture
def api_client(anonymous_client: TestClient) -> TestClient:
    """Signed-in client with the CSRF header pre-set."""
    login(anonymous_client)
    return anonymous_client


def create_project_with_video(
    api_client: TestClient, project_name: str, video_path: Path = TINY_TEST_VIDEO
) -> dict:
    with video_path.open("rb") as video_file:
        response = api_client.post(
            "/api/v1/projects",
            data={"project_name": project_name},
            files={"video_file": (video_path.name, video_file, "video/mp4")},
        )
    assert response.status_code == 201, response.text
    return response.json()


def create_project_via_requests(
    base_url: str, project_name: str, video_path: Path = TINY_TEST_VIDEO
) -> str:
    """Same as :func:`create_project_with_video` but against a real running
    server (used by browser tests, which drive a live uvicorn instance
    rather than an in-process TestClient). Returns the new project_id."""
    with video_path.open("rb") as video_file:
        response = httpx.post(
            f"{base_url}/api/v1/projects",
            data={"project_name": project_name},
            files={"video_file": (video_path.name, video_file, "video/mp4")},
            timeout=30,
        )
    assert response.status_code == 201, response.text
    return str(response.json()["project_id"])
