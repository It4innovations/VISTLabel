# VISTLabel – Verification and Test Report

**Document purpose:** describes how VISTLabel is tested and reports the results of an independent verification run of version 1.0.0 – automated test suites, static checks, coverage, HTTP security spot checks and a manual end-to-end scenario – with the exact environment and commands used.

| | |
| --- | --- |
| Software | VISTLabel – Video Instance Segmentation, Tracking and Labeling |
| Version | 1.0.0 |

---

## 1. Summary

| Check | Command | Result | Duration |
| --- | --- | --- | --- |
| CPU test suite (unit, integration, fake-backend contract) | `pytest -m "not gpu and not browser"` | **444 passed**, 0 failed, 1 module skipped (no `torch`) | 89.1 s |
| Browser tests, Chromium | `pytest -m browser` | **49 passed**, 0 failed | 105.1 s |
| Browser tests, Firefox | `pytest -m browser --browser firefox` | **49 passed**, 0 failed | 138.1 s |
| GPU tests | `pytest -m gpu` | **12 skipped** ("'gpu' extra not installed"), 0 failed; nothing downloaded | 1.8 s |
| Lint | `ruff check src tests scripts` | **All checks passed** | < 1 s |
| Type check | `mypy src` | **Success: no issues found in 93 source files** | 10.6 s |
| Help/shortcut table consistency | `python scripts/generate_help_reference.py --check` | **OK** (exit 0) | < 1 s |
| Line coverage (CPU suite) | `pytest -m "not gpu and not browser" --cov=vistlabel` | **86 %** (5 997 statements, 854 missed) | 116.2 s |
| Wheel build | `pip wheel --no-deps -w dist .` | `vistlabel-1.0.0-py3-none-any.whl` built | – |
| HTTP security spot checks | `curl` against the running server | 11/11 behaved as specified (§5) | – |
| Manual end-to-end scenario | Playwright-driven browser walkthrough | **All 16 steps as expected** (§6) | 1 min 55 s |

No test failed. Every one of the 505 collected tests was executed in exactly one of the runs above (444 + 49 + 12 = 505); the browser tests were run twice (two browsers). The real model backends (SAM 2.1, SAM 3, YOLO + SAM) were **not** exercised, because the optional GPU stack and the licence-encumbered checkpoints were deliberately not installed (see §7).

The automated checks (test suites, lint, type check, Help consistency, coverage, wheel build) reflect the current source, including the Help pages that show the user guide's screenshots and keep the keyboard-shortcut table only in the user guide. The HTTP security spot checks (§5) and the end-to-end scenario (§6) come from the first verification run of version 1.0.0 and were not repeated after that change.

---

## 2. Test strategy

VISTLabel's test suite (`tests/`, 78 Python files, 11 096 lines) is organised in four layers, selected with pytest markers defined in `pyproject.toml`:

| Layer | Directory | Tests | What it covers | Needs |
| --- | --- | --- | --- | --- |
| Unit | `tests/unit/` (37 files) | 278 (+3 skipped without torch) | Pure functions and single classes: RLE codec, connected components, PNG mask validation, atomic writes, JSONL, migrations, locks, ids, sanitising, path containment, settings, auth (Argon2id, sessions, rate limiter), video upload planning, ffprobe validation, frame index, chunk ranges, job manager, manifests, occlusion/review reasons, frame selection and splits, YOLO/COCO writers, polygons and bridging, validation engine, dataset import parsing, YOLO class mapping, YOLO+SAM orchestration (with a scripted detector), backend configuration/unavailability, checkpoint hashing, help renderer (including image handling) and shortcut table. | – |
| Integration | `tests/integration/` (19 files) | 165 (+2 GPU) | The real FastAPI app through `TestClient` with a temporary workspace and a signed-in throwaway account: projects and upload (real FFmpeg on fixture videos), dataset import round trip, annotation workflow, class operations with undo, multi-part and review, propagation and tracking runs (chunks, pause/resume/cancel, restart recovery, manual-mask protection), a stress test with many objects and repeated interruption, exports (all policies and layouts), auth and CLI, path-traversal regression tests, help pages and images, health/metrics. | FFmpeg |
| Model contract | `tests/model_contract/` (5 files) | 1 (+10 GPU) | A shared backend contract (`backend_contract.py`) run against the fake backend, and – marked `gpu` – against SAM 2.1 (incl. the pipette occlusion fixture) and a real Ultralytics YOLO. | GPU tests: CUDA + checkpoints |
| Browser | `tests/browser/` (10 files) | 49 | Real Chromium/Firefox via Playwright against a live uvicorn server: sign-in form, upload with options, annotation (prompts, brush, right-drag erase, zoom/brush wheel, undo/redo, objects/classes), concept search UI, export dialog and box overlay, help, hide/deactivate/review/delete frame, multi-part and review, playback, propagation incl. notifications, smoke. | Playwright browsers, FFmpeg |

Test data: two small synthetic clips in `tests/fixtures/` – `tiny_test_video.mp4` (64×64, 10 frames, 10 fps) and `pipette_occlusion_video.mp4` (128×64, 20 frames; a "finger" occludes a horizontal "pipette" on frames 7–13, generated by `scripts/generate_pipette_fixture.py`). No real recordings are used.

Test isolation: every test builds its own `AppSettings` with a `tmp_path` workspace and a throwaway account with cheap Argon2 parameters; the free-disk-space floor is set to 0 by an autouse fixture; tests needing FFmpeg skip with a reason when it is missing. No `config/app.yaml` is used.

Requirement coverage: the traceability matrix in [functional-requirements.md §7](functional-requirements.md#7-traceability-matrix) maps every functional requirement group to its tests.

---

## 3. Test environment

| Item | Value |
| --- | --- |
| OS | Microsoft Windows 11 Pro, build 26200 |
| CPU | 2 × Intel Xeon E5-2640 v3 @ 2.60 GHz (8 cores / 16 threads each) |
| RAM | 256 GB |
| GPU | NVIDIA GeForce RTX 2080 Ti, 11 GB, driver 610.88 – present but **not used** (PyTorch not installed) |
| Python | 3.13.13 (`py -3.13`; virtual environment) |
| FFmpeg | gyan.dev "essentials" build 2022-08-13 (`ffmpeg`/`ffprobe` from `d:\apps\ffmpeg`, put on `PATH` for the tests, `runtime.ffmpeg_dir` for the server) |
| Install | fresh virtual environment, `pip install -e ".[dev]"` of the repository (no `gpu` extra) |
| Key packages | fastapi 0.162.5, starlette 1.10.0, uvicorn 0.54.0, pydantic 2.14.3, pydantic-settings 2.16.2, numpy 2.5.4, pycocotools 2.0.11, opencv-python-headless 5.0.0.93, av 19.0.2, pillow 13.0.0, argon2-cffi 25.1.0, httpx 0.28.1 |
| Test tools | pytest 9.2.0, pytest-cov 7.1.0, pytest-playwright 0.10.0, playwright 1.65.1 (Chromium build 1254, Firefox build 1572), ruff 0.17.4, mypy 2.4.1 |

---

## 4. Automated results in detail

### 4.1 Commands

Executed in the repository root, with the virtual environment active and `d:\apps\ffmpeg` prepended to `PATH` (Playwright browsers in their default location):

```bash
python -m pytest -m "not gpu and not browser" -ra --durations=15 --junitxml=pytest-cpu.xml
python -m pytest -m browser --browser chromium -ra --durations=10 --junitxml=pytest-browser-chromium.xml
python -m pytest -m browser --browser firefox  -ra --durations=10 --junitxml=pytest-browser-firefox.xml
python -m pytest -m gpu -ra --junitxml=pytest-gpu.xml
python -m pytest -m "not gpu and not browser" -q --cov=vistlabel --cov-report=term
ruff check src tests scripts
mypy src
python scripts/generate_help_reference.py --check
```

### 4.2 Results per layer

| Run | unit | integration | model_contract | browser | Total passed | Skipped | Failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `not gpu and not browser` | 278 | 165 | 1 | – | 444 | 1 module | 0 |
| `browser` (Chromium) | – | – | – | 49 | 49 | 1 module* | 0 |
| `browser` (Firefox) | – | – | – | 49 | 49 | 1 module* | 0 |
| `gpu` | – | 2 skipped | 10 skipped | – | 0 | 12 + 1 module | 0 |

\* `tests/unit/test_torch_utils.py` is skipped at module level in every run (`pytest.importorskip("torch")`); its 3 tests are not collected without the `gpu` extra.

Slowest CPU tests: `test_project_lock.py::test_zero_stale_after_is_stale_within_the_same_clock_tick` 5.28 s, `test_tracking_stress.py::test_many_objects_long_video_and_repeated_interruption` 4.26 s; all others ≤ 1.71 s. Slowest browser test (Chromium): `test_propagation.py::test_finished_propagation_chimes_and_notifies_when_the_tab_is_unfocused` 2.43 s.

Warnings (not failures): one `StarletteDeprecationWarning` (FastAPI's `TestClient` importing `httpx`) and repeated `DeprecationWarning`s from inside `pycocotools/mask.py` about the NumPy 2 `__array__` copy keyword. Both originate in third-party packages with the versions resolved for this run and do not affect results.

GPU run: all 12 GPU-marked tests skipped with "'gpu' extra not installed (pip install -e '.[gpu]')"; no checkpoint or model was downloaded.

### 4.3 Coverage

Line coverage of `src/vistlabel` by the CPU suite: **86 %** (5 997 statements, 854 not executed). Modules below 70 % are exactly the ones that need the GPU stack or are not reachable without it: `diagnostics/models.py` 0 % (CLI, not called by tests), `segmentation/_stub.py` 0 %, `segmentation/sam3.py` 32 %, `segmentation/torch_utils.py` 33 %, `segmentation/sam21.py` 36 %, `segmentation/yolo_detector.py` 36 %, `segmentation/sessions.py` 64 %. Fully covered modules include `tracking/manifests.py`, `validation/engine.py` and `video/ffmpeg.py`. Browser tests add front-end coverage that is not measured by this figure.

### 4.4 Static checks

- `ruff check src tests scripts` → "All checks passed!"
- `mypy src` → "Success: no issues found in 93 source files" (with the note that two `ignore_missing_imports` overrides, `huggingface_hub.*` and `torchvision.*`, are unused when those packages are absent).
- `python scripts/generate_help_reference.py --check` → exit 0 (the shortcut table in `docs/user-guide.md` matches `static/help/shortcuts.json`).
- Licence headers: `SPDX-License-Identifier: MIT` present in all 176 `.py`, 8 `.js`, 2 `.css` and 11 `.html` files tracked in git.

---

## 5. HTTP security spot checks

Performed with `curl` against the running server (127.0.0.1:9182) after the end-to-end scenario.

| # | Request | Expected (spec) | Observed |
| --- | --- | --- | --- |
| 1 | `GET /` without session | 303 → `/login?next=%2F` | 303 → `/login?next=%2F` |
| 2 | `GET /api/v1/projects` without session | 401 `AUTH_REQUIRED` | 401 `AUTH_REQUIRED` |
| 3 | `GET /api/v1/health`, `/api/v1/version` without session | 200 (public) | 200, `{"version":"1.0.0"}` |
| 4 | `POST /api/v1/auth/login` without `X-Requested-With` | 403 `CSRF_REJECTED` | 403 "missing x-requested-with: VISTLabel header" |
| 5 | Same with header but `Origin: http://evil.example` | 403 `CSRF_REJECTED` | 403 "cross-origin request rejected" |
| 6 | Correct sign-in | 200; cookie `HttpOnly`, `SameSite=Lax`, Max-Age 7200 | `vistlabel_session=…; HttpOnly; Max-Age=7200; Path=/; SameSite=lax` |
| 7 | `GET /api/v1/models`, `/metrics` signed in | backends with reasons; counters | sam21/sam3/yolo_sam unavailable "disabled in configuration (…enabled: false)", fake available; 2 projects, 21 annotated frames, 2 succeeded jobs |
| 8 | `DELETE /api/v1/projects/%2E%2E` signed in | no deletion, 404 | 404 "Not Found"; workspace intact |
| 9 | `POST /api/v1/auth/logout` | 200, session revoked | 200 `authenticated: false` |
| 10 | 11 failed sign-ins | attempts 1–10 → 401, then 429 | attempts 1–10: 401; attempt 11: 429 |
| 11 | Correct password while blocked | 429 with `Retry-After` | 429, `retry-after: 297` |

CLI checks: `vistlabel version` → `1.0.0`; `vistlabel --version` → `vistlabel 1.0.0`; `auth status` before account → exit 1 with instructions; `auth set-password --username demo` with `VISTLABEL_PASSWORD` → exit 0, account written; `auth status` → `username: demo`, `algorithm: argon2id`, `active sessions: 0`; `scripts/create_demo_project.py --ffmpeg-dir …` → demo project created. Diagnostics: `diagnostics.models --backend fake` → `ok: true` in 0.039 s, exit 0; `--backend sam21|sam3|yolo_sam` → `available: false` with the reason, exit 1, as documented.

---

## 6. Manual end-to-end verification scenario

### 6.1 Set-up

- Server: `vistlabel serve --host 127.0.0.1 --port 9182 --config <work>/config/app.yaml`, with `workspace.root` in the work area, `runtime.default_backend: fake`, `runtime.ffmpeg_dir: d:\apps\ffmpeg`, and two settings changed from the defaults to make the 20-frame clip demonstrate more behaviour: `tracking.default_chunk_frames: 5` (several checkpoints) and `tracking.review_sampling_stride: 5` (frames 5, 10, 15 flagged for review).
- Account `demo`, created with `VISTLABEL_PASSWORD=… vistlabel auth set-password --username demo`.
- Second project `demo_isolator` created beforehand with `scripts/create_demo_project.py`.
- Input video: `pipette_occlusion_1536x768.mp4` – the repository fixture `pipette_occlusion_video.mp4` upscaled 12× per side (128×64 → 1536×768, nearest-neighbour, H.264) with FFmpeg so that it is visible on screen and so that the upload's downscale option can be demonstrated. 20 frames, 10 fps.
- Driver: Playwright, Chromium, viewport 1600×1000; every step is performed through the real UI. The screenshots in [images/](images/) were taken during this run.

**Backend caveat.** All model operations used the `fake` backend: a point prompt yields a small disk (radius 4 px) around the click and propagation shifts the mask 2 px per frame. The scenario therefore verifies the application workflow, persistence and export – not segmentation quality.

### 6.2 Steps, expected and observed results

| # | Step | Expected | Observed | Result |
| --- | --- | --- | --- | --- |
| 1 | Open `/` | Redirect to sign-in form | Sign-in form shown ([01-sign-in.png](images/01-sign-in.png)) | Pass |
| 2 | Sign in as `demo` | Projects page with existing project card | Projects page with `demo_isolator` card (thumbnail, 64×64, 10 FPS, 10 frames) ([02](images/02-projects.png)) | Pass |
| 3 | Create project "Pipette occlusion demo" from the MP4, Resolution *Custom* 768×384, keep every frame | Upload, single re-encode to 768×384, 20 frames, card status *New* | Card "768×384 · 10.0 FPS · 20 frames"; `events.jsonl`: `video_transcoded` (1536×768 → 768×384), `video_finalized` (`frame_count` 20); `project.json` keeps `original` 1536×768 ([03](images/03-create-project.png), [04](images/04-projects-after-upload.png)) | Pass |
| 4 | Open the project | Workspace, status line "Ready", Fake Backend selected | As expected ([05](images/05-workspace-empty.png)) | Pass |
| 5 | Create object `pipette` | One object, class registered | Object list shows `pipette`; `labels.json` contains `pipette` | Pass |
| 6 | Positive point on the pipette | Candidate shown, **Accept candidate** enabled, nothing saved yet | Candidate disk shown ([06](images/06-point-prompt.png)) | Pass |
| 7 | Accept candidate | Mask saved on frame 0 | Status line "Saved" | Pass |
| 8 | Brush stroke (radius 24) along the pipette | Mask extended, saved on release, `manually_modified` | Frame 0: `source: manual_brush`, `manually_modified: true` ([07](images/07-brush.png)) | Pass |
| 9 | Propagate forward 0 → 19 | Background job, chunks of 5, completion message, hand-edited frame 0 not overwritten | "Propagation complete"; run manifest: `status: completed`, `chunk_frames: 5`, `last_completed_frame: 19`, `manual_masks_protected_count: 1`; frames 1–19 `source: fake_propagation` ([08](images/08-propagation.png)) | Pass |
| 10 | Play to the end, go to frame 10, erase the occluded middle of the pipette | One object with two visible parts | Object state `partially_occluded`; frame 10: `connected_component_count: 2`, `source: manual_eraser` | Pass |
| 11 | Create object `finger`, paint it, select pipette, Occluder = finger, **Mark** | Occlusion edge stored, no mask change | "finger in front of pipette"; `occlusion_edges` with `relation: in_front_of` in frame 10 ([09](images/09-occlusion-multipart.png)) | Pass |
| 12 | **Runs** dialog | Run listed with state | "forward 0–19 — completed · fake · 1 object(s) · committed up to frame 19 · 1 manually edited mask(s) left untouched" ([10](images/10-tracking-runs.png)) | Pass |
| 13 | **Queue** tab, open entry, **Solved**; Review tab, **Mark all annotated frames reviewed** | Frames 5 and 15 flagged by sampling; frame 10 flagged; queue emptied; all annotated frames reviewed | Queue (3): frame 5 `manual_review_sampling`, frame 10 `mask_touches_frame_border` (the painted finger reaches the top and bottom edge), frame 15 `manual_review_sampling` ([11](images/11-review-queue.png)); after Solved + Mark all: all 20 frames `reviewed` ([12](images/12-review-tab.png)) | Pass |
| 14 | **Splits**: 0–14 train, 15–19 val, **Save splits** | Two ranges saved | "Saved 2 range(s)"; `project.json` `split_assignments` as entered ([13](images/13-splits.png)) | Pass |
| 15 | **Export**: reviewed frames, segmentation, Ultralytics YOLO, bridge multi-part, exact COCO | Job succeeds; 20 frames, 15 train / 5 val; frame 10 bridged; leakage warning at 14/15 | "Reviewed frames only (20)"; history "YOLO masks, Reviewed frames only — succeeded · 20 frames (15 train / 5 val)"; validation report: `multipart_instances` = frame 10, `component_count: 2`, `bridge_pixel_count: 162`; `frame_leakage_risks`: "frame 14 (train) and frame 15 (val) are adjacent …"; objects per class pipette 20, finger 1 ([14](images/14-export-dialog.png), [15](images/15-export-done.png)) | Pass |
| 16 | Download and inspect `dataset.zip`; System status; Help | 45 files: `images/{train,val}`, `labels/{train,val}`, `dataset.yaml` (`0: pipette`, `1: finger`), `annotations/instances_{train,val}.json`, manifest and report; health and backend reasons shown; user guide rendered | As expected ([16](images/16-exported-files.png), [17](images/17-system-status.png), [18](images/18-help.png)) | Pass |

---

## 7. Limitations of this verification

- **Real model backends not run.** SAM 2.1, SAM 3 and YOLO + SAM require the `gpu` extra, a CUDA build of PyTorch and checkpoints with their own licences (SAM 3 gated). These were intentionally not installed; the 12 GPU-marked tests skipped. Their configuration/unavailability logic and the YOLO+SAM orchestration are covered by CPU tests with test doubles. Segmentation quality and GPU performance are therefore **not** assessed here.
- **Single platform.** Only Windows 11 / Python 3.13 was used. The README states verification on Python 3.12 and on macOS (Apple silicon, fake backend) by the developers; this was not repeated.
- **Fixture-scale data.** All videos are tiny synthetic clips (≤ 20 frames). No load or long-video test beyond the repository's own stress test (`test_tracking_stress.py`) was performed.
- **Dataset import** was verified only by the automated tests (`test_dataset_import_api.py`, including an export → import round trip), not in the manual scenario.

## 8. Observations (no code was changed)

1. `python -m vistlabel.diagnostics.system` builds `AppSettings()` directly instead of `load_settings()`, so it ignores `config/app.yaml` and only honours `VISTLABEL_*` environment variables. Observed: with `runtime.ffmpeg_dir` set only in the config file it reported FFmpeg "missing (checked PATH)"; with `VISTLABEL_RUNTIME__FFMPEG_DIR` set it reported "ok". The web endpoint `/api/v1/health` uses the loaded settings and was correct.
2. The built wheel (`vistlabel-1.0.0-py3-none-any.whl`) contains no `docs/` directory. `api/help.py` looks for `user-guide.md` in a packaged `vistlabel/docs` or the repository's `docs/`, so in a non-editable wheel installation the in-app user guide would be reported unavailable (by code inspection; not run). The README's installation uses an editable install, where it works.
3. Masks written by propagation carry `source` and `confidence` but no `backend_provenance`; the validation report's `model_provenance_summary` therefore stays empty for propagated-only data (observed `{}` in the demo export).
4. Configuration keys `upload.allowed_extensions`, `tracking.checkpoint_after_chunk`, `models.sam3.require_startup_diagnostic` and `models.yolo_sam.detection_stride_frames` are defined but not referenced by the code (see [functional-requirements.md §6](functional-requirements.md#6-constraints-and-assumptions)).
5. Third-party deprecation warnings (pycocotools with NumPy 2, Starlette `TestClient`/httpx) appear in the test output; they do not affect behaviour with the versions installed for this run but may need attention when those packages change.
