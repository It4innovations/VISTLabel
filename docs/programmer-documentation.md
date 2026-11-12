# VISTLabel – Programmer Documentation

**Document purpose:** guide for developers who build, test, maintain or extend VISTLabel: repository layout, development setup, build and installation, coding conventions, how to run the checks, extension points, command-line reference, and annotated excerpts of the key source code.

| | |
| --- | --- |
| Software | VISTLabel – Video Instance Segmentation, Tracking and Labeling |
| Version | 1.0.0 |
| Repository | <https://github.com/It4innovations/VISTLabel> |
| Licence | MIT, © 2026 IT4Innovations, VSB – Technical University of Ostrava |

For the architecture and data formats see [technical-documentation.md](technical-documentation.md); for end-user operation see [user-guide.md](user-guide.md).

---

## 1. Repository layout

```
VISTLabel/
├─ pyproject.toml              # package metadata, version (1.0.0), extras, ruff/mypy/pytest config
├─ requirements-lock.txt       # exact versions tested together on the Windows workstation (incl. torch+cu130)
├─ requirements-lumi.txt, requirements-p11.txt   # pinned sets for other machines
├─ LICENSE                     # MIT
├─ README.md                   # installation, configuration, running, checks
├─ run-vistlabel.sh            # start pinned to one GPU (Git Bash / Linux), needs nvidia-smi
├─ config/
│  ├─ app.example.yaml         # copy to config/app.yaml (git-ignored)
│  └─ yolo_class_mapping.example.json
├─ docs/user-guide.md          # end-user guide, also served at /help/user-guide
├─ models/README.md            # where checkpoints go (never committed)
├─ scripts/
│  ├─ download_models.py       # fetch SAM 2.1 / SAM 3 snapshots at pinned HF revisions
│  ├─ create_demo_project.py   # demo project from the tiny fixture video (no server needed)
│  ├─ generate_help_reference.py   # regenerate/check the shortcut table in docs/user-guide.md
│  ├─ generate_pipette_fixture.py  # regenerate tests/fixtures/pipette_occlusion_video.mp4
│  ├─ benchmark_sam21.py       # SAM 2.1 timing on a video
│  └─ win32/run-vistlabel.ps1  # PowerShell counterpart of run-vistlabel.sh
├─ src/vistlabel/              # the package (see module map in technical-documentation.md §3)
│  ├─ main.py  settings.py  logging_config.py
│  ├─ api/ auth/ annotations/ diagnostics/ domain/ exports/ help/ imports/
│  ├─ jobs/ persistence/ projects/ segmentation/ tracking/ validation/ video/
│  ├─ templates/               # Jinja2 HTML
│  └─ static/                  # css/, js/, help/shortcuts.json, img/it4i-logo.png
└─ tests/
   ├─ conftest.py              # throwaway account, signed-in clients, ffmpeg fixture
   ├─ fixtures/                # tiny_test_video.mp4 (64×64, 10 frames), pipette_occlusion_video.mp4 (128×64, 20 frames)
   ├─ unit/ (41 files)  integration/ (19)  model_contract/ (5)  browser/ (10)
```

Size: 93 Python modules (12 388 lines), 8 JavaScript modules (4 189 lines), 2 CSS files, 11 templates; 78 test files (11 096 lines, 505 tests).

---

## 2. Development setup

Always use a virtual environment. The commands below are the ones used for the verification in [verification.md](verification.md) (Windows 11, Python 3.13); the README gives the equivalent Linux and macOS commands.

```powershell
git clone https://github.com/It4innovations/VISTLabel.git vistlabel
cd vistlabel
py -3.13 -m venv .venv            # 3.11-3.13 are supported; 3.12 is the README default
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"           # pytest, pytest-cov, pytest-playwright, httpx, ruff, mypy, type stubs
playwright install chromium firefox   # browsers for the browser tests
```

FFmpeg (`ffmpeg` and `ffprobe`) must be on `PATH`, or its directory set as `runtime.ffmpeg_dir` (server) / `VISTLABEL_RUNTIME__FFMPEG_DIR` (tests) / `--ffmpeg-dir` (`scripts/create_demo_project.py`).

Optional GPU stack: install the PyTorch build matching the NVIDIA driver first (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`, or `cu126` / `cu121`), then `pip install -e ".[gpu]"`, then download checkpoints with `scripts/download_models.py` (see README, section "Optional: model backends").

Run a development server:

```bash
cp config/app.example.yaml config/app.yaml
vistlabel auth set-password --username dev        # or VISTLABEL_PASSWORD=... without a terminal
python scripts/create_demo_project.py --workspace ./workspace
vistlabel serve --host 127.0.0.1 --port 9080
```

Static files and templates are read from the source tree in an editable install, so front-end changes need only a browser reload (asset URLs carry a version query, `asset_version()` in `api/pages.py`).

---

## 3. Build and installation

- **Build backend:** hatchling (`[build-system] requires = ["hatchling"]`); the wheel contains `src/vistlabel` including `templates/` and `static/`.
- **Build a wheel:** `pip wheel --no-deps -w dist .` → `dist/vistlabel-1.0.0-py3-none-any.whl` (verified: 286 897 bytes, 121 entries). Note: the wheel does not contain `docs/`; the in-app user guide (`/help/user-guide`) and its images are found only in a source checkout or editable install (`api/help.py`, `docs_directories()`).
- **Entry point:** `[project.scripts] vistlabel = "vistlabel.main:cli"`.
- **Version:** defined only in `pyproject.toml`, read at runtime with `importlib.metadata.version("vistlabel")` (`src/vistlabel/__init__.py`); reinstall after changing it.
- **Dependencies:** base – fastapi, uvicorn[standard], pydantic, pydantic-settings, pyyaml, numpy, pycocotools, opencv-python-headless, av, pillow, python-multipart, jinja2, argon2-cffi; `gpu` extra – torch, torchvision, transformers ≥ 5.14, huggingface_hub, accelerate, ultralytics; `dev` extra – pytest, pytest-cov, pytest-playwright, httpx, ruff, mypy, types-PyYAML, types-pycocotools.

---

## 4. Conventions

| Topic | Convention |
| --- | --- |
| Formatting / lint | `ruff` with line length 100, target `py311`, rule sets `E`, `F`, `I` (import order), `UP` (pyupgrade), `B` (bugbear). |
| Typing | `mypy src` with `disallow_untyped_defs`, `no_implicit_optional`, `warn_unused_ignores`, `warn_redundant_casts`; optional heavy modules (`torch`, `transformers`, `ultralytics`, `cv2`, …) are `ignore_missing_imports`. Every module starts with `from __future__ import annotations`. |
| Licence headers | Every source file starts with `Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava` and `SPDX-License-Identifier: MIT` in the file type's comment syntax (verified present in all 176 `.py`, 8 `.js`, 2 `.css` and 11 `.html` files tracked in git). |
| Schemas | Persistent data are Pydantic models with `schema_version`; new fields are additive with defaults; incompatible changes need a registered migration (`persistence/migrations.py`). |
| Paths | Never build workspace paths by hand – use `persistence/paths.py`. Never use client-supplied names in paths; validate ids and use `is_safe_child_path` before deleting. |
| Writes | Authoritative JSON only via `save_model` / `atomic_write_text`; logs only via `append_jsonl`. |
| Concurrency | Read-modify-write through revision-checked helpers (`ProjectStore.update`, `save_frame_annotation`); long work through `JobManager`. |
| Errors | Domain errors are typed exceptions; routers map them to HTTP codes. Backends raise `BackendCapabilityError` / `BackendUnavailableError`, never silently degrade. |
| Front end | Plain ES modules, no build step, no external assets; all API calls through `apiFetch()` (`static/js/session.js`), which adds `X-Requested-With: VISTLabel`. |
| Shortcuts | Defined once in `static/help/shortcuts.json`; run `python scripts/generate_help_reference.py` after editing it (`--check` in CI). |
| Commits | Development used AI agent harnesses for selected tasks (README, "Development"). |

---

## 5. Running the checks

```bash
ruff check src tests scripts
mypy src
pytest -m "not gpu and not browser"                 # unit + integration + fake-backend contract
pytest -m browser                                   # Playwright, Chromium (default)
pytest -m browser --browser firefox
pytest -m gpu                                       # real models; skips without gpu extra/checkpoints
pytest -m "not gpu and not browser" --cov=vistlabel # coverage
python scripts/generate_help_reference.py --check
```

Markers (`pyproject.toml`): `gpu` (real CUDA GPU and checkpoints), `browser` (Playwright), `signed_out` (browser test starting without a session). Tests build their own `AppSettings` with a temporary workspace and a throwaway account (cheap Argon2 parameters for speed), so no `config/app.yaml` is needed. The autouse fixture `_no_free_disk_space_floor` sets `VISTLABEL_UPLOAD__MINIMUM_FREE_DISK_SPACE_GB=0`. Tests that need FFmpeg skip with a reason when it is missing. Results of running all of these are in [verification.md](verification.md).

---

## 6. Extension points

### 6.1 Adding a model backend

1. Create `src/vistlabel/segmentation/<name>.py` with a class implementing every method of `VideoSegmentationBackend` ([excerpt 6](#excerpt-6--backend-interface)). Return a `BackendDescriptor` whose `capabilities` lists only what the backend really supports, and raise `BackendCapabilityError` for anything else.
2. Report unavailability instead of failing at import: import heavy libraries lazily (see `torch_utils.py`) and compute an `unavailable_reason` (disabled, checkpoint missing, dependency missing). The `UnavailableBackendMixin` in `_stub.py` shows the pattern of raising `BackendUnavailableError` with that reason from every operation.
3. Add a settings model to `settings.py` (`ModelsSettings`) with `enabled: bool = False` and the checkpoint path, and document it in `config/app.example.yaml`.
4. Register the instance in `segmentation/registry.build_registry()`; dictionary order is the UI order.
5. Fill `backend_provenance` fields: `model_name`, `checkpoint_sha256` (use `checkpoints.hash_checkpoint_file`), `code_revision`.
6. Tests: reuse the shared contract in `tests/model_contract/backend_contract.py` (as `test_fake_backend_contract.py` and the GPU-marked `test_sam21_backend_contract.py` do); add CPU-only tests for the descriptor/unavailability logic (cf. `tests/unit/test_sam3_backend_config.py`); mark tests that need a GPU with `@pytest.mark.gpu`.
7. Optionally add the backend to `python -m vistlabel.diagnostics.models --backend <name>` (the command uses the registry, so registration is usually enough).

The UI reads capabilities from `GET /api/v1/models` and enables or disables point prompts, text search, propagation and **Auto** accordingly, so no front-end change is needed for a backend that fits the existing capabilities.

### 6.2 Adding an export format or layout

1. Extend the `DatasetLayout` (or `AnnotationType`) `Literal` in `domain/exports.py` and the matching field in `ExportSettings` (`settings.py`).
2. Implement a writer next to `exports/yolo_writer.py` / `exports/coco_writer.py`. Writers receive frame annotations whose `label_id` has already been renumbered to the class position (`_number_classes_by_position`), so they emit dense class ids 0..N-1. Never modify the canonical mask; record any lossy conversion in the `ValidationReport`.
3. Branch on the new layout in `exports/service._run_export_unsafe()` (image directory via `_image_dir`, label/annotation files, `expected_files` for the missing-file check).
4. Add the option to `templates/_export_dialog.html` and `static/js/exports.js`.
5. Add unit tests for the writer and an integration test in `tests/integration/test_exports_api.py`; consider a round-trip test through the dataset import (`tests/integration/test_dataset_import_api.py::test_vistlabel_export_imports_back_with_exact_masks` is the model).

### 6.3 Other extension points

- **Review reasons:** add a value to `ReviewReason` (`domain/frames.py`, additive) and a pure function in `annotations/occlusion.py`; call it from `_run_chunked_propagation`.
- **Schema change:** bump `CURRENT_SCHEMA_VERSIONS[kind]` and `register_migration(kind, old, fn)` ([excerpt 8](#excerpt-8--schema-migration-interface)).
- **Dataset import format:** add a reader producing `imports/dataset.ParsedDataset` and select it in `imports/service.py`.
- **Help pages:** add a `HelpPage` to `HELP_INDEX` in `api/help.py` and the Markdown file to `docs/`.

---

## 7. Command-line reference

### 7.1 `vistlabel`

| Command | Options | Effect | Exit code |
| --- | --- | --- | --- |
| `vistlabel serve` | `--host` (default `127.0.0.1`), `--port` (default `9080`), `--config PATH` | Load settings, build the app, run uvicorn. | 0 on shutdown |
| `vistlabel version` / `vistlabel --version` | – | Print `1.0.0` / `vistlabel 1.0.0`. | 0 |
| `vistlabel auth set-password` | `--username` (default `vistlabel`), `--config` | Create the account or change its name and password; prompts twice, or reads `VISTLABEL_PASSWORD` without a terminal; revokes all sessions. | 0 ok, 2 weak password / mismatch / no terminal |
| `vistlabel auth status` | `--config` | Print user name, algorithm, `updated_at`, active session count. | 0, or 1 if no account |
| `vistlabel auth revoke-sessions` | `--config` | Sign out every browser session. | 0 |

### 7.2 Diagnostics and scripts

| Command | Effect |
| --- | --- |
| `python -m vistlabel.diagnostics.system` | Health report as JSON (workspace, FFmpeg, Python, CUDA). Builds `AppSettings()` directly, so it honours `VISTLABEL_*` environment variables but **not** `config/app.yaml` (observed during verification). Exit 1 only on status `error`. |
| `python -m vistlabel.diagnostics.models --backend <id> [--config PATH]` | Load the backend, run a minimal image prompt and video propagation, report device, dtype, peak VRAM, checkpoint hash, duration. Exit 0 usable, 1 unavailable with reason. |
| `python scripts/download_models.py --model sam21-base-plus|sam21-large|sam3 [--dest ./models]` | Download a pinned Hugging Face snapshot (`HF_TOKEN` needed for the gated `sam3`). |
| `python scripts/create_demo_project.py [--workspace DIR] [--video MP4] [--ffmpeg-dir DIR]` | Create the `demo_isolator` project with one `pipette` object and a fake-backend mask on frame 0, directly through the service layer. FFmpeg only from `PATH` or `--ffmpeg-dir`. |
| `python scripts/generate_help_reference.py [--check]` | Rewrite (or verify) the shortcut table between `<!-- shortcuts:begin/end -->` in `docs/user-guide.md`. |
| `python scripts/generate_pipette_fixture.py` | Regenerate the synthetic occlusion fixture (finger occludes the pipette on frames 7–13). |
| `python scripts/benchmark_sam21.py --video MP4` | Time SAM 2.1 on a video (requires the GPU stack). |
| `./run-vistlabel.sh --gpu N [serve args]`, `scripts\win32\run-vistlabel.ps1 -GpuIndex N` | Start the server pinned to one GPU via `CUDA_VISIBLE_DEVICES`; requires `nvidia-smi`. |

---

## 8. Representative source code excerpts

The excerpts are copied verbatim from version 1.0.0; line numbers refer to the file named in each heading.

### Excerpt 1 – Application factory

`src/vistlabel/main.py`, lines 57–100

```python
def create_app(settings: AppSettings | None = None) -> FastAPI:
    settings = settings or load_settings()
    configure_logging(settings.logging)

    app = FastAPI(title="VISTLabel", version=__version__)
    app.state.started_at_monotonic = time.monotonic()
    app.state.settings = settings
    app.state.registry = build_registry(settings)
    app.state.accounts = AccountStore(settings.workspace.root)
    app.state.sessions = SessionStore(
        settings.workspace.root, timeout_minutes=settings.security.session_timeout_minutes
    )
    app.state.login_limiter = LoginRateLimiter(
        limit=settings.security.login_rate_limit,
        window_seconds=settings.security.login_rate_window_seconds,
    )
    app.state.project_store = ProjectStore(settings.workspace.root)
    app.state.project_store.recover_incomplete_projects()
    app.state.project_store.recover_interrupted_tracking_runs()
    app.state.backend_sessions = BackendSessionManager(
        max_sessions=settings.tracking.max_open_backend_sessions
    )
    app.state.job_manager = JobManager(
        settings.workspace.root, max_workers=settings.runtime.maximum_concurrent_gpu_jobs
    )
    app.state.job_manager.recover_interrupted_jobs()
    app.add_middleware(AuthMiddleware, sessions=app.state.sessions, accounts=app.state.accounts)
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(health_router)
    app.include_router(auth_router)
    # Before projects_router, so /projects/import-dataset is never read as
    # a project_id.
    app.include_router(dataset_imports_router)
    app.include_router(projects_router)
    app.include_router(frame_annotations_router)
    app.include_router(objects_router)
    app.include_router(segmentation_router)
    app.include_router(propagation_router)
    app.include_router(review_queue_router)
    app.include_router(auto_preannotate_router)
    app.include_router(exports_router)
    app.include_router(help_router)
    app.include_router(pages_router)
    return app
```

**Explanation.** All long-lived services are created once and kept on `app.state`, which makes the app testable: the test suite calls `create_app(AppSettings(workspace=…))` with a temporary workspace. The three `recover_*` calls implement crash recovery at every start (staging directories removed, running propagation runs → `interrupted`, pending/running jobs → `failed`). Router order matters where paths overlap (`/projects/import-dataset` vs. `/projects/{project_id}`).

### Excerpt 2 – Authentication middleware and CSRF guard

`src/vistlabel/auth/middleware.py`, lines 65–107

```python
def csrf_violation(request: Request) -> str | None:
    if request.method not in UNSAFE_METHODS:
        return None
    if request.headers.get(CSRF_HEADER) != CSRF_HEADER_VALUE:
        return f"missing {CSRF_HEADER}: {CSRF_HEADER_VALUE} header"
    if not _same_origin(request):
        return "cross-origin request rejected"
    return None


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, *, sessions: SessionStore, accounts: AccountStore) -> None:
        super().__init__(app)
        self._sessions = sessions
        self._accounts = accounts

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        is_api = path.startswith("/api/")
        session = self._sessions.resolve(request.cookies.get(SESSION_COOKIE_NAME))
        request.state.session = session

        if is_api:
            violation = csrf_violation(request)
            if violation is not None:
                return JSONResponse(
                    {"detail": violation, "code": "CSRF_REJECTED"}, status_code=403
                )

        if _is_public(path) or session is not None:
            return await call_next(request)

        if is_api:
            return JSONResponse(
                {"detail": "authentication required: sign in at /login", "code": "AUTH_REQUIRED"},
                status_code=401,
            )
        if not self._accounts.is_configured():
            return RedirectResponse("/setup", status_code=303)
        target = path
        if request.url.query:
            target += "?" + request.url.query
        return RedirectResponse(f"/login?next={quote(target, safe='')}", status_code=303)
```

**Explanation.** One middleware enforces both authentication and the CSRF policy. The CSRF check runs *before* the public-path check, so even the public `POST /api/v1/auth/login` needs the custom header and same origin (verified: HTTP 403 without it). API clients get JSON errors with a machine-readable `code`; browsers get redirects that preserve the requested page.

### Excerpt 3 – Server-side session store

`src/vistlabel/auth/sessions.py`, lines 92–128

```python
    def create(self, username: str) -> tuple[str, Session]:
        token = secrets.token_urlsafe(32)
        now = _utc_now()
        session = Session(
            session_id=secrets.token_hex(6),
            token_hash=_hash_token(token),
            username=username,
            created_at=now.isoformat(timespec="milliseconds"),
            last_seen_at=now.isoformat(timespec="milliseconds"),
            expires_at=(now + self._timeout).isoformat(timespec="milliseconds"),
        )
        with self._lock:
            self._sessions[session.token_hash] = session
            self._save()
        return token, session

    def resolve(self, token: str | None) -> Session | None:
        if not token:
            return None
        token_hash = _hash_token(token)
        with self._lock:
            session = self._sessions.get(token_hash)
            if session is None:
                return None
            now = _utc_now()
            if datetime.fromisoformat(session.expires_at) <= now:
                del self._sessions[token_hash]
                self._save()
                return None
            # Sliding inactivity timeout.
            session.last_seen_at = now.isoformat(timespec="milliseconds")
            session.expires_at = (now + self._timeout).isoformat(timespec="milliseconds")
            last = self._last_persisted_touch.get(token_hash, 0.0)
            if time.monotonic() - last > _TOUCH_PERSIST_INTERVAL_SECONDS:
                self._last_persisted_touch[token_hash] = time.monotonic()
                self._save()
            return session
```

**Explanation.** The cookie carries only a random token; the server keys sessions by its SHA-256, so a leaked `sessions.json` does not reveal usable tokens. `resolve()` implements the sliding timeout in memory on every request and persists it at most once a minute per session (`_TOUCH_PERSIST_INTERVAL_SECONDS = 60`), because the frame-by-frame UI issues many requests. Sessions are loaded from disk at start, so a restart does not sign users out.

### Excerpt 4 – Atomic JSON write

`src/vistlabel/persistence/atomic.py`, lines 59–95

```python
def _replace_with_retry(tmp_name: str, path: Path) -> None:
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(tmp_name, path)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_RETRY_DELAY_SECONDS)


def atomic_write_text(path: Path, text: str, *, fsync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        _replace_with_retry(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def atomic_write_json(path: Path, data: dict, *, fsync: bool = True) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=False), fsync=fsync)


def save_model(path: Path, model: BaseModel, *, fsync: bool = True) -> None:
    atomic_write_text(path, model.model_dump_json(indent=2), fsync=fsync)


def load_model(path: Path, model_cls: type[T]) -> T:
    return model_cls.model_validate_json(_read_text_with_retry(path))
```

**Explanation.** The temporary file is created in the destination directory so that `os.replace` is an atomic rename on the same volume; a crash leaves either the old or the new complete file, never a partial one. `PermissionError` retries (5 × 10 ms) absorb a Windows-specific race between a reader and `os.replace` that frequent job/manifest polling made realistic. Every persistent Pydantic document goes through `save_model`.

### Excerpt 5 – Mask encoding (canonical RLE and browser PNG)

`src/vistlabel/persistence/rle.py`, lines 23–40

```python
def encode_mask(mask: np.ndarray) -> RleMask:
    """Encode a 2D boolean/uint8 visible mask as canonical COCO RLE."""
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    encoded = _mask_utils.encode(fortran_mask)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    height, width = encoded["size"]
    return RleMask(size=(height, width), counts=counts)


def decode_mask(rle: RleMask) -> np.ndarray:
    """Decode canonical COCO RLE back into a 2D boolean visible mask."""
    coco_rle = {"size": list(rle.size), "counts": rle.counts.encode("ascii")}
    decoded = _mask_utils.decode(coco_rle)  # type: ignore[arg-type]
    return decoded.astype(bool)
```

`src/vistlabel/annotations/png_mask.py`, lines 34–52

```python
def decode_mask_png_base64(data: str, *, expected_shape: tuple[int, int]) -> np.ndarray:
    try:
        raw = base64.b64decode(data, validate=True)
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:  # noqa: BLE001 - any malformed input is a client error
        raise MaskDecodeError(f"could not decode mask PNG: {exc}") from exc

    array = np.array(image.convert("L"))
    if array.shape != expected_shape:
        raise MaskDecodeError(
            f"mask shape {array.shape} does not match expected frame shape {expected_shape}"
        )
    unique_values = set(np.unique(array).tolist())
    if not unique_values <= {0, 255}:
        raise MaskDecodeError(
            f"mask must be strictly binary (0/255 grayscale), got values {sorted(unique_values)}"
        )
    return array > 0
```

**Explanation.** pycocotools expects column-major (Fortran-order) `uint8` arrays; the resulting compressed `counts` string is stored as-is, which is exactly COCO's compressed-RLE `segmentation` – the COCO writer copies it verbatim. Masks from the browser are accepted only if they decode, match the frame size and are strictly binary; otherwise the `PATCH` fails with HTTP 422 before anything is written.

### Excerpt 6 – Backend interface

`src/vistlabel/segmentation/base.py`, lines 129–167

```python
class VideoSegmentationBackend(Protocol):
    def descriptor(self) -> BackendDescriptor: ...

    def open_video(self, project_id: str, video_path: Path) -> str: ...

    def close_video(self, session_id: str) -> None: ...

    def segment_interactive(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        points: list[PointPrompt],
        box_xyxy: tuple[float, float, float, float] | None,
        previous_mask: np.ndarray | None,
    ) -> MaskPrediction: ...

    def propagate(
        self,
        session_id: str,
        object_ids: list[str],
        start_frame: int,
        end_frame: int,
        direction: str,
    ) -> Iterable[MaskPrediction]: ...

    def replace_object_mask(
        self,
        session_id: str,
        frame_index: int,
        object_id: str,
        visible_mask: np.ndarray,
    ) -> MaskPrediction: ...

    def forget_object(self, session_id: str, object_id: str) -> None:
        """Drop everything the session remembers about ``object_id``, so the
        next :meth:`replace_object_mask` is its only seed. Other objects are
        untouched."""
        ...
```

(The protocol continues with `auto_preannotate`, `find_matching_objects` and `find_and_track_matching_objects`, lines 169–202.)

**Explanation.** A structural `Protocol`: any class with these methods is a backend – `fake.py` ends with `_conforms: type[VideoSegmentationBackend] = FakeBackend` so mypy checks conformance. The API layer deals only in `MaskPrediction`/`ConceptProposal` and NumPy masks; model-specific state stays inside the backend session. `replace_object_mask` and `forget_object` exist so the server can re-seed a session from the canonical saved masks (after a restart or a hand correction).

### Excerpt 7 – Chunked propagation with checkpoints

`src/vistlabel/api/propagation.py`, lines 79–104 (chunk generator)

```python
def _chunk_ranges(
    start_frame: int, end_frame: int, direction: str, chunk_frames: int
) -> Iterator[tuple[int, int]]:
    """Consecutive chunks *share* their boundary frame -- chunk N+1 starts
    exactly where chunk N ended, not one step past it. ``backend.propagate``
    requires its ``start_frame`` to already have a seeded mask in the
    backend's session (``segment_interactive`` must be called first); the
    only frame guaranteed to satisfy that for chunk N+1 is
    chunk N's own last output. ``chunk_frames`` counts new frames of
    progress per chunk, not counting the reused seed."""
    advance = max(1, chunk_frames)
    current = start_frame
    if current == end_frame:
        yield current, end_frame
        return
    if direction == "forward":
        while current < end_frame:
            chunk_end = min(current + advance, end_frame)
            yield current, chunk_end
            current = chunk_end
    else:
        while current > end_frame:
            chunk_end = max(current - advance, end_frame)
            yield current, chunk_end
            current = chunk_end
```

`src/vistlabel/api/propagation.py`, lines 137–166 and 216–238 (run loop, abridged between the two ranges)

```python
    try:
        for chunk_start, chunk_end in _chunk_ranges(
            resume_start_frame, end_frame, direction, chunk_frames
        ):
            context.raise_if_paused()
            context.raise_if_cancelled()
            previous_by_object: dict[str, PreviousFrameContext] = {}

            for prediction in backend.propagate(
                session_id, object_ids, chunk_start, chunk_end, direction
            ):
                context.raise_if_cancelled()
                label_id = label_by_object[prediction.object_id]
                previous = previous_by_object.get(prediction.object_id)

                existing = get_frame_annotation(
                    pdir, prediction.frame_index, timestamps[prediction.frame_index]
                )
                existing_entry = next(
                    (o for o in existing.objects if o.object_id == prediction.object_id), None
                )
                # Deactivating an object on a frame is a manual edit too.
                inactive_here = any(
                    o.object_id == prediction.object_id for o in existing.inactive_objects
                )
                manual_here = existing_entry is not None and existing_entry.manually_modified
                if (inactive_here or manual_here) and not overwrite_manual_masks:
                    manual_protected += 1
                    completed_frames += 1
                    continue
```

```python
            # Chunk fully processed -- this is the checkpoint: results are
            # persisted incrementally by chunk. A crash mid-chunk
            # simply redoes this one chunk on resume; nothing before it is
            # ever lost.
            manifests.update_run(
                pdir,
                manifest_run_id,
                last_completed_frame=chunk_end,
                manual_masks_protected_count=manual_protected,
            )
    except JobPaused:
        manifests.update_run(pdir, manifest_run_id, status=TrackingRunStatus.PAUSED)
        raise
    except JobCancelled:
        manifests.update_run(pdir, manifest_run_id, status=TrackingRunStatus.CANCELLED)
        raise
    except Exception as exc:
        manifests.update_run(
            pdir, manifest_run_id, status=TrackingRunStatus.FAILED, error=str(exc)
        )
        raise

    manifests.update_run(pdir, manifest_run_id, status=TrackingRunStatus.COMPLETED)
```

**Explanation.** The run loop is the heart of FR-PRP: pause is honoured only at chunk boundaries (so a paused run always has a clean checkpoint), cancel also between predictions. A frame where the object was edited by hand or deactivated is skipped and counted, never overwritten, unless the run was explicitly started with `overwrite_manual_masks`. Lines 168–214 (omitted) compute the review reasons and write the propagated mask with `apply_frame_annotation`. The manifest update after each chunk is the checkpoint used by resume; the exception handlers keep the manifest status in step with the job status set by `JobManager`.

### Excerpt 8 – Schema migration interface

`src/vistlabel/persistence/migrations.py`, lines 44–69

```python
def migrate(document_kind: str, data: dict) -> dict:
    if document_kind not in CURRENT_SCHEMA_VERSIONS:
        raise KeyError(f"unknown document kind: {document_kind!r}")
    current_version = CURRENT_SCHEMA_VERSIONS[document_kind]
    version = data.get("schema_version", 1)
    if version > current_version:
        raise UnsupportedSchemaVersionError(
            f"{document_kind}: file schema_version {version} is newer than "
            f"the supported version {current_version}; refusing to guess"
        )
    migrations_for_kind = _migrations.get(document_kind, {})
    while version < current_version:
        migration_fn = migrations_for_kind.get(version)
        if migration_fn is None:
            raise UnsupportedSchemaVersionError(
                f"{document_kind}: no migration registered from schema_version {version} "
                f"to reach {current_version}"
            )
        data = migration_fn(data)
        new_version = data.get("schema_version", version + 1)
        if new_version <= version:
            raise UnsupportedSchemaVersionError(
                f"{document_kind}: migration from {version} did not advance schema_version"
            )
        version = new_version
    return data
```

**Explanation.** Documents are migrated on read, one version step at a time, by functions registered with `register_migration(kind, from_version, fn)`. A file written by a newer VISTLabel is refused rather than misread. All kinds are at version 1 in this release, so the loop is currently a no-op; the interface keeps future schema changes out of ad-hoc `if "field" in data` checks.

### Excerpt 9 – Train/val split assignment for export

`src/vistlabel/exports/frame_selection.py`, lines 124–149

```python
def assign_splits(
    *,
    frame_indices: list[int],
    split_assignments: list[FrameRangeAssignment],
) -> tuple[dict[int, Literal["train", "val"]], list[int]]:
    """Returns ``(split_by_frame, excluded_frame_indices)``. When the
    project has no explicit split assignments at all, falls back to a
    contiguous temporal block rather than a random per-frame split, which
    would leak near-duplicate neighboring frames across train/val."""
    if not split_assignments:
        ordered = sorted(frame_indices)
        val_count = max(1, round(len(ordered) * _DEFAULT_VAL_FRACTION)) if ordered else 0
        val_set = set(ordered[len(ordered) - val_count :]) if val_count else set()
        return {idx: ("val" if idx in val_set else "train") for idx in ordered}, []

    split_by_frame: dict[int, Literal["train", "val"]] = {}
    excluded: list[int] = []
    for idx in frame_indices:
        label = _split_label_for_frame(idx, split_assignments)
        if label is None or label is SplitLabel.EXCLUDED:
            excluded.append(idx)
        elif label is SplitLabel.TRAIN:
            split_by_frame[idx] = "train"
        else:
            split_by_frame[idx] = "val"
    return split_by_frame, excluded
```

**Explanation.** Implements FR-SPL-02/03: without user ranges the last 20 % of the exported frames form one contiguous validation block (`_DEFAULT_VAL_FRACTION = 0.2`); with ranges, only covered frames are exported and the rest are returned as excluded so the export report can count them.

### Excerpt 10 – YOLO multi-part policies

`src/vistlabel/exports/yolo_writer.py`, lines 63–104

```python
        # Multipart: obj.connected_component_count >= 2.
        conversion = MultipartConversion(
            frame_index=frame.frame_index,
            object_id=obj.object_id,
            label_name=label_name,
            component_count=obj.connected_component_count,
        )
        multipart_conversions.append(conversion)

        if multipart_policy == "strict_error":
            dropped.append(
                DroppedInstance(
                    frame_index=frame.frame_index,
                    object_id=obj.object_id,
                    reason=(
                        "multipart instance excluded from YOLO export under the "
                        "strict_error policy (exact mask remains in the COCO export)"
                    ),
                )
            )
        elif multipart_policy == "minimal_bridge":
            bridged, bridge_pixels = bridge_mask(mask)
            contours = mask_to_contours(bridged)
            if not contours:
                invalid_polygons.append(
                    f"frame {frame.frame_index} object {obj.object_id}: bridged mask has "
                    "no traceable contour"
                )
                continue
            polygon = normalize_polygon(contours[0], width, height)
            lines.append(_format_line(obj.label_id, polygon))
            multipart_conversions[-1] = conversion.model_copy(
                update={"bridge_pixel_count": bridge_pixels}
            )
        elif multipart_policy == "split_components":
            for contour in mask_to_contours(mask):
                polygon = normalize_polygon(contour, width, height)
                lines.append(_format_line(obj.label_id, polygon))
        else:
            raise ValueError(f"unknown multipart policy: {multipart_policy!r}")  # pragma: no cover

    return lines, multipart_conversions, dropped, invalid_polygons
```

**Explanation.** The Ultralytics polygon format cannot express "two contours, one instance", so each policy is an explicit, reported answer: drop (COCO still has it), bridge with minimal 1-px lines (`exports/polygon.bridge_mask`, number of added pixels reported), or split into separate rows. Every conversion is listed in the validation report; in the verification run, the pipette split by the finger on frame 10 appears under `multipart_instances` with `component_count: 2` and its bridge pixel count.
