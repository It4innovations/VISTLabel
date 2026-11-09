# VISTLabel

VISTLabel (Video Instance Segmentation, Tracking and Labeling) is a
browser-controlled, FastAPI-backed tool that turns any video into
instance-segmentation datasets. You outline objects on a frame with SAM
prompts and a brush, propagate the masks through the video, review what
the model produced and export the result for training.

- Browser UI (Chrome/Firefox), no CDN assets, no network access at runtime.
- MP4 upload with optional downscaling and frame thinning, or import of an
  existing COCO or YOLO dataset.
- SAM 2.1 / SAM 3 point prompts, SAM 3 text search ("find the pipette"),
  brush and eraser, multi-part objects through occlusions, optional
  YOLO + SAM automatic pre-annotation.
- Chunked, checkpointed video propagation that can be paused, resumed and
  survives restarts. It never overwrites hand-edited masks.
- Review queue for doubtful frames, train / val / excluded splits, and
  export to Ultralytics YOLO or COCO (RF-DETR layout), as masks or boxes.
- Filesystem projects, atomic writes, schema migrations; everything stays on
  the machine.
- Single-account sign-in (Argon2id password, server-side sessions,
  same-origin request checks); the account is managed on the server with
  `vistlabel auth`.

**How to use the application** is described in
[`docs/user-guide.md`](docs/user-guide.md); the same page is served inside
the application under **Help**, together with the keyboard shortcuts. This
file covers installation and starting the server.

## Prerequisites

- Python **3.11 to 3.13** (verified on 3.12 and 3.13).
  `GET /api/v1/health` reports the interpreter status.
- **FFmpeg** (`ffmpeg` and `ffprobe`) for upload validation and
  re-encoding. Put both on `PATH`, or point `runtime.ffmpeg_dir` at their
  directory.
- Optional model support needs an NVIDIA GPU with a CUDA build of PyTorch
  (see [Optional: model backends](#optional-model-backends)). Without one
  the application still runs, and everything except the real model
  backends works.
- Disk space for the workspace: every project keeps its video, decoded
  frame cache and annotations there. Uploads are refused below 20 GB free
  (`upload.minimum_free_disk_space_gb`).

Always install into a virtual environment, never into the system Python.

## Install on Windows (PowerShell)

```powershell
git clone <repository-url> vistlabel
cd vistlabel
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Use `py -3.13` where 3.12 isn't installed (`py -0` lists the
interpreters). If PowerShell refuses to run the activation script, start it
once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

FFmpeg: `winget install Gyan.FFmpeg`, or unpack a build from
<https://www.gyan.dev/ffmpeg/builds/> anywhere and set `runtime.ffmpeg_dir`
to its `bin` directory.

## Install on Linux

```bash
git clone <repository-url> vistlabel
cd vistlabel
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install FFmpeg from the distribution (`apt install ffmpeg`,
`dnf install ffmpeg`). On a cluster without Internet access on the compute
nodes, install and download models on a login node; the application itself
never downloads anything at runtime.

`requirements-lock.txt` pins the exact versions tested together on the
Windows development workstation, including `torch==…+cu130`. It installs
only where that CUDA 13 build exists (Windows or Linux x86-64, with
`--extra-index-url https://download.pytorch.org/whl/cu130`); elsewhere,
macOS included, `pip install -r requirements-lock.txt` stops at the torch
pin and installs nothing, so use the plain `pip install -e` above.

## Install on macOS

The application runs on macOS (verified on Apple silicon, macOS 26, Python
3.12) with the `fake` backend; the real model backends need an NVIDIA GPU
with CUDA, so they stay unavailable there.

```bash
brew install python@3.12 ffmpeg
git clone <repository-url> vistlabel
cd vistlabel
python3.12 -m venv .venv            # /opt/homebrew/bin/python3.12 if not on PATH
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

The system `/usr/bin/python3` (3.9) is too old. In a non-interactive SSH
session Homebrew's `/opt/homebrew/bin` is often not on `PATH`; call the
interpreter by its full path then. PyAV and OpenCV each bundle their own
`libavdevice`, so every start prints `objc[…]: Class AVFFrameReceiver is
implemented in both …` warnings. VISTLabel never captures from a device,
and the test suite passes with them.

## Configure

```bash
cp config/app.example.yaml config/app.yaml     # PowerShell: Copy-Item
vistlabel auth set-password --username lab     # creates the single account
```

VISTLabel has exactly one user account, and every page and API call needs
a signed-in session (see [Account and sign-in](#account-and-sign-in)).
`set-password` prompts twice for the password (at least 8 characters);
without a terminal (scripts, batch jobs) it reads `VISTLABEL_PASSWORD` from
the environment instead. Run it with the same `config/app.yaml` (or
`--config`) as the server: the account is stored under `workspace.root`.

Every value in `config/app.yaml` can be overridden with a
`VISTLABEL_<SECTION>__<KEY>` environment variable, e.g.
`VISTLABEL_WORKSPACE__ROOT=D:\vistlabel-workspace`. The application also starts
with safe built-in defaults when the file is absent (bind to localhost, all
real model backends disabled). `config/app.yaml` is machine specific and
git-ignored. Important keys:

| Key | Purpose |
| --- | --- |
| `workspace.root` | Directory holding all projects, their videos, annotations and exports. |
| `runtime.ffmpeg_dir` | Directory with `ffmpeg`/`ffprobe`, checked before `PATH`. |
| `runtime.default_dtype` | `float16` (default) or `bfloat16` for GPUs that support it. |
| `upload.maximum_file_size_gb` | Largest accepted MP4 (default 50). |
| `upload.normalize_variable_frame_rate` | Re-time variable-frame-rate recordings to a constant rate before indexing. |
| `models.sam21` / `sam3` / `yolo_sam` | Optional local model backends: `enabled` and checkpoint paths. |
| `export.*` | Defaults the export dialog's "Project default" stands for (format, layout, image type, multi-part policy). |
| `security.session_timeout_minutes` | Sliding inactivity timeout of a browser session (default 120). |
| `security.secure_cookies` | Mark the session cookie `Secure`; set `true` behind an HTTPS reverse proxy. |
| `security.login_rate_limit` / `login_rate_window_seconds` | Failed sign-in attempts allowed per client within the window (default 10 per 300 s). |

The password is never read from `config/app.yaml`. The former `auth`
section (shared `Authorization: Bearer` API keys) was replaced by the
sign-in; a config that still has it is rejected at start with a message
saying so.

`config/app.example.yaml` shows the common keys; every key with its
default is defined in `src/vistlabel/settings.py`.

## Run

```bash
vistlabel version                               # prints the installed version
vistlabel serve --host 127.0.0.1 --port 9080
```

Then open <http://127.0.0.1:9080>, sign in and follow the
[user guide](docs/user-guide.md). Useful endpoints: `/` (UI), `/help` (the
guides and shortcuts), `/api/v1/health`, `/api/v1/version`,
`/api/v1/models`, `/api/v1/metrics`. Only `/login`, `/setup`,
`/api/v1/health`, `/api/v1/version` and the static assets answer without a
session; until an account exists, every page redirects to `/setup`, which
says how to create it.

On a machine with several NVIDIA GPUs, `./run-vistlabel.sh --gpu 0` (Git
Bash or Linux) or `.\scripts\win32\run-vistlabel.ps1 -GpuIndex 0` pins the
server to one card; other arguments go to `vistlabel serve`. Both scripts
need `nvidia-smi` and refuse to start without it (e.g. on macOS); start
`vistlabel serve` directly there.

Remote access over SSH:

```bash
ssh -N -L 9080:127.0.0.1:9080 user@workstation     # then open http://127.0.0.1:9080
```

### Account and sign-in

| Command | Effect |
| --- | --- |
| `vistlabel auth set-password --username <name>` | Creates the account, or changes its name and password; also signs out every browser session. |
| `vistlabel auth status` | Shows the account and the number of active sessions (exit code `1` when no account exists). |
| `vistlabel auth revoke-sessions` | Signs out every browser session. |

Each command takes `--config` like `serve`. The account lives in
`workspace.root/auth/account.json` as an Argon2id hash, never as the
password; sessions are server-side (`auth/sessions.json`, only a hash of
each token) and survive a server restart. The browser holds an opaque
`HttpOnly`, `SameSite=Lax` cookie, and every state-changing API request
must carry `X-Requested-With: VISTLabel` from the same origin, which blocks
cross-site request forgery. Failed sign-ins are rate limited per client.
Scripts that call the API sign in the same way: `POST /api/v1/auth/login`
with `{"username": …, "password": …}` and that header, then reuse the
`vistlabel_session` cookie.

The server speaks plain HTTP. Keep it on `127.0.0.1` and reach it through
the SSH tunnel above, or put a TLS reverse proxy in front and set
`security.secure_cookies: true`.

For a project with data to look at straight away:

```bash
python scripts/create_demo_project.py --workspace ./workspace
```

The script looks for FFmpeg on `PATH` only; where it isn't there, pass
`--ffmpeg-dir <dir>` (it reads neither `runtime.ffmpeg_dir` nor the
environment).

Everything the application stores (the account and sessions in `auth/`,
projects with their videos, frame caches, annotations and exports in
`projects/`) lives under `workspace.root`; back up that directory while the
server is stopped.

### Version

The version is defined once, in `pyproject.toml` (currently `1.0.0`), and
read from the installed package metadata: `vistlabel version` or
`vistlabel --version`, `GET /api/v1/version`, `GET /api/v1/health`
(`package_version`) and the **System status** dialog all show it. After
changing it, reinstall (`pip install -e .`) so the metadata follows.

### Optional: model backends

| Backend | What it does | Checkpoint |
| --- | --- | --- |
| `fake` | Deterministic synthetic masks for testing; always available. | none |
| `sam21` | SAM 2.1 point prompts and video propagation. | `--model sam21-base-plus` or `sam21-large` |
| `sam3` | SAM 3 text search, plus point, box and mask prompts and video propagation through its tracker. Only one of the two models is in GPU memory at a time. | `--model sam3` (gated) |
| `yolo_sam` | YOLO detection refined by `sam21` or `sam3` (`refiner_backend`) for automatic pre-annotation. | your own trained YOLO checkpoint + class mapping |

Install the PyTorch build matching your **driver** first, then the `gpu`
extra. A plain `pip install -e ".[gpu]"` pulls the CPU-only wheel from
PyPI. Read `CUDA Version` in the `nvidia-smi` header and pick the index
that doesn't exceed it:

| `nvidia-smi` CUDA Version | Index URL |
| --- | --- |
| 12.1 (driver 5xx) | `https://download.pytorch.org/whl/cu121` (torch 2.5.1) |
| 12.6 (driver 560+) | `https://download.pytorch.org/whl/cu126` |
| 13.x (driver 580+) | `https://download.pytorch.org/whl/cu130` |

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[gpu]"                                 # leaves torch alone
python -c "import torch; print(torch.randn(8, device='cuda').sum())"
export HF_TOKEN=hf_...                                  # gated repository (sam3)
python scripts/download_models.py --model sam21-base-plus
python scripts/download_models.py --model sam3
```

Test with an allocation as above, not only `torch.cuda.is_available()`: a
wheel built for a newer CUDA than the driver supports still reports the
GPU, but every allocation then fails with `cudaErrorDevicesUnavailable`.

Downloads are pinned to exact Hugging Face revisions and land in
`./models/`. Enable the backend in `config/app.yaml`, point its
`checkpoint` at the downloaded directory and verify it with
`python -m vistlabel.diagnostics.models --backend sam21` (exit code `0`
usable, `1` unavailable with the reason). A backend without a valid
checkpoint reports itself unavailable with the exact reason; it is never
silently replaced by another model. Checkpoints are licence-encumbered
binaries and must not be committed or redistributed; see
[`models/README.md`](models/README.md).

## Checks and tests

```bash
ruff check src tests scripts
mypy src
pytest -m "not gpu and not browser"                  # CPU-only suite
playwright install --with-deps chromium firefox      # once
pytest -m browser                                    # Chromium
pytest -m browser --browser firefox
pytest -m gpu                                        # real models; skips without checkpoints
python scripts/generate_help_reference.py --check    # shortcut table in the user guide
```

The tests create their own throwaway account in a temporary workspace and
sign in with it (the browser tests through the real sign-in form), so they
need no `config/app.yaml` and no account. Tests that upload a video need
FFmpeg and skip with a clear reason without it. The test videos under `tests/fixtures/` are small synthetic clips, not
real recordings. After editing `src/vistlabel/static/help/shortcuts.json`, run
`python scripts/generate_help_reference.py` to refresh the user guide's
table.

## Diagnostics

```bash
python -m vistlabel.diagnostics.system                  # health report as JSON
python -m vistlabel.diagnostics.models --backend fake   # or sam21, sam3, yolo_sam
python scripts/benchmark_sam21.py --video tests/fixtures/tiny_test_video.mp4
```

The **System status** button on the Projects page shows the same health
checks, backend availability and counters in the browser.

## Development

AI agent harnesses were used for selected development, testing, and
optimization tasks.

## License

MIT, Copyright (c) 2026 IT4Innovations, VSB-Technical University of
Ostrava (see [`LICENSE`](LICENSE)).

The MIT license covers the VISTLabel source code only. The IT4Innovations
logo (`src/vistlabel/static/img/it4i-logo.png`) is a trademark of
IT4Innovations and is not licensed under MIT.

Third-party components are not part of this repository and keep their own
licenses:

- **Model checkpoints** are downloaded separately by the user and are never
  redistributed here. SAM 2.1 (`facebook/sam2.1-hiera-*`) is released under
  Apache-2.0; SAM 3 (`facebook/sam3`) is gated and under Meta's own SAM
  License, which you accept on its Hugging Face page. Review each model's
  terms before use.
- **Ultralytics** (`ultralytics`, pulled in by the optional `gpu` extra and
  used only by the `yolo_sam` backend) is licensed under **AGPL-3.0**, with a
  commercial Enterprise License available from Ultralytics. Enabling
  `yolo_sam` makes your deployment subject to those terms, including when
  you offer it as a network service. The rest of VISTLabel does not import it.
- The remaining Python dependencies (FastAPI, Pydantic, NumPy, OpenCV,
  PyAV, Pillow, PyTorch, Transformers, ...) use permissive licenses
  (MIT, BSD, Apache-2.0 and similar).

## Acknowledgements

This work was supported by the Ministry of Education, Youth and Sports of
the Czech Republic through the e-INFRA CZ (ID:90254).