# VISTLabel – Documentation Package

**Document purpose:** entry point to the documentation of the software result VISTLabel: what it is, where to obtain it, which documents describe it and licensing (affiliation and funding are in the repository README).

| | |
| --- | --- |
| Software | VISTLabel – Video Instance Segmentation, Tracking and Labeling |
| Type of result | Software |
| Version | 1.0.0 |
| Licence | MIT |
| Repository | <https://github.com/It4innovations/VISTLabel> |

## Description

VISTLabel is a locally run, browser-operated application that turns video recordings into instance-segmentation and object-detection training datasets. An annotator creates a project from an MP4 video (optionally downscaled and frame-thinned at upload) or from an existing COCO/YOLO dataset, outlines each object on a key frame with SAM 2.1 / SAM 3 point prompts or SAM 3 text search and corrects the result with a brush and an eraser, and lets the video model propagate the masks through the rest of the video as a chunked, checkpointed background job that can be paused, resumed, survives restarts and never overwrites hand-edited masks. Frames where the propagation is doubtful (an object disappears or reappears, its area or number of visible parts changes, …) are flagged for review; objects split by occlusions stay one instance with several visible parts. The reviewed annotations are exported, with train/val splits, to the Ultralytics YOLO or COCO (RF-DETR) layout as masks or bounding boxes, with a validation report. The server (Python, FastAPI) stores everything as files in a local workspace with atomic writes and schema versioning, needs no network access at runtime, protects access with a single Argon2id-hashed account and server-side sessions, and runs without a GPU (the real model backends are optional and need an NVIDIA GPU).

## Documents

| Document | Contents |
| --- | --- |
| [user-guide.md](user-guide.md) | End-user guide for annotators: projects, annotation workspace, SAM prompts, brush and eraser, propagation, review, splits, export, keyboard shortcuts (the same text is served inside the application under **Help**). |
| [functional-requirements.md](functional-requirements.md) | Purpose and scope, actors, use cases, numbered functional (FR) and non-functional (NFR) requirements, constraints and a traceability matrix requirement → modules → tests. |
| [technical-documentation.md](technical-documentation.md) | Design and principles: architecture, module map, data model and on-disk format, atomic writes and migrations, REST API, model backend interface, propagation algorithm and conflict rules, export formats, configuration, security design, design decisions. |
| [programmer-documentation.md](programmer-documentation.md) | Repository layout, development set-up, build, conventions, running the checks, extension points, CLI reference and ten annotated source-code excerpts. |
| [verification.md](verification.md) | Test strategy and the results of an independent verification run (automated suites, static checks, coverage, security spot checks, end-to-end scenario), with limitations and observations. |
| [images/](images/) | Architecture, workflow, data-model and state-machine diagrams (SVG) and application screenshots (PNG). |
| [../README.md](../README.md) | The repository README (installation, configuration, running), as released. |

## Download and usage

- **Source code and releases:** <https://github.com/It4innovations/VISTLabel>
- **Licence:** MIT, Copyright (c) 2026 IT4Innovations, VSB – Technical University of Ostrava (file `LICENSE` in the repository).
- **Installation summary** (details, Linux/macOS variants and GPU set-up in [../README.md](../README.md)):
  1. Prerequisites: Python 3.11–3.13, FFmpeg (`ffmpeg` and `ffprobe`); optionally an NVIDIA GPU with a CUDA build of PyTorch for the SAM / YOLO backends.
  2. `git clone https://github.com/It4innovations/VISTLabel.git` and, in a virtual environment, `pip install -e ".[dev]"` (add `".[gpu]"` after installing the matching PyTorch build for real models; download checkpoints with `python scripts/download_models.py --model sam21-base-plus` or `--model sam3`).
  3. `cp config/app.example.yaml config/app.yaml`, then `vistlabel auth set-password --username <name>` to create the single account.
  4. `vistlabel serve --host 127.0.0.1 --port 9080` and open <http://127.0.0.1:9080> (remotely through an SSH tunnel or a TLS reverse proxy).
  5. Optional demo data: `python scripts/create_demo_project.py --workspace ./workspace`.

## Third-party components and licences

The MIT licence covers the VISTLabel source code only. The IT4Innovations logo, the model checkpoints (SAM 2.1, SAM 3; never redistributed with VISTLabel) and the optional Ultralytics dependency of the `yolo_sam` backend (AGPL-3.0) keep their own terms; see the *License* section of [../README.md](../README.md#license).
