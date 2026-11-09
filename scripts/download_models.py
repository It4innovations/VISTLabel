# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

"""Explicit, administrator-run model checkpoint download. Checkpoints are
never downloaded automatically during application startup: this script is
never invoked by ``vistlabel serve`` itself.

Downloads via the Hugging Face Hub (``huggingface_hub.snapshot_download``),
pinned to an exact commit ``revision`` per model -- not "main" -- so a
later upstream change to the HF repo cannot silently swap in different
weights.

Usage::

    python scripts/download_models.py --model sam21-base-plus
    python scripts/download_models.py --model sam21-large --dest ./models
    python scripts/download_models.py --model sam3            # gated, needs a token

``facebook/sam3`` is a gated repository: accept its licence on the model
page, then expose a Hugging Face token (``HF_TOKEN`` in the environment, or
``huggingface-cli login``) or the download fails with ``GatedRepoError``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str
    dest_subdir: str
    #: config/app.yaml key this checkpoint belongs to, printed on success so
    #: the operator is not told to configure the wrong backend.
    config_key: str = "models.sam21.checkpoint"
    #: transformers Sam2VideoModel only needs the HF-format files, not
    #: Meta's original *.pt / *.yaml (also present in these repos).
    allow_patterns: tuple[str, ...] = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    )


#: SAM 3's processor is text-conditioned, so its snapshot needs the
#: tokenizer/vocab assets the SAM 2.1 image-only models have no use for.
_SAM3_PATTERNS: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

MODEL_SPECS: dict[str, ModelSpec] = {
    "sam21-base-plus": ModelSpec(
        repo_id="facebook/sam2.1-hiera-base-plus",
        revision="b7320756a13354e7530a63935656d35b2f91a290",
        dest_subdir="sam2.1-hiera-base-plus",
    ),
    "sam21-large": ModelSpec(
        repo_id="facebook/sam2.1-hiera-large",
        revision="665f8e2ad61cf5f53d65644ff27c8ee525124610",
        dest_subdir="sam2.1-hiera-large",
    ),
    # Backs the ``sam3`` backend. This is SAM 3, not SAM 3.1: Meta ships
    # 3.1 as a raw ``sam3.1_multiplex.pt`` with, in its own README's words,
    # "no Hugging Face Transformers integration". SAM 3 has both text and
    # visual-tracker Transformers integrations.
    # Gated repo -- accept the licence on the model page and expose a token
    # (``HF_TOKEN`` or ``huggingface-cli login``) before downloading.
    "sam3": ModelSpec(
        repo_id="facebook/sam3",
        revision="3c879f39826c281e95690f02c7821c4de09afae7",
        dest_subdir="sam3",
        config_key="models.sam3.checkpoint",
        allow_patterns=_SAM3_PATTERNS,
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--dest", type=Path, default=Path("./models"))
    args = parser.parse_args(argv)

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    spec = MODEL_SPECS[args.model]
    target_dir = args.dest / spec.dest_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {spec.repo_id}@{spec.revision} -> {target_dir}")
    try:
        local_path = snapshot_download(
            repo_id=spec.repo_id,
            revision=spec.revision,
            local_dir=target_dir,
            allow_patterns=list(spec.allow_patterns),
        )
    except GatedRepoError:
        # A gated repo fails with a bare 401; say what to actually do about
        # it rather than leaving an HTTP status to interpret.
        print(
            f"error: {spec.repo_id} is gated.\n"
            f"  1. Accept its licence at https://huggingface.co/{spec.repo_id}\n"
            "  2. Provide a token: set HF_TOKEN, or run `huggingface-cli login`",
            file=sys.stderr,
        )
        return 1
    print(f"Done: {local_path}")
    print(f"Set {spec.config_key}: {target_dir} in config/app.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
