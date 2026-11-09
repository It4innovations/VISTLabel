#!/usr/bin/env bash
# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT
# Starts VISTLabel pinned to a single GPU. Bash counterpart of
# scripts/win32/run-vistlabel.ps1; works from Git Bash on Windows and on Linux.
#
# Usage:
#   ./run-vistlabel.sh                           # 127.0.0.1:9080 on GPU 1 (GPU 0 if it's the only one)
#   ./run-vistlabel.sh --gpu 0                   # use GPU 0 instead
#   ./run-vistlabel.sh --gpu 0 --port 9090       # anything else goes to "vistlabel serve"
#
# Pinning uses CUDA_VISIBLE_DEVICES rather than runtime.cuda_device in
# config/app.yaml, so the chosen card stays cuda:0 for the diagnostics that
# query device 0 directly (see the .ps1 script). Leave runtime.cuda_device
# at "auto".

set -euo pipefail

gpu_index=""
serve_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            gpu_index="${2:?--gpu needs a GPU index}"
            shift 2
            ;;
        *)
            serve_args+=("$1")
            shift
            ;;
    esac
done

# Pinning a GPU that doesn't exist hides every card, and VISTLabel would
# silently run its models on CPU -- so default to what's actually there
# (GPU 1 on the two-GPU workstation, GPU 0 on a laptop) and reject the rest.
if ! command -v nvidia-smi >/dev/null; then
    echo "nvidia-smi not found -- no NVIDIA driver, so VISTLabel would run its models on CPU." >&2
    exit 1
fi
mapfile -t gpus < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
if [[ -z "$gpu_index" ]]; then
    gpu_index=$(( ${#gpus[@]} >= 2 ? 1 : 0 ))
fi
if (( gpu_index >= ${#gpus[@]} )); then
    echo "GPU $gpu_index not present; nvidia-smi lists: ${gpus[*]}. Pass --gpu with one of those." >&2
    exit 1
fi

# PCI_BUS_ID makes the index line up with nvidia-smi; CUDA's default
# "fastest first" order is ambiguous with two identical cards.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$gpu_index"

# vistlabel resolves config/app.yaml relative to the working directory.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

vistlabel=""
for candidate in .venv/Scripts/vistlabel.exe .venv/bin/vistlabel; do
    if [[ -x "$candidate" ]]; then
        vistlabel="$candidate"
        break
    fi
done
if [[ -z "$vistlabel" ]]; then
    echo "vistlabel not found in $repo_root/.venv -- create the venv and run 'pip install -e .[gpu]' first." >&2
    exit 1
fi

echo "GPU $gpu_index pinned; visible to the process as cuda:0"
exec "$vistlabel" serve "${serve_args[@]}"
