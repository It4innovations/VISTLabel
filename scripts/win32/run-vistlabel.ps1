# Copyright (c) 2026 IT4Innovations, VSB-Technical University of Ostrava
# SPDX-License-Identifier: MIT

<#
.SYNOPSIS
    Starts VISTLabel pinned to a single GPU.

.DESCRIPTION
    This workstation has two identical Quadro RTX 6000s. Pinning happens
    through CUDA_VISIBLE_DEVICES rather than runtime.cuda_device in
    config/app.yaml, because two diagnostics query device index 0 directly
    (diagnostics/system.py check_cuda, diagnostics/models.py peak VRAM) and
    would report the wrong card under a "cuda:1" setting. Hiding the other
    card instead keeps the chosen GPU as cuda:0, so those stay correct.

    Leave runtime.cuda_device at "auto" in config/app.yaml.

.PARAMETER GpuIndex
    Which physical GPU to use, matching the index shown by nvidia-smi.
    Named only -- bare arguments are forwarded to "vistlabel serve".
    Defaults to GPU 1 on a machine with two or more GPUs and GPU 0 on a
    single-GPU machine (e.g. a laptop). An index nvidia-smi doesn't list is
    an error: pinning a missing GPU hides every card, and VISTLabel would
    silently run its models on CPU.

.PARAMETER ServeArgs
    Forwarded verbatim to "vistlabel serve" (e.g. --host, --port).

.EXAMPLE
    .\scripts\win32\run-vistlabel.ps1
    Serves on the default 127.0.0.1:9080 using GPU 1 (GPU 0 on a
    single-GPU machine).

.EXAMPLE
    .\scripts\win32\run-vistlabel.ps1 -GpuIndex 0 --port 9090
    Serves on port 9090 using GPU 0 instead.
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]]$ServeArgs = @(),

    [int]$GpuIndex = -1
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi not found -- no NVIDIA driver, so VISTLabel would run its models on CPU."
}
$gpus = @(& nvidia-smi --query-gpu=index,name --format=csv,noheader)
if ($GpuIndex -lt 0) {
    $GpuIndex = if ($gpus.Count -ge 2) { 1 } else { 0 }
}
if ($GpuIndex -ge $gpus.Count) {
    throw "GPU $GpuIndex not present; nvidia-smi lists: $($gpus -join '; '). Pass -GpuIndex with one of those."
}

# PCI_BUS_ID makes GpuIndex line up with nvidia-smi. CUDA's default
# ordering is "fastest first", which is ambiguous with two identical
# cards -- index 1 could land on either one.
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = "$GpuIndex"

# This script lives in scripts/win32, so the repo root is two levels up.
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$vistlabel = Join-Path $repoRoot '.venv\Scripts\vistlabel.exe'
if (-not (Test-Path $vistlabel)) {
    throw "vistlabel.exe not found at $vistlabel -- create the venv and run 'pip install -e .[gpu]' first."
}

# vistlabel resolves config/app.yaml relative to the working directory.
Push-Location $repoRoot
try {
    Write-Host "GPU $GpuIndex pinned; visible to the process as cuda:0" -ForegroundColor Cyan
    & $vistlabel serve @ServeArgs
}
finally {
    Pop-Location
}
