"""Which model weights are on this machine's disk (the Hugging Face cache, demo/hf-cache/hub).

Shared by scripts/common/download_weights.py (what to fetch) and the worker (/health "available"), so
the operator page can say "not on this machine" before anyone presses Load.
"""
from __future__ import annotations

import os
from pathlib import Path

# Models that fetch further repos at load time (from their own pipeline config).
DEPENDENCIES = {
    "trellis2-4b": ["facebook/dinov3-vitl16-pretrain-lvd1689m", "briaai/RMBG-2.0"],
    # SF3D config: image tokenizer = facebook/dinov2-large; its ClipBasedHeadEstimator calls
    # open_clip ViT-B-32 "laion2b_s34b_b79k" -> laion/CLIP-ViT-B-32-laion2B-s34B-b79K (found by verify_offline)
    "sf3d": ["facebook/dinov2-large", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"],
}


def hub_cache() -> Path:
    return Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "hub").expanduser()


WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".bin", ".ckpt", ".pt", ".pth", ".onnx"}


def _repo_on_disk(repo: str, file_glob: str | None = None) -> bool:
    """A snapshot counts only if it holds weights: a gated download that was refused leaves
    a snapshot with just LICENSE.md / README.md in it."""
    snaps = hub_cache() / f"models--{repo.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return False
    for snap in snaps.iterdir():
        files = snap.rglob(file_glob) if file_glob else snap.rglob("*")
        if any(f.suffix in WEIGHT_SUFFIXES and f.is_file() for f in files):
            return True
    return False


def on_disk(spec: dict, key: str = "") -> bool:
    """True when the model's repo (and the repos it pulls at load) are in the cache. A repo
    without an org ('music-stub', a local path) is not a hub download and counts as present."""
    repo = spec.get("repo") or ""
    if "/" not in repo or Path(repo).exists():
        return True
    if not _repo_on_disk(repo, spec.get("file")):
        return False
    return all(_repo_on_disk(r) for r in DEPENDENCIES.get(key, []))
