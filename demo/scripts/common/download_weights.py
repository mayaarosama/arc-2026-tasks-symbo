"""Download every model a machine needs into the Hugging Face cache. Run with network, days ahead.

    export HF_TOKEN=hf_...           # from config.txt (the CPU-box scripts do this); account must accept each gated licence
    python -m scripts.common.download_weights --machine dgx --task t2i
    python -m scripts.common.download_weights --machine workstation --task i23d --dry-run
    python -m scripts.common.download_weights --cutout      # operator laptop: background removal for uploads

The model list is derived from config/demo.yaml (every model that machine is configured to run
for that task, in any protocol) plus the declared fallbacks, so config and downloads cannot drift.
Gated repos (accept on the model page first): all stabilityai/*, facebook/dinov3-*, briaai/RMBG-2.0.
GGUF repos hold every quantisation: a model's `file` glob in models.yaml limits the download to one.
"""
import argparse
import sys
from pathlib import Path

from arcdemo.conductor.config import DemoConfig
from arcdemo.weights import DEPENDENCIES  # noqa: F401  (also read by plan_downloads.py)

# Fallback tiers that are not in the settings matrix but must be on disk.
FALLBACKS = {("cpu", "t2i"): ["sd15"]}
# Skip duplicate single-file checkpoints: diffusers loads the per-component folders.
IGNORE = {
    "stabilityai/stable-diffusion-3.5-large": ["sd3.5_large.safetensors", "text_encoders/*"],
    "stabilityai/stable-diffusion-3.5-large-turbo": ["sd3.5_large_turbo.safetensors", "text_encoders/*"],
    "stabilityai/stable-diffusion-3.5-medium": ["sd3.5_medium.safetensors", "text_encoders/*"],
    "stable-diffusion-v1-5/stable-diffusion-v1-5": ["*.ckpt", "v1-5-pruned*.safetensors", "*.bin",
                                                    "*fp16*", "*non_ema*"],
    "facebook/dinov2-large": ["pytorch_model.bin"],                 # transformers loads model.safetensors
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": ["model.safetensors", "pytorch_model.bin",   # open_clip reads
                                              "open_clip_model.safetensors"],             # open_clip_pytorch_model.bin
}


def model_keys(cfg: DemoConfig, machine: str, task: str) -> set[str]:
    keys = set(FALLBACKS.get((machine, task), []))
    for protocol in (cfg.raw.get("settings") or {}).get(task, {}):
        s = cfg.settings(task, protocol, machine)
        if s.get("mode") != "replay":
            keys.add(s["model"])
    return keys


def repos_for(cfg: DemoConfig, keys: set[str]) -> list[str]:
    repos = []
    for k in sorted(keys):
        repo = cfg.models[k]["repo"]
        if "/" in repo:
            repos.append(repo)
        repos += DEPENDENCIES.get(k, [])
    return sorted(set(repos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/demo.yaml")
    ap.add_argument("--machine", help="with --task: everything this machine runs for the task")
    ap.add_argument("--task")
    ap.add_argument("--model", action="append", help="instead: just these models.yaml keys (repeatable)")
    ap.add_argument("--cutout", action="store_true",
                    help="the operator laptop's background-removal model (image-to-3D uploads)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.cutout:
        fetch_cutout(a.dry_run)
        return
    cfg = DemoConfig(a.config)
    if a.model:
        keys, what = set(a.model), "models " + ", ".join(a.model)
    elif a.machine and a.task:
        keys, what = model_keys(cfg, a.machine, a.task), f"{a.machine}/{a.task}"
    else:
        ap.error("give --machine and --task, or --model")
    repos = repos_for(cfg, keys)
    allow = {cfg.models[k]["repo"]: [cfg.models[k]["file"]] for k in keys if cfg.models[k].get("file")}
    print(f"{what}: {len(repos)} repos")
    for r in repos:
        print("  ", r, "(ignoring " + ", ".join(IGNORE[r]) + ")" if r in IGNORE else
              "(only " + ", ".join(allow[r]) + ")" if r in allow else "")
    if a.dry_run:
        return
    failed = []
    for r in repos:
        status, detail = fetch(r, IGNORE.get(r), allow.get(r))
        print(status if status != "failed" else "FAILED", r, "->" if status != "failed" else "-", detail)
        if status == "failed":
            failed.append(r)
    if failed:
        sys.exit(f"{len(failed)} repo(s) not downloaded: {', '.join(failed)}")


# U^2-Net, the model rembg (and SF3D's run.py) use by default, from rembg's GitHub release. Not on
# the Hugging Face Hub under an official repo. Saved where arcdemo/imageprep.py looks for it.
CUTOUT_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
CUTOUT_BYTES = 175_997_641


def fetch_cutout(dry_run: bool = False) -> None:
    import urllib.request
    dest = Path(__file__).resolve().parents[2] / "weights" / "u2net" / "u2net.onnx"
    print(f"cut-out model: {CUTOUT_URL} -> {dest}")
    if dest.is_file() and dest.stat().st_size == CUTOUT_BYTES:
        print("cached", dest)
        return
    if dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    try:
        urllib.request.urlretrieve(CUTOUT_URL, part)
    except OSError as e:
        sys.exit(f"FAILED cut-out model: {e}")
    if part.stat().st_size != CUTOUT_BYTES:
        part.unlink()
        sys.exit(f"FAILED cut-out model: expected {CUTOUT_BYTES} bytes")
    part.replace(dest)
    print("ok", dest)


def fetch(repo: str, ignore=None, allow=None) -> tuple[str, str]:
    """Download one repo into the cache: ("ok", path) | ("cached", path + why) | ("failed", error).

    Safe to re-run: files already in the cache are skipped and partial ones resume. If the Hub
    cannot be reached (no network, gated licence revoked) a repo already on disk is kept as is."""
    from huggingface_hub import snapshot_download
    # Windows without symlink rights: huggingface_hub marks a new repo folder "symlinks OK" before
    # testing it, so parallel workers race the test and fail with WinError 1314. One worker avoids it.
    workers = 1 if sys.platform == "win32" else 8
    kw = dict(ignore_patterns=ignore, allow_patterns=allow)
    try:
        return "ok", snapshot_download(repo, max_workers=workers, **kw)
    except Exception as e:   # noqa: BLE001 - any Hub error: try the local cache before giving up
        try:
            path = snapshot_download(repo, local_files_only=True, **kw)
            return "cached", f"{path} (Hub check failed: {type(e).__name__})"
        except Exception:    # noqa: BLE001
            return "failed", f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"


if __name__ == "__main__":
    main()
