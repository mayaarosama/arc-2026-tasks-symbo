"""Download every model a machine needs into the Hugging Face cache. Run with network, days ahead.

    export HF_TOKEN=hf_...           # from config.txt (the CPU-box scripts do this); account must accept each gated licence
    python scripts/download_weights.py --machine dgx --task t2i
    python scripts/download_weights.py --machine workstation --task i23d --dry-run

The model list is derived from config/demo.yaml (every model that machine is configured to run
for that task, in any protocol) plus the declared fallbacks, so config and downloads cannot drift.
Gated repos (accept on the model page first): all stabilityai/*, facebook/dinov3-*, briaai/RMBG-2.0.
GGUF repos hold every quantisation: a model's `file` glob in models.yaml limits the download to one.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arcdemo.conductor.config import DemoConfig   # noqa: E402

# Fallback tiers that are not in the settings matrix but must be on disk.
FALLBACKS = {("cpu", "t2i"): ["sd15"]}
# Models that fetch further repos at load time (from their own pipeline config).
DEPENDENCIES = {
    "trellis2-4b": ["facebook/dinov3-vitl16-pretrain-lvd1689m", "briaai/RMBG-2.0"],
    # SF3D config: image tokenizer = facebook/dinov2-large; its ClipBasedHeadEstimator calls
    # open_clip ViT-B-32 "laion2b_s34b_b79k" -> laion/CLIP-ViT-B-32-laion2B-s34B-b79K (found by verify_offline)
    "sf3d": ["facebook/dinov2-large", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"],
}
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
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
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
