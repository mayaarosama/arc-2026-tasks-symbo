"""Check this machine's capacity, then download only the models that fit. Run by setup_gpu_machine.sh.

    python -m scripts.gpu.plan_downloads --machine workstation --vram-gb 16 \
        --config config/demo.yaml --config config/demo-full.yaml --out-dir results/setup-<time>

For every model the configs give this machine (first config first: the show before the full matrix):
  1. GPU memory: a model whose known need is larger than this GPU's memory is not downloaded, unless
     the config makes it fail on purpose (`prepare: false`, e.g. TRELLIS.2 on the 16 GB workstation).
     Known needs: GGUF files (file size + 1.5 GB) and the Task 2 figures below. Others: not checked.
  2. Disk: the bytes still missing from hf-cache (files already cached are not counted) must fit in
     the free space minus --reserve-gb. Checked again before every repo, so a repo that no longer
     fits is skipped and the next smaller one may still go in.
  3. Download (scripts/common/download_weights.py fetch(): skips cached files, resumes partial ones).
Writes plan.json, models.tsv (config, task, model, status, detail) and downloads.md (report section).
--dry-run checks and plans but downloads nothing.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from arcdemo.conductor.config import DemoConfig  # importing arcdemo first sets HF_HUB_CACHE
from scripts.common import download_weights as dw
from scripts.common.settings import SCRIPTS

DEMO = Path(__file__).resolve().parents[2]

# GPU memory a model needs, GB, and the margins: config/scripts.yaml plan_downloads.
VRAM_GB = SCRIPTS.plan_downloads.vram_gb
GGUF_OVERHEAD_GB = SCRIPTS.plan_downloads.gguf_overhead_gb
VRAM_SLACK_GB = SCRIPTS.plan_downloads.vram_slack_gb
GB = 1e9


def gb(n: float) -> str:
    return f"{n / GB:.1f} GB"


def remote_files(api, repo: str, ignore, allow) -> tuple[list[tuple[str, int]], str]:
    """(filename, size) of every file the download would fetch, and the commit."""
    from huggingface_hub.utils import filter_repo_objects  # deferred: see main()
    info = api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    names = filter_repo_objects(list(sizes), allow_patterns=allow, ignore_patterns=ignore)
    return [(n, sizes[n]) for n in names], info.sha


def missing_bytes(repo: str, files: list[tuple[str, int]], sha: str) -> int:
    from huggingface_hub import try_to_load_from_cache  # deferred: see main()
    return sum(size for name, size in files
               if not isinstance(try_to_load_from_cache(repo, name, revision=sha), str))


def classify(detail: str) -> str:
    d = detail.lower()
    if "gated" in d or "401" in d or "403" in d or "unauthorized" in d or "forbidden" in d:
        return "needs_licence"
    return "failed"


def main() -> int:
    os.chdir(DEMO)                    # config paths are relative to demo/
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True)
    ap.add_argument("--config", action="append", required=True, help="in priority order (repeatable)")
    ap.add_argument("--task", action="append", help="default: all five")
    ap.add_argument("--vram-gb", type=float, required=True, help="memory of one GPU (the smallest)")
    ap.add_argument("--reserve-gb", type=float, default=10, help="disk to leave free")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tasks = a.task or ["t2i", "i23d", "music", "llm", "layout"]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Deferred imports: huggingface_hub reads HF_HUB_CACHE when it is imported, and importing arcdemo
    # (above) is what sets it; a top-level import would be sorted above arcdemo's and use ~/.cache.
    from huggingface_hub import HfApi
    from huggingface_hub.constants import HF_HUB_CACHE
    api = HfApi()                             # HF_TOKEN from the environment
    cache = Path(HF_HUB_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    free_start = shutil.disk_usage(cache).free
    reserve = a.reserve_gb * GB
    print(f"cache {cache}: {gb(free_start)} free, keeping {gb(reserve)} spare; GPU {a.vram_gb:.0f} GB")

    repo_state: dict[str, dict] = {}          # repo -> {status, detail, bytes}
    models: list[dict] = []
    planned = [0]                             # dry run: bytes it would already have used

    probes: dict[str, dict] = {}

    def probe(repo: str, ignore, allow) -> dict:
        """Size on the Hub and bytes still missing locally, or the error that stopped the check."""
        if repo not in probes:
            try:
                files, sha = remote_files(api, repo, ignore, allow)
                probes[repo] = {"total": sum(s for _, s in files), "missing": missing_bytes(repo, files, sha)}
            except Exception as e:  # noqa: BLE001
                probes[repo] = {"error": e}
        return probes[repo]

    def do_repo(repo: str, ignore, allow) -> dict:
        if repo in repo_state:
            return repo_state[repo]
        st: dict = {"repo": repo}
        p = probe(repo, ignore, allow)
        if "error" not in p:
            need = p["missing"]
            st["total"], st["missing"] = p["total"], need
        else:  # no network or no access: is it on disk already?
            e = p["error"]
            status, detail = dw.fetch(repo, ignore, allow) if not a.dry_run else ("failed", str(e))
            if status == "cached":
                st.update(status="cached", detail="on disk; size not checked (Hub unreachable)")
            else:
                msg = f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}"
                st.update(status=classify(msg), detail=msg)
            repo_state[repo] = st
            return st
        free = shutil.disk_usage(cache).free - planned[0]
        if need == 0:
            st.update(status="cached", detail=f"all {gb(st['total'])} on disk")
        elif need > free - reserve:
            st.update(status="skipped_disk",
                      detail=f"needs {gb(need)} more, {gb(free)} free (keeping {gb(reserve)} spare)")
        elif a.dry_run:
            planned[0] += need
            st.update(status="would_download", detail=f"{gb(need)} to download")
        else:
            print(f"downloading {repo}: {gb(need)}")
            status, detail = dw.fetch(repo, ignore, allow)
            st.update(status="downloaded" if status == "ok" else status if status == "cached"
                      else classify(detail), detail=f"{gb(need)}" if status == "ok" else detail)
        print(f"  {st['status']:14s} {repo}  {st['detail']}")
        repo_state[repo] = st
        return st

    for cpath in a.config:
        cfg = DemoConfig(cpath)
        for task in tasks:
            if task not in (cfg.raw.get("settings") or {}):
                continue
            for key in sorted(dw.model_keys(cfg, a.machine, task)):
                spec = cfg.models[key]
                deliberate = any(cfg.settings(task, p, a.machine).get("model") == key
                                 and cfg.settings(task, p, a.machine).get("prepare") is False
                                 for p in cfg.raw["settings"][task])
                allow = [spec["file"]] if spec.get("file") else None
                repos = ([spec["repo"]] if "/" in spec["repo"] else []) + dw.DEPENDENCIES.get(key, [])
                m = {"config": cpath, "task": task, "model": key, "label": spec.get("label", key),
                     "deliberate_oom": deliberate, "repos": repos}
                models.append(m)
                # 1. GPU memory
                need_vram = VRAM_GB.get(key)
                if need_vram is None and allow and "/" in spec["repo"]:
                    try:
                        files, _ = remote_files(api, spec["repo"], None, allow)
                        need_vram = sum(s for _, s in files) / GB + GGUF_OVERHEAD_GB
                    except Exception:  # noqa: BLE001 - size unknown: not checked
                        pass
                m["vram_need_gb"] = round(need_vram, 1) if need_vram else None
                # a "16 GB" card reports 15.9 (GiB): the needs above are nominal sizes, so allow the gap
                if need_vram and need_vram > a.vram_gb + VRAM_SLACK_GB and not deliberate:
                    m.update(status="skipped_vram",
                             detail=f"needs about {need_vram:.1f} GB of GPU memory, this GPU has {a.vram_gb:.1f} GB")
                    print(f"  skipped_vram   {key}  {m['detail']}")
                    continue
                # 2. disk: everything this model still needs must fit, or none of it is downloaded
                pats = {r: (dw.IGNORE.get(r), allow if r == spec["repo"] else None) for r in repos}
                need = sum(probe(r, *pats[r]).get("missing", 0) for r in repos if r not in repo_state)
                free = shutil.disk_usage(cache).free - planned[0]
                if need > free - reserve:
                    m.update(status="skipped_disk", detail=f"needs {gb(need)} more, {gb(free)} free "
                                                           f"(keeping {gb(reserve)} spare)")
                    print(f"  skipped_disk   {key}  {m['detail']}")
                    continue
                # 3. download, repo by repo
                states = [do_repo(r, *pats[r]) for r in repos]
                bad = [s for s in states if s["status"] not in ("cached", "downloaded", "would_download")]
                if not bad:
                    m.update(status="ready" if not a.dry_run else "would_be_ready", detail="")
                else:
                    m.update(status=bad[0]["status"], detail="; ".join(f"{s['repo']}: {s['detail']}" for s in bad))
                if deliberate:
                    note = "meant to run out of memory on stage" + (
                        f" (needs ~{need_vram:.0f} GB)" if need_vram else "")
                    m["detail"] = f"{m['detail']}; {note}" if m["detail"] else note

    free_end = shutil.disk_usage(cache).free
    (out / "plan.json").write_text(json.dumps(
        {"machine": a.machine, "vram_gb": a.vram_gb, "cache": str(cache), "free_start": free_start,
         "free_end": free_end, "repos": list(repo_state.values()), "models": models}, indent=2))
    with open(out / "models.tsv", "w", encoding="utf-8") as f:
        for m in models:
            # "-" for empty fields: bash's `read` merges two tabs in a row and shifts the columns
            f.write("\t".join([m["config"], m["task"], m["model"], m["status"],
                               "deliberate_oom" if m["deliberate_oom"] else "-",
                               m["detail"].replace("\t", " ") or "-"]) + "\n")
    lines = [f"Cache `{cache}`: {gb(free_start)} free before, {gb(free_end)} after "
             f"(keeping {a.reserve_gb:.0f} GB spare). GPU memory used for the check: {a.vram_gb:.0f} GB.", "",
             "| Config | Task | Model | Download | Detail |", "| --- | --- | --- | --- | --- |"]
    for m in models:
        lines.append(f"| {Path(m['config']).name} | {m['task']} | {m['label']} | {m['status']} | "
                     f"{m['detail'].replace('|', '/')} |")
    (out / "downloads.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = [m for m in models if m["status"] not in ("ready", "would_be_ready")]
    print(f"{len(models) - len(bad)} of {len(models)} models ready; {gb(free_end)} free")
    return 0


if __name__ == "__main__":
    sys.exit(main())
