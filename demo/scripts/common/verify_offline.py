"""Acceptance check: every model this worker will run loads and generates with networking OFF.

    python -m scripts.common.verify_offline --config config/worker-workstation.yaml --task t2i

Runs through the real worker code path (load -> job -> output), in-process, with
HF_HUB_OFFLINE=1. Any model that tries to reach the network fails here, not on stage.
Run after download_weights.py, and again after any pip change. Writes a line per model to
<results_dir>/verify-<task>.log.
"""
import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from arcdemo import imageprep
from arcdemo.conductor.config import DemoConfig
from arcdemo.logs import setup_logging
from arcdemo.worker.app import ROOT, create_app, load_worker_config

TINY = {"t2i": {"prompt": "a red cube on a table", "width": 512, "height": 512, "steps": 2, "batch": 1},
        "i23d": {"image_path": "data/i23d/inputs/i23d-01.png"},
        "music": {"prompt": "test", "length_s": 5},
        "llm": {"prompt": "Say hello in five words.", "max_tokens": 16},
        "layout": {"room_type": "bedroom", "floor_plan": [[0, 0], [3.8, 0], [3.8, 3.4], [0, 3.4]],
                   "door": [0.6, 0], "max_objects": 3, "max_tokens": 120}}


def verify_cutout() -> int:
    """The conductor's background removal for uploaded pictures (arcdemo/imageprep.py): the model is on
    disk, onnxruntime loads it, and it finds a known object's outline in a stage picture put on a
    plain background and saved as JPEG (no network involved at any point)."""
    path = imageprep.cutout_model_path()
    if path is None:
        print("FAIL  cutout                 model not found (python -m scripts.common.download_weights --cutout)")
        return 1
    try:
        src = Image.open(ROOT / "data/i23d/inputs/i23d-03.png").convert("RGBA").resize((1024, 1024))
        bg = Image.new("RGBA", src.size, (200, 200, 205, 255))
        bg.alpha_composite(src)
        buf = io.BytesIO()
        bg.convert("RGB").save(buf, "JPEG", quality=90)
        t = time.time()
        prep = imageprep.prepare(buf.getvalue())
        seconds = time.time() - t
        mask = np.asarray(imageprep.u2net_mask(bg.convert("RGB"))) >= 128
        truth = np.asarray(src.getchannel("A")) >= 128
        iou = float((mask & truth).sum() / (mask | truth).sum())
    except Exception as e:  # noqa: BLE001 — any failure is the answer
        print(f"FAIL  cutout                 {type(e).__name__}: {e}")
        return 1
    ok = prep.cutout == "u2net" and iou > 0.9
    print(f"{'PASS' if ok else 'FAIL'}  cutout                 outline match {iou:.3f} · {seconds:.1f}s · "
          f"{imageprep.cutout_device()} · {path}")
    return 0 if ok else 1


def main() -> int:
    if "huggingface_hub" in sys.modules:   # read HF_HUB_OFFLINE at its import: offline would not be enforced
        sys.exit("huggingface_hub was imported before HF_HUB_OFFLINE was set: offline mode is not enforced")
    os.environ["HF_HUB_OFFLINE"] = "1"          # before any model library is imported: nothing may
    os.environ["TRANSFORMERS_OFFLINE"] = "1"    # touch the network (the backends import them lazily)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="worker config, e.g. config/worker-dgx.yaml")
    ap.add_argument("--task")
    ap.add_argument("--demo", default="config/demo.yaml")
    ap.add_argument("--model", action="append", help="check only these models.yaml keys (repeatable)")
    ap.add_argument("--cutout", action="store_true",
                    help="instead: the conductor's background removal for uploaded pictures")
    a = ap.parse_args()
    if a.cutout:
        os.chdir(ROOT)
        return verify_cutout()
    if not a.config or not a.task:
        ap.error("give --config and --task (or --cutout)")
    cfg = load_worker_config(str(Path(a.config).resolve()), [a.task])
    os.chdir(ROOT)
    setup_logging(Path(cfg["results_dir"]) / f"verify-{a.task}.log")
    demo = DemoConfig(a.demo)
    keys = sorted({demo.settings(a.task, p, cfg["machine"])["model"]
                   for p in demo.raw["settings"][a.task]
                   if demo.settings(a.task, p, cfg["machine"]).get("mode") != "replay"})
    if a.model:
        keys = sorted(set(a.model))
    failures = 0
    token = os.environ.get("ARC_DEMO_TOKEN")
    headers = {"X-Demo-Token": token} if token else {}
    with TestClient(create_app(cfg), headers=headers) as c:
        for key in keys:
            job = {"task": a.task, "model": key, "input_id": "verify", "seed": 1, **TINY[a.task]}
            rid = c.post("/jobs", json=job).json()["run_id"]
            while (res := c.get(f"/jobs/{rid}").json())["status"] not in (
                    "ok", "budget_exceeded", "out_of_memory", "error", "cancelled"):
                time.sleep(0.5)
            ok = res["status"] == "ok"
            failures += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {key:22s} {res['status']:14s} load={res['load_seconds']}s "
                  f"gen={res['all_outputs_seconds']}s peak={res['peak_memory_mb']}MB {res['message']}")
            c.post("/unload", json={"model": key})
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
