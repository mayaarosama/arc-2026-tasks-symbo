"""Acceptance check: every model this worker will run loads and generates with networking OFF.

    python scripts/verify_offline.py --config config/worker-workstation.yaml --task t2i

Runs through the real worker code path (load -> job -> output), in-process, with
HF_HUB_OFFLINE=1. Any model that tries to reach the network fails here, not on stage.
Run after download_weights.py, and again after any pip change. Writes a line per model to
<results_dir>/verify-<task>.log.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient                          # noqa: E402

from arcdemo.conductor.config import DemoConfig                    # noqa: E402
from arcdemo.logs import setup_logging                             # noqa: E402
from arcdemo.worker.app import ROOT, create_app, load_worker_config  # noqa: E402

TINY = {"t2i": {"prompt": "a red cube on a table", "width": 512, "height": 512, "steps": 2, "batch": 1},
        "i23d": {"image_path": "data/i23d/inputs/i23d-01.png"},
        "music": {"prompt": "test", "length_s": 5}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="worker config, e.g. config/worker-dgx.yaml")
    ap.add_argument("--task", required=True)
    ap.add_argument("--demo", default="config/demo.yaml")
    a = ap.parse_args()
    cfg = load_worker_config(str(Path(a.config).resolve()), [a.task])
    os.chdir(ROOT)
    setup_logging(Path(cfg["results_dir"]) / f"verify-{a.task}.log")
    demo = DemoConfig(a.demo)
    keys = sorted({demo.settings(a.task, p, cfg["machine"])["model"]
                   for p in demo.raw["settings"][a.task]
                   if demo.settings(a.task, p, cfg["machine"]).get("mode") != "replay"})
    failures = 0
    with TestClient(create_app(cfg)) as c:
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
