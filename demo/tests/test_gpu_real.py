"""Real models on this machine's GPU (marker `gpu`, skipped by default).

One tiny job per task through the real worker code path (load -> job -> output), offline, with the
workstation worker config. Each test skips when its library is not in the running venv or the weights
are not on disk, so run it once per task venv:

    scripts/common/run_gpu_tests.sh        # every task venv in turn

Uses the small models (SD 1.5, SF3D, Stable Audio 3 Small, Qwen3.5 4B) so it fits next to other GPU users.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arcdemo.weights import on_disk
from arcdemo.worker.app import create_app, load_worker_config
from tests.helpers import CONFIG, ROOT

pytestmark = pytest.mark.gpu

CASES = {
    "t2i": ("sd15", "diffusers", {"prompt": "a red cube on a table", "width": 512, "height": 512, "steps": 4}),
    "i23d": ("sf3d", "sf3d", {"image_path": "data/i23d/inputs/i23d-01.png"}),
    "music": ("sa3-small-music", "stable_audio_3", {"prompt": "calm piano", "length_s": 5}),
    "llm": ("qwen35-4b", "httpx", {"prompt": "Say hello in five words.", "max_tokens": 16}),
    "layout": ("qwen35-4b-layout", "httpx", {"room_type": "bedroom", "floor_plan": [[0, 0], [3.8, 0], [3.8, 3.4], [0, 3.4]],
                                              "door": [0.6, 0], "max_objects": 3, "max_tokens": 120}),
}


VENV_OF = {"t2i": "t2i", "i23d": "i23d", "music": "music", "llm": "llm", "layout": "llm"}


def _wrong_venv(task: str) -> str | None:
    """Running in another task's venv (.venv-<task>): that venv lacks this task's libraries."""
    name = Path(sys.prefix).name
    if name.startswith(".venv-") and name != f".venv-{VENV_OF[task]}":
        return name
    return None


@pytest.mark.parametrize("task", list(CASES))
def test_real_model_runs_offline(task, tmp_path, monkeypatch):
    key, lib, job = CASES[task]
    if _wrong_venv(task):
        pytest.skip(f"runs in .venv-{VENV_OF[task]}, not {_wrong_venv(task)}")
    if lib != "sf3d" and importlib.util.find_spec(lib) is None:
        pytest.skip(f"{lib} not installed in this venv (run from .venv-{VENV_OF[task]})")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(ROOT)                      # job paths and llama_server are relative to demo/
    cfg = load_worker_config(str(CONFIG / "worker-workstation.yaml"), [task])
    cfg["results_dir"] = str(tmp_path)
    if not on_disk(cfg["models"][key], key):
        pytest.skip(f"{key} weights are not in {os.environ.get('HF_HUB_CACHE')}")
    with TestClient(create_app(cfg)) as c:
        rid = c.post("/jobs", json={"task": task, "model": key, "input_id": "gpu-test", "seed": 1, **job}).json()["run_id"]
        end = time.time() + 900
        while (res := c.get(f"/jobs/{rid}").json())["status"] not in ("ok", "budget_exceeded", "out_of_memory",
                                                                         "error", "cancelled"):
            assert time.time() < end, "no result after 15 minutes"
            time.sleep(0.5)
        c.post("/unload", json={})
    assert res["status"] == "ok", res["message"]
    assert res["completed"] >= 1 and all((tmp_path / n).is_file() for n in res["outputs"])
    if task in ("llm", "layout"):
        assert res["tokens"] > 0 and res["tokens_per_second"]
    if task == "layout":
        assert res["objects"]
