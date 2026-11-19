"""Worker job lifecycle against the mock backend (no GPU): the acceptance checks
'a run that exceeds the budget returns cleanly rather than hanging' and
'the TRELLIS.2 memory failure is reported cleanly as out_of_memory'."""
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arcdemo.worker.app import create_app, load_worker_config

CONFIG = Path(__file__).resolve().parents[1] / "config"


def client(machine: str, speed: float, tmp_path: Path) -> TestClient:
    cfg = load_worker_config(str(CONFIG / f"worker-mock-{machine}.yaml"), ["t2i", "i23d", "music"])
    cfg["mock_speed"] = speed
    cfg["results_dir"] = str(tmp_path)
    return TestClient(create_app(cfg))


def run(c: TestClient, job: dict, timeout: float = 20) -> dict:
    r = c.post("/jobs", json=job)
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    end = time.time() + timeout
    while time.time() < end:
        res = c.get(f"/jobs/{rid}").json()
        if res["status"] in ("ok", "budget_exceeded", "out_of_memory", "error", "cancelled"):
            return res
        time.sleep(0.05)
    pytest.fail(f"job did not finish: {res}")


T2I = {"task": "t2i", "model": "sd35-medium-not5", "input_id": "t2i-01", "prompt": "a test",
       "seed": 7, "width": 512, "height": 512, "steps": 4}


def test_ok_batch_records_every_output(tmp_path):
    with client("dgx", 0.02, tmp_path) as c:
        res = run(c, {**T2I, "batch": 8, "budget_seconds": 30})
    assert res["status"] == "ok"
    assert res["completed"] == 8 and len(res["output_times"]) == 8
    assert res["first_output_seconds"] <= res["all_outputs_seconds"]
    assert all((tmp_path / name).exists() for name in res["outputs"])


def test_budget_cutoff_returns_finished_images_only(tmp_path):
    with client("workstation", 0.5, tmp_path) as c:   # ~0.06 s/step at 512² -> ~0.25 s per image
        res = run(c, {**T2I, "batch": 20, "budget_seconds": 1.0})
    assert res["status"] == "budget_exceeded"
    assert 0 < res["completed"] < 20
    assert res["all_outputs_seconds"] <= 1.0 + 0.3


def test_trellis2_on_16gb_is_out_of_memory_not_error(tmp_path):
    with client("workstation", 0.02, tmp_path) as c:
        res = run(c, {"task": "i23d", "model": "trellis2-4b", "input_id": "i23d-01",
                      "image_path": "unused-by-mock.png", "budget_seconds": None})
        health = c.get("/health").json()
    assert res["status"] == "out_of_memory", res
    assert "trellis2-4b" not in health["loaded"]          # memory released after the OOM


def test_unknown_model_is_an_error_with_message(tmp_path):
    with client("cpu", 0.02, tmp_path) as c:
        res = run(c, {**T2I, "model": "no-such-model"})
    assert res["status"] == "error" and "no-such-model" in res["message"]


def test_unknown_job_field_is_rejected_not_ignored(tmp_path):
    with client("cpu", 0.02, tmp_path) as c:
        r = c.post("/jobs", json={**T2I, "step": 4})
    assert r.status_code == 422 and "step" in r.text


def test_exclusive_models_never_holds_two(tmp_path):
    with client("workstation", 0.02, tmp_path) as c:      # worker-mock-workstation: exclusive_models
        assert run(c, {**T2I, "model": "sd35-medium-not5"})["status"] == "ok"
        assert run(c, {**T2I, "model": "sd35-medium-t5int8"})["status"] == "ok"
        assert c.get("/health").json()["loaded"] == ["sd35-medium-t5int8"]


def test_second_job_while_busy_is_rejected(tmp_path):
    with client("cpu", 1.0, tmp_path) as c:
        first = c.post("/jobs", json={**T2I, "steps": 50}).json()
        assert c.post("/jobs", json=T2I).status_code == 409
        c.post(f"/jobs/{first['run_id']}/cancel")
