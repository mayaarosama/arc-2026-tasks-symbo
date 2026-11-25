"""Worker app pieces not covered by test_worker_mock.py: config loading, per-task settings echo,
auth, file serving, health / env reports, and the guards on every route."""
from __future__ import annotations

import base64
import time

import pytest
import yaml
from fastapi.testclient import TestClient

from arcdemo.schemas import JobRequest
from arcdemo.worker import app as W
from arcdemo.worker import tasks
from tests.helpers import CONFIG, mock_worker_cfg, wait_job
from tests.test_conductor_unit import png_bytes


def test_load_worker_config_applies_overrides_and_catalogue(tmp_path):
    raw = yaml.safe_load((CONFIG / "worker-mock-dgx.yaml").read_text())
    raw["model_overrides"] = {"sd15": {"dtype": "fp32"}}
    raw["results_dir"] = str(tmp_path / "res")
    p = tmp_path / "w.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = W.load_worker_config(str(p), ["t2i", "layout"])
    assert cfg["models"]["sd15"]["dtype"] == "fp32" and cfg["tasks"] == ["t2i", "layout"]
    assert len(cfg["layout_catalog_sha"]) == 12 and (tmp_path / "res").is_dir()
    assert "layout_catalog_sha" not in W.load_worker_config(str(p), ["t2i"])


@pytest.mark.parametrize("job,keys", [
    ({"task": "t2i"}, {"width", "height", "steps", "guidance", "batch", "seed"}),
    ({"task": "i23d", "resolution": 512}, {"resolution", "seed"}),
    ({"task": "music", "length_s": 5}, {"length_s", "seed", "repeat_until_budget"}),
    ({"task": "llm"}, {"streams", "max_tokens", "temperature", "seed", "repeat_until_budget"}),
    ({"task": "layout", "repairs": 1, "layout_prompt": "layoutgpt"},
     {"room_type", "max_objects", "streams", "max_tokens", "temperature", "seed", "repeat_until_budget",
      "repairs", "review_rounds", "layout_prompt"})])
def test_request_settings_echo_per_task(job, keys):
    assert set(tasks.settings_echo(JobRequest(model="m", input_id="i", **job))) == keys


def test_save_upload_guards(tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 10
    p = tasks.save_upload(base64.b64encode(png).decode(), tmp_path, "run_1-a")
    assert p == tmp_path / "inputs" / "run_1-a.png" and p.read_bytes() == png
    with pytest.raises(ValueError, match="run_id"):
        tasks.save_upload(base64.b64encode(png).decode(), tmp_path, "../evil")
    with pytest.raises(ValueError, match="base64"):
        tasks.save_upload("@@@", tmp_path, "r")
    with pytest.raises(ValueError, match="PNG"):
        tasks.save_upload(base64.b64encode(b"GIF89a").decode(), tmp_path, "r")


def test_model_manager_refuses_unknown_and_other_task_models(tmp_path):
    mm = W.ModelManager(mock_worker_cfg("dgx", tmp_path, tasks=["t2i"]))
    with pytest.raises(KeyError):
        mm.get("no-such-model")
    llm_key = next(k for k, s in mm.cfg["models"].items() if s["task"] == "llm")
    with pytest.raises(ValueError, match="llm model"):
        mm.get(llm_key)
    assert mm.devices_for("sd15") == mm.cfg["devices"]["t2i"]


def test_env_and_memory_reports():
    env = W.env_report({"machine": "m"})
    assert env["machine"] == "m" and len(env["env_hash"]) == 12
    assert not any(p.startswith(("torch", "nvidia-", "triton")) for p in env["packages"])
    mem = W.memory_report()
    assert mem["rss_mb"] > 0 and mem["ram_free_mb"] > 0


def test_short_error_text_is_capped():
    assert W._short(ValueError("x" * 1000)).endswith("…") and len(W._short(ValueError("x" * 1000))) == 401


def test_health_env_and_files(worker_client, tmp_path):
    c = worker_client("workstation")
    h = c.get("/health").json()
    assert h["machine"] == "workstation" and h["mock"] and not h["busy"]
    assert set(h["available"]) == {k for k, s in mock_worker_cfg("workstation", tmp_path)["models"].items()
                                   if s["task"] in h["tasks"]}
    assert "last_error" in c.get("/env").json()
    (tmp_path / "out.txt").write_text("hi")
    assert c.get("/files/out.txt").text == "hi"
    assert c.get("/files/.hidden").status_code == 400
    assert c.get("/files/missing.txt").status_code == 404
    assert c.get("/files/..%5Cx").status_code in (400, 404)


def test_job_route_guards(worker_client):
    c = worker_client("dgx", tasks=["t2i"])
    assert c.post("/jobs", json={"task": "llm", "model": "x", "input_id": "i"}).status_code == 400
    assert c.get("/jobs/nope").status_code == 404
    assert c.post("/jobs/nope/cancel").status_code == 404


def test_load_then_unload_through_the_api(worker_client):
    c = worker_client("dgx", tasks=["t2i"])
    assert c.post("/load", json={"model": "sd15"}).json() == {"model": "sd15", "status": "loading"}
    end = time.time() + 10
    while "sd15" not in c.get("/health").json()["loaded"]:
        assert time.time() < end
        time.sleep(0.05)
    r = c.post("/unload", json={"keep_task": "t2i"}).json()
    assert r["unloaded"] == [] and r["loaded"] == ["sd15"]
    assert c.post("/unload", json={}).json()["unloaded"] == ["sd15"]


def test_failed_load_is_reported_in_env(worker_client):
    c = worker_client("cpu", tasks=["i23d"])
    c.post("/load", json={"model": "trellis2-4b"})          # CUDA-only model on the CPU mock
    end = time.time() + 10
    while not c.get("/env").json()["last_error"]:
        assert time.time() < end
        time.sleep(0.05)
    assert "requires CUDA" in c.get("/env").json()["last_error"]


def test_token_is_required_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_DEMO_TOKEN", "s3cret")
    with TestClient(W.create_app(mock_worker_cfg("dgx", tmp_path, tasks=["t2i"]))) as c:
        assert c.get("/health").status_code == 401
        assert c.get("/health", headers={"X-Demo-Token": "wrong"}).status_code == 401
        assert c.get("/health", headers={"X-Demo-Token": "s3cret"}).status_code == 200


def test_worker_honours_the_configured_token_env(monkeypatch, tmp_path):
    """demo.yaml names the variable (token_env); a worker started with the same name must use it."""
    monkeypatch.delenv("ARC_DEMO_TOKEN", raising=False)
    monkeypatch.setenv("MY_TOKEN", "abc")
    cfg = mock_worker_cfg("dgx", tmp_path, tasks=["t2i"])
    cfg["token_env"] = "MY_TOKEN"
    with TestClient(W.create_app(cfg)) as c:
        assert c.get("/health").status_code == 401


def test_job_with_uploaded_picture_is_written_before_the_clock(worker_client, tmp_path):
    c = worker_client("dgx", tasks=["i23d"])
    job = {"task": "i23d", "model": "sf3d", "input_id": "up-x", "image_b64": base64.b64encode(png_bytes()).decode()}
    rid = c.post("/jobs", json=job).json()["run_id"]
    assert wait_job(c, rid)["status"] == "ok"
    assert (tmp_path / "inputs" / f"{rid}.png").is_file()
