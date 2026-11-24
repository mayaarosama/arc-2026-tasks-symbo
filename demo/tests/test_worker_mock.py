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
    cfg = load_worker_config(str(CONFIG / f"worker-mock-{machine}.yaml"), ["t2i", "i23d", "music", "llm", "layout"])
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


def test_unload_keep_task_frees_the_other_tasks_only(tmp_path):
    with client("dgx", 0.02, tmp_path) as c:
        c.post("/unload", json={})                          # drop the preload, start from nothing
        assert run(c, {**T2I, "model": "sd15"})["status"] == "ok"
        assert run(c, {"task": "i23d", "model": "sf3d", "input_id": "i23d-01",
                       "image_path": "data/i23d/inputs/i23d-01.png"})["status"] == "ok"
        # exclusive by default: loading sf3d already dropped sd15 in this one worker
        assert c.get("/health").json()["loaded"] == ["sf3d"]
        r = c.post("/unload", json={"keep_task": "i23d"}).json()
        assert r["unloaded"] == [] and r["loaded"] == ["sf3d"]
        r = c.post("/unload", json={"keep_task": "layout"}).json()
        assert r["unloaded"] == ["sf3d"] and r["loaded"] == []
        h = c.get("/health").json()
        assert h["rss_mb"] > 0 and h["ram_free_mb"] > 0      # shown in the operator's table


def test_unload_is_refused_while_a_job_runs(tmp_path):
    with client("cpu", 1.0, tmp_path) as c:
        first = c.post("/jobs", json={**T2I, "steps": 50}).json()
        assert c.post("/unload", json={"keep_task": "llm"}).status_code == 409
        c.post(f"/jobs/{first['run_id']}/cancel")


def test_second_job_while_busy_is_rejected(tmp_path):
    with client("cpu", 1.0, tmp_path) as c:
        first = c.post("/jobs", json={**T2I, "steps": 50}).json()
        assert c.post("/jobs", json=T2I).status_code == 409
        c.post(f"/jobs/{first['run_id']}/cancel")


LLM = {"task": "llm", "model": "qwen35-4b", "input_id": "llm-01", "prompt": "a test", "seed": 3}


def test_llm_single_reply_counts_tokens(tmp_path):
    with client("dgx", 0.02, tmp_path) as c:
        res = run(c, {**LLM, "max_tokens": 20})
    assert res["status"] == "ok" and res["completed"] == 1 == res["requested"]
    assert res["tokens"] == 20 and res["tokens_per_second"] > 0
    assert res["first_token_seconds"] <= res["all_outputs_seconds"]
    assert res["text"].startswith("[MOCK]") and (tmp_path / res["outputs"][0]).exists()


def test_llm_throughput_stops_at_the_bell(tmp_path):
    with client("workstation", 0.5, tmp_path) as c:      # ~0.013 s/token per stream
        res = run(c, {**LLM, "streams": 4, "max_tokens": 30, "repeat_until_budget": True, "budget_seconds": 1.5})
    assert res["status"] == "ok" and res["requested"] == 0
    assert res["completed"] >= 4                          # whole replies finished before the bell
    assert res["tokens"] > 30 * res["completed"] - 1      # plus the partial replies cut at the bell
    assert res["all_outputs_seconds"] <= 1.5


ROOM = {"task": "layout", "model": "qwen35-4b-layout", "input_id": "lay-01", "seed": 8, "room_type": "bedroom",
        "floor_plan": [[0, 0], [3.8, 0], [3.8, 3.4], [0, 3.4]], "door": [0.6, 0], "max_objects": 5,
        "max_tokens": 400}


def test_layout_places_objects_one_at_a_time(tmp_path):
    import json
    with client("dgx", 0.02, tmp_path) as c:
        res = run(c, ROOM)
    assert res["status"] == "ok" and res["completed"] == 1 == res["requested"]
    objs = res["objects"]
    assert 1 <= len(objs) <= 5 and [o["i"] for o in objs] == list(range(len(objs)))
    assert all(not o["issues"] for o in objs)                 # the mock packer never breaks a rule
    assert [o["t"] for o in objs] == sorted(o["t"] for o in objs)
    saved = json.loads((tmp_path / res["outputs"][0]).read_text())
    assert saved["summary"]["objects"] == len(objs) and res["tokens"] > 0


def test_layout_throughput_counts_rooms_at_the_bell(tmp_path):
    with client("workstation", 0.1, tmp_path) as c:
        res = run(c, {**ROOM, "streams": 4, "temperature": 0.7, "repeat_until_budget": True, "budget_seconds": 1.5})
    assert res["status"] == "ok" and res["requested"] == 0
    assert res["completed"] >= 4 and res["all_outputs_seconds"] <= 1.5
    assert max(o["room"] for o in res["objects"]) >= 1      # stream 0 moved on to its next room


def test_layout_without_floor_plan_is_an_error(tmp_path):
    with client("cpu", 0.02, tmp_path) as c:
        res = run(c, {**ROOM, "floor_plan": None})
    assert res["status"] == "error" and "floor_plan" in res["message"]


# ---------------------------------------------------------------- layout self-correction ------
def mistakes_client(tmp_path: Path) -> TestClient:
    """A mock layout worker that writes its first object twice: an overlap to fix."""
    cfg = load_worker_config(str(CONFIG / "worker-mock-dgx.yaml"), ["layout"])
    cfg.update(mock_speed=0.01, results_dir=str(tmp_path), mock_layout_mistakes=True)
    return TestClient(create_app(cfg))


def test_layout_mistake_is_only_flagged_without_self_correction(tmp_path):
    with mistakes_client(tmp_path) as c:
        res = run(c, ROOM)
    objs = res["objects"]
    assert objs[1]["issues"] == ["overlap"] and "history" not in objs[1]
    assert "repairs" not in res["settings"]


def test_layout_repair_rewrites_the_flagged_line_at_once(tmp_path):
    import json
    with mistakes_client(tmp_path) as c:
        res = run(c, {**ROOM, "repairs": 2})
    assert res["status"] == "ok" and res["settings"]["repairs"] == 2
    objs = res["objects"]
    bad = objs[1]
    assert [h["op"] for h in bad["history"]] == ["place", "repair"]
    assert bad["history"][0]["issues"] == ["overlap"] and bad["issues"] == [] and bad["fixes"] == 1
    assert bad["history"][1]["t"] >= bad["t"]
    assert all(not o["issues"] for o in objs) and len(objs) == ROOM["max_objects"]
    assert [o["i"] for o in objs] == list(range(len(objs)))       # the room went on after the repair
    saved = json.loads((tmp_path / res["outputs"][0]).read_text())
    s = saved["summary"]
    # every problem of the first pass was fixed (the repaired copy can push a later mock object into trouble too)
    assert s["first_flagged"] >= 1 and s["fixed"] == s["first_flagged"]
    assert (s["clean"], s["removed"]) == (len(objs), 0)
    assert "[repair]" in saved["raw"]


def test_layout_review_fixes_the_finished_room(tmp_path):
    import json
    with mistakes_client(tmp_path) as c:
        res = run(c, {**ROOM, "review_rounds": 2})
    objs = res["objects"]
    live = [o for o in objs if o.get("removed") is None]
    assert all(not o["issues"] for o in live)
    bad = objs[1]
    assert bad["history"][0]["issues"] == ["overlap"] and bad["history"][-1]["op"] in ("move", "remove")
    saved = json.loads((tmp_path / res["outputs"][0]).read_text())
    assert "[review]" in saved["raw"] and saved["summary"]["first_flagged"] == 1


def test_uploaded_image_is_checked_before_it_is_saved(tmp_path):
    import base64

    from arcdemo.worker.app import save_upload
    good = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 10).decode()
    assert save_upload(good, tmp_path, "abc123").read_bytes().startswith(b"\x89PNG")
    with pytest.raises(ValueError, match="PNG"):
        save_upload(base64.b64encode(b"GIF89a").decode(), tmp_path, "abc")
    with pytest.raises(ValueError, match="run_id"):
        save_upload(good, tmp_path, "../../evil")
    with pytest.raises(ValueError, match="base64"):
        save_upload("***", tmp_path, "abc")


def test_layout_with_layoutgpt_prompt_places_objects_from_css_lines(tmp_path):
    with client("dgx", 0.02, tmp_path) as c:
        res = run(c, {**ROOM, "layout_prompt": "layoutgpt", "max_tokens": 800})
    assert res["status"] == "ok" and res["settings"]["prompt"] == "layoutgpt"
    assert res["settings"]["layout_prompt"] == "layoutgpt"
    objs = res["objects"]
    assert objs and all(o["css"].endswith("m;}") and "written_size" in o for o in objs)
