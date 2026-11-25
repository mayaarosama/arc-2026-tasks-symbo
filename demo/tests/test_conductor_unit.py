"""Conductor logic and HTTP routes without any worker process (every worker URL is unreachable):
config resolution, the plan, model picks, job requests per task, fallback to replay, uploads,
and every /api route."""
from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from arcdemo.conductor.app import create_app
from arcdemo.conductor.config import DemoConfig
from arcdemo.conductor.core import Conductor
from arcdemo.conductor.tasks.i23d import add_upload
from arcdemo.conductor.tasks.layout import layout_prompt_text, layout_summary, layout_view
from tests.helpers import CONFIG, ROOT


def png_bytes() -> bytes:
    img = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((250, 150, 550, 450), fill=(200, 40, 40, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def segment_of(cfg: DemoConfig, task: str) -> dict:
    return next(s for s in cfg.segments if s["task"] == task)


# ------------------------------------------------------------------ config
def test_extends_is_a_shallow_merge(tmp_path):
    (tmp_path / "base.yaml").write_text(yaml.safe_dump({"a": {"x": 1}, "b": 2}))
    (tmp_path / "child.yaml").write_text(yaml.safe_dump({"extends": "base.yaml", "a": {"y": 3}}))
    from arcdemo.conductor.config import _load_yaml
    assert _load_yaml(tmp_path / "child.yaml") == {"a": {"y": 3}, "b": 2}


def test_lookups_raise_key_error_with_the_name(offline_cfg):
    for call in (lambda: offline_cfg.machine("nope"), lambda: offline_cfg.segment("nope"),
                 lambda: offline_cfg.input("t2i", "nope"), lambda: offline_cfg.settings("t2i", "nope", "dgx")):
        with pytest.raises(KeyError, match="nope"):
            call()


def test_add_input_replaces_same_id(offline_cfg):
    offline_cfg.add_input("i23d", {"id": "up-1", "label": "a"})
    offline_cfg.add_input("i23d", {"id": "up-1", "label": "b"})
    assert [x["label"] for x in offline_cfg.inputs["i23d"] if x["id"] == "up-1"] == ["b"]


def test_model_pick_applies_that_models_settings_but_keeps_protocol_keys(offline_cfg):
    task = next(t for t, ch in offline_cfg.choices.items() if ch)
    seg = segment_of(offline_cfg, task)
    mid = offline_cfg.machines[0]["id"]
    planned = offline_cfg.settings(task, seg["protocol"], mid)
    pick = next(k for k in offline_cfg.choices[task] if k != planned["model"])
    s = offline_cfg.settings(task, seg["protocol"], mid, pick)
    assert s["model"] == pick
    keep = (offline_cfg.raw["settings"][task][seg["protocol"]].get("keep_for_any_model") or [])
    assert all(s[k] == planned[k] for k in keep if k in planned)


def test_public_config_lists_model_options_per_task(offline_cfg):
    pub = offline_cfg.public()
    assert set(pub) == {"machines", "segments", "inputs", "model_labels", "model_options"}
    assert set(pub["model_options"]) == {s["task"] for s in offline_cfg.segments}


def test_unknown_model_choice_and_bad_room_are_reported(tmp_path):
    raw = yaml.safe_load((CONFIG / "demo.yaml").read_text(encoding="utf-8"))
    raw["model_choices"] = {"t2i": {"no-such-model": {}}}
    rooms = json.loads((ROOT / raw["inputs"]["layout"]).read_text(encoding="utf-8"))
    rooms[0]["room_type"] = "garage"
    (tmp_path / "rooms.json").write_text(json.dumps(rooms))
    raw["inputs"]["layout"] = str(tmp_path / "rooms.json")
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError) as e:
        DemoConfig(p)
    assert "model_choices t2i/no-such-model: not in models.yaml" in str(e.value)
    assert "room_type 'garage'" in str(e.value)


# ------------------------------------------------------------------ layout helpers
def test_layout_view_and_prompt_text(offline_cfg):
    item = offline_cfg.inputs["layout"][0]
    v = layout_view(item)
    assert set(v) == {"walls", "door_wall", "ask"} and v["ask"].startswith("Furnish this")
    assert all(len(ends) == 2 for ends in v["walls"].values())
    ours = layout_prompt_text(item)
    gpt = layout_prompt_text(item, 4, "layoutgpt")
    assert ours.startswith("# prompt v") and "=== grammar" in ours
    assert gpt.startswith("# prompt layoutgpt") and "length:" in gpt


def test_layout_summary_counts_the_last_room_and_fixes():
    assert layout_summary(None) is None and layout_summary([]) is None
    objs = [{"room": 0, "issues": ["x"]}, {"room": 1, "issues": []}, {"room": 1, "issues": ["y"]},
            {"room": 1, "removed": 3.0, "issues": []}]
    assert layout_summary(objs) == {"objects": 2, "flagged": 1}
    objs[1]["history"] = [{"issues": ["z"]}]
    objs[1]["fixes"] = 1
    assert layout_summary(objs) == {"objects": 2, "flagged": 1, "first_flagged": 2, "fixes": 1}


# ------------------------------------------------------------------ conductor core
@pytest.fixture()
def cond(offline_cfg, tmp_path):
    c = Conductor(offline_cfg, picks_path=tmp_path / "picks.json")
    yield c
    asyncio.run(c.http.aclose())


def test_job_request_per_task(cond, offline_cfg):
    mid = offline_cfg.machines[0]["id"]
    for task in ("t2i", "i23d", "llm", "layout"):
        seg = segment_of(offline_cfg, task)
        item = offline_cfg.input(task, seg["inputs"][0]) if seg["inputs"][0] != "live" else \
            {"id": "live", "prompt": "a cat"}
        s = cond.settings(task, seg["protocol"], mid)
        job = cond._job_request(task, seg["protocol"], item, s, 30)
        assert job["task"] == task and job["model"] == s["model"] and job["budget_seconds"] == 30
        assert "mode" not in job and "prepare" not in job
        if task in ("t2i", "llm"):
            assert job["prompt"] == item["prompt"]
        if task == "i23d":
            assert job["image_path"] == item["input"]
        if task == "layout":
            assert job["floor_plan"] == item["floor_plan"] and job["room_type"] == item["room_type"]


def test_job_request_music_and_uploaded_picture(cond, offline_cfg, tmp_path):
    job = cond._job_request("music", "iso_time", {"id": "m", "prompt": "oud", "mode": "text", "length_s": 10},
                            {"model": "sa3-small-music"}, 60)
    assert (job["prompt"], job["mode"], job["length_s"]) == ("oud", "text", 10)
    (offline_cfg.session_dir / "uploads").mkdir(parents=True)
    (offline_cfg.session_dir / "uploads" / "a.png").write_bytes(b"\x89PNG")
    job = cond._job_request("i23d", "iso_time", {"id": "up", "upload": "uploads/a.png"}, {"model": "sf3d"}, None)
    assert base64.b64decode(job["image_b64"]) == b"\x89PNG" and "image_path" not in job


@pytest.mark.parametrize("res,budget,expected", [
    ({"status": "ok", "all_outputs_seconds": 4.0}, 10, 4.0),
    ({"status": "ok", "all_outputs_seconds": None, "elapsed_seconds": 2.0}, 10, 2.0),
    ({"status": "budget_exceeded", "all_outputs_seconds": 12.0}, 10, 12.0),
    ({"status": "budget_exceeded", "all_outputs_seconds": 8.0}, 10, 10),
    ({"status": "out_of_memory", "elapsed_seconds": 1.5}, 10, 1.5)])
def test_final_seconds(res, budget, expected):
    assert Conductor._final_seconds(res, budget) == expected


def test_plan_lists_every_machine_and_its_recording(cond, offline_cfg, tmp_path):
    seg = segment_of(offline_cfg, "t2i")
    item = next(i for i in seg["inputs"] if i != "live")
    mid = offline_cfg.machines[0]["id"]
    f = tmp_path / "x.png"
    f.write_bytes(b"x")
    cond.replay.save("t2i", seg["protocol"], mid, item, cond.settings("t2i", seg["protocol"], mid),
                     {"status": "ok", "all_outputs_seconds": 3.0, "completed": 1, "requested": 1}, [f], variant="full")
    cond.replay.save("t2i", seg["protocol"], mid, item, cond.settings("t2i", seg["protocol"], mid),
                     {"status": "ok", "all_outputs_seconds": 3.0, "completed": 1, "requested": 1}, [f])
    plan = cond.plan(seg["id"], item)
    assert [m["id"] for m in plan["machines"]] == [m["id"] for m in offline_cfg.machines]
    row = plan["machines"][0]
    assert row["recording"] == {"seconds": 3.0, "status": "ok", "completed": 1, "requested": 1, "stale": False}
    assert all(v is None for v in row["on_disk"].values())          # health unknown: nothing blocked


def test_set_model_validates_and_persists(cond, offline_cfg, tmp_path):
    task = next(t for t, ch in offline_cfg.choices.items() if ch)
    mid = offline_cfg.machines[0]["id"]
    pick = next(iter(offline_cfg.choices[task]))
    with pytest.raises(ValueError):
        cond.set_model(task, mid, "not-offered")
    with pytest.raises(KeyError):
        cond.set_model("video", mid, None)
    with pytest.raises(KeyError):
        cond.set_model(task, "nope", None)
    cond.health[f"{mid}:{task}"] = {"ok": True, "available": []}
    with pytest.raises(ValueError, match="not downloaded"):
        cond.set_model(task, mid, pick)
    cond.health.clear()
    assert cond.set_model(task, mid, pick) == {task: {mid: pick}}
    assert json.loads((tmp_path / "picks.json").read_text()) == {task: {mid: pick}}
    again = Conductor(offline_cfg, picks_path=tmp_path / "picks.json")
    assert again.model_picks == {task: {mid: pick}}
    assert cond.set_model(task, mid, None) == {}


def test_unreadable_or_stale_picks_are_dropped(offline_cfg, tmp_path):
    p = tmp_path / "picks.json"
    p.write_text("{not json")
    assert Conductor(offline_cfg, picks_path=p).model_picks == {}
    p.write_text(json.dumps({"t2i": {"no-machine": "sd15", offline_cfg.machines[0]["id"]: "no-model"}}))
    assert Conductor(offline_cfg, picks_path=p).model_picks == {}


def test_env_warnings_flag_differing_library_versions(cond):
    cond.health = {"a:t2i": {"ok": True, "tasks": ["t2i"], "machine": "a", "env_hash": "1"},
                   "b:t2i": {"ok": True, "tasks": ["t2i"], "machine": "b", "env_hash": "2"},
                   "c:t2i": {"ok": False}}
    assert len(cond.env_warnings()) == 1 and "t2i" in cond.env_warnings()[0]


def test_round_against_dead_workers_falls_back_and_logs(cond, offline_cfg, tmp_path):
    seg = segment_of(offline_cfg, "llm")
    item = seg["inputs"][0]
    mid = offline_cfg.machines[0]["id"]
    f = tmp_path / "r.txt"
    f.write_text("hello")
    cond.replay.save("llm", seg["protocol"], mid, item, cond.settings("llm", seg["protocol"], mid),
                     {"status": "ok", "all_outputs_seconds": 2.0, "completed": 1, "tokens": 7, "text": "hello"}, [f])

    async def go():
        rnd = await cond.start_round(seg["id"], item)
        with pytest.raises(RuntimeError):
            await cond.start_round(seg["id"], item)            # one round at a time
        await cond.wait_round(30)
        return rnd
    rnd = asyncio.run(go())
    cell = rnd["cells"][mid]
    assert cell["mode"] == "REPLAY" and cell["status"] == "ok" and cell["tokens"] == 7
    assert "unreachable" in cell["note"]
    others = [c for m, c in rnd["cells"].items() if m != mid and not c["full_recording"]]
    assert all(c["status"] == "unavailable" for c in others)
    lines = (offline_cfg.session_dir / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["badge"] == "REPLAY"
    assert cond.history[-1]["round"] == rnd["id"]


def test_live_round_needs_a_clean_prompt(cond, offline_cfg):
    seg = segment_of(offline_cfg, "t2i")      # the operator may type a prompt on any t2i segment

    async def go(prompt, override=False):
        return await cond.start_round(seg["id"], "live", prompt, override)
    with pytest.raises(ValueError):
        asyncio.run(go("  "))
    with pytest.raises(PermissionError, match="gun"):
        asyncio.run(go("a gun"))


def test_add_upload_offers_the_picture_and_survives_a_restart(cond, offline_cfg):
    item = add_upload(cond, "my photo!.png", png_bytes())
    assert item["id"].startswith("up-") and item["tier"] == "upload" and item["label"] == "my photo"
    assert (offline_cfg.session_dir / item["upload"]).is_file()
    assert any(x["id"] == item["id"] for x in offline_cfg.inputs["i23d"])
    offline_cfg.inputs["i23d"] = [x for x in offline_cfg.inputs["i23d"] if x["id"] != item["id"]]
    Conductor(offline_cfg)
    assert any(x["id"] == item["id"] for x in offline_cfg.inputs["i23d"])


def test_prepare_against_dead_workers_reports_errors(cond, offline_cfg):
    seg = segment_of(offline_cfg, "t2i")

    async def go():
        st = await cond.prepare(seg["id"])
        cond._prepare_task.cancel()
        return st
    st = asyncio.run(go())
    assert set(st) == {m["id"] for m in offline_cfg.machines}
    assert all(p["status"] in ("error", "skipped") for p in st.values())
    assert any("not reachable" in p.get("message", "") for p in st.values())


# ------------------------------------------------------------------ HTTP routes
@pytest.fixture()
def api(offline_cfg):
    with TestClient(create_app(offline_cfg)) as c:
        yield c


def test_pages_static_and_config(api):
    assert api.get("/", follow_redirects=False).headers["location"] == "/operator"
    for page in ("/stage", "/operator"):
        r = api.get(page)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    assert api.get("/static/common.js").headers["content-type"].startswith("text/javascript")
    assert api.get("/api/config").json()["machines"]
    st = api.get("/api/state").json()
    assert st["view"] == "title" and st["round"] is None


def test_view_route(api):
    assert api.post("/api/view", json={"view": "summary"}).json() == {"view": "summary"}
    assert api.post("/api/view", json={"view": "disco"}).status_code == 400


def test_moderate_route(api):
    assert api.post("/api/moderate", json={"prompt": "a cat"}).json()["ok"]
    assert not api.post("/api/moderate", json={"prompt": "a gun"}).json()["ok"]


def test_plan_and_prompt_routes(api, offline_cfg):
    assert api.get("/api/plan", params={"segment_id": "nope"}).status_code == 404
    seg = segment_of(offline_cfg, "llm")
    assert api.get("/api/plan", params={"segment_id": seg["id"]}).json()["task"] == "llm"
    r = api.get("/api/prompt", params={"segment_id": seg["id"], "input_id": seg["inputs"][0]}).json()
    assert r["task"] == "llm" and "=== system ===" in r["text"]
    lay = segment_of(offline_cfg, "layout")
    r = api.get("/api/prompt", params={"segment_id": lay["id"], "input_id": lay["inputs"][0]}).json()
    assert r["task"] == "layout" and "=== grammar" in r["text"]
    t2i = segment_of(offline_cfg, "t2i")
    item = next(i for i in t2i["inputs"] if i != "live")
    assert api.get("/api/prompt", params={"segment_id": t2i["id"], "input_id": item}).json()["text"] is None
    assert api.get("/api/prompt", params={"segment_id": seg["id"], "input_id": "nope"}).status_code == 404


def test_layout_routes(api, offline_cfg):
    item = offline_cfg.inputs["layout"][0]
    r = api.get("/api/layout_view", params={"input_id": item["id"]}).json()
    assert r["id"] == item["id"] and "walls" in r
    assert api.get("/api/layout_view", params={"input_id": "nope"}).status_code == 404
    cat = api.get("/api/layout_catalog").json()
    assert cat["assets_url"] == "/layout-assets/" and cat["furniture"]
    model = next(iter(cat["furniture"].values()))["model"]
    assert api.get(f"/layout-assets/{model}").status_code == 200


def test_model_route_errors(api, offline_cfg):
    mid = offline_cfg.machines[0]["id"]
    assert api.post("/api/model", json={"task": "video", "machine": mid}).status_code == 404
    assert api.post("/api/model", json={"task": "t2i", "machine": mid, "model": "nope"}).status_code == 400


def test_upload_routes(api):
    spec = api.get("/api/i23d/spec").json()
    assert spec["spec"] and isinstance(spec["cutout_installed"], bool)
    assert api.post("/api/i23d/upload", json={"data_url": "hello"}).status_code == 400
    assert api.post("/api/i23d/upload", json={"data_url": "data:image/png;base64,@@@"}).status_code == 400
    assert api.post("/api/i23d/upload", json={"data_url": "data:image/png;base64,"
                                              + base64.b64encode(b"not a picture").decode()}).status_code == 422
    good = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    r = api.post("/api/i23d/upload", json={"name": "mug.png", "data_url": good})
    assert r.status_code == 200 and r.json()["id"].startswith("up-")
    assert api.get(r.json()["input_url"]).status_code == 200


def test_round_cancel_and_cell_routes(api, offline_cfg):
    seg = segment_of(offline_cfg, "t2i")
    item = next(i for i in seg["inputs"] if i != "live")
    assert api.post("/api/round", json={"segment_id": "nope", "input_id": item}).status_code == 409
    r = api.post("/api/round", json={"segment_id": seg["id"], "input_id": item})
    assert r.status_code == 200
    mid = offline_cfg.machines[0]["id"]
    assert api.post(f"/api/cell/{mid}/stop").json() == {"ok": True}
    assert api.post(f"/api/cell/{mid}/replay").json() == {"ok": True}
    assert api.post("/api/cancel").json() == {"ok": True}
    assert api.post("/api/prepare", json={"segment_id": seg["id"]}).status_code == 200


def test_input_media_is_served(api, offline_cfg):
    rel = Path(offline_cfg.raw["inputs"]["t2i"]).as_posix()
    assert api.get(f"/media-input/{rel}").status_code == 200
