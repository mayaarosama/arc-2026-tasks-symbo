"""End to end: conductor + three mock worker processes. Covers the failure drill:
a dead worker must never hang a round — the cell falls back and says why."""
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from arcdemo.conductor.config import DemoConfig
from arcdemo.conductor.core import Conductor

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def stack(tmp_path, request):
    procs, machines = {}, []
    speeds = getattr(request, "param", {"dgx": 0.01, "workstation": 0.02, "cpu": 0.05})
    for mid, speed in speeds.items():
        wcfg = yaml.safe_load((ROOT / f"config/worker-mock-{mid}.yaml").read_text())
        wcfg.update(mock_speed=speed, results_dir=str(tmp_path / f"results-{mid}"))
        wpath = tmp_path / f"worker-{mid}.yaml"
        wpath.write_text(yaml.safe_dump(wcfg))
        port = free_port()
        procs[mid] = subprocess.Popen([sys.executable, "-m", "arcdemo.worker", "--config", str(wpath),
                                       "--task", "t2i,i23d,music,llm,layout", "--host", "127.0.0.1", "--port", str(port)],
                                      cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        machines.append({"id": mid, "label": mid, "workers": {"t2i": url, "i23d": url, "music": url, "llm": url, "layout": url},
                         "show_full_recording": mid == "cpu"})
    for m in machines:                                   # wait until every worker answers
        end = time.time() + 30
        while True:
            try:
                httpx.get(m["workers"]["t2i"] + "/health", timeout=1).raise_for_status()
                break
            except httpx.HTTPError:
                assert time.time() < end, f"{m['id']} worker did not start"
                time.sleep(0.3)
    cfg_path = tmp_path / "demo-test.yaml"
    cfg_path.write_text(yaml.safe_dump({"extends": str(ROOT / "config/demo.yaml"), "machines": machines,
                                        "session_dir": str(tmp_path / "session"),
                                        "replay_dir": str(tmp_path / "replay")}))
    yield DemoConfig(cfg_path), procs
    for p in procs.values():
        p.terminate()


async def one_round(cfg: DemoConfig, segment: str, input_id: str) -> dict:
    cond = Conductor(cfg)
    await cond.refresh_health()
    rnd = await cond.start_round(segment, input_id)
    await cond.wait_round(60)
    await cond.http.aclose()
    return rnd


def test_round_completes_live_on_all_machines(stack):
    cfg, _ = stack
    rnd = asyncio.run(one_round(cfg, "seg2", "t2i-01"))
    cells = rnd["cells"]
    assert {c["status"] for c in cells.values()} == {"ok"}, {m: c["note"] for m, c in cells.items()}
    assert all(c["completed"] == 4 and len(c["outputs"]) == 4 for c in cells.values())   # same batch everywhere
    assert all(Path(o["path"]).exists() for c in cells.values() for o in c["outputs"])
    log = (Path(cfg.session_dir) / "runs.jsonl").read_text().splitlines()
    assert len(log) == 3                                 # one measurement record per cell


def test_dead_worker_falls_back_with_a_reason(stack):
    cfg, procs = stack
    procs["cpu"].terminate()
    procs["cpu"].wait()
    rnd = asyncio.run(one_round(cfg, "seg2", "t2i-02"))
    cpu = rnd["cells"]["cpu"]
    assert cpu["mode"] == "REPLAY" and cpu["status"] == "unavailable"   # no rehearsal record in tmp
    assert "unreachable" in cpu["note"]
    assert rnd["cells"]["dgx"]["status"] == "ok"           # the others are unaffected


def test_operator_model_pick_runs_on_that_machine_only(stack, tmp_path):
    cfg, _ = stack
    picks = tmp_path / "picks.json"

    async def go():
        cond = Conductor(cfg, picks_path=picks)
        cond.set_model("t2i", "workstation", "sd35-medium-not5")
        await cond.refresh_health()
        rnd = await cond.start_round("seg2", "t2i-01")
        with pytest.raises(RuntimeError):                  # no switching mid-round
            cond.set_model("t2i", "workstation", None)
        await cond.wait_round(60)
        await cond.http.aclose()
        return rnd

    rnd = asyncio.run(go())
    ws, dgx = rnd["cells"]["workstation"], rnd["cells"]["dgx"]
    assert ws["status"] == "ok", ws["note"]
    assert ws["model"] == "sd35-medium-not5" and ws["settings"]["width"] == 512   # seg2 keeps the size
    assert dgx["model"] == "sd15" and dgx["settings"]["width"] == 512
    log = [json.loads(x) for x in (Path(cfg.session_dir) / "runs.jsonl").read_text().splitlines()]
    assert {r["machine"]: r["model"] for r in log}["workstation"] == "sd35-medium-not5"
    assert Conductor(cfg, picks_path=picks).model_picks == {"t2i": {"workstation": "sd35-medium-not5"}}


# ---------------------------------------------------------------- stop + full recordings ----
SLOW_CPU = {"dgx": 0.01, "workstation": 0.02, "cpu": 50}     # cpu: ~28 s per image


def save_full_recording(cfg: DemoConfig, tmp_path: Path, seconds: float = 123.4) -> None:
    """What `rehearse.py --full` leaves behind for the CPU on seg2 / t2i-01."""
    from PIL import Image
    img = tmp_path / "rec-0.png"
    Image.new("RGB", (8, 8)).save(img)
    Conductor(cfg).replay.save("t2i", "iso_time", "cpu", "t2i-01", cfg.settings("t2i", "iso_time", "cpu"),
                               {"status": "ok", "all_outputs_seconds": seconds, "completed": 1, "requested": 1,
                                "output_times": [seconds]}, [img], variant="full")


@pytest.mark.parametrize("stack", [SLOW_CPU], indirect=True)
def test_cpu_time_up_shows_full_recording_with_its_time(stack, tmp_path):
    cfg, _ = stack
    save_full_recording(cfg, tmp_path)

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        plan = cond.plan("seg2", "t2i-01")
        rnd = await cond.start_round("seg2", "t2i-01", budget_override=1)
        await cond.wait_round(30)
        await cond.http.aclose()
        return plan, rnd

    plan, rnd = asyncio.run(go())
    cpu_plan = next(m for m in plan["machines"] if m["id"] == "cpu")
    assert cpu_plan["recording"]["seconds"] == 123.4 and cpu_plan["full_recording"]
    cpu = rnd["cells"]["cpu"]
    assert cpu["mode"] == "REPLAY" and cpu["status"] == "ok", cpu["note"]
    assert cpu["recorded"]["seconds"] == 123.4 and cpu["final_seconds"] == 123.4
    assert "Time up at 1 s" in cpu["note"]
    assert cpu["outputs"][0]["url"].endswith("/t2i/iso_time/cpu/t2i-01/full/rec-0.png")
    assert cpu["outputs"][0]["reveal_at"] is None                     # shown at once
    assert rnd["cells"]["dgx"]["mode"] == "LIVE"                       # fast machines unaffected


@pytest.mark.parametrize("stack", [SLOW_CPU], indirect=True)
def test_stop_one_machine_and_stop_all(stack, tmp_path):
    cfg, _ = stack
    save_full_recording(cfg, tmp_path, seconds=200.0)

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        rnd = await cond.start_round("seg1", "t2i-01")                 # no time limit
        await asyncio.sleep(1.5)
        await cond.stop_cell("workstation")                            # may already be done: no-op then
        await cond.cancel_round()                                      # the CPU is still working
        await cond.wait_round(10)
        await cond.http.aclose()
        return rnd

    rnd = asyncio.run(go())
    assert rnd["done"]
    cpu = rnd["cells"]["cpu"]
    # seg1 is iso_quality and has no recording -> the CPU shows "stopped" with the live time
    assert cpu["status"] == "cancelled" and cpu["recorded"] is None
    assert cpu["note"].startswith("Still running when stopped")


@pytest.mark.parametrize("stack", [SLOW_CPU], indirect=True)
def test_stop_cpu_with_recording_switches_to_it(stack, tmp_path):
    cfg, _ = stack
    save_full_recording(cfg, tmp_path, seconds=200.0)

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        rnd = await cond.start_round("seg2", "t2i-01")                 # 60 s limit, CPU needs ~28 s+
        await asyncio.sleep(1.0)
        await cond.stop_cell("cpu")
        await cond.wait_round(30)
        await cond.http.aclose()
        return rnd

    rnd = asyncio.run(go())
    cpu = rnd["cells"]["cpu"]
    assert cpu["recorded"]["seconds"] == 200.0 and cpu["recorded"]["why"].startswith("Stopped at")
    assert rnd["cells"]["dgx"]["status"] == "ok"


def test_full_rehearsal_runs_only_flagged_machines_without_limit(stack):
    cfg, _ = stack

    async def go():
        cond = Conductor(cfg, allow_replay=False)
        await cond.refresh_health()
        rnd = await cond.start_round("seg2", "t2i-01", run_to_completion=True, machines=["cpu"])
        await cond.wait_round(30)
        await cond.http.aclose()
        return rnd

    rnd = asyncio.run(go())
    assert rnd["order"] == ["cpu"] and rnd["budget_seconds"] is None
    assert rnd["cells"]["cpu"]["status"] == "ok"


def test_llm_round_streams_tokens_on_every_machine(stack):
    cfg, _ = stack

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        rnd = await cond.start_round("seg7", "llm-02", budget_override=2)
        await cond.wait_round(30)
        await cond.http.aclose()
        return rnd

    cells = asyncio.run(go())["cells"]
    assert {c["status"] for c in cells.values()} == {"ok"}, {m: c["note"] for m, c in cells.items()}
    assert all(c["tokens"] > 0 and c["text"].startswith("[MOCK]") for c in cells.values())
    assert cells["dgx"]["tokens"] > cells["cpu"]["tokens"]          # 64 streams vs 1


def test_layout_round_builds_the_same_room_everywhere(stack):
    cfg, _ = stack

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        rnd = await cond.start_round("seg8", "lay-01")
        await cond.wait_round(60)
        await cond.http.aclose()
        return cond, rnd

    cond, rnd = asyncio.run(go())
    cells = rnd["cells"]
    assert {c["status"] for c in cells.values()} == {"ok"}, {m: c["note"] for m, c in cells.items()}
    rooms = {m: [(o["cls"], o["x"], o["y"], o["rot"]) for o in c["objects"]] for m, c in cells.items()}
    assert len(rooms["dgx"]) >= 3 and rooms["dgx"] == rooms["workstation"] == rooms["cpu"]   # greedy
    times = [o["t"] for o in cells["cpu"]["objects"]]
    assert times == sorted(times) and times[0] < times[-1]          # arrived one at a time
    assert all(c["outputs"][0]["url"].endswith(".json") for c in cells.values())
    assert cond.history[-1]["cells"]["dgx"]["layout"]["objects"] == len(rooms["dgx"])


def test_layout_replay_keeps_object_times(stack, tmp_path):
    cfg, procs = stack
    rec_file = tmp_path / "room-0-0.json"
    rec_file.write_text("{}")
    objs = [{"cls": "double_bed", "x": 2, "y": 2.4, "w": 1.6, "d": 2, "h": 0.5, "rot": 180, "i": 0, "room": 0,
             "t": 4.0, "issues": []}]
    Conductor(cfg).replay.save("layout", "iso_quality", "workstation", "lay-01",
                               cfg.settings("layout", "iso_quality", "workstation"),
                               {"status": "ok", "all_outputs_seconds": 9.0, "completed": 1, "requested": 1,
                                "output_times": [9.0], "objects": objs, "tokens": 40}, [rec_file])
    procs["workstation"].terminate()
    procs["workstation"].wait()

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        rnd = await cond.start_round("seg8", "lay-01")
        await cond.wait_round(60)
        await cond.http.aclose()
        return rnd

    ws = asyncio.run(go())["cells"]["workstation"]
    assert ws["mode"] == "REPLAY" and ws["objects"] == objs and ws["final_seconds"] == 9.0


def test_model_not_on_disk_is_refused_and_never_sent(stack):
    """A worker that reports the model missing: no pick, no load, no job — a plain message."""
    cfg, _ = stack

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        h = cond.health["workstation:t2i"]
        assert "sd35-medium-not5" in h["available"]          # mock workers have every model
        h["available"] = [k for k in h["available"] if k != "sd35-medium-not5"]
        with pytest.raises(ValueError, match="not downloaded"):
            cond.set_model("t2i", "workstation", "sd35-medium-not5")
        cond.model_picks = {"t2i": {"workstation": "sd35-medium-not5"}}   # e.g. from an older picks file
        prep = await cond.prepare("seg2")
        rnd = await cond.start_round("seg2", "t2i-01")
        await cond.wait_round(60)
        await cond.http.aclose()
        return prep, rnd

    prep, rnd = asyncio.run(go())
    assert prep["workstation"]["status"] == "missing"
    ws = rnd["cells"]["workstation"]
    assert ws["status"] == "unavailable" and ws["missing_model"] and ws["run_id"] is None
    assert "not downloaded" in ws["note"]
    assert rnd["cells"]["dgx"]["status"] == "ok"


def test_on_disk_reads_the_hub_cache(tmp_path, monkeypatch):
    from arcdemo.weights import on_disk
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    gguf = {"repo": "org/m-GGUF", "file": "*Q4_K_M.gguf"}
    assert not on_disk(gguf)
    snap = tmp_path / "models--org--m-GGUF" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "m-Q8_0.gguf").write_text("x")
    assert not on_disk(gguf)                                   # another quantisation only
    (snap / "m-Q4_K_M.gguf").write_text("x")
    assert on_disk(gguf)
    gated = tmp_path / "models--stabilityai--stable-diffusion-3.5-medium" / "snapshots" / "b9"
    gated.mkdir(parents=True)
    (gated / "LICENSE.md").write_text("x")                     # a refused gated download
    assert not on_disk({"repo": "stabilityai/stable-diffusion-3.5-medium"})
    sf3d = {"repo": "stabilityai/stable-fast-3d"}
    (tmp_path / "models--stabilityai--stable-fast-3d" / "snapshots" / "a").mkdir(parents=True)
    (tmp_path / "models--stabilityai--stable-fast-3d" / "snapshots" / "a" / "model.safetensors").write_text("x")
    assert not on_disk(sf3d, "sf3d")                           # its encoders are missing
    assert on_disk({"repo": "music-stub"})


# ---------------------------------------------------------------- start right after Stop (409) ----
def occupy(cfg: DemoConfig, mid: str, seconds: float) -> str:
    """Keep a worker busy with a real job for `seconds` (a run that is still ending). -> run_id"""
    url = cfg.worker_url(mid, "llm")
    r = httpx.post(f"{url}/jobs", json={"task": "llm", "model": "qwen35-4b", "input_id": "busy", "prompt": "x",
                                         "budget_seconds": seconds, "repeat_until_budget": True})
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    end = time.time() + 10
    while httpx.get(f"{url}/health").json()["busy"] is False:
        assert time.time() < end, "the occupying job never started"
        time.sleep(0.05)
    return rid


def test_start_while_last_run_still_ending_waits_then_runs_live(stack):
    cfg, _ = stack
    occupy(cfg, "workstation", 2.0)

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        rnd = await cond.start_round("seg2", "t2i-01")       # the worker answers 409 at first
        await cond.wait_round(60)
        await cond.http.aclose()
        return rnd

    ws = asyncio.run(go())["cells"]["workstation"]
    assert ws["mode"] == "LIVE" and ws["status"] == "ok", ws["note"]
    assert ws["completed"] == ws["requested"]


def test_start_gives_up_on_a_worker_that_stays_busy(stack):
    cfg, _ = stack
    rid = occupy(cfg, "workstation", 40.0)

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        t0 = time.time()
        rnd = await cond.start_round("seg2", "t2i-01")
        await cond.wait_round(60)
        await cond.http.aclose()
        return rnd, time.time() - t0

    try:
        rnd, seconds = asyncio.run(go())
    finally:
        httpx.post(f"{cfg.worker_url('workstation', 'llm')}/jobs/{rid}/cancel")
    ws = rnd["cells"]["workstation"]
    assert ws["status"] == "unavailable" and "HTTP 409" in ws["note"], ws["note"]
    assert seconds < 25                                       # ~15 s of retries, never a hang
    assert rnd["cells"]["dgx"]["status"] == "ok"


def test_prepare_while_last_run_still_ending_loads_once_idle(stack):
    cfg, _ = stack
    occupy(cfg, "workstation", 2.0)

    async def go():
        cond = Conductor(cfg)
        prep = await cond.prepare("seg2")
        first = dict(prep["workstation"])
        end = time.time() + 20
        while time.time() < end and any(p["status"] == "loading" for p in cond.prepare_state.values()):
            await asyncio.sleep(0.2)
        await cond.http.aclose()
        return first, cond.prepare_state

    first, final = asyncio.run(go())
    assert first["status"] == "loading" and first["message"] == "waiting for the last run to stop"
    ws = final["workstation"]
    assert ws["status"] == "ready" and "message" not in ws and "_retry" not in ws, ws
    assert final["dgx"]["status"] == "ready"


async def _prepared(cond: Conductor, segment: str) -> dict:
    await cond.prepare(segment)
    await asyncio.wait_for(cond._prepare_task, 20)       # the watcher ends when every machine is settled
    return {mid: dict(p) for mid, p in cond.prepare_state.items()}


def test_prepare_frees_every_other_task_on_every_machine(stack):
    """Load t2i, then press Load for the layout segment: only the layout model stays in memory,
    on every machine, except one marked keep_other_tasks."""
    cfg, _ = stack
    cfg.machine("workstation")["keep_other_tasks"] = True

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        await _prepared(cond, "seg2")                     # t2i everywhere
        prep = await _prepared(cond, "seg8")              # layout
        loaded = {m["id"]: httpx.get(cfg.worker_url(m["id"], "layout") + "/health").json()["loaded"]
                  for m in cfg.machines}
        await cond.http.aclose()
        return prep, loaded

    prep, loaded = asyncio.run(go())
    t2i = cfg.settings("t2i", cfg.segment("seg2")["protocol"], "dgx")["model"]
    lay = cfg.settings("layout", cfg.segment("seg8")["protocol"], "dgx")["model"]
    assert prep["dgx"]["status"] == "ready" and t2i in prep["dgx"]["freed"], prep["dgx"]
    assert loaded["dgx"] == [lay] and loaded["cpu"] == [lay], loaded
    # keep_other_tasks: nothing freed by the conductor (the worker itself is still exclusive)
    assert prep["workstation"]["freed"] == [] and prep["workstation"]["status"] == "ready"


def test_prepare_says_why_it_failed_and_times_out(stack):
    cfg, procs = stack
    procs["cpu"].terminate()
    procs["cpu"].wait()

    async def go():
        cond = Conductor(cfg)
        prep = await cond.prepare("seg2")
        cpu = dict(prep["cpu"])
        await cond._watch_prepare("t2i", timeout=0)           # let the real loads finish first
        # a load that never completes (no error reported either): must not stay "loading" forever
        cond.prepare_state["dgx"] = {"status": "loading", "model": "sd35-large", "since": time.time()}
        await cond._watch_prepare("t2i", timeout=1.5)
        cond._prepare_task.cancel()                          # the background watcher, before closing http
        await asyncio.gather(cond._prepare_task, return_exceptions=True)
        await cond.http.aclose()
        return cpu, cond.prepare_state["dgx"]

    cpu, dgx = asyncio.run(go())
    assert cpu["status"] == "error" and cpu["message"].startswith("the machine is not reachable"), cpu
    assert dgx["status"] == "error" and dgx["message"].startswith("still not loaded after"), dgx


def test_furnish_and_fix_round_sends_the_self_correction_settings(stack):
    cfg, _ = stack
    assert cfg.settings("layout", "iso_quality+fix", "dgx")["repairs"] == 2
    rnd = asyncio.run(one_round(cfg, "seg8b", "lay-01"))
    for mid, c in rnd["cells"].items():
        assert c["status"] == "ok", (mid, c["note"])
        assert (c["result"]["settings"]["repairs"], c["result"]["settings"]["review_rounds"]) == (2, 3)
        assert c["objects"] and not any(o["issues"] for o in c["objects"])   # the mock room is clean


def test_uploaded_picture_runs_live_and_every_machine_gets_the_same_file(stack, tmp_path):
    import base64
    import io

    from PIL import Image, ImageDraw
    cfg, _ = stack
    img = Image.new("RGBA", (800, 600), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((250, 150, 550, 450), fill=(30, 120, 200, 255))
    buf = io.BytesIO()
    img.save(buf, "PNG")

    async def go():
        cond = Conductor(cfg)
        await cond.refresh_health()
        item = cond.add_upload("my mug.png", buf.getvalue())
        rnd = await cond.start_round("seg3", item["id"])
        await cond.wait_round(60)
        again = Conductor(cfg)                                # a restart still offers the upload
        await cond.http.aclose()
        await again.http.aclose()
        return item, rnd, again.cfg.input("i23d", item["id"])

    item, rnd, reloaded = asyncio.run(go())
    assert item["id"].startswith("up-") and item["label"] == "my mug" and reloaded["upload"] == item["upload"]
    sent = (cfg.session_dir / item["upload"]).read_bytes()
    for mid, c in rnd["cells"].items():
        assert c["mode"] == "LIVE" and c["status"] in ("ok", "budget_exceeded"), (mid, c["note"])
        got = list((tmp_path / f"results-{mid}" / "inputs").glob("*.png"))
        assert len(got) == 1 and got[0].read_bytes() == sent, mid
        assert "score" not in c or not c["score"]                 # no reference mesh for an upload
    assert rnd["input"]["input_url"].startswith("/media/uploads/")
    assert rnd["caption"].endswith("so no error is measured.")
