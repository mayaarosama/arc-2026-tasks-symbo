"""Shared modules with no model: moderation, logging, wire schemas, weights-on-disk, replay table,
timing / memory metrics, and the base backend helpers."""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from arcdemo import logs, weights
from arcdemo.conductor import moderation
from arcdemo.conductor.replay import ReplayStore, settings_hash
from arcdemo.schemas import TERMINAL, HealthInfo, JobRequest, JobResult, Progress
from arcdemo.worker.backends import base
from arcdemo.worker.metrics import Clock, MemoryMeter, _cuda_devices, cuda_sync

from tests.helpers import make_context


# ------------------------------------------------------------------ moderation (Task 1 live prompt)
@pytest.mark.parametrize("prompt", ["a red fox in snow", "a teapot on a wooden table, studio light"])
def test_clean_prompt_passes(prompt):
    assert moderation.check_prompt(prompt) == {"ok": True, "hits": []}


@pytest.mark.parametrize("prompt,hit", [("A NAKED man", "naked"), ("gun on a table", "gun"),
                                        ("Mickey mouse", "mickey"), ("decapitated statue", "decapitat")])
def test_blocklist_hits_are_case_insensitive_and_match_word_starts(prompt, hit):
    r = moderation.check_prompt(prompt)
    assert not r["ok"] and hit in r["hits"]


def test_blocklist_does_not_match_inside_words():
    assert moderation.check_prompt("a begun sketch")["ok"]          # "gun" only at a word start


def test_too_long_prompt_is_blocked():
    r = moderation.check_prompt("a" * 301)
    assert not r["ok"] and r["hits"] == ["too long (>300 chars)"]
    assert moderation.check_prompt("a" * 300)["ok"]


def test_extra_blocklist_file_is_read_on_every_check(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "blocklist.txt").write_text("# comment\nzebra\n\n", encoding="utf-8")
    monkeypatch.setattr(moderation, "ROOT", tmp_path)
    assert moderation.check_prompt("a zebra")["hits"] == ["zebra"]
    assert "# comment" not in moderation._terms()


# ------------------------------------------------------------------ logging
def test_setup_logging_writes_the_file_and_honours_the_level(tmp_path, monkeypatch):
    root = logging.getLogger()
    before = list(root.handlers), root.level
    monkeypatch.setenv("ARC_LOG_LEVEL", "debug")
    try:
        logs.setup_logging(tmp_path / "sub" / "x.log")
        assert root.level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.WARNING
        logging.getLogger("arcdemo.test").info("hello 3f9c2a")
        for h in root.handlers:
            h.flush()
        assert "hello 3f9c2a" in (tmp_path / "sub" / "x.log").read_text(encoding="utf-8")
    finally:
        for h in root.handlers[len(before[0]):]:
            h.close()
        root.handlers[:] = before[0]
        root.setLevel(before[1])


# ------------------------------------------------------------------ schemas
def test_job_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        JobRequest(task="t2i", model="sd15", input_id="x", stepz=3)


def test_job_request_rejects_unknown_task_and_gets_a_run_id():
    with pytest.raises(ValidationError):
        JobRequest(task="video", model="m", input_id="x")
    a, b = JobRequest(task="llm", model="m", input_id="x"), JobRequest(task="llm", model="m", input_id="x")
    assert a.run_id != b.run_id and len(a.run_id) == 32


def test_terminal_statuses_are_statuses():
    statuses = set(JobResult.model_fields["status"].annotation.__args__)
    assert TERMINAL <= statuses and {"queued", "loading", "running"}.isdisjoint(TERMINAL)


def test_result_and_health_defaults():
    r = JobResult(run_id="r", machine="m", task="t2i", model="x", input_id="i", protocol="iso_time")
    assert r.status == "queued" and r.outputs == [] and isinstance(r.progress, Progress)
    h = HealthInfo(machine="m", tasks=["t2i"], mock=True, busy=False, loaded=[], env_hash="abc")
    assert h.available is None


# ------------------------------------------------------------------ weights on disk
def _snapshot(cache: Path, repo: str, files: dict[str, bytes]) -> None:
    snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / "abc"
    for name, data in files.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_bytes(data)


def test_on_disk_needs_weights_not_just_a_readme(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _snapshot(tmp_path, "org/gated", {"README.md": b"x"})
    _snapshot(tmp_path, "org/ok", {"model.safetensors": b"x"})
    assert not weights.on_disk({"repo": "org/gated"})
    assert weights.on_disk({"repo": "org/ok"})
    assert not weights.on_disk({"repo": "org/missing"})


def test_on_disk_honours_file_glob_and_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _snapshot(tmp_path, "org/llm", {"q8/model-Q8_0.gguf": b"x"})
    assert weights.on_disk({"repo": "org/llm", "file": "*Q8_0.gguf"})
    assert not weights.on_disk({"repo": "org/llm", "file": "*Q4_K_M.gguf"})
    _snapshot(tmp_path, "stabilityai/stable-fast-3d", {"model.safetensors": b"x"})
    assert not weights.on_disk({"repo": "stabilityai/stable-fast-3d"}, "sf3d")      # dinov2 / CLIP missing
    for dep in weights.DEPENDENCIES["sf3d"]:
        _snapshot(tmp_path, dep, {"pytorch_model.bin": b"x"})
    assert weights.on_disk({"repo": "stabilityai/stable-fast-3d"}, "sf3d")


def test_repo_without_org_counts_as_present():
    assert weights.on_disk({"repo": "music-stub"}) and weights.on_disk({})


def test_hub_cache_falls_back_to_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    assert weights.hub_cache() == tmp_path / "hub"


# ------------------------------------------------------------------ replay table
def test_settings_hash_ignores_mode_and_prepare_and_key_order():
    a = settings_hash({"model": "m", "steps": 4, "mode": "replay", "prepare": False})
    assert a == settings_hash({"steps": 4, "model": "m"}) and len(a) == 10
    assert a != settings_hash({"model": "m", "steps": 5})


def test_replay_store_round_trip_and_stale_flag(tmp_path):
    store = ReplayStore(tmp_path / "replay")
    out = tmp_path / "out.png"
    out.write_bytes(b"png")
    s = {"model": "sd15", "steps": 4}
    d = store.save("t2i", "iso_time", "dgx", "t2i-01", s, {"status": "ok"}, [out], extra={"score": 1})
    assert (d / "out.png").read_bytes() == b"png"
    rec = store.get("t2i", "iso_time", "dgx", "t2i-01", s)
    assert rec["files"] == ["out.png"] and rec["score"] == 1 and rec["stale"] is False and rec["variant"] == ""
    assert store.get("t2i", "iso_time", "dgx", "t2i-01", {**s, "steps": 8})["stale"] is True
    assert store.get("t2i", "iso_time", "dgx", "other") is None


def test_get_full_prefers_the_full_run_then_a_complete_normal_one(tmp_path):
    store = ReplayStore(tmp_path)
    f = tmp_path / "a.glb"
    f.write_bytes(b"g")
    store.save("i23d", "iso_time", "cpu", "i", {}, {"status": "budget_exceeded"}, [f])
    assert store.get_full("i23d", "iso_time", "cpu", "i") is None          # incomplete and no full/
    store.save("i23d", "iso_time", "cpu", "i", {}, {"status": "ok"}, [f], variant="full")
    assert store.get_full("i23d", "iso_time", "cpu", "i")["variant"] == "full"
    store.save("i23d", "iso_time", "cpu", "j", {}, {"status": "ok"}, [f])
    assert store.get_full("i23d", "iso_time", "cpu", "j")["variant"] == ""


def test_re_rehearsing_keeps_the_full_sub_record(tmp_path):
    store = ReplayStore(tmp_path)
    f = tmp_path / "a.wav"
    f.write_bytes(b"w")
    store.save("music", "iso_time", "cpu", "m", {}, {"status": "ok"}, [f], variant="full")
    store.save("music", "iso_time", "cpu", "m", {}, {"status": "ok"}, [f])
    store.save("music", "iso_time", "cpu", "m", {}, {"status": "ok"}, [])
    assert store.get("music", "iso_time", "cpu", "m", variant="full") is not None
    assert json.loads((store.cell_dir("music", "iso_time", "cpu", "m") / "record.json").read_text())["files"] == []


# ------------------------------------------------------------------ metrics
def test_clock_reads_zero_until_started():
    c = Clock()
    assert c.now() == 0.0
    c.start()
    time.sleep(0.02)
    assert c.now() > 0.01 and c.wall0 is not None


def test_cuda_device_indices_and_sync_is_a_noop_on_cpu():
    assert _cuda_devices(["cpu", "cuda", "cuda:3"]) == [0, 3]
    cuda_sync(["cpu"])                      # no torch import, no error


def test_memory_meter_samples_rss_on_cpu():
    with MemoryMeter(["cpu"], sample_interval=0.01) as m:
        blob = bytearray(30 * 2**20)        # 30 MB, held while the sampler runs
        time.sleep(0.05)
    del blob
    assert m.peak_allocated_mb and m.peak_allocated_mb > 30 and m.peak_reserved_mb is None


# ------------------------------------------------------------------ backend base
def test_job_context_budget_progress_and_outputs(tmp_path):
    req = JobRequest(task="t2i", model="m", input_id="i", budget_seconds=0.05, batch=2)
    ctx = make_context(req, tmp_path)
    ctx.result.requested = 2
    ctx.progress(step=3, steps=10, stage="denoising")
    assert (ctx.result.progress.step, ctx.result.progress.steps, ctx.result.progress.stage) == (3, 10, "denoising")
    p = ctx.output_path(".txt")
    p.write_text("x")
    ctx.record_output(p)
    assert ctx.result.completed == 1 and ctx.result.outputs == [p.name]
    assert ctx.result.first_output_seconds == ctx.result.all_outputs_seconds
    assert not ctx.should_stop()
    time.sleep(0.06)
    assert ctx.budget_hit() and ctx.should_stop()


def test_job_context_cancel_and_no_budget(tmp_path):
    ctx = make_context(JobRequest(task="llm", model="m", input_id="i"), tmp_path)
    assert ctx.budget is None and not ctx.budget_hit()
    ctx.cancel.set()
    assert ctx.should_stop()


def test_record_output_is_thread_safe(tmp_path):
    ctx = make_context(JobRequest(task="llm", model="m", input_id="i"), tmp_path)

    def many(k):
        for j in range(50):
            ctx.record_output(tmp_path / f"{k}-{j}")
    ts = [threading.Thread(target=many, args=(k,)) for k in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert ctx.result.completed == 200 == len(ctx.result.output_times)


@pytest.mark.parametrize("exc,expected", [(MemoryError(), True), (RuntimeError("CUDA out of memory. Tried"), True),
                                          (OSError("Cannot allocate memory"), True), (ValueError("bad"), False)])
def test_is_oom(exc, expected):
    assert base.is_oom(exc) is expected


def test_backend_registry_names_resolve_without_heavy_imports():
    for name in ("mock", "music_stub", "llm_mock", "layout_mock", "llama_server", "llama_layout"):
        assert issubclass(base.backend_class(name), base.Backend)
    assert set(base._BACKENDS) >= {"diffusers_t2i", "trellis2", "trellis1", "sf3d", "stable_audio3"}
    with pytest.raises(KeyError):
        base.backend_class("nope")


def test_base_backend_is_abstract_and_free_memory_is_safe():
    b = base.Backend("k", {"repo": "org/r"}, {}, ["cpu"])
    assert b.repo == "org/r" and b.describe(None) == {}
    with pytest.raises(NotImplementedError):
        b.load()
    with pytest.raises(NotImplementedError):
        b.generate(None, None)
    b.model = object()
    b.unload()
    assert b.model is None
