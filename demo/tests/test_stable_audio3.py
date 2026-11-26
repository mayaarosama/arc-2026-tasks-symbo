"""Stable Audio 3 backend without the library or weights: a fake model stands in for
StableAudioModel, so the throughput loop, step default, precision choice and WAV output are tested
anywhere. Also: mock workers must never load real music weights."""
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from arcdemo.schemas import JobRequest, JobResult
from arcdemo.worker.app import create_app, load_worker_config
from arcdemo.worker.backends import stable_audio3
from arcdemo.worker.backends.base import JobContext
from arcdemo.worker.backends.stable_audio3 import StableAudio3
from arcdemo.worker.metrics import Clock

torch = pytest.importorskip("torch")      # real tensors: run in the music venv (test_backends_unit.py has no torch)

SPEC = {"task": "music", "backend": "stable_audio3", "repo": "stabilityai/stable-audio-3-small-music",
        "sa3_name": "small-music", "dtype": "auto"}


class FakeModel:
    model_config = {"sample_rate": 44100}

    def __init__(self):
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        return torch.full((1, 2, int(kw["duration"] * 44100)), 0.25)


@pytest.fixture
def fake_library(monkeypatch):
    made = {}

    class StableAudioModel:
        @staticmethod
        def from_pretrained(name, device=None, model_half=True):
            made.update(name=name, device=device, half=model_half, model=FakeModel())
            return made["model"]

    monkeypatch.setattr(stable_audio3, "StableAudioModel", StableAudioModel)
    return made


def run_job(backend, tmp_path, **job):
    req = JobRequest(task="music", model="sa3-small-music", input_id="mus-01", prompt="oud", seed=5, **job)
    res = JobResult(run_id=req.run_id, machine="t", task="music", model=req.model, input_id="mus-01",
                    protocol="iso_time")
    clock = Clock()
    ctx = JobContext(req, res, clock, tmp_path, backend.devices, threading.Event())
    clock.start()
    backend.generate(req, ctx)
    return res


def test_cpu_load_is_fp32_and_writes_stereo_wav(fake_library, tmp_path):
    b = StableAudio3("sa3-small-music", SPEC, {}, ["cpu"])
    b.load()
    assert fake_library == {**fake_library, "name": "small-music", "device": "cpu", "half": False}
    res = run_job(b, tmp_path, length_s=2)
    assert res.completed == 1
    call = fake_library["model"].calls[0]
    assert call["steps"] == 8 and call["seed"] == 5 and call["duration"] == 2
    with wave.open(str(tmp_path / res.outputs[0])) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (2, 44100, 88200)
        peak = np.abs(np.frombuffer(w.readframes(10), dtype=np.int16)).max()
    assert peak == 32767                      # peak-normalised like the upstream example


def test_cuda_precision_follows_dtype(fake_library):
    StableAudio3("k", SPEC, {}, ["cuda:0"]).load()
    assert fake_library["half"] is True
    StableAudio3("k", {**SPEC, "dtype": "fp32"}, {}, ["cuda:0"]).load()   # Pascal override
    assert fake_library["half"] is False


def test_explicit_steps_and_repeat_until_budget(fake_library, tmp_path):
    b = StableAudio3("sa3-small-music", SPEC, {}, ["cpu"])
    b.load()
    res = run_job(b, tmp_path, length_s=0.1, steps=4, budget_seconds=0.5, repeat_until_budget=True)
    calls = fake_library["model"].calls
    assert len(calls) > 1 and {c["steps"] for c in calls} == {4}
    assert [c["seed"] for c in calls[:3]] == [5, 6, 7]      # new seed per track
    assert res.completed in (len(calls), len(calls) - 1)  # a track finished after the bell is dropped


def test_mock_worker_uses_stub_not_real_weights(tmp_path, monkeypatch):
    def real_load(self):
        raise AssertionError("a mock worker loaded the real Stable Audio 3 weights")
    monkeypatch.setattr(StableAudio3, "load", real_load)
    cfg = load_worker_config(str(Path(__file__).resolve().parents[1] / "config/worker-mock-cpu.yaml"), ["music"])
    cfg["results_dir"] = str(tmp_path)
    with TestClient(create_app(cfg)) as c:
        r = c.post("/load", json={"model": "sa3-small-music"})
        assert r.status_code == 200, r.text
        for _ in range(100):
            h = c.get("/health").json()
            if "sa3-small-music" in h["loaded"]:
                break
            time.sleep(0.05)
        assert "sa3-small-music" in h["loaded"]
        assert c.get("/env").json()["last_error"] is None
