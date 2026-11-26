"""The real model backends (Tasks 1, 2, 3, 4, 8) with their libraries replaced by stand-ins, so the
wiring — arguments passed, budget handling, progress, outputs, timings — is tested on any machine,
torch or not. Real weights are exercised by test_gpu_real.py (-m gpu)."""
from __future__ import annotations

import contextlib
import os
import stat
import sys
import textwrap
import threading
import time
import types
import weakref
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from arcdemo.schemas import JobRequest
from arcdemo.worker.backends import base, diffusers_t2i, llm, sf3d, stable_audio3
from arcdemo.worker.backends.diffusers_t2i import DiffusersT2I
from arcdemo.worker.backends.layout import LayoutServer
from arcdemo.worker.backends.llm import LlamaServer
from arcdemo.worker.backends.sf3d import StableFast3D
from arcdemo.worker.backends.stable_audio3 import StableAudio3
from arcdemo.worker.backends.trellis import Trellis1, Trellis2
from tests.helpers import ROOT, FakeLlama, make_context


# ------------------------------------------------------------------ a stand-in torch
class _Gen:
    def __init__(self, device="cpu"):
        self.device, self.seed = device, None

    def manual_seed(self, s):
        self.seed = s
        return self


def fake_torch() -> types.ModuleType:
    t = types.ModuleType("torch")
    t.bfloat16, t.float16, t.float32 = "torch.bfloat16", "torch.float16", "torch.float32"
    t.Generator = _Gen
    t.inference_mode = t.no_grad = contextlib.nullcontext
    t.autocast = lambda **kw: contextlib.nullcontext()
    t.cuda = types.SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False,
                                   synchronize=lambda i: None, OutOfMemoryError=type("OOM", (RuntimeError,), {}))
    t.from_numpy = lambda a: a
    return t


@pytest.fixture()
def torch(monkeypatch):
    """The stand-in torch, bound where the backends look it up (they import torch at the top)."""
    t = fake_torch()
    for mod in (base, diffusers_t2i, sf3d, stable_audio3):
        monkeypatch.setattr(mod, "torch", t)
    return t


# ------------------------------------------------------------------ Task 1: diffusers t2i
class FakePipe:
    calls: list = []

    def __init__(self, delay=0.0):
        self.delay = delay

    @classmethod
    def from_pretrained(cls, repo, **kw):
        cls.calls.append(("from_pretrained", repo, kw))
        return cls()

    def to(self, dev):
        self.dev = dev
        return self

    def set_progress_bar_config(self, **kw):
        pass

    def __call__(self, prompt, num_inference_steps, generator=None, callback_on_step_end=None, **kw):
        FakePipe.calls.append(("call", prompt, num_inference_steps, generator, kw))
        for s in range(num_inference_steps):
            time.sleep(self.delay)
            if callback_on_step_end:
                callback_on_step_end(self, s, 0, {})
        n = len(prompt) if isinstance(prompt, list) else 1
        return types.SimpleNamespace(images=[Image.new("RGB", (8, 8)) for _ in range(n)])


@pytest.fixture()
def diffusers(monkeypatch, torch):
    FakePipe.calls = []
    monkeypatch.setattr(diffusers_t2i, "AutoPipelineForText2Image", FakePipe)
    monkeypatch.setattr(diffusers_t2i, "StableDiffusion3Pipeline", FakePipe)
    return FakePipe


def test_t2i_load_warms_up_each_device_and_drops_t5_when_asked(diffusers):
    b = DiffusersT2I("sd35-medium-not5", {"repo": "stabilityai/stable-diffusion-3.5-medium", "t5": "none"},
                     {}, ["cpu", "cpu"])
    b.load()
    loads = [c for c in diffusers.calls if c[0] == "from_pretrained"]
    assert len(loads) == 2 and loads[0][2]["text_encoder_3"] is None
    warm = [c for c in diffusers.calls if c[0] == "call"]
    assert len(warm) == 2 and warm[0][2] == 2                      # 2-step warm-up per device
    assert b.describe(JobRequest(task="t2i", model="m", input_id="i"))["devices"] == 2


def test_t2i_int8_t5_needs_cuda(diffusers):
    b = DiffusersT2I("k", {"repo": "stabilityai/stable-diffusion-3.5-large", "t5": "int8"}, {}, ["cpu"])
    with pytest.raises(RuntimeError, match="int8 needs CUDA"):
        b.load()


def test_t2i_seeds_each_image_on_a_cpu_generator_round_robin(diffusers, tmp_path):
    b = DiffusersT2I("sd15", {"repo": "runwayml/sd15"}, {"micro_batch": 2}, ["cpu", "cpu"])
    b.load()
    diffusers.calls.clear()
    req = JobRequest(task="t2i", model="sd15", input_id="i", prompt="a cat", batch=4, steps=3, seed=10)
    ctx = make_context(req, tmp_path)
    b.generate(req, ctx)
    assert ctx.result.completed == 4 and len(list(tmp_path.glob("*.png"))) == 4
    seeds = sorted(g.seed for c in diffusers.calls for g in c[3])
    assert seeds == [10, 11, 12, 13] and all(g.device == "cpu" for c in diffusers.calls for g in c[3])


def test_t2i_without_prompt_fails(diffusers, tmp_path):
    b = DiffusersT2I("sd15", {"repo": "r"}, {}, ["cpu"])
    b.load()
    with pytest.raises(ValueError):
        b.generate(JobRequest(task="t2i", model="sd15", input_id="i"), make_context(
            JobRequest(task="t2i", model="sd15", input_id="i"), tmp_path))


def test_t2i_budget_aborts_the_image_in_flight(diffusers, tmp_path):
    b = DiffusersT2I("sd15", {"repo": "r"}, {}, ["cpu"])
    b.load()
    b.model = [("cpu", FakePipe(delay=0.05))]
    req = JobRequest(task="t2i", model="sd15", input_id="i", prompt="p", batch=3, steps=4, budget_seconds=0.3)
    ctx = make_context(req, tmp_path)
    b.generate(req, ctx)
    assert ctx.result.completed == 1                                # the second image is discarded


# ------------------------------------------------------------------ Task 2: TRELLIS / SF3D
def rgba(tmp_path: Path, mode="RGBA") -> str:
    p = tmp_path / f"in-{mode}.png"
    Image.new(mode, (32, 32)).save(p)
    return str(p)


class FakeGlb:
    def export(self, path, **kw):
        Path(path).write_bytes(b"glb")


@pytest.fixture()
def trellis(monkeypatch, torch):
    seen = {}

    class Pipe:
        low_vram = True

        @classmethod
        def from_pretrained(cls, repo):
            seen["repo"] = repo
            return cls()

        def cuda(self):
            seen["cuda"] = True

        def run(self, img, **kw):
            seen["run"] = kw
            mesh = types.SimpleNamespace(simplify=lambda n: None, vertices=0, faces=0, attrs=0, coords=0,
                                         layout=0, voxel_size=0)
            return [mesh] if "pipeline_type" in kw else {"gaussian": [0], "mesh": [0]}

    def to_glb(*a, **kw):
        seen["to_glb"] = kw
        return FakeGlb()
    pkg = types.SimpleNamespace(pipelines=types.SimpleNamespace(Trellis2ImageTo3DPipeline=Pipe,
                                                                TrellisImageTo3DPipeline=Pipe))
    monkeypatch.setitem(sys.modules, "trellis2", pkg)
    monkeypatch.setitem(sys.modules, "trellis2.pipelines", pkg.pipelines)
    monkeypatch.setitem(sys.modules, "trellis", pkg)
    monkeypatch.setitem(sys.modules, "trellis.pipelines", pkg.pipelines)
    utils = types.SimpleNamespace(postprocessing_utils=types.SimpleNamespace(to_glb=to_glb))
    monkeypatch.setitem(sys.modules, "trellis.utils", utils)
    monkeypatch.setitem(sys.modules, "o_voxel", types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=to_glb)))
    return seen


def test_trellis2_resolution_picks_the_pipeline_and_low_vram_override(trellis, tmp_path):
    b = Trellis2("trellis2-4b", {"repo": "microsoft/TRELLIS.2-4B", "low_vram": False}, {}, ["cuda:0"])
    b.load()
    assert b.model.low_vram is False and trellis["cuda"]
    req = JobRequest(task="i23d", model="trellis2-4b", input_id="i", image_path=rgba(tmp_path), resolution=512, seed=3)
    ctx = make_context(req, tmp_path)
    b.generate(req, ctx)
    assert trellis["run"] == {"seed": 3, "pipeline_type": "512"}
    assert trellis["to_glb"]["decimation_target"] == 300000 and trellis["to_glb"]["texture_size"] == 2048
    assert ctx.result.completed == 1 and ctx.result.export_seconds is not None


def test_trellis1_sampler_steps_and_rgba_check(trellis, tmp_path):
    b = Trellis1("trellis-image-large", {"repo": "microsoft/TRELLIS-image-large"}, {}, ["cuda:0"])
    b.load()
    req = JobRequest(task="i23d", model="m", input_id="i", image_path=rgba(tmp_path), ss_steps=5)
    b.generate(req, make_context(req, tmp_path))
    assert trellis["run"]["sparse_structure_sampler_params"] == {"steps": 5, "cfg_strength": 7.5}
    bad = JobRequest(task="i23d", model="m", input_id="i", image_path=rgba(tmp_path, "RGB"))
    with pytest.raises(ValueError, match="RGBA"):
        b.generate(bad, make_context(bad, tmp_path))


def test_sf3d_loads_on_cpu_and_exports_a_glb(monkeypatch, torch, tmp_path):
    seen = {}

    class SF3D:
        @classmethod
        def from_pretrained(cls, repo, config_name, weight_name):
            seen["args"] = (repo, config_name, weight_name)
            return cls()

        def to(self, dev):
            seen["dev"] = dev

        def eval(self):
            pass

        def run_image(self, imgs, bake_resolution, remesh, vertex_count):
            seen["run"] = (len(imgs), bake_resolution, remesh, vertex_count, imgs[0].size)
            return [FakeGlb()], {}
    monkeypatch.setitem(sys.modules, "sf3d", types.ModuleType("sf3d"))
    monkeypatch.setitem(sys.modules, "sf3d.system", types.SimpleNamespace(SF3D=SF3D))
    monkeypatch.delenv("SF3D_USE_CPU", raising=False)
    b = StableFast3D("sf3d", {"repo": "stabilityai/stable-fast-3d"}, {}, ["cpu"])
    b.load()
    assert os.environ["SF3D_USE_CPU"] == "1" and seen["args"][1:] == ("config.yaml", "model.safetensors")
    img = tmp_path / "obj.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(img)
    req = JobRequest(task="i23d", model="sf3d", input_id="i", image_path=str(img), texture_size=512)
    ctx = make_context(req, tmp_path)
    b.generate(req, ctx)
    assert seen["run"][:4] == (1, 512, "none", -1) and ctx.result.completed == 1
    assert b.describe(req)["precision"] == "fp32"


# ------------------------------------------------------------------ Task 3: Stable Audio 3
class _StackKeeper(types.ModuleType):
    """A stand-in library that, like a real one (typeguard, pulled in by SF3D's jaxtyping), keeps the
    stack it was imported from: every frame from load() up stays alive with its local variables."""
    kept: list = []

    def __init__(self, name, **attrs):
        super().__init__(name)
        self._attrs = attrs

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        f = sys._getframe(1)
        while f is not None:
            _StackKeeper.kept.append(f)
            f = f.f_back
        return self._attrs[name]


@pytest.mark.parametrize("which", ["sf3d", "trellis2", "trellis1"])
def test_unload_frees_the_model_even_when_a_library_keeps_the_load_stack(which, monkeypatch, torch, trellis):
    """The SF3D leak: 4 GB of VRAM stayed after unload, held by load()'s local `model` in a frame
    that the library kept. load() must not hold the model in a local."""
    class Model:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def to(self, dev):
            return self

        cuda = eval = lambda self: self
    if which == "sf3d":
        monkeypatch.setitem(sys.modules, "sf3d", types.ModuleType("sf3d"))
        monkeypatch.setitem(sys.modules, "sf3d.system", _StackKeeper("sf3d.system", SF3D=Model))
        b = StableFast3D("sf3d", {"repo": "r"}, {}, ["cpu"])
    else:
        pkg = "trellis2" if which == "trellis2" else "trellis"
        cls_name = "Trellis2ImageTo3DPipeline" if which == "trellis2" else "TrellisImageTo3DPipeline"
        monkeypatch.setitem(sys.modules, f"{pkg}.pipelines", _StackKeeper(f"{pkg}.pipelines", **{cls_name: Model}))
        b = (Trellis2 if which == "trellis2" else Trellis1)(which, {"repo": "r"}, {}, ["cuda:0"])
    monkeypatch.setattr(_StackKeeper, "kept", [])
    b.load()
    assert _StackKeeper.kept, "the stand-in library did not keep the stack"
    ref = weakref.ref(b.model)
    b.unload()
    assert ref() is None, "the model outlived unload(): a kept frame still references it"


def test_stable_audio3_throughput_loop_with_a_fake_library(monkeypatch, torch, tmp_path):

    class Audio:
        def __init__(self, a):
            self.a = a

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class Model:
        model_config = {"sample_rate": 8000}

        def generate(self, prompt, duration, steps, seed, **kw):
            time.sleep(0.05)
            return [Audio(np.full((2, int(duration * 8000)), 0.25, dtype=np.float32))]

    monkeypatch.setattr(stable_audio3, "StableAudioModel", types.SimpleNamespace(
        from_pretrained=lambda name, device=None, model_half=True: Model()))
    b = StableAudio3("sa3-small-music", {"sa3_name": "small-music"}, {}, ["cpu"])
    b.load()
    req = JobRequest(task="music", model="m", input_id="i", prompt="oud", length_s=0.1,
                     repeat_until_budget=True, budget_seconds=0.18)
    ctx = make_context(req, tmp_path)
    b.generate(req, ctx)
    assert 2 <= ctx.result.completed <= 3 and b.describe(req)["steps"] == 8
    assert not list(tmp_path.glob(f"*-{ctx.result.completed}.wav"))      # the late track is discarded


# ------------------------------------------------------------------ Tasks 4 and 8: llama-server
def test_chat_reconciles_with_the_servers_usage_count(fake_server, tmp_path):

    b = LlamaServer("qwen35-4b", {"task": "llm", "file": "*.gguf"}, {"results_dir": str(tmp_path)}, ["cpu"])
    b.http = httpx.Client(base_url=fake_server("complete"), trust_env=False)
    pieces = list(b._chat([{"role": "user", "content": "hi"}], 16, 0.0, 7, grammar="root ::= x"))
    assert "".join(p for p, _ in pieces) == "one two three" and sum(n for _, n in pieces) == 5
    body = FakeLlama.bodies[-1]
    assert body["seed"] == 7 and body["grammar"] == "root ::= x" and body["stream"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_llm_job_counts_tokens_and_saves_each_reply(fake_server, tmp_path):

    b = LlamaServer("qwen35-4b", {"task": "llm"}, {"results_dir": str(tmp_path)}, ["cpu"])
    b.http = httpx.Client(base_url=fake_server("complete"), trust_env=False)
    req = JobRequest(task="llm", model="m", input_id="i", prompt="hi", streams=2)
    ctx = make_context(req, tmp_path)
    b.generate(req, ctx)
    assert ctx.result.tokens == 10 and ctx.result.completed == 2 and ctx.result.text == "one two three"
    assert ctx.result.first_token_seconds is not None and ctx.result.tokens_per_second


def test_slots_and_context_come_from_config_then_model():
    assert LlamaServer("k", {}, {"llm_slots": 4}, ["cpu"])._slots() == 4
    assert LlamaServer("k", {"slots": 2}, {"llm_slots": 4}, ["cpu"])._slots() == 2
    assert LlamaServer("k", {}, {}, ["cpu"])._ctx_per_slot() == 2048
    assert LayoutServer("k", {}, {}, ["cpu"])._ctx_per_slot() == 4096
    assert LayoutServer("k", {}, {"layout_ctx": 8192}, ["cpu"])._ctx_per_slot() == 8192


FAKE_SERVER = textwrap.dedent('''\
    #!{python}
    """A stand-in llama-server binary: --version, or serve /health and chat on --port."""
    import sys
    if "--version" in sys.argv:
        print("version: 9999 (fake)")
        sys.exit(0)
    sys.path.insert(0, {root!r})
    from http.server import ThreadingHTTPServer
    from tests.helpers import FakeLlama
    port = int(sys.argv[sys.argv.index("--port") + 1])
    open({argv!r}, "w").write(" ".join(sys.argv[1:]))
    ThreadingHTTPServer(("127.0.0.1", port), type("H", (FakeLlama,), {{"mode": "complete"}})).serve_forever()
''')


def test_llama_server_load_starts_the_binary_and_unload_stops_it(monkeypatch, tmp_path):
    snap = tmp_path / "snap"
    (snap / "q").mkdir(parents=True)
    (snap / "q" / "model-Q4_K_M.gguf").write_bytes(b"gguf")
    monkeypatch.setattr(llm, "snapshot_download", lambda repo, allow_patterns: str(snap))
    exe = tmp_path / "llama-server"
    argv = tmp_path / "argv.txt"
    exe.write_text(FAKE_SERVER.format(python=sys.executable, root=str(ROOT), argv=str(argv)))
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    b = llm.LlamaServer("qwen35-4b", {"repo": "org/r", "file": "*Q4_K_M.gguf"},
                        {"results_dir": str(tmp_path), "llama_server": str(exe), "llm_slots": 3,
                         "cpu_threads": 2, "llama_args": ["--flash-attn", "on"]}, ["cpu"])
    b.load()
    try:
        args = argv.read_text()
        assert "--parallel 3" in args and "--ctx-size 6144" in args and "--n-gpu-layers 0" in args
        assert "--threads 2" in args and "--flash-attn on" in args
        assert b.version.startswith("version: 9999")
        assert b.describe(JobRequest(task="llm", model="m", input_id="i"))["slots"] == 3
    finally:
        b.unload()
    assert b.proc.poll() is not None and b.model is None


def test_llama_server_that_exits_reports_its_log(monkeypatch, tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "m.gguf").write_bytes(b"x")
    monkeypatch.setattr(llm, "snapshot_download", lambda repo, allow_patterns: str(snap))
    exe = tmp_path / "llama-server"
    exe.write_text(f"#!{sys.executable}\nimport sys\nprint('cudaMalloc failed: out of memory')\nsys.exit(1)\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    b = llm.LlamaServer("k", {"repo": "org/r", "file": "*.gguf"},
                        {"results_dir": str(tmp_path), "llama_server": str(exe)}, ["cpu"])
    with pytest.raises(MemoryError):
        b.load()
    (snap / "m.gguf").unlink()
    with pytest.raises(FileNotFoundError):
        b.load()


def test_layout_server_warms_every_slot_with_the_shared_prompt(fake_server, tmp_path):

    b = LayoutServer("qwen35-4b-layout", {"task": "layout"}, {"results_dir": str(tmp_path), "llm_slots": 3}, ["cpu"])
    b.http = httpx.Client(base_url=fake_server("complete"), trust_env=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("arcdemo.worker.backends.llm.LlamaServer.load", lambda self: None)
        b.load()
    assert b.warmed and len(FakeLlama.bodies) == 3
    assert all(body["max_tokens"] == 1 for body in FakeLlama.bodies)
    req = JobRequest(task="layout", model="m", input_id="i", layout_prompt="layoutgpt")
    assert b.describe(req)["prompt_warmed"] is False

