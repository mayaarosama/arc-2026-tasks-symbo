"""Worker service: runs on each machine under test, one process per task (per Python env).

    python -m arcdemo.worker --config config/worker-dgx.yaml --task t2i --port 8101

Endpoints (all require X-Demo-Token when ARC_DEMO_TOKEN is set):
  GET  /health                 machine, loaded models, busy flag, env hash
  GET  /env                    library versions (the conductor flags mismatches between machines)
  POST /load   {model}         load a model in the background (errors: /env last_error + log)
  POST /unload {model?}        free memory (all models if none given)
  POST /jobs   JobRequest      start a job; returns immediately (409 if busy)
  GET  /jobs/{run_id}          status, progress, timings, outputs so far
  POST /jobs/{run_id}/cancel
  GET  /files/{name}           fetch an output file
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import logging
import os
import platform
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import psutil
import yaml
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

try:                     # optional: the llm venv has no torch (llama-server does its GPU work)
    import torch
except ImportError:
    torch = None

from ..layout_prompt import DEFAULT_CATALOG, load_catalog
from ..schemas import TERMINAL, HealthInfo, JobRequest, JobResult, LoadRequest
from ..weights import on_disk
from . import tasks
from .backends.base import Backend, JobContext, backend_class, free_memory, is_oom
from .metrics import Clock

ROOT = Path(__file__).resolve().parents[2]          # demo/
log = logging.getLogger("arcdemo.worker")


def load_worker_config(path: str, tasks: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    models = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))["models"]
    for key, extra in (cfg.get("model_overrides") or {}).items():
        models[key] = {**models[key], **extra}
    cfg["models"] = models
    cfg["tasks"] = tasks
    if "layout" in tasks:     # the furniture catalogue the prompt, grammar and room checks use
        cfg["layout_catalog_sha"] = load_catalog(cfg.get("layout_catalog", DEFAULT_CATALOG))["sha"]
    cfg["results_dir"] = str((ROOT / cfg.get("results_dir", "results")).resolve())
    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


class ModelManager:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.loaded: "OrderedDict[str, Backend]" = OrderedDict()
        self.loading: Optional[str] = None
        self.last_error: Optional[str] = None
        self.lock = threading.RLock()

    def devices_for(self, key: str) -> list[str]:
        task = self.cfg["models"][key]["task"]
        return list(self.cfg["devices"].get(task, ["cpu"]))

    def get(self, key: str) -> Backend:
        """Return a loaded backend, loading it first if needed."""
        with self.lock:
            if key in self.loaded:
                return self.loaded[key]
            if key not in self.cfg["models"]:
                raise KeyError(f"unknown model '{key}' (not in config/models.yaml)")
            spec = self.cfg["models"][key]
            if spec["task"] not in self.cfg["tasks"]:
                raise ValueError(f"model '{key}' is a {spec['task']} model; this worker serves {self.cfg['tasks']}")
            name = spec["backend"]
            if self.cfg.get("mock"):   # music and llm have their own stand-ins (tone, canned text)
                name = tasks.MOCK_BACKEND[spec["task"]]
            backend = backend_class(name)(key, spec, self.cfg, self.devices_for(key))
            self.loading = key
            log.info("loading %s (%s) on %s", key, name, backend.devices)
            t = time.perf_counter()
            try:
                backend.load()
            except BaseException as e:
                log.error("load %s failed after %.1fs: %s", key, time.perf_counter() - t, _short(e))
                backend.unload()
                raise
            finally:
                self.loading = None
            log.info("loaded %s in %.1fs", key, time.perf_counter() - t)
            self.loaded[key] = backend
            return backend

    def unload(self, key: Optional[str] = None, keep_task: Optional[str] = None) -> list[str]:
        """Unload `key`, or every model (keep_task: every model except that task's). -> unloaded keys"""
        with self.lock:
            keys = [key] if key else [k for k in self.loaded
                                      if not keep_task or self.cfg["models"][k]["task"] != keep_task]
            gone = []
            for k in keys:
                b = self.loaded.pop(k, None)
                if b is not None:
                    b.unload()
                    gone.append(k)
                    log.info("unloaded %s", k)
            free_memory()
            return gone

    def ensure(self, key: str) -> Backend:
        """Load `key` if needed. Every other model in this worker is unloaded first, so two
        models never compete for the same memory (worker config exclusive_models: false keeps them)."""
        with self.lock:
            if self.cfg.get("exclusive_models", True):
                for k in [k for k in self.loaded if k != key]:
                    self.unload(k)
            return self.get(key)


class Job:
    def __init__(self, req: JobRequest, result: JobResult) -> None:
        self.req = req
        self.result = result
        self.clock = Clock()
        self.cancel = threading.Event()


class JobRunner:
    def __init__(self, cfg: dict, models: ModelManager) -> None:
        self.cfg = cfg
        self.models = models
        self.jobs: "OrderedDict[str, Job]" = OrderedDict()
        self.active: Optional[Job] = None
        self.lock = threading.Lock()

    def submit(self, req: JobRequest) -> JobResult:
        with self.lock:
            if self.active is not None and self.active.result.status not in TERMINAL:
                raise HTTPException(409, f"busy with {self.active.req.run_id}")
            # 0 = throughput run (as many as fit in the budget): nothing is "missing" at the bell
            requested = tasks.requested(req)
            result = JobResult(run_id=req.run_id, machine=self.cfg["machine"], task=req.task,
                               model=req.model, model_repo=self.cfg["models"].get(req.model, {}).get("repo", ""),
                               input_id=req.input_id, protocol=req.protocol, requested=requested)
            job = Job(req, result)
            self.jobs[req.run_id] = job
            while len(self.jobs) > 100:
                self.jobs.popitem(last=False)
            self.active = job
        log.info("[%s] submit %s model=%s input=%s budget=%s", req.run_id[:6], req.task, req.model,
                 req.input_id, req.budget_seconds)
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return result

    def snapshot(self, run_id: str) -> JobResult:
        job = self.jobs.get(run_id)
        if job is None:
            raise HTTPException(404, "unknown run_id")
        r = job.result
        if r.status == "running":
            r.elapsed_seconds = round(job.clock.now(), 3)
        return r

    def _run(self, job: Job) -> None:
        req, r = job.req, job.result
        try:
            tasks.prepare_inputs(req, Path(self.cfg["results_dir"]))   # off the clock (i23d: an upload)
            r.status = "loading"
            t = time.perf_counter()
            backend = self.models.ensure(req.model)
            r.load_seconds = round(time.perf_counter() - t, 3)
            if r.load_seconds > 0.5:
                r.message = "model was not preloaded: load time excluded from the clock but logged"
            r.settings = {**tasks.settings_echo(req), **backend.describe(req)}
            out_dir = Path(self.cfg["results_dir"])
            ctx = JobContext(req, r, job.clock, out_dir, backend.devices, job.cancel)
            r.status = "running"
            log.info("[%s] clock start | settings %s", req.run_id[:6], r.settings)
            job.clock.start()
            with backend.memory_meter() as mem:
                backend.generate(req, ctx)
            r.peak_memory_mb = round(mem.peak_allocated_mb or 0, 1)
            r.peak_reserved_mb = None if mem.peak_reserved_mb is None else round(mem.peak_reserved_mb, 1)
            over = (req.budget_seconds is not None and r.all_outputs_seconds is not None
                    and r.all_outputs_seconds > req.budget_seconds)
            short = r.requested and r.completed < r.requested
            if job.cancel.is_set():
                r.status = "cancelled"
            elif (short or over) and req.budget_seconds is not None:
                r.status = "budget_exceeded"
            elif short:          # no budget, yet outputs are missing: a backend bug, not a result
                r.status = "error"
                r.message = r.message or f"finished with {r.completed} of {r.requested} outputs"
            else:
                r.status = "ok"
        except BaseException as e:  # noqa: BLE001 — every failure must come back as a status
            r.message = _short(e)
            if is_oom(e):
                r.status = "out_of_memory"
                log.warning("[%s] out of memory: %s", req.run_id[:6], r.message)
                self.models.unload(req.model)
            else:
                r.status = "error"
                log.exception("[%s] job failed", req.run_id[:6])
        finally:
            r.elapsed_seconds = round(job.clock.now(), 3)
            log.info("[%s] %s | %d/%s outputs | all=%ss | peak=%s MB", req.run_id[:6], r.status,
                     r.completed, r.requested or "?", r.all_outputs_seconds, r.peak_memory_mb)


def memory_report() -> dict:
    """This worker's resident memory (itself plus children such as llama-server) and what the
    machine has free. VRAM only when this process already uses CUDA (asking would create a context)."""
    out: dict = {}
    try:
        p = psutil.Process()
        rss = p.memory_info().rss
        for ch in p.children(recursive=True):
            try:
                rss += ch.memory_info().rss
            except psutil.Error:
                pass
        out["rss_mb"] = round(rss / 2**20)
        out["ram_free_mb"] = round(psutil.virtual_memory().available / 2**20)
    except psutil.Error:
        pass
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            out["vram_free_mb"] = round(torch.cuda.mem_get_info()[0] / 2**20)
    except Exception:  # noqa: BLE001 — a report, never a failure
        pass
    return out


def _short(e: BaseException) -> str:
    s = f"{type(e).__name__}: {e}"
    return s if len(s) < 400 else s[:400] + "…"


def env_report(cfg: dict) -> dict:
    dists = sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in md.distributions()
                    if d.metadata["Name"]})
    # Torch / CUDA wheels legitimately differ between the CPU box and the GPU machines.
    comparable = [d for d in dists if not d.split("==")[0].startswith(("torch", "nvidia-", "triton"))]
    info: dict = {"machine": cfg["machine"], "python": platform.python_version(),
                  "platform": platform.platform(), "packages": comparable,
                  "env_hash": hashlib.sha1("\n".join(comparable).encode()).hexdigest()[:12]}
    if torch is None:
        info["torch"] = None
    else:
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpus"] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if torch.cuda.is_available() else [])
        info["threads"] = torch.get_num_threads()
    for pkg in ("diffusers", "transformers", "accelerate", "bitsandbytes"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return info


def create_app(cfg: dict) -> FastAPI:
    models = ModelManager(cfg)
    runner = JobRunner(cfg, models)
    token = os.environ.get("ARC_DEMO_TOKEN")
    env_cache: dict = {}

    def auth(x_demo_token: Optional[str] = Header(default=None)) -> None:
        if token and x_demo_token != token:
            raise HTTPException(401, "bad or missing X-Demo-Token")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        env_cache.update(env_report(cfg))   # slow (scans installed packages): do it once, here
        torch_tasks = [t for t in cfg["tasks"] if t not in tasks.OWN_THREADS]   # llama-server sets its own threads
        if any(d == "cpu" for t in torch_tasks for d in cfg["devices"].get(t, [])) and not cfg.get("mock"):
            torch.set_num_threads(int(cfg.get("cpu_threads") or psutil.cpu_count(logical=False) or 1))
            log.info("CPU threads: %d", torch.get_num_threads())
        keys = [k for t in cfg["tasks"] for k in (cfg.get("preload") or {}).get(t, [])]

        def preload() -> None:   # in the background so /health answers while weights load
            for k in keys:
                try:
                    models.get(k)
                except BaseException as e:  # noqa: BLE001 — already logged by ModelManager.get
                    models.last_error = f"preload {k}: {_short(e)}"
        threading.Thread(target=preload, daemon=True).start()
        yield

    app = FastAPI(title=f"arcdemo worker {cfg['machine']}", dependencies=[Depends(auth)], lifespan=lifespan)

    def is_busy() -> bool:
        return runner.active is not None and runner.active.result.status not in TERMINAL

    @app.get("/health", response_model=HealthInfo)
    def health() -> HealthInfo:
        return HealthInfo(machine=cfg["machine"], tasks=cfg["tasks"], mock=bool(cfg.get("mock")),
                          busy=is_busy(), loaded=list(models.loaded), loading=models.loading,
                          env_hash=env_cache["env_hash"], available=available(), **memory_report())

    disk = {"at": 0.0, "keys": []}

    def available() -> list[str]:
        """Models of this worker's tasks whose weights are on disk (re-checked every 15 s)."""
        if time.time() - disk["at"] > 15:
            disk["keys"] = [k for k, spec in cfg["models"].items() if spec.get("task") in cfg["tasks"]
                            and (cfg.get("mock") or on_disk(spec, k))]
            disk["at"] = time.time()
        return disk["keys"]

    @app.get("/env")
    def env() -> dict:
        return {**env_cache, "last_error": models.last_error}

    @app.post("/load")
    def load(req: LoadRequest) -> dict:
        if runner.active is not None and runner.active.result.status not in TERMINAL:
            raise HTTPException(409, "busy")

        def do() -> None:   # progress is visible in /health (loading, loaded) and the log
            try:
                models.ensure(req.model)
                models.last_error = None
            except BaseException as e:  # noqa: BLE001 — already logged by ModelManager.get
                models.last_error = f"load {req.model}: {_short(e)}"
                if is_oom(e):
                    models.unload(req.model)
        threading.Thread(target=do, daemon=True).start()
        return {"model": req.model, "status": "loading"}

    @app.post("/unload")
    def unload(body: Optional[dict] = None) -> dict:
        """{model}: that model; {keep_task}: every model of the other tasks; {}: everything.
        Refused while a job runs: it would pull the model out from under it."""
        if is_busy():
            raise HTTPException(409, "busy")
        body = body or {}
        gone = models.unload(body.get("model"), keep_task=body.get("keep_task"))
        return {"unloaded": gone, "loaded": list(models.loaded), **memory_report()}

    @app.post("/jobs", response_model=JobResult)
    def submit(req: JobRequest) -> JobResult:
        if req.task not in cfg["tasks"]:
            raise HTTPException(400, f"this worker serves {cfg['tasks']}, not {req.task}")
        return runner.submit(req)

    @app.get("/jobs/{run_id}", response_model=JobResult)
    def get_job(run_id: str) -> JobResult:
        return runner.snapshot(run_id)

    @app.post("/jobs/{run_id}/cancel", response_model=JobResult)
    def cancel(run_id: str) -> JobResult:
        job = runner.jobs.get(run_id)
        if job is None:
            raise HTTPException(404, "unknown run_id")
        job.cancel.set()
        return runner.snapshot(run_id)

    @app.get("/files/{name}")
    def files(name: str) -> FileResponse:
        if "/" in name or "\\" in name or name.startswith("."):
            raise HTTPException(400, "bad name")
        path = Path(cfg["results_dir"]) / name
        if not path.is_file():
            raise HTTPException(404, "no such file")
        return FileResponse(path)

    return app
