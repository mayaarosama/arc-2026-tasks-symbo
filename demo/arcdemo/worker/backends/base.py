"""Backend interface. A backend wraps one model variant (one key in config/models.yaml)."""
from __future__ import annotations

import gc
import importlib
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Optional

from ...schemas import JobRequest, JobResult
from ..metrics import Clock, cuda_sync

log = logging.getLogger("arcdemo.worker")


class JobContext:
    """What a backend may touch while generating: the clock, the budget, progress, outputs."""

    def __init__(self, req: JobRequest, result: JobResult, clock: Clock, out_dir: Path,
                 devices: list[str], cancel: threading.Event) -> None:
        self.req = req
        self.result = result
        self.clock = clock
        self.out_dir = out_dir
        self.devices = devices
        self.cancel = cancel
        self._lock = threading.Lock()

    # ---- budget -------------------------------------------------------------------------
    @property
    def budget(self) -> Optional[float]:
        return self.req.budget_seconds

    def budget_hit(self) -> bool:
        return self.budget is not None and self.clock.now() >= self.budget

    def should_stop(self) -> bool:
        return self.cancel.is_set() or self.budget_hit()

    # ---- progress -----------------------------------------------------------------------
    def progress(self, step: Optional[int] = None, steps: Optional[int] = None,
                 stage: Optional[str] = None) -> None:
        p = self.result.progress
        if step is not None:
            p.step = step
        if steps is not None:
            p.steps = steps
        if stage is not None:
            p.stage = stage

    # ---- outputs ------------------------------------------------------------------------
    def output_path(self, suffix: str) -> Path:
        return self.out_dir / f"{self.req.run_id}{suffix}"

    def record_output(self, path: Path) -> None:
        """Call after an output file is fully written. Timestamps it on the job clock."""
        cuda_sync(self.devices)
        t = self.clock.now()
        with self._lock:
            r = self.result
            r.outputs.append(path.name)
            r.output_times.append(round(t, 3))
            r.completed = len(r.outputs)
            r.progress.done = r.completed
            if r.first_output_seconds is None:
                r.first_output_seconds = round(t, 3)
            r.all_outputs_seconds = round(t, 3)
        log.info("[%s] output %d/%s %s at %.2fs", self.req.run_id[:6], r.completed, r.requested or "?", path.name, t)

    def save_image(self, index: int, image: Any) -> Path:
        path = self.output_path(f"-{index}.png")
        image.save(path)
        self.record_output(path)
        return path


class Backend:
    task: str = ""

    def __init__(self, key: str, spec: dict, worker_cfg: dict, devices: list[str]) -> None:
        self.key = key
        self.spec = spec
        self.cfg = worker_cfg
        self.devices = devices
        self.model: Any = None      # whatever load() produced; dropped by unload()

    @property
    def repo(self) -> str:
        return self.spec.get("repo", "")

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        self.model = None
        free_memory()

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        raise NotImplementedError

    def describe(self, req: JobRequest) -> dict:
        """Settings echo for the run log: what actually ran, including precision/offload."""
        return {}


def free_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    msg = str(exc).lower()
    return "out of memory" in msg or "cannot allocate memory" in msg


def resolve_dtype(spec: dict, cfg: dict, device: str):
    import torch
    d = spec.get("dtype", "auto")
    if d == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16
        return torch.bfloat16 if cfg.get("cpu_bf16") else torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[d]


_BACKENDS: dict[str, str] = {
    "diffusers_t2i": "arcdemo.worker.backends.diffusers_t2i:DiffusersT2I",
    "trellis2": "arcdemo.worker.backends.trellis:Trellis2",
    "trellis1": "arcdemo.worker.backends.trellis:Trellis1",
    "sf3d": "arcdemo.worker.backends.sf3d:StableFast3D",
    "stable_audio3": "arcdemo.worker.backends.stable_audio3:StableAudio3",
    "music_stub": "arcdemo.worker.backends.music_stub:MusicStub",
    "mock": "arcdemo.worker.backends.mock:MockBackend",   # every non-music model when mock: true (music: music_stub)
}


def backend_class(name: str) -> Callable[..., Backend]:
    """Imported lazily: the i23d env has no diffusers, the t2i env has no TRELLIS."""
    module, cls = _BACKENDS[name].split(":")
    return getattr(importlib.import_module(module), cls)
