"""Timing and memory measurement, taken on the machine being measured.

Rules from the spec, enforced here:
- the clock starts when generation starts (model already loaded) and stops when the last
  output file is written; CUDA work is synchronised before any timestamp is read;
- peak memory is reset per request: torch.cuda.max_memory_allocated() on GPU (summed over the
  devices the job used), peak process RSS on CPU (sampled — ru_maxrss cannot be reset).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import psutil


class Clock:
    def __init__(self) -> None:
        self.t0: Optional[float] = None
        self.wall0: Optional[float] = None

    def start(self) -> None:
        self.wall0 = time.time()
        self.t0 = time.perf_counter()

    def now(self) -> float:
        return 0.0 if self.t0 is None else time.perf_counter() - self.t0


def _cuda_devices(devices: list[str]) -> list[int]:
    return [int(d.split(":")[1]) if ":" in d else 0 for d in devices if d.startswith("cuda")]


def cuda_sync(devices: list[str]) -> None:
    idx = _cuda_devices(devices)
    if not idx:
        return
    import torch
    for i in idx:
        torch.cuda.synchronize(i)


class MemoryMeter:
    """Peak memory for one job. Use as a context manager around the generation."""

    def __init__(self, devices: list[str], sample_interval: float = 0.05) -> None:
        self.devices = devices
        self.gpu = _cuda_devices(devices)
        self.interval = sample_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss = 0
        self.peak_allocated_mb: Optional[float] = None
        self.peak_reserved_mb: Optional[float] = None

    def __enter__(self) -> "MemoryMeter":
        if self.gpu:
            import torch
            for i in self.gpu:
                torch.cuda.reset_peak_memory_stats(i)
        else:
            proc = psutil.Process()
            self.peak_rss = proc.memory_info().rss

            def sample() -> None:
                while not self._stop.is_set():
                    try:
                        self.peak_rss = max(self.peak_rss, proc.memory_info().rss)
                    except psutil.Error:
                        pass
                    self._stop.wait(self.interval)

            self._thread = threading.Thread(target=sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self.gpu:
            import torch
            mb = 1024 * 1024
            self.peak_allocated_mb = sum(torch.cuda.max_memory_allocated(i) for i in self.gpu) / mb
            self.peak_reserved_mb = sum(torch.cuda.max_memory_reserved(i) for i in self.gpu) / mb
        else:
            self._stop.set()
            if self._thread:
                self._thread.join(timeout=1)
            self.peak_allocated_mb = self.peak_rss / (1024 * 1024)
