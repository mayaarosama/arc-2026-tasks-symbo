"""Task 4 — language model race: one GGUF model, served by llama.cpp's llama-server, on every machine.

Why llama.cpp: the same binary and the same quantised file run on the CPU box, a 4 GB laptop GPU,
the workstation and the DGX, so only the hardware changes between cells (proposal: identical
quantisation, sampling and seed everywhere). vLLM is faster on the DGX but has no equal CPU path,
so using it there would change the software stack mid-comparison.

load() starts llama-server as a child process (off the clock) and waits for /health. A job opens
`streams` concurrent streaming chat requests; the server batches them (continuous batching).
Tokens are counted as they arrive and reconciled with the server's own usage count per reply.
  iso_quality  1 stream, fixed max_tokens, temperature 0: the same answer everywhere, compare time
  iso_time     N streams, repeat_until_budget: total tokens generated inside the budget
One stream of a small model is memory-bandwidth bound, so the DGX and the workstation nearly tie on
it; many concurrent streams are where the DGX's advantage is real.

Worker config keys: llama_server (binary, default "llama-server" on PATH), llm_slots (parallel
sequences the server holds; >= the largest `streams` this machine runs), llm_ctx (tokens per slot,
default 2048), cpu_threads, llama_args (extra flags, list), llama_env (extra environment
variables for the server process, e.g. Vulkan driver workarounds).
"""
from __future__ import annotations

import json
import logging
import os
import random
import socket
import subprocess
import threading
import time
from contextlib import closing
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterator, Optional

from ...schemas import JobRequest
from .base import Backend, JobContext

log = logging.getLogger("arcdemo.worker")
SYSTEM = ("You are a helpful assistant at a public technology demo. "
          "Answer clearly, in plain English, without tables.")


class LlmBackend(Backend):
    """Shared generation loop; subclasses provide _stream()."""
    task = "llm"

    def _stream(self, req: JobRequest, seed: int) -> Iterator[tuple[str, int]]:
        """Yield (text piece, tokens it adds) as one reply is generated."""
        raise NotImplementedError

    def describe(self, req: JobRequest) -> dict:
        return {"streams": req.streams, "max_tokens": req.max_tokens, "temperature": req.temperature,
                "quant": self.spec.get("file", "")}

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        n = max(1, req.streams)
        r = ctx.result
        lock = threading.Lock()
        last = [0.0]
        errors: list[BaseException] = []
        ctx.progress(stage=f"{n} stream{'s' if n > 1 else ''}")

        def count(k: int) -> bool:
            """Count tokens on the job clock; refuse any that arrive after the bell."""
            if ctx.should_stop():
                return False
            t = ctx.clock.now()
            with lock:
                if r.first_token_seconds is None:
                    r.first_token_seconds = round(t, 3)
                r.tokens += k
                last[0] = t
                r.tokens_per_second = round(r.tokens / t, 1) if t > 0 else None
            return True

        def run(i: int) -> None:
            k = 0
            while not ctx.should_stop():
                parts: list[str] = []
                finished = True
                with closing(self._stream(req, req.seed + i + k * n)) as gen:
                    for piece, ntok in gen:
                        if not count(ntok):
                            finished = False       # cut off by the budget: tokens count, reply is not saved
                            break
                        parts.append(piece)
                        if i == 0:
                            r.text = "".join(parts)
                if not finished:
                    return
                path = ctx.output_path(f"-{i}-{k}.txt")
                path.write_text("".join(parts), encoding="utf-8")
                ctx.record_output(path)
                k += 1
                if not req.repeat_until_budget or req.budget_seconds is None:
                    return

        def guarded(i: int) -> None:
            try:
                run(i)
            except BaseException as e:  # noqa: BLE001 — re-raised on the job thread below
                errors.append(e)
                ctx.cancel.set()           # one failed stream stops the others

        threads = [threading.Thread(target=guarded, args=(i,), daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        if r.tokens:   # the last output of an LLM run is its last token, not its last whole reply
            r.all_outputs_seconds = round(last[0], 3)
            r.tokens_per_second = round(r.tokens / last[0], 1)


class LlamaServer(LlmBackend):
    def load(self) -> None:
        import httpx
        from huggingface_hub import snapshot_download

        pattern = self.spec["file"]
        root = Path(snapshot_download(self.repo, allow_patterns=[pattern]))
        files = sorted(p for p in root.rglob("*.gguf") if fnmatch(p.relative_to(root).as_posix(), pattern))
        if not files:
            raise FileNotFoundError(f"no file matching {pattern} in {self.repo}")
        gpus = [d.split(":")[1] if ":" in d else "0" for d in self.devices if d.startswith("cuda")]
        slots, per_slot = self._slots(), self._ctx_per_slot()
        exe = self.cfg.get("llama_server", "llama-server")
        self.port = _free_port()
        args = [exe, "-m", str(files[0]),          # split GGUFs: the server finds parts 2..n itself
                "--host", "127.0.0.1", "--port", str(self.port),
                "--parallel", str(slots), "--ctx-size", str(slots * per_slot),
                "--n-gpu-layers", "999" if gpus else "0", "--jinja"]
        if self.cfg.get("cpu_threads"):
            args += ["--threads", str(self.cfg["cpu_threads"])]
        args += [str(a) for a in self.cfg.get("llama_args") or []]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(gpus),   # "" keeps a CUDA build off the GPU
               **{k: str(v) for k, v in (self.cfg.get("llama_env") or {}).items()}}
        self.log_path = Path(self.cfg["results_dir"]) / f"llama-server-{self.key}.log"
        self._logf = open(self.log_path, "ab")
        log.info("starting %s", " ".join(args))
        self.proc = subprocess.Popen(args, stdout=self._logf, stderr=subprocess.STDOUT, env=env)
        self.http = httpx.Client(base_url=f"http://127.0.0.1:{self.port}", trust_env=False,
                                 timeout=httpx.Timeout(120, connect=5))   # a stalled stream errors
        end = time.time() + 900
        while True:            # 503 while the weights load, 200 when ready
            if self.proc.poll() is not None:
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                self.unload()
                if "out of memory" in tail.lower() or "failed to allocate" in tail.lower():
                    raise MemoryError(f"llama-server out of memory loading {files[0].name}")
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}): {tail[-400:]}")
            try:
                if self.http.get("/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if time.time() > end:
                self.unload()
                raise TimeoutError("llama-server not ready after 900 s")
            time.sleep(0.5)
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
        self.version = next((ln.strip() for ln in (out.stdout + out.stderr).splitlines() if "version" in ln), "?")
        warm = JobRequest(task="llm", model=self.key, input_id="warmup", prompt="Hello", max_tokens=8)
        for _ in self._stream(warm, 0):   # first request pays one-off CUDA / page-in costs
            pass
        self.model = self.proc

    def _ctx_per_slot(self) -> int:
        return int(self.cfg.get("llm_ctx", 2048))

    def _slots(self) -> int:
        """Server slots: the worker's llm_slots, unless the model's models.yaml entry caps it with
        `slots` (a model that only fits the card with a small KV cache, e.g. Qwen3.8 27B on 16 GB)."""
        return int(self.spec.get("slots") or self.cfg.get("llm_slots", 1))

    def unload(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
        for h in (getattr(self, "http", None), getattr(self, "_logf", None)):
            if h is not None:
                h.close()
        self.model = None

    def describe(self, req: JobRequest) -> dict:
        return {**super().describe(req), "server": getattr(self, "version", "?"),
                "slots": self._slots()}

    def memory_meter(self) -> "_ServerMemory":
        return _ServerMemory(self)

    def _stream(self, req: JobRequest, seed: int) -> Iterator[tuple[str, int]]:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": req.prompt or ""}]
        yield from self._chat(messages, req.max_tokens, req.temperature, seed)

    def _chat(self, messages: list[dict], max_tokens: int, temperature: float, seed: int,
              **extra) -> Iterator[tuple[str, int]]:
        """One streaming chat completion. `extra` goes into the request body (e.g. grammar)."""
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                "stream": True, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},   # stage shows answers, not reasoning
                **extra}
        counted = 0
        with self.http.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                for ch in chunk.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:                  # llama-server streams one token per chunk
                        counted += 1
                        yield piece, 1
                used = (chunk.get("usage") or {}).get("completion_tokens")
                if used is not None and used != counted:
                    yield "", used - counted   # reconcile with the server's own count
                    counted = used


class _ServerMemory:
    """Memory held by the llama-server process: VRAM via nvidia-smi on GPU, RSS on CPU.
    Weights and KV cache are allocated at load, so the end-of-job value is the peak."""

    def __init__(self, backend: LlamaServer) -> None:
        self.b = backend
        self.peak_allocated_mb: Optional[float] = None
        self.peak_reserved_mb: Optional[float] = None

    def __enter__(self) -> "_ServerMemory":
        return self

    def __exit__(self, *exc) -> None:
        pid = self.b.proc.pid
        if any(d.startswith("cuda") for d in self.b.devices):
            try:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                mb = [float(m) for p, m in (ln.split(",") for ln in out.stdout.splitlines() if "," in ln)
                      if p.strip() == str(pid) and m.strip().replace(".", "").isdigit()]
                self.peak_allocated_mb = sum(mb) or None     # Windows WDDM may report N/A
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            import psutil
            try:
                self.peak_allocated_mb = psutil.Process(pid).memory_info().rss / (1024 * 1024)
            except psutil.Error:
                pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ mock -------------------
_MOCK_TEXT = ("[MOCK] This reply is placeholder text from a mock worker. Its timing is simulated and is "
              "never a result. On stage, the same prompt goes to every machine and each streams its answer "
              "token by token, while the counter shows tokens per second. ").split(" ")


class LlmMock(LlmBackend):
    """Canned text at a simulated rate: workstation ~40 tok/s per stream (mock_speed 1.0), slower per
    stream as streams are added, as batching does on real hardware."""

    def load(self) -> None:
        time.sleep(0.3)

    def _stream(self, req: JobRequest, seed: int) -> Iterator[tuple[str, int]]:
        delay = 0.025 * float(self.cfg.get("mock_speed", 1.0)) * (1 + req.streams / 32)
        rng = random.Random(seed)
        for j in range(req.max_tokens):
            time.sleep(delay * rng.uniform(0.8, 1.2))
            yield (" " if j else "") + _MOCK_TEXT[j % len(_MOCK_TEXT)], 1
