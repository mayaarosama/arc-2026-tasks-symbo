"""Helpers shared by the test files (import as `from tests.helpers import ...`)."""
from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from arcdemo.schemas import JobRequest, JobResult
from arcdemo.worker.app import load_worker_config
from arcdemo.worker.backends.base import JobContext
from arcdemo.worker.metrics import Clock

ROOT = Path(__file__).resolve().parents[1]           # demo/
CONFIG = ROOT / "config"
ALL_TASKS = ["t2i", "i23d", "music", "llm", "layout"]
MOCK_SPEEDS = {"dgx": 0.01, "workstation": 0.02, "cpu": 0.05}


# ------------------------------------------------------------------ helpers -------------------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_context(req: JobRequest, out_dir: Path, devices=("cpu",), start: bool = True) -> JobContext:
    """A JobContext as the worker builds one, for calling a backend's generate() directly."""
    res = JobResult(run_id=req.run_id, machine="test", task=req.task, model=req.model,
                    input_id=req.input_id, protocol=req.protocol)
    clock = Clock()
    if start:
        clock.start()
    return JobContext(req, res, clock, out_dir, list(devices), threading.Event())


def mock_worker_cfg(machine: str, tmp_path: Path, speed: float = 0.02, tasks=None) -> dict:
    cfg = load_worker_config(str(CONFIG / f"worker-mock-{machine}.yaml"), list(tasks or ALL_TASKS))
    cfg["mock_speed"] = speed
    cfg["results_dir"] = str(tmp_path)
    return cfg


def wait_job(c: TestClient, run_id: str, timeout: float = 20) -> dict:
    end = time.time() + timeout
    res = {}
    while time.time() < end:
        res = c.get(f"/jobs/{run_id}").json()
        if res["status"] in ("ok", "budget_exceeded", "out_of_memory", "error", "cancelled"):
            return res
        time.sleep(0.05)
    pytest.fail(f"job did not finish: {res}")


# ------------------------------------------------------------------ fake llama-server ---------
RELEASE = threading.Event()        # set at teardown: lets stalled handler threads exit


class FakeLlama(BaseHTTPRequestHandler):
    """Stands in for llama-server's /v1/chat/completions and /health.
    mode: tokens_then_stall (3 tokens, then nothing) | stall_before_headers | error |
          complete (3 tokens, a usage chunk that disagrees by +2, then [DONE])."""
    mode = "tokens_then_stall"
    bodies: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        type(self).bodies.append(json.loads(body or b"{}"))
        if self.mode == "error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.mode == "stall_before_headers":     # still reading the prompt: nothing sent yet
            RELEASE.wait(30)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for word in ("one", " two", " three"):
                chunk = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            if self.mode == "complete":
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 5}})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            RELEASE.wait(30)                       # then the server goes quiet
        except OSError:                            # the client closed the stream: what we want
            pass
