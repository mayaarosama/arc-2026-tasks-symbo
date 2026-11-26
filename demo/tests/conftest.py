"""Shared fixtures for the whole suite (helpers live in tests/helpers.py).

Markers (registered in pytest.ini):
  gpu   loads real model weights; skipped unless pytest is run with `-m gpu`
  slow  starts worker processes or waits on real timers (runs by default)
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from arcdemo.conductor.config import DemoConfig
from arcdemo.worker.app import create_app
from tests.helpers import ALL_TASKS, CONFIG, MOCK_SPEEDS, RELEASE, ROOT, FakeLlama, free_port, mock_worker_cfg


def pytest_collection_modifyitems(config, items):
    """Tests using the `stack` fixture (worker processes) are `slow`; `gpu` tests run only when asked
    for with -m gpu (they load real weights)."""
    for item in items:
        if "stack" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.slow)
    if "gpu" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="loads real models: run with -m gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


# ------------------------------------------------------------------ fixtures ------------------
@pytest.fixture()
def worker_client(tmp_path):
    """worker_client(machine="dgx", speed=0.02, tasks=None, **cfg_overrides) -> TestClient on a mock worker."""
    clients = []

    def make(machine: str = "dgx", speed: float = 0.02, tasks=None, **overrides) -> TestClient:
        cfg = mock_worker_cfg(machine, tmp_path, speed, tasks)
        cfg.update(overrides)
        c = TestClient(create_app(cfg))
        c.__enter__()
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def stack(tmp_path, request):
    """Three mock worker processes (dgx, workstation, cpu) and a DemoConfig pointing at them.
    -> (DemoConfig, {machine: Popen}). Pass other speeds with parametrize(..., indirect=True)."""
    procs, machines = {}, []
    speeds = getattr(request, "param", MOCK_SPEEDS)
    for mid, speed in speeds.items():
        wcfg = yaml.safe_load((CONFIG / f"worker-mock-{mid}.yaml").read_text())
        wcfg.update(mock_speed=speed, results_dir=str(tmp_path / f"results-{mid}"))
        wpath = tmp_path / f"worker-{mid}.yaml"
        wpath.write_text(yaml.safe_dump(wcfg))
        port = free_port()
        procs[mid] = subprocess.Popen([sys.executable, "-m", "arcdemo.worker", "--config", str(wpath),
                                       "--task", ",".join(ALL_TASKS), "--host", "127.0.0.1", "--port", str(port)],
                                      cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        machines.append({"id": mid, "label": mid, "workers": {t: url for t in ALL_TASKS},
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
    cfg_path.write_text(yaml.safe_dump({"extends": str(CONFIG / "demo.yaml"), "machines": machines,
                                        "session_dir": str(tmp_path / "session"),
                                        "replay_dir": str(tmp_path / "replay")}))
    yield DemoConfig(cfg_path), procs
    for p in procs.values():
        p.terminate()
        p.wait(10)


@pytest.fixture()
def offline_cfg(tmp_path):
    """A DemoConfig whose workers are all unreachable (port 9 on localhost): conductor logic and
    routes run without any worker process."""
    raw = yaml.safe_load((CONFIG / "demo.yaml").read_text(encoding="utf-8"))
    for m in raw["machines"]:
        m["workers"] = {t: "http://127.0.0.1:9" for t in (m.get("workers") or {})}
    p = tmp_path / "demo-offline.yaml"
    p.write_text(yaml.safe_dump({**raw, "session_dir": str(tmp_path / "session"),
                                 "replay_dir": str(tmp_path / "replay")}), encoding="utf-8")
    return DemoConfig(p)


@pytest.fixture()
def fake_server():
    """fake_server(mode) -> base URL of a FakeLlama. Request bodies land in FakeLlama.bodies."""
    RELEASE.clear()
    FakeLlama.bodies = []
    servers = []

    def start(mode: str) -> str:
        handler = type("H", (FakeLlama,), {"mode": mode, "bodies": FakeLlama.bodies})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    RELEASE.set()
    for srv in servers:
        srv.shutdown()
        srv.server_close()
