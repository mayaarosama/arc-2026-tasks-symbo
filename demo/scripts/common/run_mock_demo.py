"""Laptop dry-run: three mock workers + the conductor, no GPU or weights needed.

    python -m scripts.common.run_mock_demo            # then open http://127.0.0.1:8000/operator
    python -m scripts.common.run_mock_demo --no-conductor   # workers only (tests, rehearse.py)

Failure drill: kill one worker (its PID is printed) mid-round and watch the cell go REPLAY.
Logs: results-mock/<machine>/worker-*.log and session-mock/conductor.log
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from arcdemo.defaults import DEFAULTS
from scripts.common.settings import SCRIPTS

ROOT = Path(__file__).resolve().parents[2]       # demo/
WORKERS = list(SCRIPTS.run_mock_demo.workers.items())     # config/scripts.yaml (match config/demo-mock.yaml)
CONDUCTOR = f"http://127.0.0.1:{DEFAULTS.conductor.port}"


def start_workers() -> list[subprocess.Popen]:
    procs = []
    for machine, port in WORKERS:
        p = subprocess.Popen([sys.executable, "-m", "arcdemo.worker", "--config", f"config/worker-mock-{machine}.yaml",
                              "--task", "t2i,i23d,music,llm,layout", "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT)
        print(f"mock worker {machine:12s} pid {p.pid}  http://127.0.0.1:{port}/health")
        procs.append(p)
    return procs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-conductor", action="store_true")
    a = ap.parse_args()
    procs = start_workers()
    if not a.no_conductor:
        procs.append(subprocess.Popen([sys.executable, "-m", "arcdemo.conductor", "--config", SCRIPTS.run_mock_demo.conductor_config],
                                      cwd=ROOT))
        print(f"conductor: {CONDUCTOR}/operator   stage: {CONDUCTOR}/stage")
    try:   # runs until Ctrl+C; a killed worker (failure drill) does not stop the others
        reported = set()
        while True:
            for p in procs:
                if p.poll() is not None and p.pid not in reported:
                    reported.add(p.pid)
                    print(f"process {p.pid} exited with {p.returncode}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
