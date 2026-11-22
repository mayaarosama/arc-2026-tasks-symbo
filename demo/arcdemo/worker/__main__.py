"""python -m arcdemo.worker --config config/worker-dgx.yaml --task t2i --port 8101"""
import argparse
import logging
import os
from pathlib import Path

import uvicorn

from ..logs import setup_logging
from .app import ROOT, create_app, load_worker_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True, help="t2i | i23d | music | llm | layout, or a comma list (mock workers)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8101)
    a = p.parse_args()
    cfg_path = str(Path(a.config).resolve())
    os.chdir(ROOT)  # input paths in job requests are relative to demo/
    cfg = load_worker_config(cfg_path, [t.strip() for t in a.task.split(",")])
    setup_logging(Path(cfg["results_dir"]) / f"worker-{a.task.replace(',', '-')}.log")
    log = logging.getLogger("arcdemo.worker")
    log.info("worker %s | tasks %s | mock=%s | port %d", cfg["machine"], cfg["tasks"], bool(cfg.get("mock")), a.port)
    if os.environ.get("HF_HUB_OFFLINE") != "1" and not cfg.get("mock"):
        log.warning("HF_HUB_OFFLINE is not 1 — on stage nothing may touch the network")
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
