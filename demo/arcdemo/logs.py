"""One logging setup for every process: console + a file, every line tagged with time and source.

Worker lines carry the job's run_id in brackets; the conductor logs the same run_id when it
dispatches, so one grep across conductor.log and worker-*.log follows a cell end to end:
    grep 3f9c2a conductor.log results/worker-t2i.log
Set ARC_LOG_LEVEL=DEBUG to also see every failed poll.
"""
import logging
import os
from pathlib import Path


def setup_logging(logfile: Path) -> None:
    level = os.environ.get("ARC_LOG_LEVEL", "INFO").upper()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for h in (logging.StreamHandler(), logging.FileHandler(logfile, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
