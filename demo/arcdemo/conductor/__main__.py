"""python -m arcdemo.conductor --config config/demo.yaml [--port 8000]"""
import argparse

import uvicorn

from ..logs import setup_logging
from .app import create_app
from .config import DemoConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/demo.yaml")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    cfg = DemoConfig(a.config)
    setup_logging(cfg.session_dir / "conductor.log")
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
