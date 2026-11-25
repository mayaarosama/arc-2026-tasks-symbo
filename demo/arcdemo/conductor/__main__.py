"""python -m arcdemo.conductor --config config/demo.yaml [--port 8000]  (defaults: config/defaults.yaml conductor)"""
import argparse

import uvicorn

from ..defaults import DEFAULTS
from ..logs import setup_logging
from .app import create_app
from .config import DemoConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULTS.conductor.config)
    p.add_argument("--host", default=DEFAULTS.conductor.host)
    p.add_argument("--port", type=int, default=DEFAULTS.conductor.port)
    a = p.parse_args()
    cfg = DemoConfig(a.config)
    setup_logging(cfg.session_dir / "conductor.log")
    uvicorn.run(create_app(cfg), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
