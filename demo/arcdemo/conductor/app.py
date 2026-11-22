"""Conductor web app, on the operator laptop.

    python -m arcdemo.conductor --config config/demo.yaml --port 8000
    projector:  http://localhost:8000/stage
    operator:   http://localhost:8000/operator
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import ROOT, DemoConfig
from .core import Conductor
from .moderation import check_prompt

STATIC = Path(__file__).parent / "static"


class RoundBody(BaseModel):
    segment_id: str
    input_id: str
    prompt: Optional[str] = None
    override: bool = False
    force_replay: list[str] = []
    budget_override: Optional[float] = None


class ViewBody(BaseModel):
    view: str


class PromptBody(BaseModel):
    prompt: str


class SegmentBody(BaseModel):
    segment_id: str


def create_app(cfg: DemoConfig) -> FastAPI:
    cond: Optional[Conductor] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal cond
        cond = Conductor(cfg)
        task = asyncio.create_task(cond.health_loop())
        yield
        task.cancel()
        await cond.http.aclose()

    app = FastAPI(title="arcdemo conductor", lifespan=lifespan)
    cfg.session_dir.mkdir(parents=True, exist_ok=True)
    cfg.replay_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/media", StaticFiles(directory=cfg.session_dir), name="media")
    app.mount("/replay-media", StaticFiles(directory=cfg.replay_dir), name="replay")
    app.mount("/media-input/data", StaticFiles(directory=ROOT / "data"), name="inputs")

    @app.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse("/operator")

    @app.get("/stage")
    def stage() -> FileResponse:
        return FileResponse(STATIC / "stage.html")

    @app.get("/operator")
    def operator() -> FileResponse:
        return FileResponse(STATIC / "operator.html")

    @app.get("/api/config")
    def config() -> dict:
        return cfg.public()

    @app.get("/api/state")
    def state() -> dict:
        return cond.state()

    @app.post("/api/round")
    async def start_round(body: RoundBody) -> dict:
        try:
            return await cond.start_round(body.segment_id, body.input_id, body.prompt, body.override,
                                          body.force_replay, body.budget_override)
        except PermissionError as e:
            raise HTTPException(403, str(e))
        except (RuntimeError, ValueError, KeyError) as e:
            raise HTTPException(409, str(e))

    @app.post("/api/cancel")
    async def cancel() -> dict:
        await cond.cancel_round()
        return {"ok": True}

    @app.post("/api/cell/{mid}/replay")
    async def cell_replay(mid: str) -> dict:
        await cond.switch_to_replay(mid)
        return {"ok": True}

    @app.post("/api/cell/{mid}/stop")
    async def cell_stop(mid: str) -> dict:
        await cond.stop_cell(mid)
        return {"ok": True}

    @app.get("/api/plan")
    def plan(segment_id: str, input_id: Optional[str] = None) -> dict:
        try:
            return cond.plan(segment_id, input_id)
        except KeyError as e:
            raise HTTPException(404, str(e))

    @app.get("/api/layout_view")
    def layout_view_api(input_id: str) -> dict:
        """A layout input plus its wall names and one-sentence request (operator plan thumbnail)."""
        from .core import layout_view
        try:
            item = cfg.input("layout", input_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {**item, **layout_view(item)}

    @app.get("/api/prompt")
    def prompt_text(segment_id: str, input_id: str) -> dict:
        """What the model will be sent for this segment and input (layout and llm tasks)."""
        from .core import layout_prompt_text
        try:
            seg = cfg.segment(segment_id)
            item = cfg.input(seg["task"], input_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        if seg["task"] == "layout":
            mids = [m["id"] for m in cfg.machines]
            s = cfg.settings("layout", seg["protocol"], mids[0]) if mids else {}
            return {"task": "layout", "text": layout_prompt_text(item, s.get("max_objects"))}
        if seg["task"] == "llm":
            from ..worker.backends.llm import SYSTEM
            return {"task": "llm", "text": f"=== system ===\n{SYSTEM}\n\n=== user ===\n{item.get('prompt', '')}"}
        return {"task": seg["task"], "text": None}

    @app.post("/api/prepare")
    async def prepare(body: SegmentBody) -> dict:
        return await cond.prepare(body.segment_id)

    @app.post("/api/view")
    def view(body: ViewBody) -> dict:
        if body.view not in ("title", "round", "summary", "blank"):
            raise HTTPException(400, "unknown view")
        cond.view = body.view
        return {"view": cond.view}

    @app.post("/api/moderate")
    def moderate(body: PromptBody) -> dict:
        return check_prompt(body.prompt)

    return app
