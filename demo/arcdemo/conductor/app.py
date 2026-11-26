"""Conductor web app, on the operator laptop.

    python -m arcdemo.conductor --config config/demo.yaml --port 8000
    projector:  http://localhost:8000/stage
    operator:   http://localhost:8000/operator
"""
from __future__ import annotations

import asyncio
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..defaults import DEFAULTS
from ..schemas import TERMINAL
from .config import ROOT, DemoConfig
from .core import Conductor
from .tasks import HANDLERS, handler
from .urls import MEDIA_INPUT, MEDIA_REPLAY, MEDIA_SESSION
from .urls import STATIC as STATIC_URL

STATIC = Path(__file__).parent / "static"
# Windows can map .js to text/plain in the registry; browsers then refuse ES modules (layout3d.js).
mimetypes.add_type("text/javascript", ".js")


class RoundBody(BaseModel):
    segment_id: str
    input_id: str
    prompt: Optional[str] = None
    override: bool = False
    force_replay: list[str] = []
    budget_override: Optional[float] = None


class ViewBody(BaseModel):
    view: str


class SegmentBody(BaseModel):
    segment_id: str


class ModelPickBody(BaseModel):
    task: str
    machine: str
    model: Optional[str] = None     # None = back to the model planned in the config


def create_app(cfg: DemoConfig) -> FastAPI:
    cond: Optional[Conductor] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal cond
        cond = Conductor(cfg, picks_path=cfg.session_dir / "model_picks.json")
        task = asyncio.create_task(cond.health_loop())
        yield
        task.cancel()
        await cond.http.aclose()

    app = FastAPI(title="arcdemo conductor", lifespan=lifespan)
    cfg.session_dir.mkdir(parents=True, exist_ok=True)
    cfg.replay_dir.mkdir(parents=True, exist_ok=True)
    app.mount(STATIC_URL, StaticFiles(directory=STATIC), name="static")
    app.mount(MEDIA_SESSION, StaticFiles(directory=cfg.session_dir), name="media")
    app.mount(MEDIA_REPLAY, StaticFiles(directory=cfg.replay_dir), name="replay")
    app.mount(MEDIA_INPUT, StaticFiles(directory=ROOT / "data"), name="inputs")

    @app.middleware("http")
    async def revalidate_pages(request, call_next):
        """Pages and their scripts: the browser asks for the newest version each time (a 304 when
        unchanged), so a page opened before a git pull never runs an old operator.js."""
        resp = await call_next(request)
        if request.url.path.startswith(STATIC_URL + "/") or request.url.path in ("/operator", "/stage"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

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
        """The show (machines, segments, inputs, models) and the pages' own settings (config/defaults.yaml stage)."""
        st = DEFAULTS.stage
        ui = {"poll_ms": st.poll_ms, "operator_poll_ms": st.operator_poll_ms, "views": st.views,
              "terminal": sorted(TERMINAL), "media_input": MEDIA_INPUT,
              "spare_prompts": f"{MEDIA_INPUT}/{Path(st.spare_prompts).relative_to('data').as_posix()}"}
        return {**cfg.public(), "ui": ui}

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

    @app.get("/api/prompt")
    def prompt_text(segment_id: str, input_id: str) -> dict:
        """What the model will be sent for this segment and input (text: None for tasks without one)."""
        try:
            seg = cfg.segment(segment_id)
            item = cfg.input(seg["task"], input_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {"task": seg["task"], "text": handler(seg["task"]).prompt_text(cond, seg, item)}

    @app.post("/api/model")
    def pick_model(body: ModelPickBody) -> dict:
        """Operator's model choice for one machine and task; Prepare must be pressed again."""
        try:
            return {"model_picks": cond.set_model(body.task, body.machine, body.model)}
        except KeyError as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/prepare")
    async def prepare(body: SegmentBody) -> dict:
        return await cond.prepare(body.segment_id)

    @app.post("/api/view")
    def view(body: ViewBody) -> dict:
        if body.view not in DEFAULTS.stage.views:
            raise HTTPException(400, "unknown view")
        cond.view = body.view
        return {"view": cond.view}

    # ---- each task's own routes, in show order (tasks/<task>.py) ----------------------------
    for h in HANDLERS.values():
        h.routes(app, cfg, lambda: cond)
    return app
