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


class UploadBody(BaseModel):
    name: str = "picture"
    data_url: str           # data:image/...;base64,... (the browser's FileReader), up to ~14 MB


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
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/media", StaticFiles(directory=cfg.session_dir), name="media")
    app.mount("/replay-media", StaticFiles(directory=cfg.replay_dir), name="replay")
    app.mount("/media-input/data", StaticFiles(directory=ROOT / "data"), name="inputs")

    @app.middleware("http")
    async def revalidate_pages(request, call_next):
        """Pages and their scripts: the browser asks for the newest version each time (a 304 when
        unchanged), so a page opened before a git pull never runs an old operator.js."""
        resp = await call_next(request)
        if request.url.path.startswith("/static/") or request.url.path in ("/operator", "/stage"):
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
            s = cond.settings("layout", seg["protocol"], mids[0]) if mids else {}
            return {"task": "layout", "text": layout_prompt_text(item, s.get("max_objects"), s.get("layout_prompt"))}
        if seg["task"] == "llm":
            from ..worker.backends.llm import SYSTEM
            return {"task": "llm", "text": f"=== system ===\n{SYSTEM}\n\n=== user ===\n{item.get('prompt', '')}"}
        return {"task": seg["task"], "text": None}

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

    @app.get("/api/i23d/spec")
    def upload_spec() -> dict:
        """What an uploaded picture must be like (shown next to the upload box)."""
        from ..imageprep import INPUT_SPEC, cutout_model_path
        return {"spec": [{"what": a, "rule": b} for a, b in INPUT_SPEC],
                "cutout_installed": cutout_model_path() is not None}

    @app.post("/api/i23d/upload")
    async def upload(body: UploadBody) -> dict:
        """An operator's picture for image-to-3D: checked, cut out and normalised here, once, so every
        machine gets the same input. 422 with the reason when it cannot be used."""
        import base64
        import binascii

        from ..imageprep import MAX_BYTES, Rejected
        head, _, b64 = body.data_url.partition(",")
        if not head.startswith("data:") or ";base64" not in head:
            raise HTTPException(400, "expected a data URL (data:image/...;base64,...)")
        if len(b64) > MAX_BYTES * 4 // 3 + 16:
            raise HTTPException(422, f"The file is larger than {MAX_BYTES // 2**20} MB.")
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(400, "the picture is not valid base64")
        try:   # the cut-out takes a second or two: keep the conductor's loop (health, rounds) running
            return await asyncio.to_thread(cond.add_upload, body.name, data)
        except Rejected as e:
            raise HTTPException(422, str(e))

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
