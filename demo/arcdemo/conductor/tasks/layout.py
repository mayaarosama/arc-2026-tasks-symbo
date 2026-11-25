"""Task 8 — scene construction: the room, its wall names, the exact prompt, and the furnished-room summary."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles

from ...defaults import DEFAULTS
from ...layout_prompt import (PROMPT_VERSION, _angled, _door_wall, _not_floor, css_grammar, grammar, layoutgpt_messages,
                              messages, walls)
from ..config import ROOT
from ..urls import LAYOUT_ASSETS
from .base import TaskHandler

DEFAULT_MAX_OBJECTS = DEFAULTS.layout.max_objects    # a room without max_objects in its input or settings
ASSETS_URL = LAYOUT_ASSETS + "/"


def layout_view(item: dict) -> dict:
    """What the stage shows about a layout prompt: the wall names the model uses (with their
    end points, for the plan thumbnail), the door's wall, and the request in one plain sentence."""
    plan = item["floor_plan"]
    named = walls(plan)
    ends = {n: ([w["span"][0], w["fixed"]], [w["span"][1], w["fixed"]]) if w["axis"] == "x"
            else ([w["fixed"], w["span"][0]], [w["fixed"], w["span"][1]]) for n, w in named.items()}
    xs, ys = [p[0] for p in plan], [p[1] for p in plan]
    shape = ("an angled wall" if _angled(plan) else "an L-shape" if _not_floor(plan) else "")
    spec = {"floor_plan": plan, "door": item.get("door")}
    door = _door_wall(spec) if item.get("door") else None
    room = item["room_type"].replace("_", " ")
    ask = (f"Furnish this {room}: {max(xs) - min(xs):g} × {max(ys) - min(ys):g} m{' with ' + shape if shape else ''}"
           f"{', door on the ' + door + ' wall' if door else ''}, up to {item.get('max_objects', DEFAULT_MAX_OBJECTS)} "
           "pieces, one line each: a wall piece names its wall, a free piece gives its spot.")
    return {"walls": ends, "door_wall": door, "ask": ask}


def layout_prompt_text(item: dict, max_objects: Optional[int] = None, style: Optional[str] = None) -> str:
    """The exact chat the model receives for this room, as plain text for the operator page.
    style: None / "v3.1" (ours) or "layoutgpt" (LayoutGPT's prompt, layout_prompt.py)."""
    spec = {"room_type": item["room_type"], "floor_plan": item["floor_plan"], "door": item.get("door"),
            "max_objects": int(max_objects or item.get("max_objects") or DEFAULT_MAX_OBJECTS)}
    gpt = style == "layoutgpt"
    parts = ["# prompt layoutgpt (LayoutGPT, k-similar example rooms)" if gpt else f"# prompt {PROMPT_VERSION}"]
    for m in (layoutgpt_messages(spec) if gpt else messages(spec)):
        parts += ["", f"=== {m['role']} ===", m["content"].rstrip()]
    parts += ["", "=== grammar (constrains the reply's syntax only) ===", css_grammar(spec) if gpt else grammar(spec)]
    return "\n".join(parts)


def layout_summary(objects: Optional[list]) -> Optional[dict]:
    """Objects and flagged problems in the last room stream 0 worked on (None when there are none)."""
    if not objects:
        return None
    last = max(o.get("room", 0) for o in objects)
    every = [o for o in objects if o.get("room", 0) == last]
    room = [o for o in every if o.get("removed") is None]
    flagged = sum(1 for o in room if o.get("issues"))
    out = {"objects": len(room), "flagged": flagged}
    if any("history" in o for o in every):      # self-correction: the first pass and the fixes kept
        out.update(first_flagged=sum(1 for o in every if (o.get("history") or [o])[0].get("issues")),
                   fixes=sum(o.get("fixes", 0) for o in every))
    return out


class SceneLayout(TaskHandler):
    task = "layout"
    hard_budget = True

    def prepare_input(self, item: dict, prompt, override: bool) -> None:
        item.update(layout_view(item))

    def job_fields(self, cond, item: dict, settings: dict) -> dict[str, Any]:
        return {"room_type": item["room_type"], "floor_plan": item["floor_plan"], "door": item.get("door"),
                "max_objects": settings.get("max_objects", item.get("max_objects"))}

    def summary(self, cell: dict) -> dict[str, Any]:
        return {"layout": layout_summary(cell.get("objects"))}

    def prompt_text(self, cond, seg: dict, item: dict) -> Optional[str]:
        mids = [m["id"] for m in cond.cfg.machines]
        s = cond.settings("layout", seg["protocol"], mids[0]) if mids else {}
        return layout_prompt_text(item, s.get("max_objects"), s.get("layout_prompt"))

    def routes(self, app, cfg, get_cond) -> None:
        if cfg.layout_catalog:   # the furniture models, from the catalogue's assets_dir
            app.mount(LAYOUT_ASSETS, StaticFiles(directory=ROOT / cfg.layout_catalog["assets_dir"]),
                      name="layout-assets")

        @app.get("/api/layout_catalog")
        def layout_catalog() -> dict:
            """The furniture catalogue for the 3D view: each class's model file, props and rug,
            with the URL its files are served from."""
            if not cfg.layout_catalog:
                raise HTTPException(404, "no layout task in this config")
            c = cfg.layout_catalog
            return {"assets_url": ASSETS_URL, "sha": c["sha"], "furniture": c["furniture"], "rooms": c["rooms"]}

        @app.get("/api/layout_view")
        def layout_view_api(input_id: str) -> dict:
            """A layout input plus its wall names and one-sentence request (operator plan thumbnail)."""
            try:
                item = cfg.input("layout", input_id)
            except KeyError as e:
                raise HTTPException(404, str(e))
            return {**item, **layout_view(item)}
