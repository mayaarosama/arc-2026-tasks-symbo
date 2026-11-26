"""Task 8 — scene construction: the room, its wall names, the exact prompt, and the furnished-room summary.

The operator's own room: an outline typed on the operator page, checked here (check_room) and offered
like the stage rooms. Kept in session_dir/uploads/rooms.json, so a conductor restart still offers it.
The outline travels in the job, so the workers need nothing new.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ...defaults import DEFAULTS
from ...layout_prompt import (PROMPT_VERSION, ROOMS, _angled, _door_wall, _not_floor, css_grammar, grammar,
                              layoutgpt_messages, messages, walls)
from ..config import ROOT
from ..urls import LAYOUT_ASSETS
from .base import TaskHandler

log = logging.getLogger("arcdemo.conductor")
DEFAULT_MAX_OBJECTS = DEFAULTS.layout.max_objects    # a room without max_objects in its input or settings
ASSETS_URL = LAYOUT_ASSETS + "/"
CUSTOM = DEFAULTS.layout.custom                      # the operator's own room: config/defaults.yaml layout.custom
MAX_SIDE = DEFAULTS.layout.mask_metres               # LayoutGPT's floor mask covers this square
DIGEST_CHARS = 10                                    # a room's id: the first characters of its SHA-1


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


# ------------------------------------------------------------------ the operator's own room
DOOR_SNAP = 0.25        # metres: a door this close to a straight wall is moved onto it (with a warning)


class RoomBody(BaseModel):
    label: str = ""
    room_type: str
    floor_plan: list[list[float]]           # outline, metres, [[x, y], ...], in order round the room
    door: Optional[list[float]] = None      # a point on a straight wall
    max_objects: Optional[int] = None
    save: bool = True                       # False: check and describe only (the operator's preview)


def _point(p: Any, what: str) -> list[float]:
    if not (isinstance(p, (list, tuple)) and len(p) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p)):
        raise ValueError(f"{what} must be two numbers, x and y in metres.")
    return [round(float(p[0]), 2) + 0.0, round(float(p[1]), 2) + 0.0]     # + 0.0: no -0.0


def _cross(o: list, a: list, b: list) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_meet(a: list, b: list, c: list, d: list) -> bool:
    """Whether segment ab touches or crosses segment cd."""
    d1, d2, d3, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
    if ((d1 > 0) != (d2 > 0) and d1 * d2 != 0) and ((d3 > 0) != (d4 > 0) and d3 * d4 != 0):
        return True

    def on(p, q, r):      # r on segment pq, given the three are in line
        return min(p[0], q[0]) - 1e-9 <= r[0] <= max(p[0], q[0]) + 1e-9 and min(p[1], q[1]) - 1e-9 <= r[1] <= max(p[1], q[1]) + 1e-9
    return any(abs(v) < 1e-9 and on(p, q, r) for v, p, q, r in
               ((d1, c, d, a), (d2, c, d, b), (d3, a, b, c), (d4, a, b, d)))


def _outline(raw: list) -> tuple[list[list[float]], list[str]]:
    """The outline cleaned up: rounded to the centimetre, repeated corners and corners in the middle of a
    straight wall dropped. Raises ValueError when it is not a usable room."""
    if not isinstance(raw, list):
        raise ValueError("The outline must be a list of corners.")
    pts = [_point(p, f"Corner {i + 1}") for i, p in enumerate(raw)]
    warnings = []
    dedup = [p for i, p in enumerate(pts) if p != pts[i - 1]] if len(pts) > 1 else pts
    if len(dedup) < len(pts):
        warnings.append("A repeated corner was dropped.")
    pts, changed = dedup, True
    while changed and len(pts) > 3:     # a corner in the middle of a straight wall would split it in two
        changed = False
        for i in range(len(pts)):
            a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
            if abs(_cross(a, b, c)) < 1e-9 and (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) > 0:
                pts = pts[:i] + pts[i + 1:]
                warnings.append(f"Corner ({b[0]:g}, {b[1]:g}) is in the middle of a straight wall: dropped.")
                changed = True
                break
    if len(pts) < 3:
        raise ValueError("A room needs at least three corners.")
    if len(pts) > CUSTOM.max_corners:
        raise ValueError(f"A room can have at most {CUSTOM.max_corners} corners (this one has {len(pts)}).")
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        if math.dist(a, b) < CUSTOM.min_wall:
            raise ValueError(f"The wall from ({a[0]:g}, {a[1]:g}) to ({b[0]:g}, {b[1]:g}) is shorter than "
                             f"{CUSTOM.min_wall:g} m.")
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):        # neighbours share a corner
                continue
            if _segments_meet(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                raise ValueError("The outline crosses itself: list the corners in order round the room.")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    w, d = max(xs) - min(xs), max(ys) - min(ys)
    if min(w, d) < CUSTOM.min_side:
        raise ValueError(f"The room must be at least {CUSTOM.min_side:g} m each way (this one is {w:g} × {d:g} m).")
    if max(w, d) > MAX_SIDE:
        raise ValueError(f"The room must be at most {MAX_SIDE:g} m each way (this one is {w:g} × {d:g} m).")
    area = abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n))) / 2
    if area < CUSTOM.min_area:
        raise ValueError(f"The floor must be at least {CUSTOM.min_area:g} m² (this one is {area:.1f} m²).")
    if not walls(pts):
        raise ValueError("The room needs at least one straight wall running north-south or east-west: "
                         "wall pieces (beds, shelves) stand only against those.")
    return pts, warnings


def _place_door(raw: Any, pts: list) -> tuple[list[float], list[str]]:
    """The door on its straight wall, at least door_margin from the wall's ends. A door within
    DOOR_SNAP of a straight wall is moved onto it."""
    door = _point(raw, "The door")
    best = None
    for name, w in walls(pts).items():
        along, across = (door[0], door[1]) if w["axis"] == "x" else (door[1], door[0])
        a, b = w["span"]
        on = min(max(along, a), b)
        gap = math.hypot(on - along, w["fixed"] - across)
        if best is None or gap < best[0]:
            best = (gap, name, w, on)
    gap, name, w, on = best
    if gap > DOOR_SNAP:
        raise ValueError(f"The door at ({door[0]:g}, {door[1]:g}) is not on a straight wall. Put it on one of: "
                         + ", ".join(walls(pts)) + ".")
    a, b = w["span"]
    if b - a < 2 * CUSTOM.door_margin:
        raise ValueError(f"The {name} wall is too short for a door ({b - a:g} m; it needs {2 * CUSTOM.door_margin:g} m).")
    on = round(min(max(on, a + CUSTOM.door_margin), b - CUSTOM.door_margin), 2)
    moved = [on, w["fixed"]] if w["axis"] == "x" else [w["fixed"], on]
    moved = [v + 0.0 for v in moved]
    warnings = [] if moved == door else [f"The door was moved onto the {name} wall, to ({moved[0]:g}, {moved[1]:g})."]
    return moved, warnings


def check_room(body: dict) -> tuple[dict, list[str]]:
    """The operator's room as a layout input. Raises ValueError with the reason when it cannot be used.
    -> (the input item, warnings about what was changed)"""
    room = body.get("room_type")
    if room not in ROOMS:
        raise ValueError(f"Choose a room type: {', '.join(r.replace('_', ' ') for r in ROOMS)}.")
    pts, warnings = _outline(body.get("floor_plan"))
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    raw_door = body.get("door")
    if x0 or y0:        # the model's grammar writes no negative numbers: the room starts at (0, 0)
        pts = [[round(x - x0, 2) + 0.0, round(y - y0, 2) + 0.0] for x, y in pts]
        raw_door = [_point(raw_door, "The door")[0] - x0, _point(raw_door, "The door")[1] - y0] if raw_door else None
        warnings.append(f"The room was moved by ({-x0:g}, {-y0:g}) so that x and y start at 0.")
    door = None
    if raw_door is not None:
        door, moved = _place_door(raw_door, pts)
        warnings += moved
    n = body.get("max_objects") or DEFAULT_MAX_OBJECTS
    if not isinstance(n, int) or not 1 <= n <= CUSTOM.max_objects:
        raise ValueError(f"The number of pieces must be from 1 to {CUSTOM.max_objects}.")
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    key = json.dumps([room, pts, door, n])
    digest = hashlib.sha1(key.encode()).hexdigest()[:DIGEST_CHARS]
    size = f"{max(xs) - min(xs):g} × {max(ys) - min(ys):g} m"
    label = " ".join(re.sub(r"[^\w .,()·×-]+", " ", body.get("label") or "").split())[:CUSTOM.label_chars]
    item = {"id": f"my-{digest}", "label": label or f"Your {room.replace('_', ' ')} · {size}", "tier": "upload",
            "exposes": "An operator's own floor plan: live only (no recording)",
            "room_type": room, "floor_plan": pts, "door": door, "max_objects": n, "seed": int(digest[:6], 16)}
    return item, warnings


def rooms_file(cond) -> Path:
    return cond.cfg.session_dir / "uploads" / "rooms.json"


def load_rooms(cond) -> list[dict]:
    path = rooms_file(cond)
    try:
        items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError):
        log.warning("ignoring unreadable %s", path)
        return []
    return [x for x in items if x.get("room_type") in ROOMS]     # a room type the catalogue has since dropped


def add_room(cond, body: dict) -> dict:
    """Check the operator's room and offer it as a layout input. Raises ValueError with the reason.
    -> the input item + warnings"""
    item, warnings = check_room(body)
    cond.cfg.add_input("layout", item)
    path = rooms_file(cond)
    path.parent.mkdir(parents=True, exist_ok=True)
    rooms = [x for x in load_rooms(cond) if x["id"] != item["id"]] + [item]
    path.write_text(json.dumps(rooms, indent=1), encoding="utf-8")
    log.info("own room %s (%s, %d corners)", item["id"], item["label"], len(item["floor_plan"]))
    return {**item, "warnings": warnings}


class SceneLayout(TaskHandler):
    task = "layout"
    hard_budget = True

    def on_start(self, cond) -> None:
        for item in load_rooms(cond):
            cond.cfg.add_input("layout", item)

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

        @app.get("/api/layout/rules")
        def room_rules() -> dict:
            """What the operator's own room may be (shown next to the form)."""
            return {"room_types": list(ROOMS), "max_corners": CUSTOM.max_corners, "min_side": CUSTOM.min_side,
                    "max_side": MAX_SIDE, "min_wall": CUSTOM.min_wall, "min_area": CUSTOM.min_area,
                    "door_margin": CUSTOM.door_margin, "max_objects": CUSTOM.max_objects,
                    "default_objects": DEFAULT_MAX_OBJECTS}

        @app.post("/api/layout/room")
        def room(body: RoomBody) -> dict:
            """The operator's own room: checked, described for the plan thumbnail and, with save,
            offered as an input. 422 with the reason when it cannot be used."""
            try:
                if not body.save:
                    item, warnings = check_room(body.model_dump())
                    return {**item, **layout_view(item), "warnings": warnings}
                item = add_room(get_cond(), body.model_dump())
            except ValueError as e:
                raise HTTPException(422, str(e))
            return {**item, **layout_view(item)}
