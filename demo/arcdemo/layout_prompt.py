"""Task 8 prompt, room geometry and checks: everything about a layout that needs no model.

Shared by the worker (arcdemo/worker/backends/layout.py, which streams the model's reply through
parse_line and check) and the conductor (which shows the operator the exact prompt and gives the
stage the wall names). Pure Python, no dependencies. See the backend's docstring for the format.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]                  # demo/
DEFAULT_CATALOG = "data/layout/furniture.json"              # demo.yaml / worker config key: layout_catalog

# Filled from the furniture catalogue (data/layout/furniture.json) by load_catalog(), never written here.
# Updated in place, so modules that imported these names see the loaded catalogue.
FURNITURE: dict[str, tuple[float, float, float]] = {}   # fixed size (w along the front, d, h), metres
WALL_PIECES: set[str] = set()                            # placed against a wall (wall=… at=…); others free
ROOMS: dict[str, list[str]] = {}                         # vocabulary per room type, in placement order
CATALOG: dict = {}                                       # the whole document, as loaded (the stage draws from it)


def _resolve(path: Optional[str | Path]) -> Path:
    p = Path(path or DEFAULT_CATALOG)
    return p if p.is_absolute() else ROOT / p


def load_catalog(path: Optional[str | Path] = None, check_models: bool = False) -> dict:
    """Read the furniture catalogue and make it the one in use. Raises ValueError naming every problem,
    so a bad catalogue stops the conductor / worker at startup instead of one room on stage.
    check_models: also require every model file in assets_dir (the conductor serves them; a worker
    needs only sizes and room lists)."""
    p = _resolve(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"furniture catalogue {p}: {e}") from e
    furn, rooms, errors = doc.get("furniture") or {}, doc.get("rooms") or {}, []
    for cls, f in furn.items():
        size = f.get("size")
        if not (isinstance(size, list) and len(size) == 3 and all(isinstance(v, (int, float)) and v > 0 for v in size)):
            errors.append(f"{cls}: size must be [w, d, h], three positive numbers in metres")
        if f.get("place") not in ("wall", "free"):
            errors.append(f"{cls}: place must be 'wall' or 'free'")
        if not re.fullmatch(r"[a-z_]+", cls):
            errors.append(f"{cls}: class names are lower case letters and _ (the model writes them)")
    for room, vocab in rooms.items():
        errors += [f"room {room}: '{c}' is not in furniture" for c in vocab if c not in furn]
    for ex, reply in globals().get("EXAMPLES", []):     # the prompt's worked examples must stay valid
        if ex["room_type"] not in rooms:
            errors.append(f"rooms: needs '{ex['room_type']}' (used by a worked example in the prompt)")
        else:
            errors += [f"room {ex['room_type']}: needs '{c}' (used by a worked example in the prompt)"
                       for c in dict.fromkeys(ln.split()[0] for ln in reply.splitlines()) if c not in rooms[ex["room_type"]]]
    if check_models:
        assets = _resolve(doc.get("assets_dir", ""))
        for cls, f in furn.items():
            for m in [f.get("model")] + [pr.get("model") for pr in f.get("props") or []]:
                if not m:
                    errors.append(f"{cls}: no model file")
                elif not (assets / m).is_file():
                    errors.append(f"{cls}: model file {assets / m} not found")
    if not furn or not rooms:
        errors.append("needs both 'furniture' and 'rooms'")
    if errors:
        raise ValueError(f"furniture catalogue {p} is invalid:\n  " + "\n  ".join(dict.fromkeys(errors)))
    FURNITURE.clear()
    FURNITURE.update({c: tuple(float(v) for v in f["size"]) for c, f in furn.items()})
    WALL_PIECES.clear()
    WALL_PIECES.update(c for c, f in furn.items() if f["place"] == "wall")
    ROOMS.clear()
    ROOMS.update({r: list(v) for r, v in rooms.items()})
    CATALOG.clear()
    CATALOG.update(doc, path=str(p), sha=hashlib.sha1(p.read_bytes()).hexdigest()[:12])
    return CATALOG


FACING = {"north": 0, "east": 90, "south": 180, "west": 270}
DOOR_CLEARANCE = 0.8        # metres kept free in front of the door
OVERLAP_SHARE = 0.10        # overlap above this share of the smaller footprint is flagged
WALL_TOLERANCE = 0.03       # a corner this far outside the outline still counts as inside

PROMPT_VERSION = "v3.1"  # v1 raw coordinates; v2 + wall rules, sizes, free zones; v3 wall-anchored lines;
                         # v3.1 examples are room types not on stage (v3 bedroom example was copied verbatim)

SYSTEM = (
    "You are an interior designer furnishing an empty room, one object per line, the most important "
    "object first.\n"
    "Coordinates are in metres: the origin is the south-west corner, x grows east, y grows north. "
    "Sizes are fixed and listed with each object as width x depth x height; the width is the object's front.\n"
    "Wall pieces (marked [wall]) stand with their back against a wall, facing into the room. Write:\n"
    "<object> wall=<wall name> at=<position of the object's centre along that wall>\n"
    "On a north or south wall, at is an x value; on an east or west wall, at is a y value. Keep at least "
    "half the object's width between at and each end of the wall.\n"
    "Free pieces (marked [free]) stand anywhere on the floor. Write:\n"
    "<object> x=<centre x> y=<centre y> facing=<north|east|south|west>\n"
    "Rules: objects never overlap. Two pieces on the same wall need their at values apart by at least half "
    "of each width added together; a free piece keeps clear of the wall pieces (a wall piece reaches its "
    "depth into the room). Nothing goes in the door's keep-empty zone or in areas marked not floor. Put "
    "each piece where it makes sense: nightstands beside the bed on the same wall, a chair facing the desk, "
    "the coffee table in front of the sofa, the TV stand facing the sofa. Spread the furniture around the "
    "room. Write nothing except the lines."
)

# Hand-written examples (our own, not from any dataset). Room types that are not on stage (a study,
# a dining room) and plans unlike the stage rooms, so the model cannot copy a stage answer. Checked
# by tests/test_layout.py: every line resolves, inside, no overlaps, door free.
EXAMPLES = [
    ({"room_type": "study", "floor_plan": [[0, 0], [4.4, 0], [4.4, 3.2], [0, 3.2]], "door": [4.4, 2.6],
      "max_objects": 8},
     "desk wall=south at=1.20\n"
     "chair x=1.20 y=0.85 facing=south\n"
     "bookshelf wall=west at=2.00\n"
     "bookshelf wall=west at=2.80\n"
     "armchair x=2.40 y=2.60 facing=west\n"
     "floor_lamp x=2.40 y=1.90 facing=south\n"
     "cabinet wall=east at=1.00\n"
     "plant x=0.30 y=0.90 facing=south\n"),
    ({"room_type": "dining_room", "floor_plan": [[0, 0], [3, 0], [3, 1.8], [5.2, 1.8], [5.2, 4.6], [0, 4.6]],
      "door": [0, 1.0], "max_objects": 9},
     "dining_table x=2.00 y=3.00 facing=north\n"
     "dining_chair x=2.00 y=3.75 facing=south\n"
     "dining_chair x=2.00 y=2.25 facing=north\n"
     "cabinet wall=north at=4.00\n"
     "dresser wall=east2 at=3.00\n"
     "bookshelf wall=south1 at=1.80\n"
     "side_table x=4.40 y=2.10 facing=north\n"
     "floor_lamp x=4.80 y=4.20 facing=south\n"
     "plant x=0.30 y=4.30 facing=south\n"),
]

load_catalog()     # the default document; the conductor and worker reload it from their config's layout_catalog

WALL_LINE = re.compile(r"^\s*([a-z_]+)\s+wall=([a-z]+\d*)\s+at=(-?\d+(?:\.\d+)?)\s*$")
FREE_LINE = re.compile(r"^\s*([a-z_]+)\s+x=(-?\d+(?:\.\d+)?)\s+y=(-?\d+(?:\.\d+)?)\s+facing=(north|east|south|west)\s*$")


# ------------------------------------------------------------------ prompt ------------------
def walls(pts: list) -> dict[str, dict]:
    """Straight walls by name. A wall is named for the side of the room it is on (floor to its north
    -> "south"); repeated sides are numbered in outline order (south1, south2). Angled walls get no
    name: nothing can be placed against them."""
    raw = []
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
        if abs(y1 - y2) < 1e-9:
            north = _inside((x1 + x2) / 2, y1 + 0.05, pts)
            raw.append({"side": "south" if north else "north", "axis": "x", "fixed": y1,
                        "span": tuple(sorted((x1, x2))), "rot": 0 if north else 180, "sign": 1 if north else -1})
        elif abs(x1 - x2) < 1e-9:
            east = _inside(x1 + 0.05, (y1 + y2) / 2, pts)
            raw.append({"side": "west" if east else "east", "axis": "y", "fixed": x1,
                        "span": tuple(sorted((y1, y2))), "rot": 90 if east else 270, "sign": 1 if east else -1})
    count = {s: sum(1 for w in raw if w["side"] == s) for s in FACING}
    seen: dict[str, int] = {}
    out = {}
    for w in raw:
        seen[w["side"]] = seen.get(w["side"], 0) + 1
        out[w["side"] if count[w["side"]] == 1 else f"{w['side']}{seen[w['side']]}"] = w
    return out


def _angled(pts: list) -> list[str]:
    out = []
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
        if abs(y1 - y2) > 1e-9 and abs(x1 - x2) > 1e-9:
            out.append(f"angled wall from ({x1:g},{y1:g}) to ({x2:g},{y2:g}): nothing stands against it; keep clear")
    return out


def describe_room(spec: dict) -> str:
    """The room as the model sees it: size, named walls with the range of `at`, areas that are not
    floor, the door's keep-empty zone, and the objects with their fixed sizes and kind."""
    pts = spec["floor_plan"]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    lines = [f"Room: {spec['room_type'].replace('_', ' ')}",
             f"Size: {max(xs) - min(xs):g} m east-west (x {min(xs):g} to {max(xs):g}) by "
             f"{max(ys) - min(ys):g} m north-south (y {min(ys):g} to {max(ys):g})",
             "Floor outline corners: " + " ".join(f"({x:g},{y:g})" for x, y in pts),
             "Walls (name: where it is; the range of at):"]
    for name, w in walls(pts).items():
        a, b = w["span"]
        lines.append(f"- {name}: {'y' if w['axis'] == 'x' else 'x'}={w['fixed']:.2f}; at = {w['axis']} from "
                     f"{a:.2f} to {b:.2f}")
    lines += [f"- {t}" for t in _angled(pts)]
    holes = _not_floor(pts)
    if holes:
        lines.append("Not floor: " + "; ".join(holes))
    if spec.get("door"):
        dx, dy = spec["door"]
        on = _door_wall(spec)
        lines.append(f"Door: {'on the ' + on + ' wall ' if on else ''}at ({dx:g},{dy:g}). Keep this zone empty: "
                     f"x {max(min(xs), dx - DOOR_CLEARANCE):.2f} to {min(max(xs), dx + DOOR_CLEARANCE):.2f}, "
                     f"y {max(min(ys), dy - DOOR_CLEARANCE):.2f} to {min(max(ys), dy + DOOR_CLEARANCE):.2f}")
    lines.append("Allowed objects (width x depth x height): " + ", ".join(
        f"{c} {FURNITURE[c][0]:.2f}x{FURNITURE[c][1]:.2f}x{FURNITURE[c][2]:.2f} "
        f"[{'wall' if c in WALL_PIECES else 'free'}]" for c in dict.fromkeys(ROOMS[spec["room_type"]])))
    lines.append(f"Place up to {spec['max_objects']} objects.")
    return "\n".join(lines)


def _door_wall(spec: dict) -> Optional[str]:
    dx, dy = spec["door"]
    for name, w in walls(spec["floor_plan"]).items():
        along, across = (dx, dy) if w["axis"] == "x" else (dy, dx)
        if abs(across - w["fixed"]) < 1e-6 and w["span"][0] - 1e-6 <= along <= w["span"][1] + 1e-6:
            return name
    return None


def _not_floor(pts: list) -> list[str]:
    """Cells of the grid through the outline's corners that are not floor, as x/y ranges."""
    gx, gy = sorted({p[0] for p in pts}), sorted({p[1] for p in pts})
    out = []
    for xa, xb in zip(gx, gx[1:]):
        for ya, yb in zip(gy, gy[1:]):
            probes = [((xa + xb) / 2, (ya + yb) / 2)] + [
                (x + (0.02 if x == xa else -0.02), y + (0.02 if y == ya else -0.02)) for x in (xa, xb) for y in (ya, yb)]
            inside = [_inside(px, py, pts) for px, py in probes]
            if not any(inside):
                out.append(f"x {xa:.2f} to {xb:.2f}, y {ya:.2f} to {yb:.2f}")
            elif not all(inside):
                out.append(f"part of x {xa:.2f} to {xb:.2f}, y {ya:.2f} to {yb:.2f} (cut off by the angled wall)")
    return out


def messages(spec: dict) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex_spec, ex_reply in EXAMPLES:
        msgs += [{"role": "user", "content": describe_room(ex_spec)}, {"role": "assistant", "content": ex_reply}]
    return msgs + [{"role": "user", "content": describe_room(spec)}]


def grammar(spec: dict, lines: Optional[int] = None, at_least: int = 1) -> str:
    """GBNF: at_least..lines lines (default 1..max_objects; a continuation after kept lines may
    write none); wall pieces only in the wall form with this room's wall names, free pieces only
    in the x/y form. Syntax and vocabulary only, never positions."""
    vocab = list(dict.fromkeys(ROOMS[spec["room_type"]]))
    wcls = [c for c in vocab if c in WALL_PIECES]
    fcls = [c for c in vocab if c not in WALL_PIECES]
    names = list(walls(spec["floor_plan"]))
    alt = [x for x, ok in (("wline", wcls and names), ("fline", fcls)) if ok]
    top = spec["max_objects"] if lines is None else lines
    rules = [f"root ::= line{{{at_least},{top}}}", "line ::= " + " | ".join(alt)]
    if "wline" in alt:
        rules += ['wline ::= wcls " wall=" wall " at=" num "\\n"',
                  "wcls ::= " + " | ".join(f'"{c}"' for c in wcls),
                  "wall ::= " + " | ".join(f'"{n}"' for n in names)]
    if "fline" in alt:
        rules += ['fline ::= fcls " x=" num " y=" num " facing=" dir "\\n"',
                  "fcls ::= " + " | ".join(f'"{c}"' for c in fcls),
                  'dir ::= "north" | "east" | "south" | "west"']
    rules.append('num ::= [0-9] [0-9]? "." [0-9] [0-9]')
    return "\n".join(rules)


def parse_line(line: str, spec: dict) -> Optional[dict]:
    """One model line -> a placed object (centre, size, facing), or None if it is not a valid line
    for this room. Wall pieces are resolved against the named wall."""
    allowed = set(ROOMS[spec["room_type"]])
    m = WALL_LINE.match(line)
    if m:
        cls, name, at = m.group(1), m.group(2), float(m.group(3))
        wall = walls(spec["floor_plan"]).get(name)
        if cls not in allowed or wall is None:
            return None
        w, d, h = FURNITURE[cls]
        off = round(wall["fixed"] + wall["sign"] * d / 2, 3)
        x, y = (at, off) if wall["axis"] == "x" else (off, at)
        return {"cls": cls, "x": x, "y": y, "w": w, "d": d, "h": h, "rot": wall["rot"], "wall": name, "at": at}
    m = FREE_LINE.match(line)
    if m:
        cls = m.group(1)
        if cls not in allowed:
            return None
        w, d, h = FURNITURE[cls]
        return {"cls": cls, "x": float(m.group(2)), "y": float(m.group(3)), "w": w, "d": d, "h": h,
                "rot": FACING[m.group(4)]}
    return None


# ------------------------------------------------------------------ checks ------------------
def footprint(o: dict) -> tuple[float, float, float, float]:
    """Axis-aligned footprint (x0, y0, x1, y1). Off-axis rotations use the bounding box."""
    sideways = round(o["rot"] / 90) % 2 == 1
    ex, ey = (o["d"], o["w"]) if sideways else (o["w"], o["d"])
    return o["x"] - ex / 2, o["y"] - ey / 2, o["x"] + ex / 2, o["y"] + ey / 2


def _inside(px: float, py: float, poly: list) -> bool:
    hit = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > py) != (y2 > py) and px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def _near_edge(px: float, py: float, poly: list, tol: float) -> bool:
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / ((dx * dx + dy * dy) or 1)))
        if (px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2 <= tol * tol:
            return True
    return False


def _edge_dist(px: float, py: float, poly: list) -> float:
    best = float("inf")
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        dx, dy = x2 - x1, y2 - y1
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / ((dx * dx + dy * dy) or 1)))
        best = min(best, ((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2) ** 0.5)
    return best


def measure(o: dict, earlier: list[dict], spec: dict) -> list[tuple[str, float, str]]:
    """Every problem with one object, given those placed before it: (kind, amount, plain words).
    amount: metres outside the outline / metres inside the door's clearance / m² of overlap."""
    poly = spec["floor_plan"]
    x0, y0, x1, y1 = footprint(o)
    out: list[tuple[str, float, str]] = []
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    beyond = [_edge_dist(px, py, poly) for px, py in corners
              if not (_inside(px, py, poly) or _near_edge(px, py, poly, WALL_TOLERANCE))]
    if beyond:
        out.append(("outside", max(beyond), f"it sticks out of the room by {max(beyond):.2f} m"))
    door = spec.get("door")
    if door:
        cx, cy = min(max(door[0], x0), x1), min(max(door[1], y0), y1)   # nearest footprint point
        dist = ((cx - door[0]) ** 2 + (cy - door[1]) ** 2) ** 0.5
        if dist < DOOR_CLEARANCE:
            out.append(("door", DOOR_CLEARANCE - dist,
                        f"it comes within {dist:.2f} m of the door at ({door[0]:g},{door[1]:g}); "
                        f"keep {DOOR_CLEARANCE:.2f} m clear"))
    area = (x1 - x0) * (y1 - y0)
    for e in earlier:
        a0, b0, a1, b1 = footprint(e)
        ov = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
        if ov > OVERLAP_SHARE * max(1e-6, min(area, (a1 - a0) * (b1 - b0))):
            who = f"{e['i'] + 1} {e['cls']}" if "i" in e else e["cls"]
            out.append(("overlap", ov, f"it overlaps {who} by {ov:.2f} m²"))
    return out


def check(o: dict, earlier: list[dict], spec: dict) -> list[str]:
    """Problems with one object, given those placed before it in the same room."""
    return list(dict.fromkeys(kind for kind, _, _ in measure(o, earlier, spec)))


# ------------------------------------------------------------------ self-correction ---------
# The model fixes its own mistakes; the code only measures. Room cost C (0 = nothing flagged):
#   FLAG_COST per problem (so fewer problems always wins: three small ones never beat one big one)
#   + metres outside the outline + metres inside the door's clearance + m² of flagged overlap
#   + REMOVE_COST per object taken out (less than a flag, so dropping a piece that fits nowhere is
#     allowed, but more than nothing, so emptying the room is never the cheapest "fix")
#   + MUST_HAVE_COST if the room's main piece is missing.
FLAG_COST = 1.0
REMOVE_COST = 0.5
MUST_HAVE_COST = 2.0
MUST_HAVE = {"bedroom": {"double_bed", "single_bed"}, "living_room": {"sofa"}, "library": {"bookshelf"},
             "study": {"desk"}, "dining_room": {"dining_table"}}


def problem_cost(o: dict, earlier: list[dict], spec: dict) -> float:
    return sum(FLAG_COST + amount for _, amount, _ in measure(o, earlier, spec))


def room_cost(active: list[dict], spec: dict, first_count: int) -> float:
    """C for the objects still in the room, in placement order (a pair's overlap counts once,
    on the later object), against the number the model first wrote."""
    c = sum(problem_cost(o, active[:k], spec) for k, o in enumerate(active))
    c += REMOVE_COST * max(0, first_count - len(active))
    must = MUST_HAVE.get(spec["room_type"], set())
    if must and not any(o["cls"] in must for o in active):
        c += MUST_HAVE_COST
    return round(c, 4)


def line_of(o: dict) -> str:
    """An object back as the line the model writes (the inverse of parse_line)."""
    if o.get("wall"):
        return f"{o['cls']} wall={o['wall']} at={o['at']:.2f}"
    dirs = {v: k for k, v in FACING.items()}
    return f"{o['cls']} x={o['x']:.2f} y={o['y']:.2f} facing={dirs.get(o['rot'], 'north')}"


def continue_messages(spec: dict, kept: list[str]) -> list[dict]:
    """The room prompt with the reply so far prefilled: the model goes on after the kept lines."""
    msgs = messages(spec)
    return msgs + [{"role": "assistant", "content": "".join(kept)}] if kept else msgs


def repair_messages(spec: dict, kept: list[str], bad_line: str, n: int, problems: list[str]) -> list[dict]:
    cls = bad_line.split()[0]
    ask = (f"Line {n}, \"{bad_line.strip()}\", has a problem: {'; '.join(problems)}. "
           f"Write line {n} again: the same {cls.replace('_', ' ')}, placed where it fits. "
           "Write only that one line.")
    return messages(spec) + [{"role": "assistant", "content": "".join(kept) + bad_line.rstrip("\n") + "\n"},
                             {"role": "user", "content": ask}]


def review_messages(spec: dict, kept: list[str], active: list[dict], notes: list[str]) -> list[dict]:
    """After the room is written: the numbered room with every problem, and the edit commands."""
    rows = []
    for k, o in enumerate(active):
        probs = [txt for _, _, txt in measure(o, active[:k], spec)]
        rows.append(f"{o['i'] + 1} {line_of(o)}" + (f"  <- problem: {'; '.join(probs)}" if probs else "  ok"))
    missing = [m for m in MUST_HAVE.get(spec["room_type"], set())]
    ask = ["Check the room you wrote. Here it is, numbered, with the problems found:", *rows, ""]
    if missing and not any(o["cls"] in missing for o in active):
        ask.append(f"The room has no {' or '.join(sorted(missing))}.")
    ask += notes
    ask += ["Fix the problems with edits, one per line:",
            "move <n> wall=<wall name> at=<at>      (a wall piece, to a wall)",
            "move <n> x=<x> y=<y> facing=<north|east|south|west>      (a free piece)",
            "remove <n>      (only if it fits nowhere)",
            "Then write done. Change only what fixes a problem."]
    return messages(spec) + [{"role": "assistant", "content": "".join(kept)},
                             {"role": "user", "content": "\n".join(ask)}]


def _rules_num_dir() -> list[str]:
    return ['num ::= [0-9] [0-9]? "." [0-9] [0-9]', 'dir ::= "north" | "east" | "south" | "west"']


def repair_grammar(spec: dict, cls: str) -> str:
    """Exactly one line, for the same object class (the fix may not swap the object for another)."""
    names = list(walls(spec["floor_plan"]))
    if cls in WALL_PIECES and names:
        return "\n".join([f'root ::= "{cls} wall=" wall " at=" num "\\n"',
                          "wall ::= " + " | ".join(f'"{n}"' for n in names)] + _rules_num_dir())
    return "\n".join([f'root ::= "{cls} x=" num " y=" num " facing=" dir "\\n"'] + _rules_num_dir())


def edit_grammar(spec: dict, active: list[dict], max_edits: int = 6) -> str:
    """move / remove for the numbers in the room (wall pieces move along walls, free pieces on the
    floor), then done. Syntax only: which object and where are the model's choice."""
    names = list(walls(spec["floor_plan"]))
    widx = [str(o["i"] + 1) for o in active if o["cls"] in WALL_PIECES] if names else []
    fidx = [str(o["i"] + 1) for o in active if o["cls"] not in WALL_PIECES]
    alt, rules = ["remove"], []
    if widx:
        alt.append("wmove")
        rules += ['wmove ::= "move " widx " wall=" wall " at=" num "\\n"',
                  "widx ::= " + " | ".join(f'"{n}"' for n in widx),
                  "wall ::= " + " | ".join(f'"{n}"' for n in names)]
    if fidx:
        alt.append("fmove")
        rules += ['fmove ::= "move " fidx " x=" num " y=" num " facing=" dir "\\n"',
                  "fidx ::= " + " | ".join(f'"{n}"' for n in fidx)]
    rules += ['remove ::= "remove " idx "\\n"',
              "idx ::= " + " | ".join(f'"{o["i"] + 1}"' for o in active)]
    return "\n".join([f'root ::= edit{{0,{max_edits}}} "done\\n"', "edit ::= " + " | ".join(alt)]
                     + rules + _rules_num_dir())


MOVE_WALL = re.compile(r"^\s*move\s+(\d+)\s+(wall=\S+\s+at=\S+)\s*$")
MOVE_FREE = re.compile(r"^\s*move\s+(\d+)\s+(x=\S+\s+y=\S+\s+facing=\S+)\s*$")
REMOVE = re.compile(r"^\s*remove\s+(\d+)\s*$")


def parse_edit(line: str, active: list[dict], spec: dict) -> Optional[tuple]:
    """("move", object, new placed object) | ("remove", object) | ("done",) | None"""
    if line.strip() == "done":
        return ("done",)
    by_n = {o["i"] + 1: o for o in active}
    m, wall = MOVE_WALL.match(line), True
    if not m:
        m, wall = MOVE_FREE.match(line), False
    if m:
        o = by_n.get(int(m.group(1)))
        if o is None or (o["cls"] in WALL_PIECES) != wall:   # wall pieces move to walls, free pieces on the floor
            return None
        new = parse_line(f"{o['cls']} {m.group(2)}", spec)
        return ("move", o, new) if new else None
    m = REMOVE.match(line)
    if m and int(m.group(1)) in by_n:
        return ("remove", by_n[int(m.group(1))])
    return None


def summarise(objects: list[dict]) -> dict:
    """Counts for the finished room (removed objects excluded). With self-correction, also what the
    model's first pass looked like and how many of its fixes were kept."""
    active = [o for o in objects if o.get("removed") is None]
    flags = [i for o in active for i in o.get("issues", [])]
    out = {"objects": len(active), "outside": flags.count("outside"), "overlap": flags.count("overlap"),
           "door": flags.count("door"), "clean": sum(1 for o in active if not o.get("issues"))}
    if any("history" in o or o.get("removed") is not None for o in objects):
        out.update(first_flagged=sum(1 for o in objects if (o.get("history") or [o])[0].get("issues")),
                   fixes_tried=sum(o.get("tries", 0) for o in objects),
                   fixed=sum(1 for o in active if o.get("fixes") and not o.get("issues")),      # now clean
                   improved=sum(1 for o in active if o.get("fixes") and o.get("issues")),   # better, still flagged
                   removed=sum(1 for o in objects if o.get("removed") is not None))
    return out


# ------------------------------------------------------------------ LayoutGPT variant ---------
# LayoutGPT (Feng et al., NeurIPS 2023; github.com/UCSB-AI/LayoutGPT, MIT licence), 3D scenes, run
# on our local model instead of GPT-4. Kept as published where it matters:
#   - its instruction, and its CSS-style line: category {length; width; height; orientation (degrees);
#     left; top; depth}, sizes and the object's centre, from the room's min corner (depth = height
#     of the centre above the floor);
#   - its condition: room type and "max length / max width" only. The outline, the door and the
#     walls are NOT written into the prompt (as published);
#   - k-similar examples: the K rooms of the same type whose 64x64 floor masks differ least (mean
#     squared difference), shown as user "Condition: ... Layout:" / assistant lines.
# Changed: metres (their metre mode); our example rooms (data/layout/exemplars.json, built by
# scripts/build_layout_exemplars.py) instead of 3D-FRONT, which needs a signed licence; K=4 (not 8)
# so a prompt fits llama-server's 4,096-token slot; the stage's greedy decoding, not GPT-4 at 0.7.
# Sizes the model writes are kept on the object (written_size); the checks and the stage use the
# standard sizes, as for v3.1, so the comparison is about where things go. A grammar fixes the
# line's syntax and the room's vocabulary only, as in v3.1.
LAYOUTGPT_K = 4
EXEMPLARS_PATH = "data/layout/exemplars.json"
CSS_LINE = re.compile(
    r"^\s*([a-z_]+)\s*\{length:\s*(\d+(?:\.\d+)?)m;\s*width:\s*(\d+(?:\.\d+)?)m;\s*height:\s*(\d+(?:\.\d+)?)m;"
    r"\s*orientation:\s*(-?\d+)\s*degrees;\s*left:\s*(-?\d+(?:\.\d+)?)m;\s*top:\s*(-?\d+(?:\.\d+)?)m;"
    r"\s*depth:\s*(-?\d+(?:\.\d+)?)m;\s*\}\s*$")
ROOM_NAME = {"living_room": "living room", "dining_room": "dining room"}
_EXEMPLARS: Optional[list[dict]] = None


def css_line(o: dict) -> str:
    return (f"{o['cls']} {{length: {o['w']:.2f}m; width: {o['d']:.2f}m; height: {o['h']:.2f}m; "
            f"orientation: {int(o['rot'])} degrees; left: {o['x']:.2f}m; top: {o['y']:.2f}m; "
            f"depth: {o['h'] / 2:.2f}m;}}")


def parse_css(line: str, spec: dict) -> Optional[dict]:
    """A LayoutGPT line -> the same placed object as parse_line gives (standard size; facing snapped
    to the nearest quarter turn), with the model's own sizes kept as written_size."""
    m = CSS_LINE.match(line)
    if not m or m.group(1) not in set(ROOMS[spec["room_type"]]):
        return None
    cls = m.group(1)
    w, d, h = FURNITURE[cls]
    rot = (round(int(m.group(5)) / 90) * 90) % 360
    return {"cls": cls, "x": float(m.group(6)), "y": float(m.group(7)), "w": w, "d": d, "h": h, "rot": rot,
            "written_size": [float(m.group(2)), float(m.group(3)), float(m.group(4))], "css": line.strip()}


def css_grammar(spec: dict) -> str:
    vocab = list(dict.fromkeys(ROOMS[spec["room_type"]]))
    return "\n".join([
        f"root ::= line{{1,{spec['max_objects']}}}",
        'line ::= cls " {length: " num "m; width: " num "m; height: " num "m; orientation: " deg '
        '" degrees; left: " num "m; top: " num "m; depth: " num "m;}\\n"',
        "cls ::= " + " | ".join(f'"{c}"' for c in vocab),
        'deg ::= "-"? [0-9] [0-9]? [0-9]?',
        'num ::= [0-9] [0-9]? "." [0-9] [0-9]'])


def exemplars() -> list[dict]:
    global _EXEMPLARS
    if _EXEMPLARS is None:
        import json
        from pathlib import Path
        path = Path(__file__).resolve().parents[1] / EXEMPLARS_PATH
        _EXEMPLARS = json.loads(path.read_text(encoding="utf-8"))
    return _EXEMPLARS


def floor_mask(pts: list, n: int = 64, extent: float = 8.0) -> list[list[bool]]:
    """The floor as an n x n mask over an extent x extent metre square from the outline's min corner."""
    x0, y0 = min(p[0] for p in pts), min(p[1] for p in pts)
    shifted = [[x - x0, y - y0] for x, y in pts]
    step = extent / n
    return [[_inside((i + 0.5) * step, (j + 0.5) * step, shifted) for i in range(n)] for j in range(n)]


def similar_rooms(spec: dict, k: int = LAYOUTGPT_K) -> list[dict]:
    """LayoutGPT's k-similar selection: same room type, least mean squared mask difference first."""
    q = floor_mask(spec["floor_plan"])
    pool = [e for e in exemplars() if e["room_type"] == spec["room_type"]]

    def mse(e: dict) -> float:
        m = floor_mask(e["floor_plan"])
        return sum(a != b for ra, rb in zip(q, m) for a, b in zip(ra, rb)) / (len(q) * len(q[0]))
    return sorted(pool, key=lambda e: (mse(e), e["id"]))[:k]


def _condition(spec: dict) -> str:
    xs, ys = [p[0] for p in spec["floor_plan"]], [p[1] for p in spec["floor_plan"]]
    return (f"Condition:\nRoom Type: {ROOM_NAME.get(spec['room_type'], spec['room_type'])}\n"
            f"Room Size: max length {max(xs) - min(xs):.2f}m, max width {max(ys) - min(ys):.2f}m\nLayout:\n")


def layoutgpt_messages(spec: dict, k: int = LAYOUTGPT_K) -> list[dict]:
    shots = similar_rooms(spec, k)
    vocab = list(dict.fromkeys(ROOMS[spec["room_type"]]))
    freq: dict[str, int] = {}
    for e in exemplars():
        if e["room_type"] == spec["room_type"]:
            for line in e["layout"]:
                c = line.split(" ", 1)[0]
                freq[c] = freq.get(c, 0) + 1
    total = sum(freq.values()) or 1
    system = ("You are a 3D indoor scene designer. \nInstruction: synthesize the 3D layout of an indoor scene. "
              "The generated 3D layout should follow the CSS style, where each line starts with the furniture "
              "category and is followed by the 3D size, orientation and absolute position. Formally, each line "
              "should follow the template: \nFURNITURE {length: ?m; width: ?m; height: ?m; orientation: ? degrees; "
              "left: ?m; top: ?m; depth: ?m;}\nAll values are in meters but the orientation angle is in degrees.\n"
              f"Available furnitures: {', '.join(vocab)}\n"
              "Overall furniture frequencies: (" + "; ".join(f"{c}: {freq[c] / total:.2f}" for c in vocab if c in freq)
              + ")")
    msgs = [{"role": "system", "content": system}]
    for e in shots:
        msgs += [{"role": "user", "content": _condition(e)}, {"role": "assistant", "content": "\n".join(e["layout"]) + "\n"}]
    return msgs + [{"role": "user", "content": _condition(spec)}]
