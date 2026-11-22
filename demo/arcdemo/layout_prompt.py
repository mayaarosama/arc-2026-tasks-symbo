"""Task 8 prompt, room geometry and checks: everything about a layout that needs no model.

Shared by the worker (arcdemo/worker/backends/layout.py, which streams the model's reply through
parse_line and check) and the conductor (which shows the operator the exact prompt and gives the
stage the wall names). Pure Python, no dependencies. See the backend's docstring for the format.
"""
from __future__ import annotations

import re
from typing import Optional

# Fixed sizes (w along the front, d front to back, h), metres. v3: the model does not choose sizes.
FURNITURE: dict[str, tuple[float, float, float]] = {
    "double_bed": (1.60, 2.00, 0.50), "single_bed": (0.90, 2.00, 0.50), "nightstand": (0.50, 0.40, 0.55),
    "wardrobe": (1.20, 0.60, 2.00), "dresser": (1.20, 0.50, 0.80), "desk": (1.20, 0.60, 0.75),
    "chair": (0.50, 0.50, 0.90), "armchair": (0.80, 0.80, 0.90), "sofa": (2.20, 0.90, 0.85),
    "coffee_table": (1.20, 0.60, 0.45), "tv_stand": (1.50, 0.40, 0.50), "side_table": (0.50, 0.50, 0.55),
    "bookshelf": (0.80, 0.30, 1.80), "reading_table": (1.60, 0.80, 0.75), "cabinet": (1.00, 0.45, 0.90),
    "dining_table": (1.60, 0.90, 0.75), "dining_chair": (0.45, 0.50, 0.90),
    "floor_lamp": (0.40, 0.40, 1.60), "plant": (0.40, 0.40, 1.00),
}
# Placed against a wall (wall=… at=…); everything else is free-standing (x=… y=… facing=…).
WALL_PIECES = {"double_bed", "single_bed", "nightstand", "wardrobe", "dresser", "desk", "sofa", "tv_stand",
               "bookshelf", "cabinet"}
ROOMS: dict[str, list[str]] = {   # vocabulary per room type, in a sensible placement order (mock uses it)
    "bedroom": ["double_bed", "nightstand", "nightstand", "wardrobe", "dresser", "desk", "chair",
                "armchair", "bookshelf", "floor_lamp", "plant", "single_bed", "tv_stand"],
    "living_room": ["sofa", "coffee_table", "tv_stand", "armchair", "armchair", "side_table", "bookshelf",
                    "floor_lamp", "plant", "cabinet", "dining_table", "dining_chair", "dining_chair"],
    "library": ["bookshelf", "bookshelf", "bookshelf", "reading_table", "chair", "chair", "armchair",
                "side_table", "floor_lamp", "sofa", "desk", "cabinet", "plant"],
    # Used by the worked examples: room types that are NOT on stage, so there is nothing to copy
    # (with a same-type example, the 4B model reproduced the bedroom example line for line).
    "study": ["desk", "chair", "bookshelf", "bookshelf", "armchair", "floor_lamp", "cabinet", "plant",
              "side_table", "sofa"],
    "dining_room": ["dining_table", "dining_chair", "dining_chair", "cabinet", "dresser", "bookshelf",
                    "side_table", "floor_lamp", "plant", "tv_stand"],
}
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


def grammar(spec: dict) -> str:
    """GBNF: 1..max_objects lines; wall pieces only in the wall form with this room's wall names,
    free pieces only in the x/y form. Syntax and vocabulary only, never positions."""
    vocab = list(dict.fromkeys(ROOMS[spec["room_type"]]))
    wcls = [c for c in vocab if c in WALL_PIECES]
    fcls = [c for c in vocab if c not in WALL_PIECES]
    names = list(walls(spec["floor_plan"]))
    alt = [x for x, ok in (("wline", wcls and names), ("fline", fcls)) if ok]
    rules = [f"root ::= line{{1,{spec['max_objects']}}}", "line ::= " + " | ".join(alt)]
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


def check(o: dict, earlier: list[dict], spec: dict) -> list[str]:
    """Problems with one object, given those placed before it in the same room."""
    poly = spec["floor_plan"]
    x0, y0, x1, y1 = footprint(o)
    issues = []
    if not all(_inside(px, py, poly) or _near_edge(px, py, poly, WALL_TOLERANCE)
               for px, py in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
        issues.append("outside")
    door = spec.get("door")
    if door:
        cx, cy = min(max(door[0], x0), x1), min(max(door[1], y0), y1)   # nearest footprint point
        if (cx - door[0]) ** 2 + (cy - door[1]) ** 2 < DOOR_CLEARANCE ** 2:
            issues.append("door")
    area = (x1 - x0) * (y1 - y0)
    for e in earlier:
        a0, b0, a1, b1 = footprint(e)
        ov = max(0.0, min(x1, a1) - max(x0, a0)) * max(0.0, min(y1, b1) - max(y0, b0))
        if ov > OVERLAP_SHARE * max(1e-6, min(area, (a1 - a0) * (b1 - b0))):
            issues.append("overlap")
            break
    return issues


def summarise(objects: list[dict]) -> dict:
    flags = [i for o in objects for i in o.get("issues", [])]
    return {"objects": len(objects), "outside": flags.count("outside"), "overlap": flags.count("overlap"),
            "door": flags.count("door"), "clean": sum(1 for o in objects if not o.get("issues"))}
