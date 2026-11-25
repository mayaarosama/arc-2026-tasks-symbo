"""Build data/layout/exemplars.json: the example rooms for the LayoutGPT prompt variant (Task 8).

LayoutGPT (Feng et al., NeurIPS 2023; github.com/UCSB-AI/LayoutGPT, MIT) shows the model the k
training rooms most similar to the query, taken from 3D-FRONT. 3D-FRONT needs a signed licence, so
we use our own rooms instead: rectangles of varied size with a door, furnished by a few designer
rules (the bed centred on a wall without the door, nightstands either side, the coffee table in
front of the sofa and the TV stand facing it...). Every placement passes the same room checks as
the stage (layout_prompt.check): nothing outside, overlapping or in front of the door. No room
here has a stage room's floor plan.

    python -m scripts.layout.build_layout_exemplars          # rewrites data/layout/exemplars.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from arcdemo.layout_prompt import (FURNITURE, _door_wall, check, css_line, parse_line,
                                   walls)

ROOT = Path(__file__).resolve().parents[2]       # demo/
OUT = ROOT / "data" / "layout" / "exemplars.json"
OPPOSITE = {"south": "north", "north": "south", "east": "west", "west": "east"}
SIZES = [(3.2, 3.0), (3.6, 3.2), (4.0, 3.4), (4.2, 3.8), (4.6, 3.6), (5.0, 4.0), (5.4, 4.4), (6.0, 4.6)]


class Room:
    def __init__(self, room_type: str, w: float, h: float, rng: random.Random) -> None:
        self.spec = {"room_type": room_type, "floor_plan": [[0, 0], [w, 0], [w, h], [0, h]], "max_objects": 10}
        side = rng.choice(["south", "north", "east", "west"])
        t = rng.uniform(0.15, 0.85)
        self.spec["door"] = {"south": [round(w * t, 2), 0], "north": [round(w * t, 2), h],
                             "west": [0, round(h * t, 2)], "east": [w, round(h * t, 2)]}[side]
        self.w, self.h = w, h
        self.walls = walls(self.spec["floor_plan"])
        door = _door_wall(self.spec)
        free = sorted((n for n in self.walls if n != door), key=lambda n: -(self.walls[n]["span"][1] - self.walls[n]["span"][0]))
        self.main = free[0]                                  # the longest wall without the door
        self.opp = OPPOSITE[self.main]
        self.sides = [n for n in ("south", "north", "east", "west") if n not in (self.main, self.opp)]
        self.placed: list[dict] = []
        self.lines: list[str] = []

    def mid(self, wall: str) -> float:
        a, b = self.walls[wall]["span"]
        return round((a + b) / 2, 2)

    def inward(self, wall: str, along: float, dist: float) -> tuple[float, float]:
        """The point `dist` metres into the room from `wall` at position `along`."""
        wl = self.walls[wall]
        off = wl["fixed"] + wl["sign"] * dist
        return (along, off) if wl["axis"] == "x" else (off, along)

    def _try(self, line: str) -> bool:
        o = parse_line(line, self.spec)
        if o is None or check(o, self.placed, self.spec):
            return False
        self.placed.append(o)
        self.lines.append(line)
        return True

    def wall(self, cls: str, prefer: list[str], at: float | None = None) -> bool:
        for name in prefer + [n for n in self.walls if n not in prefer]:
            a, b = self.walls[name]["span"]
            half = FURNITURE[cls][0] / 2
            want = self.mid(name) if at is None or name != prefer[0] else at
            spots = [round(a + half + 0.05 + k * 0.1, 2) for k in range(int((b - a - 2 * half - 0.1) / 0.1) + 1)]
            for s in sorted(spots, key=lambda s: abs(s - want)):
                if self._try(f"{cls} wall={name} at={s:.2f}"):
                    return True
        return False

    def free(self, cls: str, x: float, y: float, facing: str) -> bool:
        pts = [(round(x + dx * 0.1, 2), round(y + dy * 0.1, 2)) for dx in range(-12, 13) for dy in range(-12, 13)]
        for px, py in sorted(pts, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2):
            if 0.2 <= px <= self.w - 0.2 and 0.2 <= py <= self.h - 0.2 and \
                    self._try(f"{cls} x={px:.2f} y={py:.2f} facing={facing}"):
                return True
        return False

    def corner(self, cls: str, k: int) -> bool:
        c = [(0.35, 0.35), (self.w - 0.35, self.h - 0.35), (self.w - 0.35, 0.35), (0.35, self.h - 0.35)][k % 4]
        return self.free(cls, *c, facing="south")


def bedroom(r: Room) -> None:
    m = r.mid(r.main)
    r.wall("double_bed", [r.main], m)
    r.wall("nightstand", [r.main], m - 1.07)
    r.wall("nightstand", [r.main], m + 1.07)
    r.wall("wardrobe", [r.sides[0], r.sides[1], r.opp])
    r.wall("dresser", [r.opp, r.sides[1]])
    if r.wall("desk", [r.sides[1], r.opp]):
        d = r.placed[-1]
        x, y = r.inward(d["wall"], d["at"], 0.85)
        r.free("chair", x, y, facing=d["wall"])
    r.corner("plant", 0)
    r.corner("floor_lamp", 1)


def living_room(r: Room) -> None:
    m = r.mid(r.main)
    r.wall("sofa", [r.main], m)
    r.free("coffee_table", *r.inward(r.main, m, 1.55), facing=OPPOSITE[r.main])
    r.wall("tv_stand", [r.opp], m)
    r.free("armchair", *r.inward(r.main, m + 1.6, 1.55), facing=r.main)
    r.free("side_table", *r.inward(r.main, m - 1.45, 0.35), facing=OPPOSITE[r.main])
    r.wall("bookshelf", [r.sides[0], r.sides[1]])
    r.wall("cabinet", [r.sides[1], r.sides[0]])
    r.corner("plant", 0)
    r.corner("floor_lamp", 2)


def library(r: Room) -> None:
    m = r.mid(r.main)
    r.wall("bookshelf", [r.main], m - 0.45)
    r.wall("bookshelf", [r.main], m + 0.45)
    r.wall("bookshelf", [r.sides[0], r.sides[1]])
    cx, cy = r.w / 2, r.h / 2
    if r.free("reading_table", cx, cy, facing="north"):
        t = r.placed[-1]
        r.free("chair", t["x"], t["y"] + 0.65, facing="south")
        r.free("chair", t["x"], t["y"] - 0.65, facing="north")
    r.wall("desk", [r.opp, r.sides[1]])
    r.corner("armchair", 1)
    r.corner("floor_lamp", 2)
    r.corner("plant", 3)


def main() -> None:
    rooms = []
    for room_type, furnish in (("bedroom", bedroom), ("living_room", living_room), ("library", library)):
        for n, (w, h) in enumerate(SIZES):
            r = Room(room_type, w, h, random.Random(f"{room_type}-{n}"))
            furnish(r)
            rooms.append({"id": f"ex-{room_type}-{n + 1}", **r.spec, "max_objects": len(r.lines),
                          "v3": r.lines, "layout": [css_line(o) for o in r.placed]})
    OUT.write_text(json.dumps(rooms, indent=1), encoding="utf-8")
    for x in rooms:
        print(f"{x['id']:22s} {x['floor_plan'][2][0]:.1f} x {x['floor_plan'][2][1]:.1f} m  door {x['door']}  "
              f"{len(x['v3'])} objects")


if __name__ == "__main__":
    main()
