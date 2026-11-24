"""Compare Task 8 review runs (scripts/sample_layout.py output folders) prompt against prompt.

    python scripts/compare_layout.py results-local/layout-compare-gpu-v31 results-local/layout-compare-gpu-layoutgpt

Per folder and room: objects, flags (outside / overlap / door), wall pieces really standing against
a wall (back within 5 cm of a straight wall, facing into the room), objects copied from an example
the model was shown (same type within 0.20 m), and the timings from summary.md. Prints Markdown.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arcdemo.layout_prompt import (EXAMPLES, WALL_PIECES, footprint, parse_line,  # noqa: E402
                                   similar_rooms, walls)


def against_wall(o: dict, spec: dict) -> bool:
    x0, y0, x1, y1 = footprint(o)
    for w in walls(spec["floor_plan"]).values():
        if o["rot"] != w["rot"]:
            continue
        a, b = w["span"]
        if w["axis"] == "x":
            back = y0 if w["sign"] > 0 else y1
            if abs(back - w["fixed"]) <= 0.05 and a - 0.05 <= x0 and x1 <= b + 0.05:
                return True
        else:
            back = x0 if w["sign"] > 0 else x1
            if abs(back - w["fixed"]) <= 0.05 and a - 0.05 <= y0 and y1 <= b + 0.05:
                return True
    return False


def shown_examples(spec: dict, style: str) -> list[dict]:
    """The example objects the model saw for this room."""
    if style == "layoutgpt":
        return [parse_line(line, e) for e in similar_rooms(spec) for line in e["v3"]]
    return [parse_line(line, s) for s, reply in EXAMPLES for line in reply.splitlines()]


def timings(folder: Path) -> dict[str, dict]:
    out = {}
    for line in (folder / "summary.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\| (lay-\d+) [^|]*\| (\w+) \| ([\d.]+) s \| ([\d.]+) s \|.*\| (\d+) \| ([\d.]+) \|$", line)
        if m:
            out[m.group(1)] = {"status": m.group(2), "time": float(m.group(3)), "first": float(m.group(4)),
                               "tokens": int(m.group(5))}
    return out


def analyse(folder: Path) -> tuple[str, list[dict]]:
    style = "layoutgpt" if "layoutgpt" in (folder / "prompt.txt").read_text(encoding="utf-8")[:200] else "v3.1"
    t = timings(folder)
    rows = []
    for f in sorted((folder / "rooms").glob("lay-*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        spec, objs = rec["room"], [o for o in rec["objects"] if o.get("removed") is None]
        wall = [o for o in objs if o["cls"] in WALL_PIECES]
        shown = shown_examples(spec, style)
        copied = sum(1 for o in objs if any(e["cls"] == o["cls"] and abs(e["x"] - o["x"]) <= 0.2
                                            and abs(e["y"] - o["y"]) <= 0.2 for e in shown))
        flags = [i for o in objs for i in o.get("issues", [])]
        sized = [o for o in objs if "written_size" in o]
        size_off = sum(1 for o in sized if any(abs(a - b) > 0.1 for a, b in zip(o["written_size"], (o["w"], o["d"], o["h"]))))
        rows.append({"room": f.stem, "objects": len(objs), "flagged": sum(1 for o in objs if o.get("issues")),
                     "outside": flags.count("outside"), "overlap": flags.count("overlap"), "door": flags.count("door"),
                     "wall": f"{sum(against_wall(o, spec) for o in wall)} of {len(wall)}", "copied": copied,
                     "sizes_off": size_off if sized else None, **t.get(f.stem, {})})
    return style, rows


def main() -> None:
    for folder in map(Path, sys.argv[1:]):
        style, rows = analyse(folder)
        print(f"\n### {style} — {folder.name}\n")
        print("| Room | Objects | Flagged | outside · overlap · door | Wall pieces against a wall | Copied from an example "
              "| Sizes not standard | Room time | First object | Tokens | Tokens / object |")
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in rows:
            print(f"| {r['room']} | {r['objects']} | {r['flagged']} | {r['outside']} · {r['overlap']} · {r['door']} | "
                  f"{r['wall']} | {r['copied']} | {'—' if r['sizes_off'] is None else r['sizes_off']} | "
                  f"{r.get('time', '?')} s | {r.get('first', '?')} s | {r.get('tokens', '?')} | "
                  f"{round(r['tokens'] / r['objects']) if r.get('tokens') and r['objects'] else '—'} |")
        tot = lambda k: sum(r[k] for r in rows)   # noqa: E731
        print(f"| **all** | {tot('objects')} | **{tot('flagged')}** | {tot('outside')} · {tot('overlap')} · {tot('door')} | "
              f"{sum(int(r['wall'].split()[0]) for r in rows)} of {sum(int(r['wall'].split()[-1]) for r in rows)} | "
              f"{tot('copied')} | | | | | |")


if __name__ == "__main__":
    main()
