"""Task 8 review run: furnish the stage rooms with the real model and save everything for review.

    python -m scripts.layout.sample_layout --config config/worker-workstation.yaml
    python -m scripts.layout.sample_layout --config config/worker-workstation.yaml --rooms lay-01,lay-03

Runs the real worker code path in-process with networking OFF and the show settings
(demo.yaml layout/iso_quality: greedy, one room at a time). Writes one folder,
results-local/layout-review-<machine>-<time>/:
  prompt.txt                the exact prompt and grammar (first room)
  summary.md                timings, tokens, objects and flags per room, plus the raw model text
  rooms/<input>.json        the worker's room record (objects with times, checks, raw reply)
  review.html (+layout.js)  the rooms drawn as on stage: 3D with furniture when served over HTTP
                            (python -m http.server -d demo 8765), canvas fallback as a plain file
  worker.log                the worker log;  llama-server-<model>.log  the server log
Use it to tune the prompt in arcdemo/worker/backends/layout.py and compare runs.
"""
import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from arcdemo.conductor.config import DemoConfig
from arcdemo.conductor.tasks.layout import layout_prompt_text
from arcdemo.logs import setup_logging
from arcdemo.worker.app import ROOT, create_app, load_worker_config

TERMINAL = ("ok", "budget_exceeded", "out_of_memory", "error", "cancelled")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="worker config, e.g. config/worker-workstation.yaml")
    ap.add_argument("--demo", default="config/demo.yaml", help="where the settings and rooms come from")
    ap.add_argument("--protocol", default="iso_quality")
    ap.add_argument("--rooms", help="comma list of input ids (default: all)")
    ap.add_argument("--out", help="output folder (default: results-local/layout-review-<machine>-<time>)")
    a = ap.parse_args()
    if "huggingface_hub" in sys.modules:   # read HF_HUB_OFFLINE at its import: offline would not be enforced
        sys.exit("huggingface_hub was imported before HF_HUB_OFFLINE was set: offline mode is not enforced")
    os.environ["HF_HUB_OFFLINE"] = "1"          # before the backend imports huggingface_hub: no network

    cfg = load_worker_config(str(Path(a.config).resolve()), ["layout"])
    os.chdir(ROOT)
    demo = DemoConfig(a.demo)
    settings = {k: v for k, v in demo.settings("layout", a.protocol, cfg["machine"]).items()
                if k not in ("mode", "prepare")}
    rooms = demo.inputs["layout"]
    if a.rooms:
        want = [r.strip() for r in a.rooms.split(",")]
        rooms = [r for r in rooms if r["id"] in want]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) if a.out else ROOT / "results-local" / f"layout-review-{cfg['machine']}-{stamp}"
    (out / "rooms").mkdir(parents=True, exist_ok=True)
    cfg["results_dir"] = str(out)               # outputs and the llama-server log land here too
    setup_logging(out / "worker.log")

    runs = []
    with TestClient(create_app(cfg)) as c:
        t = time.time()
        c.post("/load", json={"model": settings["model"]})
        while True:                             # load + prompt warm-up, off the clock
            h = c.get("/health").json()
            if settings["model"] in h["loaded"]:
                break
            err = c.get("/env").json().get("last_error")
            if err:
                print("load failed:", err)
                return 1
            time.sleep(1)
        load_s = round(time.time() - t, 1)
        print(f"loaded {settings['model']} in {load_s} s (includes prompt warm-up)")
        for room in rooms:
            job = {"task": "layout", "input_id": room["id"], "protocol": a.protocol, "seed": room.get("seed", 0),
                   "room_type": room["room_type"], "floor_plan": room["floor_plan"], "door": room.get("door"),
                   **settings, "max_objects": settings.get("max_objects", room.get("max_objects"))}
            rid = c.post("/jobs", json=job).json()["run_id"]
            while (res := c.get(f"/jobs/{rid}").json())["status"] not in TERMINAL:
                time.sleep(0.5)
            rec = {}
            if res["outputs"]:
                src = out / res["outputs"][0]
                rec = json.loads(src.read_text(encoding="utf-8"))
                shutil.move(src, out / "rooms" / f"{room['id']}.json")
            runs.append({"room": room, "result": res, "record": rec})
            s = rec.get("summary", {})
            print(f"{room['id']}: {res['status']} in {res['all_outputs_seconds']} s · {s.get('objects', 0)} objects, "
                  f"{s.get('objects', 0) - s.get('clean', 0)} flagged · {res['tokens']} tokens "
                  f"@ {res['tokens_per_second']} tok/s {res['message']}")
        c.post("/unload", json={})               # stop llama-server; it would outlive this script

    write_prompt(out, rooms, settings.get("layout_prompt"))
    write_summary(out, cfg, settings, load_s, runs)
    write_review_page(out, runs)
    print("review folder:", out)
    return 0 if all(r["result"]["status"] == "ok" for r in runs) else 1


def write_prompt(out: Path, rooms: list, style: str | None = None) -> None:
    """The exact chat messages the model saw, for each room (LayoutGPT picks different examples per room)."""
    text = "\n\n".join(f"########## {room['id']}\n" + layout_prompt_text(room, room.get("max_objects"), style)
                       for room in rooms)
    (out / "prompt.txt").write_text(text + "\n", encoding="utf-8")


def write_summary(out: Path, cfg: dict, settings: dict, load_s: float, runs: list) -> None:
    first = runs[0]["result"]["settings"] if runs else {}
    lines = [f"# Task 8 layout review — {cfg['machine']}, {datetime.now():%Y-%m-%d %H:%M}", "",
             f"- Model: `{settings['model']}` ({first.get('quant', '')}), {first.get('server', '?')}",
             f"- Settings: {json.dumps({k: v for k, v in settings.items() if k != 'model'})}",
             f"- Prompt warmed at load: {first.get('prompt_warmed')} · load + warm-up {load_s} s (off the clock)",
             "", "| Room | Status | Time | First object | Objects | Flagged | First pass flagged | Fixed · improved · removed · tries | Tokens | tok/s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in runs:
        res, s = r["result"], r["record"].get("summary", {})
        objs = r["record"].get("objects", [])
        flagged = s.get("objects", 0) - s.get("clean", 0)          # removed objects excluded
        fixes = (f"{s['fixed']} · {s['improved']} · {s['removed']} · {s['fixes_tried']}" if "fixed" in s else "—")
        lines.append(f"| {r['room']['id']} {r['room']['label']} | {res['status']} | {res['all_outputs_seconds']} s | "
                     f"{objs[0]['t'] if objs else '—'} s | {s.get('objects', 0)} | {flagged} | "
                     f"{s.get('first_flagged', flagged)} | {fixes} | {res['tokens']} | {res['tokens_per_second']} |")
    for r in runs:
        room, objs = r["room"], r["record"].get("objects", [])
        lines += ["", f"## {room['id']} · {room['label']}", "",
                  f"Outline {room['floor_plan']} · door {room.get('door')} · up to "
                  f"{room.get('max_objects')} objects", "",
                  "| # | Object | x | y | w × d × h | Facing | At | Problems | Fix history |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for o in objs:
            facing = {0: "north", 90: "east", 180: "south", 270: "west"}.get(o["rot"], f"{o['rot']}°")
            hist = " → ".join(f"{h['op']} {h['t']} s ({', '.join(h['issues']) or 'clean'})"
                              for h in o.get("history", [])) or "—"
            probs = "removed" if o.get("removed") is not None else (", ".join(o["issues"]) or "—")
            lines.append(f"| {o['i'] + 1} | {o['cls']} | {o['x']} | {o['y']} | {o['w']} × {o['d']} × {o['h']} | "
                         f"{facing} | {o['t']} s | {probs} | {hist} |")
        if r["result"]["message"] and r["result"]["status"] != "ok":
            lines += ["", f"**{r['result']['status']}**: {r['result']['message']}"]
        lines += ["", "Raw model output:", "", "```text", r["record"].get("raw", "").rstrip(), "```"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_page(out: Path, runs: list) -> None:
    """The rooms drawn with the stage renderers, finished state. Served over HTTP from demo/
    (python -m http.server -d demo 8765, then /results-local/<folder>/review.html) it uses the 3D
    view with furniture models; opened as a plain file it falls back to the canvas view."""
    shutil.copy(ROOT / "arcdemo" / "conductor" / "static" / "layout.js", out / "layout.js")   # file:// fallback
    static = Path(os.path.relpath(ROOT / "arcdemo" / "conductor" / "static", out)).as_posix() + "/"
    data = [{"id": r["room"]["id"], "label": r["room"]["label"], "input": r["room"],
             "objects": r["record"].get("objects", []), "seconds": r["result"]["all_outputs_seconds"],
             "status": r["result"]["status"]} for r in runs]
    page = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Layout review</title>
<style>
  body { margin: 0; padding: 24px; background: #0b0d10; color: #f2f4f7; font-family: "Segoe UI", system-ui, sans-serif; }
  h1 { font-size: 22px; margin: 0 0 16px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 16px; }
  figure { margin: 0; background: #14181d; border: 1px solid #262c34; border-radius: 10px; padding: 12px; }
  figcaption { font-size: 15px; margin-bottom: 8px; } figcaption small { color: #a9b1bc; display: block; margin-top: 2px; }
  .box { position: relative; height: 460px; }
  .layout-canvas { position: absolute; inset: 0; width: 100%; height: 100%; }
  .layout-labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
  .lbl { position: absolute; transform: translate(-50%, -100%); white-space: nowrap; font-size: 12px; padding: 1px 6px;
         border-radius: 4px; background: #0b0d10cc; color: #c9d1db; }
  .lbl.bad { background: #5a1515e6; color: #ffb3b3; } .lbl.door { background: none; color: #ffd24a; font-weight: 600; }
</style>
<script type="importmap">{"imports": {"three": "__STATIC__vendor/three/three.module.min.js",
                                      "three/addons/": "__STATIC__vendor/three/addons/"}}</script>
</head><body>
<h1>Task 8 layout review</h1><div class="grid" id="g"></div>
<script>window.Demo = { elapsed: () => 1e9 };</script>
<script src="layout.js"></script>
<script type="module" src="__STATIC__layout3d.js"></script>
<script>
const DATA = __DATA__;
document.addEventListener("DOMContentLoaded", () => {      // after the 3D module, if it loaded
  const views = [];
  for (const d of DATA) {
    const f = document.createElement("figure");
    const cap = document.createElement("figcaption");
    cap.textContent = `${d.id} · ${d.label}`;
    const sm = document.createElement("small");
    sm.textContent = `${d.status} in ${d.seconds} s · ${Layout.tally(d.objects)}`;
    cap.appendChild(sm);
    const box = document.createElement("div");
    box.className = "box";
    f.append(cap, box);
    document.getElementById("g").appendChild(f);
    views.push([Layout.attach(box), { input: d.input }, { objects: d.objects, mode: "LIVE" }]);
  }
  (function frame() { for (const [v, r, c] of views) Layout.draw(v, r, c); requestAnimationFrame(frame); })();
});
</script></body></html>
"""
    page = page.replace("__STATIC__", static).replace("__DATA__", json.dumps(data))
    (out / "review.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
