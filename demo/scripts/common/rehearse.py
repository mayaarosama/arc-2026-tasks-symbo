"""Rehearsal matrix pass: run segments for real, record every run, fill the replay table.

    python -m scripts.common.rehearse --config config/demo.yaml --segments seg2,seg3 --repeat 3
    python -m scripts.common.rehearse --config config/demo.yaml --segments seg1 --max-wait 14400   # overnight CPU iso-quality

- Uses the same conductor code as the show, with replay fallback OFF: failures are recorded
  as failures, never papered over.
- Every run goes to <session_dir>/runs.jsonl (the measurement record).
- Per cell, the *median* run (by time to all outputs) becomes the replay record, so a replay
  never shows a lucky best case.
- out_of_memory and budget_exceeded are real results and are saved as replays too.
- Prints a timing table to choose the budget (30-60 s) and the per-machine tiers from.

Full runs (what the CPU shows on stage when its time is up, it is stopped, or it is down):

    python -m scripts.common.rehearse --config config/demo.yaml --segments seg0,seg2,seg3 --full --repeat 1 --max-wait 14400

  runs only the machines flagged show_full_recording (override with --machines), with NO time
  limit, and saves each complete run to replay/<task>/<protocol>/<machine>/<input>/full/.
"""
import argparse
import asyncio
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from arcdemo.conductor.config import DemoConfig
from arcdemo.conductor.core import Conductor
from arcdemo.logs import setup_logging

log = logging.getLogger("arcdemo.rehearse")
SAVEABLE = {"ok", "budget_exceeded", "out_of_memory"}


def cell_time(res: dict) -> float:
    return res.get("all_outputs_seconds") or res.get("elapsed_seconds") or 0.0


async def rehearse(cfg: DemoConfig, segments: list[str], repeat: int, max_wait: float, save: bool,
                   full: bool = False, machines: list[str] | None = None) -> None:
    if full and not machines:
        machines = [m["id"] for m in cfg.machines if m.get("show_full_recording")]
        if not machines:
            raise SystemExit("--full: no machine has show_full_recording: true; pass --machines")
    cond = Conductor(cfg, allow_replay=False)
    await cond.refresh_health()
    down = [k for k, h in cond.health.items() if not h["ok"]]
    if down:
        log.warning("workers down at start: %s", down)
    runs: dict[tuple, list] = defaultdict(list)

    for sid in segments:
        seg = cfg.segment(sid)
        for input_id in seg["inputs"]:
            if input_id == "live":
                continue
            for i in range(repeat):
                log.info("=== %s %s run %d/%d", sid, input_id, i + 1, repeat)
                rnd = await cond.start_round(sid, input_id, run_to_completion=full, machines=machines)
                try:
                    await cond.wait_round(max_wait)
                except asyncio.TimeoutError:
                    log.warning("%s %s: still running after %ss, stopping", sid, input_id, max_wait)
                    await cond.cancel_round()
                    await cond.wait_round(30)
                for mid, cell in rnd["cells"].items():
                    res = cell.get("result")
                    if res and (res["status"] == "ok" if full else res["status"] in SAVEABLE):
                        runs[(seg["task"], seg["protocol"], mid, input_id)].append(
                            (res, [Path(o["path"]) for o in cell["outputs"]], cell.get("score")))
                    else:
                        log.warning("%s %s %s: not saveable (%s) %s", sid, input_id, mid,
                                    cell["status"], cell.get("note", ""))

    print("\n| task | protocol | machine | input | runs | status | median s | min s | max s | outputs | peak MB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for key, items in sorted(runs.items()):
        items.sort(key=lambda it: cell_time(it[0]))
        med_res, med_files, med_score = items[len(items) // 2]
        times = [cell_time(r) for r, _, _ in items]
        task, protocol, mid, input_id = key
        print(f"| {task} | {protocol} | {mid} | {input_id} | {len(items)} | {med_res['status']} | "
              f"{statistics.median(times):.1f} | {min(times):.1f} | {max(times):.1f} | "
              f"{med_res['completed']}/{med_res['requested']} | {med_res.get('peak_memory_mb')} |")
        if save:
            settings = cfg.settings(task, protocol, mid)
            d = cond.replay.save(task, protocol, mid, input_id, settings, med_res, med_files,
                                 extra={"score": med_score, "runs": len(items), "times": times},
                                 variant="full" if full else "")
            log.info("replay saved: %s", d)
    await cond.http.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/demo.yaml")
    ap.add_argument("--segments", required=True, help="comma list of segment ids")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--max-wait", type=float, default=600, help="seconds per round before stopping it")
    ap.add_argument("--no-save", action="store_true", help="measure only, do not touch the replay table")
    ap.add_argument("--full", action="store_true",
                    help="run to completion with no time limit and save to the full/ recording")
    ap.add_argument("--machines", help="comma list of machine ids (default: all; with --full, the "
                                       "machines flagged show_full_recording)")
    a = ap.parse_args()
    cfg = DemoConfig(a.config)
    setup_logging(cfg.session_dir / "rehearsal.log")
    machines = [m.strip() for m in a.machines.split(",")] if a.machines else None
    asyncio.run(rehearse(cfg, [s.strip() for s in a.segments.split(",")], a.repeat, a.max_wait, not a.no_save,
                         a.full, machines))


if __name__ == "__main__":
    main()
