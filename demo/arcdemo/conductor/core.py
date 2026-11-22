"""The conductor: dispatches one job per machine, follows them, falls back to replay.

A *round* = one segment × one input, sent to all machines at once. Each machine is a *cell*.
Cells are independent: a slow or dead machine never blocks the others.

Badges (shown on stage, never suppressed):
  LIVE    the output was produced now, on that machine
  REPLAY  the output and timing come from rehearsal (scheduled, or because the live run failed)
  MOCK    a mock worker produced it (laptop dry-run only)

Fallback policy:
  worker unreachable / error / silent past budget+grace   -> REPLAY (with the reason noted)
  budget_exceeded, out_of_memory                           -> shown LIVE: they are results

Machines with `show_full_recording: true` (the CPU box) are slower than any time limit, so for
them every recording is the *full* run (replay/.../full/, made by rehearse.py --full): when the
machine is down, its time is up, or the operator stops it, the stage shows that run's output at
once, badged RECORDED, with how long it took to finish.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx

from ..schemas import TERMINAL
from .config import ROOT, DemoConfig
from .moderation import check_prompt
from .replay import ReplayStore, settings_hash

log = logging.getLogger("arcdemo.conductor")
MEDIA_SESSION = "/media"
MEDIA_REPLAY = "/replay-media"
LLM_FIELDS = ("tokens", "tokens_per_second", "first_token_seconds", "text", "objects")


def _llm_fields(res: dict) -> dict:
    """llm / layout counters and stream-0 content from a worker result (empty for the other tasks)."""
    out = {k: res.get(k) for k in LLM_FIELDS}
    out["objects"] = out["objects"] or []
    return out


def layout_view(item: dict) -> dict:
    """What the stage shows about a layout prompt: the wall names the model uses (with their
    end points, for the plan thumbnail), the door's wall, and the request in one plain sentence."""
    from ..layout_prompt import _angled, _door_wall, _not_floor, walls
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
           f"{', door on the ' + door + ' wall' if door else ''}, up to {item.get('max_objects', 10)} pieces, "
           "one line each: a wall piece names its wall, a free piece gives its spot.")
    return {"walls": ends, "door_wall": door, "ask": ask}


def layout_prompt_text(item: dict, max_objects: Optional[int] = None) -> str:
    """The exact chat the model receives for this room, as plain text for the operator page."""
    from ..layout_prompt import PROMPT_VERSION, grammar, messages
    spec = {"room_type": item["room_type"], "floor_plan": item["floor_plan"], "door": item.get("door"),
            "max_objects": int(max_objects or item.get("max_objects") or 10)}
    parts = [f"# prompt {PROMPT_VERSION}"]
    for m in messages(spec):
        parts += ["", f"=== {m['role']} ===", m["content"].rstrip()]
    parts += ["", "=== grammar (constrains the reply's syntax only) ===", grammar(spec)]
    return "\n".join(parts)


def _layout_summary(objects: Optional[list]) -> Optional[dict]:
    """Objects and flagged problems in the last room stream 0 worked on (None for other tasks)."""
    if not objects:
        return None
    last = max(o.get("room", 0) for o in objects)
    room = [o for o in objects if o.get("room", 0) == last]
    flagged = sum(1 for o in room if o.get("issues"))
    return {"objects": len(room), "flagged": flagged}


class Conductor:
    def __init__(self, cfg: DemoConfig, allow_replay: bool = True) -> None:
        self.cfg = cfg
        self.allow_replay = allow_replay
        self.replay = ReplayStore(cfg.replay_dir)
        cfg.session_dir.mkdir(parents=True, exist_ok=True)
        token = os.environ.get(cfg.token_env)
        self.http = httpx.AsyncClient(headers={"X-Demo-Token": token} if token else {},
                                      timeout=httpx.Timeout(5.0, connect=2.0))
        self.round: Optional[dict] = None
        self.view = "title"
        self.history: list[dict] = []
        self.health: dict[str, dict] = {}
        self.prepare_state: dict[str, dict] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._round_done = asyncio.Event()
        self.log_path = cfg.session_dir / "runs.jsonl"

    # ================================================================== public state ======
    def state(self) -> dict:
        return {"server_now": time.time(), "view": self.view, "round": self.round,
                "history": self.history, "health": self.health, "prepare": self.prepare_state,
                "warnings": self.env_warnings()}

    def env_warnings(self) -> list[str]:
        out = []
        by_task: dict[str, dict[str, str]] = {}
        for h in self.health.values():
            if h.get("ok"):
                for t in h.get("tasks", []):
                    by_task.setdefault(t, {})[h["machine"]] = h.get("env_hash", "?")
        for task, hashes in by_task.items():
            if len(set(hashes.values())) > 1:
                out.append(f"{task}: library versions differ between machines {hashes} — "
                           "check GET /env on each worker before trusting a comparison")
        return out

    # ================================================================== health ============
    async def health_loop(self, every: float = 2.0) -> None:
        while True:
            await self.refresh_health()
            await asyncio.sleep(every)

    async def refresh_health(self) -> None:
        urls = {(m["id"], t, u) for m in self.cfg.machines for t, u in (m.get("workers") or {}).items()}

        async def one(mid: str, task: str, url: str) -> None:
            key = f"{mid}:{task}"
            was_ok = self.health.get(key, {}).get("ok")
            t0 = time.perf_counter()
            try:
                r = await self.http.get(f"{url}/health", timeout=1.5)
                r.raise_for_status()
                self.health[key] = {"ok": True, "url": url, "task": task,
                                    "latency_ms": round((time.perf_counter() - t0) * 1000), **r.json()}
            except Exception as e:  # noqa: BLE001
                self.health[key] = {"ok": False, "url": url, "task": task, "machine": mid,
                                    "error": type(e).__name__}
            if was_ok != self.health[key]["ok"]:   # log transitions only, not every 2 s poll
                log.info("worker %s %s (%s)", key, "UP" if self.health[key]["ok"] else "DOWN", url)
        await asyncio.gather(*(one(*u) for u in urls))

    async def _post_quiet(self, url: str, what: str) -> None:
        """Best-effort POST (cancel/unload). Failure is logged, not raised: the caller has
        already decided what the stage shows."""
        try:
            (await self.http.post(url, json={})).raise_for_status()
        except httpx.HTTPError as e:
            log.warning("%s failed: %s %s", what, type(e).__name__, url)

    # ================================================================== prepare ===========
    async def prepare(self, segment_id: str) -> dict:
        """Load each machine's model for a segment, off the clock. On machines flagged
        one_task_at_a_time, first unload the other task's worker (16 GB VRAM / RAM)."""
        seg = self.cfg.segment(segment_id)
        task, protocol = seg["task"], seg["protocol"]
        for m in self.cfg.machines:
            mid = m["id"]
            s = self.cfg.settings(task, protocol, mid)
            if s.get("mode") == "replay" or s.get("prepare") is False:
                self.prepare_state[mid] = {"status": "skipped", "model": s["model"]}
                continue
            url = self.cfg.worker_url(mid, task)
            try:
                if m.get("one_task_at_a_time"):
                    for other, ourl in (m.get("workers") or {}).items():
                        if other != task and ourl != url:
                            await self._post_quiet(f"{ourl}/unload", f"unload {mid}:{other}")
                r = await self.http.post(f"{url}/load", json={"model": s["model"]})
                r.raise_for_status()
                self.prepare_state[mid] = {"status": "loading", "model": s["model"], "since": time.time()}
                log.info("prepare %s: loading %s on %s", segment_id, s["model"], mid)
            except Exception as e:  # noqa: BLE001
                self.prepare_state[mid] = {"status": "error", "model": s["model"], "message": type(e).__name__}
                log.warning("prepare %s: %s could not start loading %s: %s", segment_id, mid, s["model"], e)
        asyncio.create_task(self._watch_prepare(task))
        return self.prepare_state

    async def _watch_prepare(self, task: str, timeout: float = 900) -> None:
        end = time.time() + timeout
        while time.time() < end:
            pending = [mid for mid, p in self.prepare_state.items() if p["status"] == "loading"]
            if not pending:
                return
            for mid in pending:
                url = self.cfg.worker_url(mid, task)
                try:
                    h = (await self.http.get(f"{url}/health")).json()
                    if self.prepare_state[mid]["model"] in h.get("loaded", []):
                        p = self.prepare_state[mid]
                        p.update(status="ready", seconds=round(time.time() - p["since"], 1))
                        log.info("prepare: %s ready with %s after %ss", mid, p["model"], p["seconds"])
                    elif not h.get("loading"):
                        env = (await self.http.get(f"{url}/env")).json()
                        if env.get("last_error"):
                            self.prepare_state[mid].update(status="error", message=env["last_error"])
                            log.warning("prepare: %s failed: %s", mid, env["last_error"])
                except Exception as e:  # noqa: BLE001 — keep watching; the worker may be restarting
                    log.warning("prepare: cannot reach %s: %s", mid, type(e).__name__)
            await asyncio.sleep(1)

    # ================================================================== rounds ============
    async def start_round(self, segment_id: str, input_id: str, prompt: Optional[str] = None,
                          override: bool = False, force_replay: Optional[list[str]] = None,
                          budget_override: Optional[float] = None, run_to_completion: bool = False,
                          machines: Optional[list[str]] = None) -> dict:
        """run_to_completion: no time limit (rehearse.py --full). machines: only these cells."""
        if self.round and not self.round["done"]:
            raise RuntimeError("a round is still running — cancel it first")
        seg = self.cfg.segment(segment_id)
        task, protocol = seg["task"], seg["protocol"]
        item = self.cfg.input(task, input_id)
        if input_id == "live":
            if not prompt or not prompt.strip():
                raise ValueError("live round needs a prompt")
            mod = check_prompt(prompt)
            if not mod["ok"] and not override:
                raise PermissionError(f"prompt blocked: {', '.join(mod['hits'])}")
            item["prompt"] = prompt.strip()
        if task == "layout":
            item.update(layout_view(item))
        budget = budget_override if budget_override is not None else seg.get("budget_seconds")
        if run_to_completion:
            budget = None
        ms = [m for m in self.cfg.machines if not machines or m["id"] in machines]
        rid = time.strftime("%H%M%S-") + uuid.uuid4().hex[:6]
        caption = (seg.get("caption") or "").replace("{budget}", f"{budget:g}" if budget else "")
        self.round = {
            "id": rid, "segment_id": segment_id, "title": seg["title"], "caption": caption,
            "task": task, "protocol": protocol, "budget_seconds": budget, "warmup": bool(seg.get("warmup")),
            "input": {k: v for k, v in item.items() if not k.startswith("_")},
            "started_at": time.time(), "done": False, "order": [m["id"] for m in ms],
            "cells": {m["id"]: self._new_cell(m, task, protocol) for m in ms},
        }
        self.view = "round"
        log.info("round %s | %s | %s/%s | input %s | budget %s", rid, segment_id, task, protocol, input_id, budget)
        self._round_done = asyncio.Event()
        force = set(force_replay or [])
        self._tasks = {mid: asyncio.create_task(self._run_cell(mid, mid in force)) for mid in self.round["order"]}
        asyncio.create_task(self._finish_round(self.round))
        return self.round

    def _new_cell(self, m: dict, task: str, protocol: str) -> dict:
        s = self.cfg.settings(task, protocol, m["id"])
        return {"machine": m["id"], "label": m["label"], "hardware": m.get("hardware", ""),
                "color": m.get("color", "#888"), "mode": "PENDING", "status": "queued",
                "model": s["model"], "model_label": self.cfg.model_label(s["model"]),
                "settings": {k: v for k, v in s.items() if k not in ("model", "mode", "prepare")},
                "outputs": [], "completed": 0, "requested": None, "final_seconds": None,
                "progress": None, "note": "", "reference": None, "score": None, "mock": False,
                "run_id": None, "result": None, "recorded": None,
                "full_recording": bool(m.get("show_full_recording")), **_llm_fields({})}

    async def _finish_round(self, rnd: dict) -> None:
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        rnd["done"] = True
        rnd["finished_at"] = time.time()
        log.info("round %s done | %s", rnd["id"], {m: (c["mode"], c["status"], c["final_seconds"])
                                                   for m, c in rnd["cells"].items()})
        if not rnd["warmup"]:
            self.history.append(self._summarise(rnd))
        self._round_done.set()

    async def wait_round(self, timeout: Optional[float] = None) -> None:
        await asyncio.wait_for(self._round_done.wait(), timeout)

    async def cancel_round(self) -> None:
        """Stop every machine still working. Stopped cells say how far they got; machines
        flagged show_full_recording switch to their full recording."""
        rnd = self.round
        if not rnd or rnd["done"]:
            return
        log.info("round %s stopped by operator", rnd["id"])
        await asyncio.gather(*(self.stop_cell(mid) for mid in rnd["order"]))

    async def stop_cell(self, mid: str) -> None:
        """Stop one machine: cancel its job on the worker and stop following it."""
        rnd = self.round
        if not rnd or mid not in rnd["cells"]:
            return
        cell = rnd["cells"][mid]
        if cell["status"] in TERMINAL or cell["mode"] == "REPLAY":
            return
        if cell["run_id"]:
            url = self.cfg.worker_url(mid, rnd["task"])
            await self._post_quiet(f"{url}/jobs/{cell['run_id']}/cancel", f"cancel {mid}")
        t = self._tasks.get(mid)
        if t and not t.done():
            t.cancel()
        if cell["status"] in TERMINAL or cell["mode"] == "REPLAY":   # finished while we waited
            return
        stopped_at = round(time.time() - rnd["started_at"], 1)
        log.info("round %s | %s stopped by operator at %.1fs", rnd["id"], mid, stopped_at)
        cell["status"] = "cancelled"
        cell["final_seconds"] = stopped_at
        cell["note"] = f"Still running when stopped at {stopped_at:.0f} s"
        if cell["full_recording"] and self.allow_replay:
            self._show_full_recording(rnd, mid, f"Stopped at {stopped_at:.0f} s", stopped_at)

    async def switch_to_replay(self, mid: str, reason: str = "operator switched to replay") -> None:
        rnd = self.round
        if not rnd:
            return
        cell = rnd["cells"][mid]
        if cell["run_id"] and cell["status"] not in TERMINAL:
            await self._post_quiet(f"{self.cfg.worker_url(mid, rnd['task'])}/jobs/{cell['run_id']}/cancel",
                                   f"cancel {mid}")
        cell["_operator_replay"] = True
        await self._play_replay(rnd, mid, reason)

    def plan(self, segment_id: str, input_id: Optional[str] = None) -> dict:
        """What a round would run, per machine: model, settings, and the recording on file.
        The operator page shows it before Start so the operator knows what to expect."""
        seg = self.cfg.segment(segment_id)
        task, protocol = seg["task"], seg["protocol"]
        out = {"segment_id": segment_id, "task": task, "protocol": protocol,
               "budget_seconds": seg.get("budget_seconds"), "machines": []}
        for m in self.cfg.machines:
            s = self.cfg.settings(task, protocol, m["id"])
            row = {"id": m["id"], "label": m["label"], "model": s["model"],
                   "model_label": self.cfg.model_label(s["model"]), "mode": s.get("mode", "live"),
                   "settings": {k: v for k, v in s.items() if k not in ("model", "mode", "prepare")},
                   "full_recording": bool(m.get("show_full_recording")), "recording": None}
            if input_id and input_id != "live":
                rec = (self.replay.get_full(task, protocol, m["id"], input_id, s) if row["full_recording"]
                       else self.replay.get(task, protocol, m["id"], input_id, s))
                if rec:
                    r = rec["result"]
                    row["recording"] = {"seconds": r.get("all_outputs_seconds") or r.get("elapsed_seconds"),
                                        "status": r.get("status"), "completed": r.get("completed"),
                                        "requested": r.get("requested"), "stale": rec["stale"]}
            out["machines"].append(row)
        return out

    # ------------------------------------------------------------------ one cell ---------
    async def _run_cell(self, mid: str, force_replay: bool) -> None:
        rnd = self.round
        task, protocol, item = rnd["task"], rnd["protocol"], rnd["input"]
        cell = rnd["cells"][mid]
        s = self.cfg.settings(task, protocol, mid)
        rec = (self.replay.get_full(task, protocol, mid, item["id"], s) if cell["full_recording"]
               else self.replay.get(task, protocol, mid, item["id"], s))
        if rec:
            r = rec["result"]
            cell["reference"] = {"seconds": r.get("all_outputs_seconds") or r.get("elapsed_seconds")}
        if (s.get("mode") == "replay" or force_replay) and self.allow_replay:
            await self._play_replay(rnd, mid, "scheduled replay (rehearsal result)")
            return

        url = self.cfg.worker_url(mid, task)
        job = self._job_request(task, protocol, item, s, rnd["budget_seconds"])
        cell["run_id"] = job["run_id"]
        cell["mode"] = "LIVE"
        h = self.health.get(f"{mid}:{task}", {})
        cell["mock"] = bool(h.get("mock"))
        log.info("round %s | %s -> %s run_id=%s model=%s", rnd["id"], mid, url, job["run_id"][:6], job["model"])
        try:
            r = await self.http.post(f"{url}/jobs", json=job)
        except Exception as e:  # noqa: BLE001
            await self._fallback(rnd, mid, f"worker unreachable ({type(e).__name__})")
            return
        if r.status_code != 200:
            await self._fallback(rnd, mid, f"worker rejected the job: HTTP {r.status_code} {r.text[:200]}")
            return

        last_ok = time.time()
        fetched: set[str] = set()
        budget = rnd["budget_seconds"]
        while True:
            await asyncio.sleep(0.4)
            if cell.get("_operator_replay"):
                return
            try:
                resp = await self.http.get(f"{url}/jobs/{job['run_id']}", timeout=2.5)
                last_ok = time.time()
            except Exception as e:  # noqa: BLE001
                log.debug("poll %s failed: %s", mid, type(e).__name__)
                if time.time() - last_ok > self.cfg.unreachable_after:
                    await self._fallback(rnd, mid, "lost contact with worker mid-run")
                    return
                continue
            if resp.status_code == 404:
                await self._fallback(rnd, mid, "worker no longer knows this job (restarted mid-run?)")
                return
            res = resp.json()
            if cell.get("_operator_replay"):      # operator switched while this poll was in flight
                return
            await self._apply_result(rnd, mid, url, res, fetched)
            if res["status"] in TERMINAL:
                break
            elapsed = time.time() - rnd["started_at"]
            # Slow machine still busy when the time is up: stop it and show its full recording.
            # (2 s of slack lets the worker report its own budget_exceeded with any partial output.)
            if (cell["full_recording"] and self.allow_replay and budget is not None
                    and elapsed > budget + 2 and self._has_full_recording(rnd, mid)):
                await self._post_quiet(f"{url}/jobs/{job['run_id']}/cancel", f"cancel {mid}")
                self._log(rnd, mid, {**res, "status": "cancelled", "message": f"time up at {budget:g} s"},
                          "MOCK" if cell["mock"] else "LIVE")
                self._show_full_recording(rnd, mid, f"Time up at {budget:g} s", budget)
                return
            if budget is not None and task in ("t2i", "llm", "layout") and elapsed > budget + self.cfg.grace:
                await self._post_quiet(f"{url}/jobs/{job['run_id']}/cancel", f"cancel {mid}")
                await self._fallback(rnd, mid, f"no result {self.cfg.grace:g}s after the budget")
                return

        cell["result"] = res          # full worker record: the replay table and debugging use it
        status = res["status"]
        if status == "error":
            await self._fallback(rnd, mid, f"live run failed: {res.get('message', '')[:160]}")
            return
        cell["status"] = status
        cell["final_seconds"] = self._final_seconds(res, budget)
        if status == "out_of_memory":
            cell["note"] = "Out of memory — this model does not fit on this machine"
        elif status == "budget_exceeded":
            done, req = res.get("completed", 0), res.get("requested") or 0
            if task == "i23d" and done:
                cell["note"] = f"Finished after the bell, at {res['all_outputs_seconds']:.1f} s"
            else:
                cell["note"] = (f"{done} of {req} finished in {budget:g} s" if done
                                else f"Did not finish in {budget:g} s")
        self._log(rnd, mid, res, "MOCK" if cell["mock"] else "LIVE")
        done = res.get("completed", 0)
        unfinished = done == 0 or (task == "t2i" and done < (res.get("requested") or 0))
        if (status == "budget_exceeded" and unfinished and cell["full_recording"] and self.allow_replay
                and self._has_full_recording(rnd, mid)):
            self._show_full_recording(rnd, mid, f"Time up at {budget:g} s", budget)
            return
        if task == "i23d" and cell["outputs"]:
            await self._score(rnd, mid)

    def _job_request(self, task: str, protocol: str, item: dict, s: dict, budget: Optional[float]) -> dict:
        job: dict[str, Any] = {"run_id": uuid.uuid4().hex, "task": task, "model": s["model"],
                               "input_id": item["id"], "protocol": protocol,
                               "budget_seconds": budget, "seed": int(item.get("seed") or 0)}
        for k, v in s.items():
            if k not in ("model", "mode", "prepare"):
                job[k] = v
        if task == "t2i":
            job["prompt"] = item["prompt"]
        elif task == "i23d":
            job["image_path"] = item["input"]
        elif task == "music":
            job.update(prompt=item.get("prompt"), mode=item.get("mode"), audio_path=item.get("audio"),
                       length_s=s.get("length_s", item.get("length_s")))
        elif task == "llm":
            job["prompt"] = item["prompt"]
        elif task == "layout":
            job.update(room_type=item["room_type"], floor_plan=item["floor_plan"], door=item.get("door"),
                       max_objects=s.get("max_objects", item.get("max_objects")))
        return job

    @staticmethod
    def _final_seconds(res: dict, budget: Optional[float]) -> Optional[float]:
        if res["status"] == "ok":
            return res.get("all_outputs_seconds") or res.get("elapsed_seconds")
        if res["status"] == "budget_exceeded":
            if res.get("all_outputs_seconds") and (budget is None or res["all_outputs_seconds"] > budget):
                return res["all_outputs_seconds"]
            return budget
        return res.get("elapsed_seconds")

    async def _apply_result(self, rnd: dict, mid: str, url: str, res: dict, fetched: set[str]) -> None:
        cell = rnd["cells"][mid]
        cell["status"] = res["status"]
        cell["progress"] = res.get("progress")
        cell["completed"] = res.get("completed", 0)
        cell["requested"] = res.get("requested")
        cell.update(_llm_fields(res))
        if res.get("message"):
            cell["note"] = res["message"]
        for name in res.get("outputs", []):
            if name in fetched:
                continue
            dest = self.cfg.session_dir / rnd["id"] / mid / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                async with self.http.stream("GET", f"{url}/files/{name}", timeout=60) as resp:
                    resp.raise_for_status()
                    with open(dest, "wb") as f:
                        async for chunk in resp.aiter_bytes():
                            f.write(chunk)
            except Exception as e:  # noqa: BLE001 — retried on the next poll
                log.warning("fetch %s from %s failed (%s), will retry", name, mid, type(e).__name__)
                continue
            fetched.add(name)
            cell["outputs"].append({"url": f"{MEDIA_SESSION}/{rnd['id']}/{mid}/{name}",
                                    "reveal_at": None, "path": str(dest)})

    # ------------------------------------------------------------------ replay -----------
    async def _fallback(self, rnd: dict, mid: str, reason: str) -> None:
        log.warning("round %s | %s FALLBACK: %s", rnd["id"], mid, reason)
        if self.allow_replay:
            await self._play_replay(rnd, mid, reason)
        else:
            cell = rnd["cells"][mid]
            cell.update(status="error", note=reason)
            self._log(rnd, mid, {"status": "error", "message": reason}, "LIVE")

    async def _play_replay(self, rnd: dict, mid: str, reason: str) -> None:
        cell = rnd["cells"][mid]
        if cell["full_recording"] and self._has_full_recording(rnd, mid):
            self._show_full_recording(rnd, mid, reason, None)
            return
        s = self.cfg.settings(rnd["task"], rnd["protocol"], mid)
        rec = self.replay.get(rnd["task"], rnd["protocol"], mid, rnd["input"]["id"], s)
        cell["mode"] = "REPLAY"
        cell["outputs"] = []
        if rec is None or rnd["input"]["id"] == "live":
            cell.update(status="unavailable", note=f"{reason}; no rehearsal record for this cell")
            log.warning("round %s | %s has no replay record", rnd["id"], mid)
            return
        log.info("round %s | %s REPLAY from %s (%s)", rnd["id"], mid, rec["_dir"], reason)
        r = rec["result"]
        base = self._replay_url(rec)
        times = r.get("output_times") or []
        for i, name in enumerate(rec["files"]):
            t = times[i] if i < len(times) else r.get("all_outputs_seconds")
            # Replays reveal each output at its rehearsal time, measured from the round start.
            cell["outputs"].append({"url": f"{base}/{name}", "reveal_at": t,
                                    "path": str(Path(rec["_dir"]) / name)})
        cell.update(status=r["status"], completed=r.get("completed", len(rec["files"])),
                    requested=r.get("requested"), progress=None, result=r,
                    final_seconds=self._final_seconds(r, rnd["budget_seconds"]),
                    note=reason + (" · settings changed since rehearsal" if rec["stale"] else ""),
                    score=rec.get("score"), model=rec["settings"].get("model", cell["model"]),
                    model_label=self.cfg.model_label(rec["settings"].get("model", cell["model"])), **_llm_fields(r))
        if r.get("status") == "out_of_memory":
            cell["note"] = "Out of memory (rehearsal) — this model does not fit on this machine"
        elif r.get("status") == "budget_exceeded" and not r.get("completed"):
            cell["note"] = f"Did not finish in the budget (rehearsal) · {reason}"
        self._log(rnd, mid, r, "REPLAY")

    def _replay_url(self, rec: dict) -> str:
        rel = Path(rec["_dir"]).resolve().relative_to(self.cfg.replay_dir).as_posix()
        return f"{MEDIA_REPLAY}/{rel}"

    def _has_full_recording(self, rnd: dict, mid: str) -> bool:
        if rnd["input"]["id"] == "live":
            return False
        s = self.cfg.settings(rnd["task"], rnd["protocol"], mid)
        return self.replay.get_full(rnd["task"], rnd["protocol"], mid, rnd["input"]["id"], s) is not None

    def _show_full_recording(self, rnd: dict, mid: str, why: str, live_seconds: Optional[float]) -> None:
        """Replace the cell with the recording of a complete run, all outputs shown at once,
        labelled with the time that run took to finish. Synchronous on purpose: it must land
        before _finish_round summarises the round."""
        cell = rnd["cells"][mid]
        s = self.cfg.settings(rnd["task"], rnd["protocol"], mid)
        rec = self.replay.get_full(rnd["task"], rnd["protocol"], mid, rnd["input"]["id"], s)
        if rec is None:
            return
        r = rec["result"]
        took = r.get("all_outputs_seconds") or r.get("elapsed_seconds")
        base = self._replay_url(rec)
        log.info("round %s | %s RECORDED full run from %s (%s, took %.1fs)", rnd["id"], mid, rec["_dir"], why, took or 0)
        cell.update(mode="REPLAY", status="ok", progress=None, result=r, final_seconds=took,
                    completed=r.get("completed", len(rec["files"])), requested=r.get("requested"),
                    outputs=[{"url": f"{base}/{n}", "reveal_at": None, "path": str(Path(rec["_dir"]) / n)}
                             for n in rec["files"]],
                    recorded={"seconds": took, "why": why, "live_seconds": live_seconds, "stale": rec["stale"]},
                    note=f"{why} — showing the recorded full run"
                         + (" · settings changed since recording" if rec["stale"] else ""),
                    score=rec.get("score"), model=rec["settings"].get("model", cell["model"]),
                    model_label=self.cfg.model_label(rec["settings"].get("model", cell["model"])), **_llm_fields(r))
        cell["_operator_replay"] = True       # the live poll loop, if still going, stops following
        self._log(rnd, mid, r, "REPLAY")

    # ------------------------------------------------------------------ scoring ----------
    async def _score(self, rnd: dict, mid: str) -> None:
        truth = rnd["input"].get("truth")
        cell = rnd["cells"][mid]
        if not truth or not (ROOT / truth).exists() or not cell["outputs"]:
            return
        cell["score"] = {"pending": True}
        try:
            from ..scoring import score_files
            sc = await asyncio.to_thread(score_files, cell["outputs"][0]["path"], str(ROOT / truth))
            cell["score"] = sc.as_dict()
        except Exception as e:  # noqa: BLE001
            cell["score"] = {"error": type(e).__name__}
            log.exception("scoring %s failed", mid)

    # ------------------------------------------------------------------ records ----------
    def _log(self, rnd: dict, mid: str, res: dict, badge: str) -> None:
        cell = rnd["cells"][mid]
        s = self.cfg.settings(rnd["task"], rnd["protocol"], mid)
        rec = {"ts": time.time(), "round": rnd["id"], "segment": rnd["segment_id"], "warmup": rnd["warmup"],
               "machine": mid, "task": rnd["task"], "protocol": rnd["protocol"],
               "input_id": rnd["input"]["id"], "model": s["model"], "model_repo": res.get("model_repo"),
               "tier": self.cfg.model_label(s["model"]),
               "settings": s, "settings_hash": settings_hash(s), "seed": rnd["input"].get("seed"),
               "budget_seconds": rnd["budget_seconds"], "badge": badge,
               "status": res.get("status"), "message": res.get("message"),
               "load_seconds": res.get("load_seconds"),
               "first_output_seconds": res.get("first_output_seconds"),
               "all_outputs_seconds": res.get("all_outputs_seconds"),
               "generation_seconds": res.get("generation_seconds"), "export_seconds": res.get("export_seconds"),
               "completed": res.get("completed"), "requested": res.get("requested"),
               "peak_memory_mb": res.get("peak_memory_mb"), "peak_reserved_mb": res.get("peak_reserved_mb"),
               "tokens": res.get("tokens"), "tokens_per_second": res.get("tokens_per_second"),
               "first_token_seconds": res.get("first_token_seconds"),
               "worker_settings": res.get("settings"), "run_id": cell.get("run_id"),
               "outputs": [o["path"] for o in cell["outputs"]]}
        if rnd["input"]["id"] == "live":
            rec["prompt"] = rnd["input"].get("prompt")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _summarise(self, rnd: dict) -> dict:
        return {"round": rnd["id"], "segment_id": rnd["segment_id"], "title": rnd["title"],
                "task": rnd["task"], "protocol": rnd["protocol"], "budget_seconds": rnd["budget_seconds"],
                "input_id": rnd["input"]["id"], "input_label": rnd["input"].get("label", rnd["input"]["id"]),
                "cells": {mid: {k: c.get(k) for k in ("mode", "status", "final_seconds", "completed",
                                                      "requested", "model_label", "score", "mock", "recorded",
                                                      "tokens", "tokens_per_second")}
                              | {"layout": _layout_summary(c.get("objects"))}
                          for mid, c in rnd["cells"].items()}}
