"""Task 8 — sequential 3D scene construction, driven by a local LLM (the LayoutGPT idea, our code).

The model gets a room type and a floor outline and places furniture one object per line, in one of
two forms (prompt v3, "wall-anchored"):

    wardrobe wall=east at=1.70              a wall piece: back against the named wall, centre at
                                            y=1.70 along it; the worker computes x, y and facing
    armchair x=2.10 y=3.00 facing=west      a free piece: centre and facing on the open floor

Why: the first two prompt versions asked for raw centre coordinates and facing for everything, and
a 4B model could not do that arithmetic (0 and 1 of 9 wall pieces actually against a wall; see
docs/Task 8 — Layout model review.md). Now the model decides *what goes where*; the geometry of
"against that wall" is computed. Sizes are fixed per object type (FURNITURE).

Each line is parsed the moment its newline arrives and timestamped on the job clock, so the stage
(and a replay) shows the room furnishing itself in the model's own order.

Same weights and same llama-server as Task 4 (backends/llm.py): only the prompt differs. A GBNF
grammar restricts the reply to those two line forms, the room's furniture vocabulary and this room's
wall names, so even a small model's output always parses. Where the furniture goes is still the
model's choice, and every object is checked against the room (outside the walls, blocking the door,
overlapping an earlier object). Problems are flagged on stage, never corrected.

Coordinates: metres, origin at the room's south-west corner, x east, y north. `rot` (stored per
object) is the way the front faces (0 north, 90 east, 180 south, 270 west). w runs along the front,
d front to back, so at rot 90/270 the footprint is d wide in x and w deep in y.

  iso_quality  1 room, greedy: the same room on every machine, compare the time
  iso_time     N rooms at once, repeated until the bell: rooms furnished inside the budget

Tokens are counted as in Task 4; result.objects carries stream 0's objects (all its rooms, each
tagged with its room index) for the stage.
"""
from __future__ import annotations

import json
import random
import threading
import time
from contextlib import closing
from typing import Iterator

from ...layout_prompt import (  # noqa: F401 — re-exported: tests and tools use layout.<name>
    DOOR_CLEARANCE, EXAMPLES, FACING, FURNITURE, OVERLAP_SHARE, PROMPT_VERSION, ROOMS, SYSTEM, WALL_PIECES,
    WALL_TOLERANCE, check, describe_room, footprint, grammar, messages, parse_line, summarise, walls,
)
from ...schemas import JobRequest
from .base import Backend, JobContext
from .llm import LlamaServer


def room_spec(req: JobRequest) -> dict:
    if not req.floor_plan or len(req.floor_plan) < 3:
        raise ValueError("layout job needs floor_plan: a list of at least three [x, y] points")
    room = req.room_type or "bedroom"
    if room not in ROOMS:
        raise ValueError(f"unknown room_type '{room}' (known: {', '.join(ROOMS)})")
    return {"room_type": room, "floor_plan": req.floor_plan, "door": req.door,
            "max_objects": int(req.max_objects or 10)}


# ------------------------------------------------------------------ generation --------------
class LayoutBackend(Backend):
    """Shared loop; subclasses provide _room_stream()."""
    task = "layout"

    def _room_stream(self, spec: dict, req: JobRequest, seed: int) -> Iterator[tuple[str, int]]:
        raise NotImplementedError

    def describe(self, req: JobRequest) -> dict:
        return {"streams": req.streams, "max_tokens": req.max_tokens, "temperature": req.temperature,
                "room_type": req.room_type, "max_objects": req.max_objects, "quant": self.spec.get("file", ""),
                "prompt": PROMPT_VERSION}

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        spec = room_spec(req)
        n = max(1, req.streams)
        r = ctx.result
        lock = threading.Lock()
        last = [0.0]
        errors: list[BaseException] = []
        ctx.progress(stage=f"{n} room{'s' if n > 1 else ''} at once")

        def count(k: int) -> bool:
            if ctx.should_stop():
                return False
            t = ctx.clock.now()
            with lock:
                if r.first_token_seconds is None:
                    r.first_token_seconds = round(t, 3)
                r.tokens += k
                last[0] = t
                r.tokens_per_second = round(r.tokens / t, 1) if t > 0 else None
            return True

        def run(i: int) -> None:
            k = 0
            while not ctx.should_stop():
                placed: list[dict] = []
                buf, finished = "", True
                raw: list[str] = []          # the model's reply as written, kept for review

                def take(line: str) -> None:
                    o = parse_line(line, spec)
                    if o is None:
                        return
                    o["issues"] = check(o, placed, spec)
                    o.update(i=len(placed), room=k, t=round(ctx.clock.now(), 3))
                    placed.append(o)
                    if i == 0:
                        with lock:
                            r.objects = r.objects + [o]       # new list: a poll never sees it half-built
                            r.progress.step = len(placed)

                with closing(self._room_stream(spec, req, req.seed + i + k * n)) as gen:
                    for piece, ntok in gen:
                        if not count(ntok):
                            finished = False         # cut by the bell: placed objects stay on stage, room not saved
                            break
                        buf += piece
                        raw.append(piece)
                        while "\n" in buf:
                            line, buf = buf.split("\n", 1)
                            take(line)
                if not finished:
                    return
                if buf.strip():
                    take(buf)
                path = ctx.output_path(f"-{i}-{k}.json")
                path.write_text(json.dumps({"room": spec, "objects": placed, "summary": summarise(placed),
                                            "raw": "".join(raw)},
                                           indent=1), encoding="utf-8")
                ctx.record_output(path)
                k += 1
                if not req.repeat_until_budget or req.budget_seconds is None:
                    return

        def guarded(i: int) -> None:
            try:
                run(i)
            except BaseException as e:  # noqa: BLE001 — re-raised on the job thread below
                errors.append(e)
                ctx.cancel.set()

        ctx.progress(step=0, steps=spec["max_objects"])
        threads = [threading.Thread(target=guarded, args=(i,), daemon=True) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        if r.tokens:
            r.tokens_per_second = round(r.tokens / last[0], 1)
            if req.repeat_until_budget:   # throughput: the run ends with its last token, as in Task 4
                r.all_outputs_seconds = round(last[0], 3)


class LayoutServer(LayoutBackend, LlamaServer):
    """The Task 4 llama-server, with the layout prompt and grammar.

    Prompt warm-up: the rules and the two worked examples (~1,100 tokens) are the same text for
    every room, so load() sends them off the clock and llama-server keeps their KV cache. A room
    then only reads its own few lines of prompt (~130 tokens; on the laptop CPU 29 s -> 4 s).
    The cache is per slot and is NOT shared between slots (measured, build 12608), so one warm-up
    request goes to every slot at once: concurrent requests each take a free slot. Disclosed in
    every run record as settings.prompt_warmed; worker config layout_warm_prompt: false puts the
    full prompt read back on the clock."""

    def _ctx_per_slot(self) -> int:
        """The layout prompt (rules, two worked examples, the room) is ~2,000-2,800 tokens before the
        reply, more than the Task 4 default of 2,048 per slot. Worker config key layout_ctx overrides it."""
        return int(self.cfg.get("layout_ctx", 4096))

    def load(self) -> None:
        super().load()
        self.warmed = bool(self.cfg.get("layout_warm_prompt", True))
        if not self.warmed:
            return
        spec = {**EXAMPLES[0][0], "max_objects": 1}   # any room: only the shared prefix matters
        errors: list[BaseException] = []

        def warm() -> None:
            try:
                for _ in self._chat(messages(spec), 1, 0.0, 0):
                    pass
            except BaseException as e:  # noqa: BLE001 — re-raised below
                errors.append(e)

        threads = [threading.Thread(target=warm, daemon=True) for _ in range(self._slots())]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]

    def describe(self, req: JobRequest) -> dict:
        return {**LayoutBackend.describe(self, req), "server": getattr(self, "version", "?"),
                "slots": self._slots(), "prompt_warmed": getattr(self, "warmed", False)}

    def _room_stream(self, spec: dict, req: JobRequest, seed: int) -> Iterator[tuple[str, int]]:
        yield from self._chat(messages(spec), req.max_tokens, req.temperature, seed, grammar=grammar(spec))


# ------------------------------------------------------------------ mock --------------------
_MOCK_CACHE: dict[str, list[str]] = {}
_MOCK_LOCK = threading.Lock()


def mock_layout(spec: dict, seed: int) -> list[str]:
    """Cached, and computed once even when 64 mock streams ask at the same moment."""
    key = json.dumps([spec, seed], sort_keys=True)
    with _MOCK_LOCK:
        if key not in _MOCK_CACHE:
            _MOCK_CACHE[key] = _mock_layout(spec, seed)
        return _MOCK_CACHE[key]


def _mock_layout(spec: dict, seed: int) -> list[str]:
    """A plausible room in the v3 line format: wall pieces at the first free spot along a wall,
    free pieces at the first free spot on a 10 cm grid. Placeholder output, never shown as a result."""
    poly = spec["floor_plan"]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    named = walls(poly)
    rng = random.Random(seed)
    placed: list[dict] = []
    lines: list[str] = []
    for cls in ROOMS[spec["room_type"]]:
        if len(lines) >= spec["max_objects"]:
            break
        w = FURNITURE[cls][0]
        cands = []
        if cls in WALL_PIECES:
            for name, wl in named.items():
                a, b = wl["span"]
                at = a + w / 2
                while at <= b - w / 2 + 1e-9:
                    cands.append(f"{cls} wall={name} at={at:.2f}")
                    at += 0.1
        else:
            x = min(xs) + 0.3
            while x <= max(xs) - 0.3:
                y = min(ys) + 0.3
                while y <= max(ys) - 0.3:
                    cands.append(f"{cls} x={x:.2f} y={y:.2f} facing={rng.choice(list(FACING))}")
                    y += 0.1
                x += 0.1
        rng.shuffle(cands)
        for line in cands:
            o = parse_line(line, spec)
            if o and not check(o, placed, spec):
                placed.append(o)
                lines.append(line + "\n")
                break
    return lines


class LayoutMock(LayoutBackend):
    """Wall-packed placeholder rooms at the Task 4 mock token rate (~4 characters per token)."""

    def load(self) -> None:
        time.sleep(0.3)

    def _room_stream(self, spec: dict, req: JobRequest, seed: int) -> Iterator[tuple[str, int]]:
        delay = 0.025 * float(self.cfg.get("mock_speed", 1.0)) * (1 + req.streams / 32)
        rng = random.Random(seed)
        sent = 0
        for line in mock_layout(spec, 0):     # one placeholder room per floor plan: packing is slow-ish
            for j in range(0, len(line), 4):
                if sent >= req.max_tokens:
                    return
                time.sleep(delay * rng.uniform(0.8, 1.2))
                sent += 1
                yield line[j:j + 4], 1
