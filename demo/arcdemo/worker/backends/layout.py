"""Task 8 — sequential 3D scene construction, driven by a local LLM (the LayoutGPT idea, our code).

The model gets a room type and a floor outline and places furniture one object per line, in one of
two forms (prompt v3, "wall-anchored"):

    wardrobe wall=east at=1.70              a wall piece: back against the named wall, centre at
                                            y=1.70 along it; the worker computes x, y and facing
    armchair x=2.10 y=3.00 facing=west      a free piece: centre and facing on the open floor

Why: the first two prompt versions asked for raw centre coordinates and facing for everything, and
a 4B model could not do that arithmetic (0 and 1 of 9 wall pieces actually against a wall; see
docs/Task 8 — Layout model review.md). Now the model decides *what goes where*; the geometry of
"against that wall" is computed. Sizes are fixed per object type, read from the furniture catalogue
(data/layout/furniture.json; worker config key layout_catalog).

Each line is parsed the moment its newline arrives and timestamped on the job clock, so the stage
(and a replay) shows the room furnishing itself in the model's own order.

Same weights and same llama-server as Task 4 (backends/llm.py): only the prompt differs. A GBNF
grammar restricts the reply to those two line forms, the room's furniture vocabulary and this room's
wall names, so even a small model's output always parses. Where the furniture goes is still the
model's choice, and every object is checked against the room (outside the walls, blocking the door,
overlapping an earlier object). Problems are flagged on stage. The code never moves furniture;
with self-correction on (req.repairs, req.review_rounds; segment seg8b) the model is told each problem
and fixes it itself, and every attempt is kept and shown (see _Room).

Coordinates: metres, origin at the room's south-west corner, x east, y north. `rot` (stored per
object) is the way the front faces (0 north, 90 east, 180 south, 270 west). w runs along the front,
d front to back, so at rot 90/270 the footprint is d wide in x and w deep in y.

  iso_quality  1 room, greedy: the same room on every machine, compare the time
  iso_time     N rooms at once, repeated until the bell: rooms furnished inside the budget

Tokens are counted as in Task 4; result.objects carries stream 0's objects (all its rooms, each
tagged with its room index) for the stage.
"""
from __future__ import annotations

import copy
import json
import random
import threading
import time
from contextlib import closing
from typing import Callable, Iterator, Optional

from ...layout_prompt import (  # noqa: F401 — re-exported: tests and tools use layout.<name>
    CATALOG, DOOR_CLEARANCE, EXAMPLES, FACING, FURNITURE, MUST_HAVE_COST, OVERLAP_SHARE, PROMPT_VERSION, REMOVE_COST,
    ROOMS, SYSTEM, WALL_PIECES,
    WALL_TOLERANCE, check, continue_messages, css_grammar, css_line, layoutgpt_messages, parse_css, describe_room, edit_grammar, footprint, grammar, line_of, measure,
    messages, parse_edit, parse_line, problem_cost, repair_grammar, repair_messages, review_messages, room_cost,
    summarise, walls,
)
from ...schemas import JobRequest
from .base import Backend, JobContext
from .llm import LlamaServer, run_streams


def room_spec(req: JobRequest) -> dict:
    if not req.floor_plan or len(req.floor_plan) < 3:
        raise ValueError("layout job needs floor_plan: a list of at least three [x, y] points")
    room = req.room_type or "bedroom"
    if room not in ROOMS:
        raise ValueError(f"unknown room_type '{room}' (known: {', '.join(ROOMS)})")
    style = req.layout_prompt or PROMPT_VERSION
    if style not in (PROMPT_VERSION, "layoutgpt"):
        raise ValueError(f"unknown layout_prompt '{style}' (known: {PROMPT_VERSION}, layoutgpt)")
    if style == "layoutgpt" and (req.repairs or req.review_rounds):
        raise ValueError("self-correction (repairs, review_rounds) works with the v3.1 prompt only")
    return {"room_type": room, "floor_plan": req.floor_plan, "door": req.door,
            "max_objects": int(req.max_objects or 10)}


POSE = ("x", "y", "rot", "wall", "at")    # what a fix may change; size and class stay
REPAIR_TOKENS = 40      # one line is 13-16 tokens
REVIEW_TOKENS = 200     # up to 6 edits and "done"



# ------------------------------------------------------------------ generation --------------
class _Cut(Exception):
    """The bell or a stop ended the room mid-way: its placed objects stay on stage, the room is not saved."""


class _Room:
    """One room, built by the model: placement, then (optionally) self-correction.

    Phase A, repair (req.repairs > 0): when a line is flagged, the stream is closed and the model is
    told exactly what is wrong ("it overlaps 3 sofa by 0.42 m²") and asked for that line again, the
    same object (grammar). Up to req.repairs tries; the attempt with the lowest problem cost is kept.
    Then the room continues from the kept lines (the reply so far is prefilled).
    Phase B, review (req.review_rounds > 0): the finished room, numbered with every problem, goes back
    to the model, which answers with move / remove edits. An edit stays only if it lowers the room
    cost C (layout_prompt.room_cost); one that does not is undone and the model is told so.

    The code never places anything: every position is the model's. Each change is kept in the
    object's `history` (with its time on the job clock), so the stage and replays show every
    attempt, and the first pass stays on record (summary: first_flagged, fixes_tried, fixed)."""

    def __init__(self, backend: "LayoutBackend", spec: dict, req: JobRequest, ctx: JobContext, seed: int,
                 k: int, count: Callable[[int], bool], publish: Callable[[list[dict]], None]) -> None:
        self.b, self.spec, self.req, self.ctx, self.seed, self.k = backend, spec, req, ctx, seed, k
        self.count, self.publish_fn = count, publish
        self.placed: list[dict] = []          # every object in placement order, removed ones included
        self.lines: dict[int, str] = {}       # object index -> its current line, as the model wrote it
        self.raw: list[str] = []              # every reply, as written, for review
        self.used = 0                         # tokens this room

    # ---- helpers ------------------------------------------------------------------------
    def active(self) -> list[dict]:
        return [o for o in self.placed if o.get("removed") is None]

    def kept(self, objs: Optional[list[dict]] = None) -> list[str]:
        return [self.lines[o["i"]] for o in (self.active() if objs is None else objs)]

    def now(self) -> float:
        return round(self.ctx.clock.now(), 3)

    def publish(self) -> None:
        self.publish_fn([copy.deepcopy(o) for o in self.placed])   # a poll never sees a half-made change

    def stream(self, kind: str, msgs: list[dict], gram: str, max_tokens: int, info: dict) -> Iterator[str]:
        """Complete lines of one model reply. Raises _Cut when the bell rings."""
        if kind != "room" or self.raw:
            self.raw.append(f"\n[{kind}]\n")
        buf = ""
        with closing(self.b._reply(kind, self.spec, self.req, msgs, gram, max_tokens, self.seed, info)) as gen:
            for piece, ntok in gen:
                if not self.count(ntok):
                    raise _Cut
                self.used += ntok
                buf += piece
                self.raw.append(piece)
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    yield line
        if buf.strip():
            yield buf

    @staticmethod
    def _first(o: dict) -> None:
        """Keep the model's first pass, as it appeared, before anything about the object changes."""
        if "history" not in o:
            o["history"] = [{"t": o["t"], **{k: o.get(k) for k in POSE}, "issues": list(o["issues"]), "op": "place"}]

    def _state(self, o: dict, op: str) -> None:
        o["history"].append({"t": self.now(), **{k: o.get(k) for k in POSE}, "issues": list(o["issues"]), "op": op})

    def move(self, o: dict, new: dict, op: str) -> None:
        self._first(o)
        for key in POSE:
            o[key] = new.get(key)
        o["issues"] = check(o, [e for e in self.active() if e["i"] < o["i"]], self.spec)
        self._state(o, op)

    def refresh(self) -> None:
        """Re-check every object against the earlier ones still in the room (a move or removal can
        clear, or cause, a later object's problem)."""
        act = self.active()
        for n, o in enumerate(act):
            issues = check(o, act[:n], self.spec)
            if issues != o["issues"]:
                self._first(o)
                o["issues"] = issues
                self._state(o, "recheck")

    # ---- the room -----------------------------------------------------------------------
    def build(self) -> None:
        spec, req = self.spec, self.req
        while len(self.placed) < spec["max_objects"] and req.max_tokens - self.used > 0:
            if req.layout_prompt == "layoutgpt":    # LayoutGPT's prompt: one request, no continuation
                msgs, gram, parse = layoutgpt_messages(spec), css_grammar(spec), parse_css
            else:
                msgs = continue_messages(spec, self.kept())
                gram = grammar(spec, lines=spec["max_objects"] - len(self.placed), at_least=0 if self.placed else 1)
                parse = parse_line
            again = False
            for line in self.stream("room", msgs, gram, req.max_tokens - self.used, {"done": len(self.placed)}):
                o = parse(line, spec)
                if o is None:
                    continue
                o["issues"] = check(o, self.active(), spec)
                o.update(i=len(self.placed), room=self.k, t=self.now())
                self.placed.append(o)
                self.lines[o["i"]] = line.strip() + "\n"
                self.publish()
                if o["issues"] and req.repairs > 0:
                    self.repair(o)
                    again = True          # the stream was closed: continue the room from the kept lines
                    break
            if not again:
                return

    def repair(self, o: dict) -> None:
        spec = self.spec
        earlier = [e for e in self.active() if e["i"] < o["i"]]
        before = self.kept(earlier)
        first_cost = problem_cost(o, earlier, spec)
        best = (first_cost, self.lines[o["i"]], {k: o.get(k) for k in POSE})
        line = self.lines[o["i"]]
        o["tries"] = 0
        for _ in range(self.req.repairs):
            probs = [txt for _, _, txt in measure(o, earlier, spec)]
            msgs = repair_messages(spec, before, line, o["i"] + 1, probs)
            new = None
            for got in self.stream("repair", msgs, repair_grammar(spec, o["cls"]), REPAIR_TOKENS,
                                   {"cls": o["cls"], "earlier": earlier}):
                new = parse_line(got, spec)
                if new is not None and new["cls"] == o["cls"]:
                    line = got.strip() + "\n"
                    break
                new = None
            o["tries"] += 1
            if new is None:
                break
            self.move(o, new, "repair")
            self.publish()
            cost = problem_cost(o, earlier, spec)
            if cost < best[0]:
                best = (cost, line, {k: o.get(k) for k in POSE})
            if not o["issues"]:
                break
        if {k: o.get(k) for k in POSE} != best[2]:     # a later try was worse: back to the model's best one
            self.move(o, best[2], "best")
            self.publish()
        self.lines[o["i"]] = best[1]
        o["fixes"] = int(best[0] < first_cost)

    def review(self) -> None:
        spec, first = self.spec, len(self.placed)
        notes: list[str] = []
        idle = 0
        for _ in range(self.req.review_rounds):
            self.refresh()
            act = self.active()
            cost = room_cost(act, spec, first)
            if cost <= 0:
                return
            msgs = review_messages(spec, self.kept(act), act, notes)
            notes, kept_any = [], False
            for got in self.stream("review", msgs, edit_grammar(spec, act), REVIEW_TOKENS, {"active": act}):
                e = parse_edit(got, self.active(), spec)
                if e is None:
                    continue
                if e[0] == "done":
                    break
                o = e[1]
                trial = [e[2] if x is o else x for x in self.active() if not (x is o and e[0] == "remove")]
                new_cost = room_cost(trial, spec, first)
                o["tries"] = o.get("tries", 0) + 1
                if new_cost < cost - 1e-6:
                    if e[0] == "move":
                        self.move(o, e[2], "move")
                        self.lines[o["i"]] = line_of(o) + "\n"
                    else:
                        self._first(o)
                        o["removed"] = self.now()
                        self._state(o, "remove")
                    o["fixes"] = o.get("fixes", 0) + 1
                    self.refresh()
                    self.publish()
                    cost, kept_any = new_cost, True
                else:
                    notes.append(f"Your edit \"{got.strip()}\" was undone: it made the room worse.")
            idle = 0 if kept_any else idle + 1
            if idle >= 2:                 # two rounds without a kept edit: the model has no better idea
                return

    def run(self) -> None:
        self.build()
        if self.req.review_rounds > 0 and self.placed:
            self.review()


class LayoutBackend(Backend):
    """Shared loop; subclasses provide _reply()."""
    task = "layout"

    def _reply(self, kind: str, spec: dict, req: JobRequest, msgs: list[dict], gram: str, max_tokens: int,
                seed: int, info: dict) -> Iterator[tuple[str, int]]:
        """One model reply as (text piece, tokens). kind: room | repair | review (the mock uses it)."""
        raise NotImplementedError

    def describe(self, req: JobRequest) -> dict:
        d = {"streams": req.streams, "max_tokens": req.max_tokens, "temperature": req.temperature,
             "room_type": req.room_type, "max_objects": req.max_objects, "quant": self.spec.get("file", ""),
             "prompt": "layoutgpt" if req.layout_prompt == "layoutgpt" else PROMPT_VERSION,
             "catalog": CATALOG.get("sha")}      # which furniture catalogue the prompt was built from
        if req.repairs or req.review_rounds:    # self-correction: disclosed in every run record
            d.update(repairs=req.repairs, review_rounds=req.review_rounds)
        return d

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
                with lock:
                    base = [o for o in r.objects if o.get("room") != k] if i == 0 else []

                def publish(objs: list[dict], base: list[dict] = base) -> None:
                    if i == 0:
                        with lock:
                            r.objects = base + objs     # new list: a poll never sees it half-built
                            r.progress.step = sum(1 for o in objs if o.get("removed") is None)

                room = _Room(self, spec, req, ctx, req.seed + i + k * n, k, count, publish)
                try:
                    room.run()
                except _Cut:
                    return
                path = ctx.output_path(f"-{i}-{k}.json")
                path.write_text(json.dumps({"room": spec, "objects": room.placed, "summary": summarise(room.placed),
                                            "raw": "".join(room.raw)},
                                           indent=1), encoding="utf-8")
                ctx.record_output(path)
                k += 1
                if not req.repeat_until_budget or req.budget_seconds is None:
                    return

        def guarded(i: int) -> None:
            try:
                run(i)
            except BaseException as e:  # noqa: BLE001 — re-raised on the job thread below
                if ctx.should_stop():      # the stream was closed by the stop / bell: not a failure
                    return
                errors.append(e)
                ctx.cancel.set()

        ctx.progress(step=0, steps=spec["max_objects"])
        run_streams(self, ctx, n, guarded)
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
    full prompt read back on the clock.

    Self-correction requests (repair, review, and the room's continuation after a repair) start
    with the same prompt and the reply so far, so the slot's cache covers all but the new turn.
    A continuation sends the reply so far as a final assistant message, which llama-server
    continues (assistant prefill)."""

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
                "slots": self._slots(),   # the warm-up holds the v3.1 prefix; LayoutGPT's examples differ per room
                "prompt_warmed": getattr(self, "warmed", False) and req.layout_prompt != "layoutgpt"}

    def _reply(self, kind: str, spec: dict, req: JobRequest, msgs: list[dict], gram: str, max_tokens: int,
                seed: int, info: dict) -> Iterator[tuple[str, int]]:
        yield from self._chat(msgs, max_tokens, req.temperature, seed, grammar=gram)


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


def _mock_fit(spec: dict, cls: str, placed: list[dict], rng: random.Random) -> Optional[str]:
    """The first clean spot for `cls` among shuffled candidates (wall spots every 10 cm along each
    wall, free spots on a 10 cm grid), as a v3 line; None if nothing fits."""
    poly = spec["floor_plan"]
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    w = FURNITURE[cls][0]
    cands = []
    if cls in WALL_PIECES:
        for name, wl in walls(poly).items():
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
            return line
    return None


def _mock_layout(spec: dict, seed: int) -> list[str]:
    """A plausible room in the v3 line format: each object at the first free spot. Placeholder
    output, never shown as a result."""
    rng = random.Random(seed)
    placed: list[dict] = []
    lines: list[str] = []
    for cls in ROOMS[spec["room_type"]]:
        if len(lines) >= spec["max_objects"]:
            break
        line = _mock_fit(spec, cls, placed, rng)
        if line:
            placed.append(parse_line(line, spec))
            lines.append(line + "\n")
    return lines


class LayoutMock(LayoutBackend):
    """Wall-packed placeholder rooms at the Task 4 mock token rate (~4 characters per token).

    Worker config mock_layout_mistakes: true makes the mock write its first object twice (an
    overlap), so the self-correction path runs without a model: a repair answers with a clean spot
    for the same object, a review moves each flagged object to a clean spot (or removes it)."""

    def load(self) -> None:
        time.sleep(0.3)

    def _drip(self, lines: list[str], req: JobRequest, max_tokens: int, seed: int) -> Iterator[tuple[str, int]]:
        delay = 0.025 * float(self.cfg.get("mock_speed", 1.0)) * (1 + req.streams / 32)
        rng = random.Random(seed)
        sent = 0
        for line in lines:
            for j in range(0, len(line), 4):
                if sent >= max_tokens:
                    return
                time.sleep(delay * rng.uniform(0.8, 1.2))
                sent += 1
                yield line[j:j + 4], 1

    def _reply(self, kind: str, spec: dict, req: JobRequest, msgs: list[dict], gram: str, max_tokens: int,
                seed: int, info: dict) -> Iterator[tuple[str, int]]:
        rng = random.Random(seed)
        if kind == "repair":
            line = _mock_fit(spec, info["cls"], info["earlier"], rng)
            lines = [line + "\n"] if line else []
        elif kind == "review":
            lines, act = [], info["active"]
            for n, o in enumerate(act):
                if not check(o, act[:n], spec):
                    continue
                line = _mock_fit(spec, o["cls"], [x for x in act if x is not o], rng)
                lines.append(f"move {o['i'] + 1} {line.split(' ', 1)[1]}\n" if line else f"remove {o['i'] + 1}\n")
            lines.append("done\n")
        else:                                  # one placeholder room per floor plan: packing is slow-ish
            room = list(mock_layout(spec, 0))
            if self.cfg.get("mock_layout_mistakes") and room:
                room = (room[:1] + room)[: spec["max_objects"]]
            if req.layout_prompt == "layoutgpt":
                room = [css_line(parse_line(x.strip(), spec)) + "\n" for x in room]
            lines = room[info.get("done", 0):]
        yield from self._drip(lines, req, max_tokens, seed)
