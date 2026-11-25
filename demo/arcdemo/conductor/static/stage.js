// Stage renderer. Builds the three cells once per round, then updates in place:
// re-creating an <img> or <model-viewer> on every poll would flicker and reload meshes.
//
// Shared rendering first; then one section per task in show order (Task 1 t2i, 2 i23d, 3 music,
// 4 llm, 8 layout). Each fills STAGE_VIEW[task] with the hooks where it differs; a missing hook
// falls back to the shared behaviour:
//   input(box, inp)              the round header's input (picture, plan, prompt)
//   finishedResult(round, c)     the result line of a finished cell, before the shared wording
//   okResult(c, total)           the result line of a cell that finished ok
//   recordedCount(c, total)      "3 of 4 " in "Recorded run: 3 of 4 finished in …"
//   outputs(b)                   draw the outputs area yourself (true = handled)
//   outputCount(total)           how many outputs the grid lays out
//   progress(round, c, p)        progress bar fraction (undefined = shared)
//   stageLine(round, c, p)       the line under the bar while running (undefined = shared)
//   headline(round, c, fin)      this cell's words in the iso-time headline
//   summaryCount(c)              the count part of a summary bar's label (undefined = shared)
"use strict";

// ---------------------------------------------------------------- shared -----------------
const PROTOCOL_LABEL = { iso_time: "ISO-TIME", iso_quality: "ISO-QUALITY" };
const STAGE_VIEW = {};
const view = (task) => STAGE_VIEW[task] || {};
let built = { roundId: null, cells: {} };
let cfg = null;

fetch("/api/config").then((r) => r.json()).then((c) => {
  cfg = c;
  const box = document.getElementById("title-machines");
  for (const m of c.machines) {
    const s = el("span", null, m.label);
    s.style.setProperty("--c", m.color);
    box.appendChild(s);
  }
});

function showView(name) {
  for (const v of ["title", "round", "summary", "blank"]) {
    document.getElementById("v-" + v).hidden = v !== name;
  }
}

// ---- round view
function buildRound(round) {
  const box = document.getElementById("r-cells");
  box.replaceChildren();
  box.style.gridTemplateColumns = `repeat(${round.order.length}, 1fr)`;   // one column per machine
  built = { roundId: round.id, cells: {} };
  for (const mid of round.order) {
    const c = round.cells[mid];
    const root = el("div", "cell task-" + round.task);
    root.style.setProperty("--c", c.color);
    const head = el("div", "cell-head");
    const who = el("div");
    who.append(el("div", "name", c.label), el("div", "hw", c.hardware));
    const badge = el("span", "badge PENDING", "…");
    head.append(who, badge);
    const model = el("div", "model");
    const clock = el("div", "clock mono", "0.0 s");
    const bar = el("div", "bar");
    const fill = el("i");
    bar.appendChild(fill);
    const stageText = el("div", "stage-text");
    const outputs = el("div", "outputs");
    const result = el("div", "result");
    const score = el("div", "score");
    const note = el("div", "note");
    root.append(head, model, clock, bar, stageText, outputs, result, score, note);
    box.appendChild(root);
    built.cells[mid] = { badge, model, clock, fill, stageText, outputs, result, score, note, shown: new Set() };
  }
  const inp = round.input;
  document.getElementById("r-seg").textContent = round.warmup ? "Warm-up" : round.segment_id.replace("seg", "Segment ");
  document.getElementById("r-title").textContent = round.title;
  const models = [...new Set(round.order.map((m) => round.cells[m].model_label))];
  const tl = document.getElementById("r-task");
  tl.replaceChildren(el("b", null, "Task "), document.createTextNode(TASK_NAME[round.task] || round.task),
    el("b", null, "   ·   Model "), document.createTextNode(models.length === 1 ? `${models[0]}, same on every machine` : models.join(" / ")));
  const ibox = document.getElementById("r-input");
  ibox.replaceChildren();
  ibox.classList.remove("layout-ask");
  const v = view(round.task);
  if (!(v.input && v.input(ibox, inp))) {
    ibox.append(el("span", null, inp.prompt ? `“${inp.prompt}”` : inp.label || inp.id));
  }
  document.getElementById("r-protocol").textContent = PROTOCOL_LABEL[baseProtocol(round.protocol)] || round.protocol;
  document.getElementById("r-caption").textContent = round.caption || "";
}

function updateCellStatic(round, mid) {
  const c = round.cells[mid];
  const b = built.cells[mid];
  const v = view(round.task);
  const finished = Demo.cellFinished(round, c);
  const badge = c.mock && c.mode === "LIVE" ? "MOCK" : c.mode === "LIVE" && finished ? "FINISHED" : c.mode;
  b.badge.className = "badge " + badge;
  b.badge.textContent = badge === "PENDING" ? "…" : c.recorded ? "RECORDED" : badge;
  b.model.replaceChildren(document.createTextNode(c.model_label),
    el("span", "settings", "Expected: " + expectedOutput(round.task, c.settings, round.budget_seconds)));

  const total = c.requested || c.settings.batch || 1;
  let res = "", cls = "result";
  const own = finished && v.finishedResult ? v.finishedResult(round, c) : undefined;
  if (c.recorded) {
    const what = v.recordedCount ? v.recordedCount(c, total) : "";
    res = `Recorded run: ${what}finished in ${fmtShort(c.recorded.seconds)}`;
    cls += " recorded";
  } else if (own !== undefined) {
    res = own;
  } else if (finished) {
    if (c.status === "out_of_memory") { res = "Out of memory"; cls += " oom"; }
    else if (c.status === "ok") res = v.okResult ? v.okResult(c, total) : `Finished in ${fmtShort(c.final_seconds)}`;
    else if (c.status === "budget_exceeded") res = c.completed ? `${c.completed} of ${total} in budget` : "Did not finish in time";
    else if (c.status === "cancelled") res = "Still running";
    else if (c.status === "unavailable" || c.status === "error") res = noResult(c);
  } else if (c.status === "cancelled") {
    res = "Still running";
  }
  b.result.className = cls;
  b.result.textContent = res;
  const ref = c.reference && !finished && c.reference.seconds
    ? `Rehearsal: finished in ${fmtShort(c.reference.seconds)}` : "";
  b.note.textContent = [ref, c.mode === "REPLAY" && !c.recorded ? "Recorded in rehearsal" : "", finished ? c.note : ""]
    .filter(Boolean).join(" · ");

  if (c.score && c.score.chamfer_x100 != null && finished) {
    b.score.replaceChildren(document.createTextNode("Error vs scan  "), el("b", null, c.score.chamfer_x100.toFixed(2)),
      document.createTextNode(`  ·  F-score ${c.score.fscore.toFixed(2)}`));
  } else {
    b.score.textContent = c.score && c.score.pending ? "Scoring against the scan…" : "";
  }

  // outputs: append only what is new and already revealed
  const vis = Demo.visibleOutputs(round, c);
  if (c.mode === "REPLAY" && b.outputs.dataset.mode !== "REPLAY") {   // switched mid-round
    b.outputs.replaceChildren();
    b.shown.clear();
  }
  b.outputs.dataset.mode = c.mode;
  if (v.outputs && v.outputs(b)) return;      // llm text, layout scene: drawn per frame in tick()
  const n = v.outputCount ? v.outputCount(total) : 1;
  b.outputs.className = "outputs n" + (n >= 8 ? 8 : n >= 3 ? 4 : n);
  for (const o of vis) {
    if (b.shown.has(o.url)) continue;
    b.shown.add(o.url);
    if (o.url.endsWith(".glb")) {
      const mv = document.createElement("model-viewer");
      mv.setAttribute("src", o.url);
      mv.setAttribute("auto-rotate", "");
      mv.setAttribute("camera-controls", "");
      mv.setAttribute("interaction-prompt", "none");
      mv.setAttribute("rotation-per-second", "30deg");
      mv.setAttribute("shadow-intensity", "0.6");
      mv.setAttribute("exposure", "1.1");
      b.outputs.appendChild(mv);
    } else if (o.url.endsWith(".wav")) {
      if (!b.outputs.querySelector("audio")) {
        const a = el("audio");
        a.controls = true;
        a.src = o.url;
        b.outputs.appendChild(a);
      }
    } else {
      const img = el("img");
      img.src = o.url;
      b.outputs.appendChild(img);
    }
  }
}

// Per frame: clocks, progress, budget countdown, headline.
function tick() {
  const s = Demo.state;
  if (s && s.view === "round" && s.round && built.roundId === s.round.id) {
    const round = s.round;
    for (const mid of round.order) {
      const c = round.cells[mid];
      const b = built.cells[mid];
      const t = Demo.cellClock(round, c);
      const finished = Demo.cellFinished(round, c);
      if (b.text) drawText(round, c, b);
      if (b.layout) { b.layoutImpl.draw(b.layout, round, c); drawConsole(round, c, b, finished); }
      if (b.layout && throughput(round, c) && c.mode !== "PENDING") {   // layout iso-time: rooms finished
        const n = c.mode === "REPLAY" && !c.recorded ? Demo.visibleOutputs(round, c).length : c.completed;
        b.clock.replaceChildren(document.createTextNode(fmtInt(n)), el("small", null, n === 1 ? " room" : " rooms"));
      } else if (b.text && round.budget_seconds && c.mode !== "PENDING") {   // llm iso-time: the clocks tie, tokens differ
        b.clock.replaceChildren(document.createTextNode(fmtInt(b.tokens)), el("small", null, " tokens"));
      } else {
        b.clock.textContent = c.mode === "PENDING" ? "—" : fmtClock(t);
      }
      b.clock.classList.toggle("done", finished && c.status === "ok");
      b.fill.style.width = (100 * progressFraction(round, c, finished)).toFixed(1) + "%";
      b.stageText.textContent = finished ? "" : stageLine(round, c);
    }
    // Header clock: countdown for iso-time, elapsed for iso-quality; frozen once the round is over.
    const bud = document.getElementById("r-budget");
    const t = round.done ? round.finished_at - round.started_at : Demo.elapsed(round);
    if (round.budget_seconds) {
      const left = round.budget_seconds - t;
      bud.replaceChildren(document.createTextNode(left > 0 ? fmtClock(left) : "Time"), el("small", null, left > 0 ? "left" : ""));
      bud.classList.toggle("over", left <= 0 || round.done);
    } else {
      bud.replaceChildren(document.createTextNode(fmtClock(t)), el("small", null, "elapsed"));
      bud.classList.toggle("over", round.done);
    }
    document.getElementById("r-headline").textContent = headline(round);
  }
  requestAnimationFrame(tick);
}

function progressFraction(round, c, finished) {
  if (finished) return 1;
  if (c.mode === "REPLAY" && c.final_seconds) return Math.min(1, Demo.elapsed(round) / c.final_seconds);
  const p = c.progress;
  if (!p) return 0;
  const v = view(round.task);
  const own = v.progress ? v.progress(round, c, p) : undefined;
  if (own !== undefined) return own;
  // no step count (3D, music): advance against the rehearsal time, never past 95% until it lands
  if (c.reference && c.reference.seconds) return Math.min(0.95, Demo.elapsed(round) / c.reference.seconds);
  return 0;
}

function stageLine(round, c) {
  if (c.mode === "REPLAY") return "";
  const p = c.progress;
  if (!p) return c.status === "loading" ? "Loading model…" : "";
  const v = view(round.task);
  const own = v.stageLine ? v.stageLine(round, c, p) : undefined;
  return own !== undefined ? own : p.stage || "";
}

// A cell that ended without output: say why in a few words, never a Python error.
function noResult(c) {
  if (c.missing_model) return "Model not on this machine";
  return c.status === "error" ? "Could not run" : "No result";
}

function headline(round) {
  const parts = [];
  const v = view(round.task);
  for (const mid of round.order) {
    const c = round.cells[mid];
    const fin = Demo.cellFinished(round, c);
    let words;
    if (c.recorded) {
      words = `recorded run took ${fmtShort(c.recorded.seconds)}`;
    } else if (fin && (c.status === "unavailable" || c.status === "error")) {
      words = noResult(c).toLowerCase();
    } else if (baseProtocol(round.protocol) === "iso_quality") {
      words = fin && c.status === "ok" ? fmtShort(c.final_seconds)
        : c.status === "out_of_memory" && fin ? "out of memory" : "still running";
    } else if (v.headline) {
      words = v.headline(round, c, fin);
    } else {
      words = !fin ? "…" : c.status === "ok" ? "done" : c.status === "out_of_memory" ? "out of memory" : "no result";
    }
    parts.push(`${c.label}: ${words}`);
  }
  return parts.join("   ·   ");
}

// ---- summary view
function renderSummary(history) {
  const rows = document.getElementById("s-rows");
  rows.replaceChildren();
  const times = history.flatMap((h) => Object.values(h.cells).map((c) => c.final_seconds).filter((t) => t > 0));
  const max = Math.max(10, ...times) * 1.15;
  const scale = (t) => Math.max(0.02, Math.log10(Math.max(1, t)) / Math.log10(max));   // log, 1 s .. max
  // latest run per (segment, input): a re-run replaces the earlier bar instead of stacking
  const latest = new Map();
  for (const h of history) { latest.delete(h.segment_id + "/" + h.input_id); latest.set(h.segment_id + "/" + h.input_id, h); }
  for (const h of [...latest.values()].slice(-6)) {
    const row = el("div", "srow");
    row.appendChild(el("div", "label",
      `${h.title} · ${h.input_label} · ${PROTOCOL_LABEL[baseProtocol(h.protocol)]}${h.budget_seconds ? " " + h.budget_seconds + " s" : ""}`));
    for (const m of cfg.machines) {
      const c = h.cells[m.id];
      if (!c) continue;
      const line = el("div", "sbar");
      line.style.setProperty("--c", m.color);
      line.appendChild(el("div", "who", m.label));
      const track = el("div", "track");
      const ok = c.final_seconds > 0 && c.status !== "unavailable" && c.status !== "error";
      const w = ok ? scale(c.final_seconds) * 78 : 0;     // leave room for the label
      const fill = el("div", "fill");
      fill.style.width = w + "%";
      const val = el("div", "val mono", summaryLabel(c, h.task));
      val.style.left = `calc(${w}% + 0.6vw)`;
      track.append(fill, val);
      line.appendChild(track);
      line.title = `${m.label} · ${c.model_label} · ${c.status} · ${fmtClock(c.final_seconds)}`;
      row.appendChild(line);
    }
    rows.appendChild(row);
  }
}

function summaryLabel(c, task) {
  const badge = c.mock ? "MOCK" : c.recorded ? "RECORDED" : c.mode;
  if (c.recorded) return `${c.requested > 1 ? `${c.completed}/${c.requested} · ` : ""}finished in ${fmtShort(c.recorded.seconds)} · ${c.model_label} · ${badge}`;
  if (c.status === "out_of_memory") return `out of memory · ${badge}`;
  if (c.status === "unavailable" || c.status === "error") return noResult(c).toLowerCase();
  const v = view(task);
  const own = v.summaryCount ? v.summaryCount(c) : undefined;
  const count = own !== undefined ? own
    : c.tokens ? tokenText(c.tokens, c.tokens_per_second) + " · "
    : c.requested > 1 ? `${c.completed}/${c.requested} · ` : "";
  const sc = c.score && c.score.chamfer_x100 != null ? ` · error ${c.score.chamfer_x100.toFixed(2)}` : "";
  const t = c.status === "cancelled" ? `still running at ${fmtShort(c.final_seconds)}`
    : c.status === "budget_exceeded" && !c.completed ? "did not finish" : fmtShort(c.final_seconds);
  return `${count}${t}${sc} · ${c.model_label} · ${badge}`;
}

// ---------------------------------------------------------------- Task 1: text to image ----
STAGE_VIEW.t2i = {
  recordedCount: (c, total) => `${c.completed} of ${total} `,
  okResult: (c, total) => `${c.completed} of ${total} in ${fmtShort(c.final_seconds)}`,
  outputCount: (total) => total,
  progress: (round, c, p) => (p.total ? Math.min(1, (p.done + (p.steps ? p.step / p.steps : 0)) / p.total) : undefined),
  stageLine: (round, c, p) => (p.steps ? `Image ${Math.min(p.done + 1, p.total)} of ${p.total} · step ${p.step}/${p.steps}` : undefined),
  headline: (round, c, fin) => (fin ? `${c.status === "ok" || c.status === "budget_exceeded" ? c.completed : 0} image${c.completed === 1 ? "" : "s"}` : "…"),
};

// ---------------------------------------------------------------- Task 2: image to 3D ------
STAGE_VIEW.i23d = {
  input(box, inp) {
    if (!(inp.input || inp.input_url)) return false;
    const img = el("img");
    img.src = inp.input_url || "/media-input/" + inp.input;     // input_url: an operator's own picture
    img.onerror = () => img.remove();
    box.append(img, el("span", null, inp.tier === "upload" ? `${inp.label} · a picture brought today`
      : `${inp.label} · ${inp.tier} · from Google Scanned Objects`));
    return true;
  },
  headline: (round, c, fin) => (!fin ? "…" : c.status === "ok" ? (c.score && c.score.chamfer_x100 != null ? `error ${c.score.chamfer_x100.toFixed(2)}` : "done")
    : c.status === "out_of_memory" ? "out of memory" : "no mesh"),
};

// ---------------------------------------------------------------- Task 3: music ------------
// Shared behaviour throughout: an <audio> player per cell, the shared headline and result.
STAGE_VIEW.music = {};

// ---------------------------------------------------------------- Task 4: language model ---
STAGE_VIEW.llm = {
  finishedResult(round, c) {
    if (c.status === "out_of_memory" || !c.tokens) return undefined;
    return baseProtocol(round.protocol) === "iso_quality" ? `Finished in ${fmtShort(c.final_seconds)} · ${tpsText(c.tokens_per_second)}`
      : tokenText(c.tokens, c.tokens_per_second);
  },
  outputs(b) {   // text, not files: drawn per frame in tick() so replays type out
    if (!b.text || !b.text.isConnected) { b.text = el("div", "llm-text"); b.outputs.replaceChildren(b.text); }
    b.outputs.className = "outputs n1";
    return true;
  },
  progress: (round, c) => (round.budget_seconds ? Math.min(1, Demo.elapsed(round) / round.budget_seconds)
    : Math.min(1, (c.tokens || 0) / ((c.settings.max_tokens || 256) * (c.settings.streams || 1)))),
  stageLine: (round, c, p) => `${p.stage} · ${tokenText(c.tokens, c.tokens_per_second)}`,
  headline: (round, c, fin) => (fin ? (c.status === "out_of_memory" ? "out of memory" : tpsText(c.tokens_per_second) || "no tokens") : "…"),
};

// Live: the worker's current reply. Replay: the recorded reply types out over its recorded time.
function drawText(round, c, b) {
  const full = c.text || "";
  const frac = c.mode === "REPLAY" && !c.recorded && c.final_seconds ? Math.min(1, Demo.elapsed(round) / c.final_seconds) : 1;
  const txt = full.slice(0, Math.round(full.length * frac));
  if (b.text.textContent !== txt) {
    b.text.textContent = txt;
    b.text.scrollTop = b.text.scrollHeight;
  }
  b.tokens = Math.round((c.tokens || 0) * frac);
}

// ---------------------------------------------------------------- Task 8: scene construction
STAGE_VIEW.layout = {
  input(box, inp) {   // what the model is asked: the plan with its wall names, in words
    box.classList.add("layout-ask");
    const txt = el("div", "ask");
    txt.append(el("div", "ask-title", inp.label), el("div", "ask-line", inp.ask || ""),
      el("div", "ask-note", "The model sees the walls by these names and writes one line per object."));
    box.append(planSvg(inp), txt);
    return true;
  },
  finishedResult(round, c) {
    if (c.status === "out_of_memory" || !(c.objects || []).length) return undefined;
    const objs = Layout.visible(round, c);
    return throughput(round, c) ? `${roomsText(c.completed)} in ${round.budget_seconds} s`
                                : `Finished in ${fmtShort(c.final_seconds)} · ${Layout.tally(objs)}`;
  },
  outputs(b) {   // objects, not files: drawn per frame in tick() so replays build up
    // scene on the left, the model's own lines on the right
    if (!b.layoutWrap || !b.layoutWrap.isConnected) {
      b.layoutWrap = el("div", "layout-wrap");
      b.scene = el("div", "layout-scene");
      b.console = el("div", "layout-console");
      b.layoutWrap.append(b.scene, b.console);
      b.outputs.replaceChildren(b.layoutWrap);
      b.layout = null;
      b.consoleKey = null;
    }
    // re-attach when the 3D view (a module, loaded after this script) takes over from the canvas one
    if (!b.layout || b.layoutImpl !== Layout) {
      b.layoutImpl = Layout;
      b.layout = Layout.attach(b.scene);
    }
    b.outputs.className = "outputs n1";
    return true;
  },
  progress: (round, c, p) => (round.budget_seconds ? Math.min(1, Demo.elapsed(round) / round.budget_seconds)
    : Math.min(1, (p.step || 0) / (p.steps || 10))),
  stageLine: (round, c, p) => (throughput(round, c) ? `${p.stage} · ${fmtInt(c.completed)} finished so far`
    : `Placing object ${(p.step || 0) + 1}${p.steps ? ` of up to ${p.steps}` : ""} · ${tpsText(c.tokens_per_second)}`),
  headline: (round, c, fin) => (!fin ? "…" : c.status === "out_of_memory" ? "out of memory"
    : throughput(round, c) ? roomsText(c.completed) : Layout.tally(Layout.visible(round, c))),
  summaryCount: (c) => (!c.layout ? undefined
    : c.requested === 0 ? `${roomsText(c.completed)} · ` : `${c.layout.objects} objects, ${c.layout.flagged} flagged`
      + (c.layout.fixes != null ? ` (first pass ${c.layout.first_flagged}, ${c.layout.fixes} fixed)` : "") + " · "),
};

// Layout throughput run (rooms repeated until the bell), as opposed to one room per machine.
function throughput(round, c) { return round.task === "layout" && !!round.budget_seconds && !!c.settings.repeat_until_budget; }
function roomsText(n) { return `${fmtInt(n || 0)} room${n === 1 ? "" : "s"} furnished`; }

// The lines the model has written so far, numbered like the labels in the scene; red when the room
// checks flag the object. Rebuilt only when something changes.
function drawConsole(round, c, b, finished) {
  const objs = Layout.visible(round, c);
  const writing = !finished && c.mode !== "PENDING" && !c.recorded;
  const key = `${objs.map((o) => `${o.i}:${layoutLine(o)}:${(o.issues || []).length}`).join(",")}|${writing}|${c.mode}|${c.status}|${objs.length ? objs[0].room || 0 : 0}`;
  if (key === b.consoleKey) return;
  b.consoleKey = key;
  const body = el("div", "con-body");
  objs.forEach((o, k) => {
    const bad = o.issues && o.issues.length;
    const line = el("div", "con-line" + (bad ? " bad" : "") + (k === objs.length - 1 && writing ? " new" : ""));
    line.append(el("span", "n", String(o.i + 1)), el("span", "t", layoutLine(o)));
    if (bad) line.appendChild(el("span", "why", " ← " + o.issues.join(", ") + (o.fixing ? " · fixing" : "")));
    else if (o.fixing) line.appendChild(el("span", "fix", " ↻ fixed by the model"));
    body.appendChild(line);
  });
  if (writing) body.appendChild(el("div", "con-line caret", c.status === "loading" ? "loading the model…" : "▍"));
  if (!objs.length && !writing) body.appendChild(el("div", "con-line", "—"));
  b.console.replaceChildren(el("div", "con-head", "What the model wrote"), body);
}

// ---------------------------------------------------------------- wiring --------------
let lastHistoryLen = -1;
Demo.onState((s) => {
  showView(s.view === "round" && !s.round ? "title" : s.view);
  if (s.view === "round" && s.round) {
    if (built.roundId !== s.round.id) buildRound(s.round);
    for (const mid of s.round.order) updateCellStatic(s.round, mid);
  }
  if (s.view === "summary" && cfg && s.history.length !== lastHistoryLen) {
    lastHistoryLen = s.history.length;
    renderSummary(s.history);
  }
});
Demo.poll(300);
requestAnimationFrame(tick);
