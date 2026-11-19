// Stage renderer. Builds the three cells once per round, then updates in place:
// re-creating an <img> or <model-viewer> on every poll would flicker and reload meshes.
"use strict";

const PROTOCOL_LABEL = { iso_time: "ISO-TIME", iso_quality: "ISO-QUALITY" };
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

// ---------------------------------------------------------------- round view ----------
function buildRound(round) {
  const box = document.getElementById("r-cells");
  box.replaceChildren();
  box.style.gridTemplateColumns = `repeat(${round.order.length}, 1fr)`;   // one column per machine
  built = { roundId: round.id, cells: {} };
  for (const mid of round.order) {
    const c = round.cells[mid];
    const root = el("div", "cell");
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
  if (round.task === "i23d" && inp.input) {
    const img = el("img");
    img.src = "/media-input/" + inp.input;
    img.onerror = () => img.remove();
    ibox.append(img, el("span", null, `${inp.label} · ${inp.tier} · from Google Scanned Objects`));
  } else {
    ibox.append(el("span", null, inp.prompt ? `“${inp.prompt}”` : inp.label || inp.id));
  }
  document.getElementById("r-protocol").textContent = PROTOCOL_LABEL[round.protocol] || round.protocol;
  document.getElementById("r-caption").textContent = round.caption || "";
}

function updateCellStatic(round, mid) {
  const c = round.cells[mid];
  const b = built.cells[mid];
  const badge = c.mock && c.mode === "LIVE" ? "MOCK" : c.mode;
  b.badge.className = "badge " + badge;
  b.badge.textContent = badge === "PENDING" ? "…" : c.recorded ? "RECORDED" : badge;
  b.model.replaceChildren(document.createTextNode(c.model_label),
    el("span", "settings", "Expected: " + expectedOutput(round.task, c.settings, round.budget_seconds)));

  const finished = Demo.cellFinished(round, c);
  const total = c.requested || c.settings.batch || 1;
  let res = "", cls = "result";
  if (c.recorded) {
    const what = round.task === "t2i" ? `${c.completed} of ${total} ` : "";
    res = `Recorded run: ${what}finished in ${fmtShort(c.recorded.seconds)}`;
    cls += " recorded";
  } else if (finished) {
    if (c.status === "out_of_memory") { res = "Out of memory"; cls += " oom"; }
    else if (c.status === "ok") res = round.task === "t2i" ? `${c.completed} of ${total} in ${fmtShort(c.final_seconds)}`
                                                            : `Finished in ${fmtShort(c.final_seconds)}`;
    else if (c.status === "budget_exceeded") res = c.completed ? `${c.completed} of ${total} in budget` : "Did not finish in time";
    else if (c.status === "cancelled") res = "Still running";
    else if (c.status === "unavailable") res = "No result";
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
  const n = round.task === "t2i" ? total : 1;
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
      b.clock.textContent = c.mode === "PENDING" ? "—" : fmtClock(t);
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
  if (round.task === "t2i" && p.total) return Math.min(1, (p.done + (p.steps ? p.step / p.steps : 0)) / p.total);
  // 3D has no step count: advance against the rehearsal time, never past 95% until it lands
  if (c.reference && c.reference.seconds) return Math.min(0.95, Demo.elapsed(round) / c.reference.seconds);
  return 0;
}

function stageLine(round, c) {
  if (c.mode === "REPLAY") return "";
  const p = c.progress;
  if (!p) return c.status === "loading" ? "Loading model…" : "";
  if (round.task === "t2i" && p.steps) return `Image ${Math.min(p.done + 1, p.total)} of ${p.total} · step ${p.step}/${p.steps}`;
  return p.stage || "";
}

function headline(round) {
  const parts = [];
  for (const mid of round.order) {
    const c = round.cells[mid];
    const fin = Demo.cellFinished(round, c);
    let v;
    if (c.recorded) {
      v = `recorded run took ${fmtShort(c.recorded.seconds)}`;
    } else if (round.protocol === "iso_quality") {
      v = fin && c.status === "ok" ? fmtShort(c.final_seconds)
        : c.status === "out_of_memory" && fin ? "out of memory" : "still running";
    } else if (round.task === "t2i") {
      v = fin ? `${c.status === "ok" || c.status === "budget_exceeded" ? c.completed : 0} image${c.completed === 1 ? "" : "s"}` : "…";
    } else {
      v = !fin ? "…" : c.status === "ok" ? (c.score && c.score.chamfer_x100 != null ? `error ${c.score.chamfer_x100.toFixed(2)}` : "done")
        : c.status === "out_of_memory" ? "out of memory" : "no mesh";
    }
    parts.push(`${c.label}: ${v}`);
  }
  return parts.join("   ·   ");
}

// ---------------------------------------------------------------- summary view --------
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
      `${h.title} · ${h.input_label} · ${PROTOCOL_LABEL[h.protocol]}${h.budget_seconds ? " " + h.budget_seconds + " s" : ""}`));
    for (const m of cfg.machines) {
      const c = h.cells[m.id];
      if (!c) continue;
      const line = el("div", "sbar");
      line.style.setProperty("--c", m.color);
      line.appendChild(el("div", "who", m.label));
      const track = el("div", "track");
      const ok = c.final_seconds > 0 && c.status !== "unavailable";
      const w = ok ? scale(c.final_seconds) * 78 : 0;     // leave room for the label
      const fill = el("div", "fill");
      fill.style.width = w + "%";
      const val = el("div", "val mono", summaryLabel(c));
      val.style.left = `calc(${w}% + 0.6vw)`;
      track.append(fill, val);
      line.appendChild(track);
      line.title = `${m.label} · ${c.model_label} · ${c.status} · ${fmtClock(c.final_seconds)}`;
      row.appendChild(line);
    }
    rows.appendChild(row);
  }
}

function summaryLabel(c) {
  const badge = c.mock ? "MOCK" : c.recorded ? "RECORDED" : c.mode;
  if (c.recorded) return `${c.requested > 1 ? `${c.completed}/${c.requested} · ` : ""}finished in ${fmtShort(c.recorded.seconds)} · ${c.model_label} · ${badge}`;
  if (c.status === "out_of_memory") return `out of memory · ${badge}`;
  if (c.status === "unavailable") return "no result";
  const count = c.requested > 1 ? `${c.completed}/${c.requested} · ` : "";
  const sc = c.score && c.score.chamfer_x100 != null ? ` · error ${c.score.chamfer_x100.toFixed(2)}` : "";
  const t = c.status === "cancelled" ? `still running at ${fmtShort(c.final_seconds)}`
    : c.status === "budget_exceeded" && !c.completed ? "did not finish" : fmtShort(c.final_seconds);
  return `${count}${t}${sc} · ${c.model_label} · ${badge}`;
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
