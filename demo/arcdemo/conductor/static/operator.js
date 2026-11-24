// Operator console. Every button maps to one conductor endpoint; errors are shown inline.
// Labels are written for the operator on the night, not for the developer: no segment ids,
// protocol names or input ids on buttons.
"use strict";

let cfg = null;
let segId = null;
let spares = [];          // pre-vetted spare prompts (t2i), typed as a custom prompt
let planSeq = 0;          // ignore plan answers that arrive after a newer selection

const VIEW_NAMES = [["title", "Title screen"], ["round", "Live results"], ["summary", "Summary chart"], ["blank", "Blank screen"]];

async function api(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

function say(id, text, cls) {
  const e = document.getElementById(id);
  e.textContent = text;
  e.className = "msg " + (cls || "");
}

const seg = () => cfg.segments.find((s) => s.id === segId);
const partName = (s, i) => `Part ${i + 1} · ${s.title}${s.warmup ? " (warm-up)" : ""}`;

// ---------------------------------------------------------------- choosing --------------
function selectSegment(id) {
  segId = id;
  const s = seg();
  document.querySelectorAll("#segments button").forEach((b) => b.classList.toggle("sel", b.dataset.id === id));
  fillExamples(s);
  onChoice();
}

// The dropdown: this part's planned inputs first, then the task's other examples, then
// (text-to-image only) the spare prompts and a free-text option.
function fillExamples(s) {
  const sel = document.getElementById("example");
  sel.replaceChildren();
  const all = (cfg.inputs[s.task] || []).filter((x) => x.id !== "live" && !x._todo);
  const describe = (x) => {
    if (s.task === "i23d") return `${x.label}${x.tier ? " (" + x.tier + ")" : ""}`;
    return x.prompt ? `${x.label} — “${x.prompt}”` : x.label || x.id;
  };
  const group = (label, items) => {
    if (!items.length) return;
    const g = el("optgroup");
    g.label = label;
    for (const [value, text] of items) {
      const o = el("option", null, text);
      o.value = value;
      g.appendChild(o);
    }
    sel.appendChild(g);
  };
  const planned = all.filter((x) => s.inputs.includes(x.id));
  const others = all.filter((x) => !s.inputs.includes(x.id));
  group("Planned for this part", planned.map((x) => ["input:" + x.id, describe(x)]));
  group("Other examples (rehearsed)", others.map((x) => ["input:" + x.id, describe(x)]));
  const canType = (cfg.inputs[s.task] || []).some((x) => x.id === "live");
  if (canType) {
    group("Spare prompts", spares.map((p, i) => ["spare:" + i, p]));
    group("Audience prompt", [["custom", "✎ Type a new prompt…"]]);
  }
  sel.value = s.inputs.includes("live") ? "custom" : sel.options[0] ? sel.options[0].value : "";
}

function choice() {
  const v = document.getElementById("example").value || "";
  if (v.startsWith("input:")) return { inputId: v.slice(6) };
  if (v.startsWith("spare:")) return { inputId: "live", prompt: spares[Number(v.slice(6))] };
  if (v === "custom") return { inputId: "live", prompt: null };
  return { inputId: null };
}

function onChoice() {
  const c = choice();
  const live = c.inputId === "live";
  document.getElementById("live-box").hidden = !live;
  if (live && c.prompt != null) document.getElementById("prompt").value = c.prompt;
  say("mod-msg", "");
  loadPlan();
}

// ---------------------------------------------------------------- "what will run" -------
async function loadPlan() {
  const s = seg();
  const c = choice();
  const my = ++planSeq;
  const q = new URLSearchParams({ segment_id: s.id });
  if (c.inputId) q.set("input_id", c.inputId);
  let plan;
  try {
    plan = await (await fetch("/api/plan?" + q)).json();
  } catch (e) {
    return;
  }
  if (my !== planSeq) return;
  renderPlan(s, c, plan);
}

// Task 8 and Task 4: the exact text the model is sent, so the operator can show or explain it.
async function showPrompt(s, c) {
  const dt = document.getElementById("p-prompt-dt"), dd = document.getElementById("p-prompt-dd");
  const show = (s.task === "layout" || s.task === "llm") && c.inputId && c.inputId !== "live";
  dt.hidden = dd.hidden = !show;
  if (!show) return;
  const pre = document.getElementById("p-prompt");
  pre.textContent = "loading…";
  try {
    const r = await (await fetch(`/api/prompt?segment_id=${encodeURIComponent(s.id)}&input_id=${encodeURIComponent(c.inputId)}`)).json();
    pre.textContent = r.text || "—";
  } catch (e) {
    pre.textContent = "could not load the prompt";
  }
}

// The plan thumbnail needs the wall names, which the conductor derives (same as the stage header).
async function loadLayoutView(s, c, box) {
  try {
    const v = await (await fetch(`/api/layout_view?input_id=${encodeURIComponent(c.inputId)}`)).json();
    box.appendChild(planSvg(v));
    box.appendChild(el("div", "muted", v.ask));
  } catch (e) { /* the label alone is enough */ }
}

function renderPlan(s, c, plan) {
  const budget = plan.budget_seconds;
  document.getElementById("p-task").textContent = TASK_NAME[s.task] || s.task;
  renderModelPicker(s, plan);

  const inp = document.getElementById("p-input");
  inp.replaceChildren();
  const item = (cfg.inputs[s.task] || []).find((x) => x.id === c.inputId) || {};
  if (c.inputId === "live") {
    inp.textContent = "The prompt typed below (live — no recording exists for it)";
  } else if (s.task === "i23d" && item.input) {
    inp.append(document.createTextNode(`Photo of a ${item.label.toLowerCase()}`));
    const img = el("img", "input-img");
    img.src = "/media-input/" + item.input;
    img.onerror = () => img.remove();
    inp.appendChild(img);
  } else if (s.task === "layout" && item.floor_plan) {
    inp.append(document.createTextNode(item.label));
    loadLayoutView(s, c, inp);
  } else {
    inp.textContent = item.prompt ? `“${item.prompt}”` : item.label || "—";
  }
  showPrompt(s, c);
  document.getElementById("p-protocol").textContent = protocolText(plan.protocol, budget);

  const ul = document.getElementById("p-expected");
  ul.replaceChildren();
  for (const m of plan.machines) {
    const li = el("li");
    const color = (cfg.machines.find((x) => x.id === m.id) || {}).color;
    if (color) li.style.setProperty("--c", color);
    li.append(el("b", null, m.label + ": "), document.createTextNode(expectedOutput(s.task, m.settings, budget)));
    li.appendChild(el("div", "rec " + recordingClass(m, c), recordingText(m, c, budget)));
    ul.appendChild(li);
  }
}

// One dropdown per machine: the planned model first, then every other model the config offers
// for this task (model_choices). A pick holds for every part of the same task until changed back.
function renderModelPicker(s, plan) {
  const dd = document.getElementById("p-model");
  dd.replaceChildren();
  const label = (k) => (cfg.model_labels && cfg.model_labels[k]) || k;
  const options = (cfg.model_options && cfg.model_options[s.task]) || [];
  const models = [...new Set(plan.machines.map((m) => m.model))];
  if (models.length === 1) dd.appendChild(el("div", "muted", "The same on every machine"));
  else dd.appendChild(el("div", "warn", "Different models on the machines — the comparison is not like for like"));
  for (const m of plan.machines) {
    const row = el("div", "model-pick");
    const sel = el("select");
    sel.title = `AI model ${m.label} runs for ${TASK_NAME[s.task] || s.task}`;
    for (const k of [m.planned_model, ...options.filter((k) => k !== m.planned_model)]) {
      const missing = (m.on_disk || {})[k] === false;    // null = worker down: not known, allowed
      const o = el("option", null, (k === m.planned_model ? `${label(k)} (planned)` : label(k))
        + (missing ? " — not on this machine" : ""));
      o.value = k;
      o.disabled = missing && k !== m.model;
      sel.appendChild(o);
    }
    sel.value = m.model;
    sel.onchange = async () => {
      try {
        await api("/api/model", { task: s.task, machine: m.id, model: sel.value === m.planned_model ? null : sel.value });
        say("run-msg", `${m.label} now runs ${label(sel.value)}. Press “Load the AI model” again before starting.`, "warn");
      } catch (e) {
        say("run-msg", e.message, "down");
      }
      loadPlan();
    };
    row.append(el("span", null, m.label), sel);
    dd.appendChild(row);
  }
}

function recordingText(m, c, budget) {
  if (c.inputId === "live") return m.mode === "replay" ? "Planned as a recording — no recording exists for a new prompt" : "";
  const r = m.recording;
  const plannedRec = m.mode === "replay" ? "Shows the recording (planned). " : "";
  if (!r) {
    return plannedRec + (m.full_recording
      ? "No full recording on file: if it runs out of time it will just say “did not finish”."
      : "No recording on file: if this machine fails there is nothing to fall back to.");
  }
  const took = fmtClock(r.seconds);
  const stale = r.stale ? " (settings changed since — re-record)" : "";
  if (m.full_recording) {
    const late = budget && r.seconds > budget ? ` — longer than the ${budget} s limit, so the recording will be shown when time is up` : "";
    return `${plannedRec}Recording on file: finished in ${took}${late}${stale}`;
  }
  return `${plannedRec}Recording on file: ${r.status === "ok" ? "finished in " + took : r.status.replace(/_/g, " ")}${stale}`;
}

function recordingClass(m, c) {
  if (c.inputId === "live") return "muted";
  if (!m.recording) return "warn";
  return m.recording.stale ? "warn" : "muted";
}

// ---------------------------------------------------------------- start / stop ---------
async function start() {
  const c = choice();
  if (!c.inputId) return say("run-msg", "Choose an example first.", "down");
  const force = [...document.querySelectorAll("#force input:checked")].map((x) => x.value);
  const budget = document.getElementById("budget").value;
  const body = { segment_id: segId, input_id: c.inputId, force_replay: force,
    budget_override: budget ? Number(budget) : null };
  if (c.inputId === "live") {
    body.prompt = document.getElementById("prompt").value;
    body.override = document.getElementById("override").checked;
  }
  try {
    await api("/api/round", body);
    say("run-msg", "Started. Watch the projector preview.", "up");
  } catch (e) {
    say("run-msg", e.message, "down");
  }
}

// ---------------------------------------------------------------- live state -----------
const STATE_WORDS = { queued: "Waiting to start", loading: "Loading the model", running: "Working",
  ok: "Finished", budget_exceeded: "Time up", out_of_memory: "Out of memory", cancelled: "Stopped",
  error: "Failed", unavailable: "No result (no recording on file)" };

function cellWords(round, c) {
  if (c.recorded) return `Showing the recording — took ${fmtClock(c.recorded.seconds)} to finish`;
  if (c.mode === "PENDING") return "Starting…";
  const w = STATE_WORDS[c.status] || c.status;
  return c.mode === "REPLAY" ? `Showing the recording · ${w.toLowerCase()}` : `Live · ${w}`;
}

function renderHealth(s) {
  const tb = document.getElementById("health");
  tb.replaceChildren();
  const labels = Object.fromEntries((cfg ? cfg.machines : []).map((m) => [m.id, m.label]));
  for (const [key, h] of Object.entries(s.health).sort()) {
    const [mid, task] = key.split(":");
    const tr = el("tr");
    const state = !h.ok ? `Not reachable (${h.error})` : h.busy ? "Working" : h.loading ? "Loading model…" : "Ready";
    tr.append(el("td", null, `${labels[mid] || mid} · ${TASK_NAME[task] || task}`),
      el("td", h.ok ? "up" : "down", state + (h.mock ? " · practice mode" : "")),
      el("td", null, (h.loaded || []).map((k) => (cfg && cfg.model_labels && cfg.model_labels[k]) || k).join(", ") || "—"),
      el("td", null, h.env_hash || ""),
      el("td", null, h.latency_ms != null ? h.latency_ms + " ms" : ""));
    tb.appendChild(tr);
  }
  document.getElementById("warnings").textContent = (s.warnings || []).join("\n");
  const PREP = { loading: "loading…", ready: "ready", error: "could not load", skipped: "uses the recording, nothing to load",
    missing: "not downloaded" };
  const prep = Object.entries(s.prepare || {}).map(([m, p]) =>
    `${labels[m] || m}: ${PREP[p.status] || p.status}${p.seconds ? " (" + fmtShort(p.seconds) + ")" : ""}${p.message ? " — " + p.message : ""}`);
  document.getElementById("prepare-state").textContent = prep.join("  |  ");
}

function renderCells(s) {
  const box = document.getElementById("cellctl");
  const info = document.getElementById("round-info");
  const r = s.round;
  const running = !!(r && !r.done);
  document.getElementById("cancel").disabled = !running;
  document.getElementById("start").disabled = running;
  if (!r) { box.replaceChildren(); info.textContent = "Nothing has run yet."; return; }
  const what = r.input.prompt ? `“${r.input.prompt}”` : r.input.label || r.input.id;
  info.textContent = `${r.title} · ${TASK_NAME[r.task] || r.task} · ${what} · ${running ? "running" : "finished"}`;
  // Build the rows once per round and update them in place, so a click on a button is never
  // lost to a re-render between mouse-down and mouse-up.
  if (box.dataset.round !== r.id) {
    box.replaceChildren();
    box.dataset.round = r.id;
    for (const mid of r.order) {
      const c = r.cells[mid];
      const line = el("div", "cell");
      line.dataset.mid = mid;
      const txt = el("div");
      txt.append(el("div", "who", c.label), el("div", "what"), el("div", "what note"));
      const stop = el("button", "danger stop", "■ Stop");
      stop.title = `Stop ${c.label} now`;
      stop.onclick = () => api(`/api/cell/${mid}/stop`).catch((e) => say("run-msg", e.message, "down"));
      const rec = el("button", "rec", "Show recording");
      rec.title = `Stop ${c.label} and show its rehearsal recording instead`;
      rec.onclick = () => api(`/api/cell/${mid}/replay`).catch((e) => say("run-msg", e.message, "down"));
      line.append(txt, stop, rec);
      box.appendChild(line);
    }
  }
  for (const line of box.children) {
    const c = r.cells[line.dataset.mid];
    const total = c.requested || (r.task === "t2i" ? c.settings.batch : null);
    const count = total ? ` · ${c.completed} of ${total} done` : "";
    const [state, note] = line.querySelectorAll(".what");
    state.textContent = `${cellWords(r, c)} · ${fmtClock(Demo.cellClock(r, c))}${count}`;
    note.textContent = c.note || "";
    const busy = c.mode !== "REPLAY" && !["ok", "budget_exceeded", "out_of_memory", "cancelled", "error", "unavailable"].includes(c.status);
    line.querySelector(".stop").disabled = !running || !busy;
    line.querySelector(".rec").disabled = c.mode === "REPLAY";
  }
}

// ---------------------------------------------------------------- wiring ---------------
Promise.all([
  fetch("/api/config").then((r) => r.json()),
  fetch("/media-input/data/t2i/live_fallback_prompts.json").then((r) => (r.ok ? r.json() : { prompts: [] })).catch(() => ({ prompts: [] })),
]).then(([c, fb]) => {
  cfg = c;
  spares = fb.prompts || [];
  const segs = document.getElementById("segments");
  c.segments.forEach((s, i) => {
    const b = el("button", null, partName(s, i));
    b.dataset.id = s.id;
    b.title = s.caption ? s.caption.replace("{budget}", s.budget_seconds ?? "") : "";
    b.onclick = () => selectSegment(s.id);
    segs.appendChild(b);
  });
  const force = document.getElementById("force");
  for (const m of c.machines) {
    const lab = el("label");
    const cb = el("input");
    cb.type = "checkbox";
    cb.value = m.id;
    lab.append(cb, document.createTextNode(" " + m.label + "  "));
    force.appendChild(lab);
  }
  const views = document.getElementById("views");
  for (const [v, name] of VIEW_NAMES) {
    const b = el("button", null, name);
    b.onclick = () => api("/api/view", { view: v });
    views.appendChild(b);
  }
  // open on a part / input from the URL, e.g. /operator#seg=seg8&input=lay-02
  const h = new URLSearchParams(location.hash.slice(1));
  const start = c.segments.find((x) => x.id === h.get("seg")) || c.segments[0];
  selectSegment(start.id);
  if (h.get("input") && document.querySelector(`#example option[value="input:${h.get("input")}"]`)) {
    document.getElementById("example").value = "input:" + h.get("input");
    onChoice();
  }
  Demo.onState((s) => { renderHealth(s); renderCells(s); });
  Demo.poll(500);
});

document.getElementById("example").onchange = onChoice;
document.getElementById("start").onclick = start;
document.getElementById("prepare").onclick = () =>
  api("/api/prepare", { segment_id: segId }).then(() => say("run-msg", "Loading the model on every machine…"))
    .catch((e) => say("run-msg", e.message, "down"));
document.getElementById("cancel").onclick = () =>
  api("/api/cancel").then(() => say("run-msg", "Stopped.")).catch((e) => say("run-msg", e.message, "down"));
document.getElementById("check").onclick = async () => {
  try {
    const r = await api("/api/moderate", { prompt: document.getElementById("prompt").value });
    say("mod-msg", r.ok ? "Safe to run" : "Blocked: " + r.hits.join(", "), r.ok ? "up" : "down");
  } catch (e) {
    say("mod-msg", e.message, "down");
  }
};
