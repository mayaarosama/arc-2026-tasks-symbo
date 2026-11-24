// Shared by stage.html and operator.html: state polling, clock maths, formatting.
"use strict";

const Demo = {
  state: null,
  offset: 0,           // server_now - client now, so clocks agree with the conductor
  listeners: [],

  onState(fn) { this.listeners.push(fn); },

  async poll(intervalMs = 300) {
    for (;;) {
      try {
        const r = await fetch("/api/state", { cache: "no-store" });
        const s = await r.json();
        this.offset = s.server_now - Date.now() / 1000;
        this.state = s;
        for (const fn of this.listeners) fn(s);
      } catch (e) {
        console.warn("state poll failed", e);   // conductor restarting: keep the last frame
      }
      await new Promise((res) => setTimeout(res, intervalMs));
    }
  },

  now() { return Date.now() / 1000 + this.offset; },

  // Seconds since the round started.
  elapsed(round) { return round ? Math.max(0, this.now() - round.started_at) : 0; },

  // What the clock of one cell shows right now.
  //  live, running  -> elapsed            live, finished -> worker-measured final time
  //  replay         -> elapsed, stopping at the rehearsal time
  //  full recording -> the time the recorded run took to finish (shown at once)
  cellClock(round, cell) {
    const t = this.elapsed(round);
    if (cell.recorded) return cell.recorded.seconds;
    if (cell.final_seconds == null) return t;
    if (cell.mode === "REPLAY") return Math.min(t, cell.final_seconds);
    return cell.final_seconds;
  },

  cellFinished(round, cell) {
    if (cell.recorded) return true;
    if (cell.status === "unavailable" || cell.status === "error") return true;   // ended, no result
    if (cell.final_seconds == null) return false;
    return cell.mode !== "REPLAY" || this.elapsed(round) >= cell.final_seconds;
  },

  visibleOutputs(round, cell) {
    const t = this.elapsed(round);
    return cell.outputs.filter((o) => o.reveal_at == null || t >= o.reveal_at);
  },
};

function fmtClock(s) {
  if (s == null || isNaN(s)) return "—";
  if (s < 60) return s.toFixed(1) + " s";
  if (s < 3600) return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}m`;
}

function fmtShort(s) {
  if (s == null) return "—";
  if (s < 60) return s.toFixed(s < 10 ? 1 : 0) + " s";
  if (s < 3600) return Math.round(s / 60) + " min";
  return (s / 3600).toFixed(1) + " h";
}

function settingsText(task, st) {
  if (!st) return "";
  if (task === "t2i") return `${st.width}×${st.height} · ${st.steps} steps · ×${st.batch}`;
  if (task === "i23d") {
    const parts = [];
    if (st.resolution) parts.push(`${st.resolution}³`);
    if (st.texture_size) parts.push(`${st.texture_size}px texture`);
    return parts.join(" · ");
  }
  if (task === "music") return `${st.length_s || "?"} s tracks`;
  if (task === "llm") return `${st.streams || 1} stream${st.streams > 1 ? "s" : ""} · ${st.max_tokens} tokens`;
  if (task === "layout") return `${st.streams || 1} room${st.streams > 1 ? "s" : ""} at once`;
  return "";
}

// Plain-language names, shared by the operator page and the stage.
const TASK_NAME = { t2i: "Text → Image", i23d: "Image → 3D model", music: "Text → Music", llm: "Language model", layout: "Floor plan → furnished room" };

// What one machine is expected to produce with these settings.
function expectedOutput(task, st, budget) {
  if (!st) return "";
  if (task === "t2i") {
    const n = st.batch || 1;
    return `${n} image${n === 1 ? "" : "s"} · ${st.width}×${st.height} · ${st.steps} steps`;
  }
  if (task === "i23d") return `1 textured 3D model${st.texture_size ? ` · ${st.texture_size}px texture` : ""}`;
  if (task === "music") {
    const len = st.length_s || "?";
    return st.repeat_until_budget && budget ? `As many ${len}-second tracks as fit in ${budget} s` : `1 track · ${len} s`;
  }
  if (task === "llm") {
    const n = st.streams || 1;
    return st.repeat_until_budget && budget ? `${n === 1 ? "1 reply" : n + " replies"} at once, as many tokens as fit in ${budget} s`
      : `${n} repl${n === 1 ? "y" : "ies"} · ${st.max_tokens} tokens`;
  }
  if (task === "layout") {
    const n = st.streams || 1;
    return st.repeat_until_budget && budget ? `${n === 1 ? "1 room" : n + " rooms"} at once, as many as fit in ${budget} s`
      : "1 furnished room, one object at a time";
  }
  return "";
}

function protocolText(protocol, budget) {
  if (protocol === "iso_quality" || !budget)
    return "Same work on every machine, no time limit — we compare how long each one takes.";
  return `Same time limit for every machine (${budget} s) — each does as much as it can.`;
}

function fmtInt(n) { return n == null ? "—" : Math.round(n).toLocaleString("en-US"); }

// llm counters: "1,234 tokens · 85 tok/s"
function tpsText(tps) { return tps ? `${tps >= 100 ? Math.round(tps) : tps.toFixed(1)} tok/s` : ""; }
function tokenText(tokens, tps) { return `${fmtInt(tokens)} tokens${tps ? " · " + tpsText(tps) : ""}`; }

// Task 8: the floor plan as the model is told about it: outline, wall names, door. Used in the stage
// header and on the operator page. item: a layout input (floor_plan, door) plus the conductor's
// layout_view() extras (walls: name -> [[x, y], [x, y]], door_wall).
function planSvg(item) {
  const NS = "http://www.w3.org/2000/svg";
  const plan = item.floor_plan || [];
  const xs = plan.map((p) => p[0]), ys = plan.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const m = 1.3;                                    // margin for the wall names, metres
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `${x0 - m} ${-y1 - m} ${x1 - x0 + 2 * m} ${y1 - y0 + 2 * m}`);   // y flipped: north up
  svg.setAttribute("class", "plan-svg");
  const mk = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    svg.appendChild(e);
    return e;
  };
  mk("polygon", { points: plan.map(([x, y]) => `${x},${-y}`).join(" "), class: "floor" });
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  for (const [name, [a, b]] of Object.entries(item.walls || {})) {
    const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
    const horiz = Math.abs(a[1] - b[1]) < 1e-6;
    // push the label outward, away from the room's centre
    const ox = horiz ? 0 : (mx < cx ? -0.6 : 0.6), oy = horiz ? (my < cy ? -0.5 : 0.5) : 0;
    mk("text", { x: mx + ox, y: -(my + oy), class: "wname" + (name === item.door_wall ? " door" : "") }, name);
  }
  if (item.door) mk("circle", { cx: item.door[0], cy: -item.door[1], r: 0.28, class: "door" });
  mk("text", { x: x0 - m + 0.1, y: -y1 - m + 0.55, class: "north" }, "N ↑");
  return svg;
}

// Task 8: one placed object back as the line the model wrote (stage console).
function layoutLine(o) {
  const n = (v) => Number(v).toFixed(2);
  if (o.wall) return `${o.cls} wall=${o.wall} at=${n(o.at)}`;
  const dir = { 0: "north", 90: "east", 180: "south", 270: "west" }[o.rot] || `${o.rot}`;
  return `${o.cls} x=${n(o.x)} y=${n(o.y)} facing=${dir}`;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
