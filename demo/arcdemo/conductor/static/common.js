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
  return "";
}

// Plain-language names, shared by the operator page and the stage.
const TASK_NAME = { t2i: "Text → Image", i23d: "Image → 3D model", music: "Text → Music" };

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
  return "";
}

function protocolText(protocol, budget) {
  if (protocol === "iso_quality" || !budget)
    return "Same work on every machine, no time limit — we compare how long each one takes.";
  return `Same time limit for every machine (${budget} s) — each does as much as it can.`;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}
