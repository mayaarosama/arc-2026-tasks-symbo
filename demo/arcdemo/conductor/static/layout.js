// Task 8 stage view (canvas fallback; layout3d.js is the 3D view with furniture models): the floor plan as a cut-away "dollhouse" on a <canvas>, furniture dropping in
// as the model places it. Plain 2D canvas, no 3D library: nothing to vendor, works offline, and
// three cells of a dozen boxes redraw every frame at no cost.
//   world: metres, x east, y north, z up (the worker's convention, backends/layout.py)
//   view:  rotate about the room centre, then oblique projection; far side drawn first.
"use strict";

window.LayoutCanvas = (() => {
  const WALL_H = 1.1;          // back walls are cut down so they never hide the furniture
  const FIT_H = 2.1;           // headroom kept above the floor: the tallest furniture (wardrobe, 2 m)
  const DROP_MS = 450;         // a new object falls into place over this long
  const TILT = 0.55;           // vertical squash of the floor plane
  const ZS = 0.85;             // screen height of one metre of z, relative to one metre of floor
  const HUES = [205, 28, 140, 265, 350, 175, 48, 300, 95, 15, 230, 320];
  const BAD = "#ff5d5d";

  function hue(cls) {
    let h = 0;
    for (const ch of cls) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    return HUES[h % HUES.length];
  }

  function footprint(o) {
    const side = Math.round(o.rot / 90) % 2 === 1;
    const ex = side ? o.d : o.w, ey = side ? o.w : o.d;
    return [o.x - ex / 2, o.y - ey / 2, o.x + ex / 2, o.y + ey / 2];
  }

  // An object as it stood at time t (seconds on the job clock), or null if it was not placed yet or
  // was removed by then. Self-correction keeps every change in o.history (place, repair, move,
  // recheck, remove), so a replay shows each fix at its recorded time. fixing: fix attempts so far.
  const FIX_OPS = new Set(["repair", "move", "best", "remove"]);
  function at(o, t) {
    if (o.t != null && o.t > t) return null;
    if (o.removed != null && o.removed <= t) return null;
    if (!o.history) return o;
    let h = o.history[0], fixing = 0;
    for (const e of o.history) if (e.t <= t) { h = e; if (e.op === "repair" || e.op === "move") fixing++; }
    return { ...o, x: h.x, y: h.y, rot: h.rot, wall: h.wall, at: h.at, issues: h.issues, fixing,
      first_bad: !!(o.history[0].issues && o.history[0].issues.length) };
  }

  // Objects to draw now: replays reveal each object (and each fix) at its recorded time; throughput
  // runs show the room stream 0 is currently working on.
  function visible(round, c) {
    const all = c.objects || [];
    const t = c.mode === "REPLAY" && !c.recorded && window.Demo ? Demo.elapsed(round) : Infinity;
    const shown = all.filter((o) => o.t == null || o.t <= t);
    if (!shown.length) return [];
    const room = Math.max(...shown.map((o) => o.room || 0));
    return shown.filter((o) => (o.room || 0) === room).map((o) => at(o, t)).filter(Boolean);
  }

  // what a label shows: placement number, name, "!" while flagged, a turn arrow once the model fixed it
  const mark = (o) => (o.issues && o.issues.length ? " !" : "") + (o.fixing ? " ↻" + (o.fixing > 1 ? o.fixing : "") : "");

  function attach(outputs) {
    const cv = document.createElement("canvas");
    cv.className = "layout-canvas";
    outputs.replaceChildren(cv);
    return { canvas: cv, seen: new Map(), phase: Math.random() * 10 };
  }

  function draw(view, round, c) {
    const cv = view.canvas;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.max(1, Math.round(cv.clientWidth * dpr)), H = Math.max(1, Math.round(cv.clientHeight * dpr));
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    const g = cv.getContext("2d");
    g.clearRect(0, 0, W, H);
    const plan = round.input.floor_plan;
    if (!plan || plan.length < 3) return [];

    // --- camera: a slow sway around a three-quarter view
    const now = performance.now();
    const th = (40 + 10 * Math.sin(now / 7000 + view.phase)) * Math.PI / 180;
    const cos = Math.cos(th), sin = Math.sin(th);
    const xs = plan.map((p) => p[0]), ys = plan.map((p) => p[1]);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    const rot = (x, y) => [(x - cx) * cos - (y - cy) * sin, (x - cx) * sin + (y - cy) * cos];
    // fit the outline plus anything placed outside it (so a misplaced object stays in view)
    const objs = visible(round, c);
    const pts = plan.concat(objs.flatMap((o) => { const [a, b, c2, d] = footprint(o); return [[a, b], [c2, d], [a, d], [c2, b]]; }));
    const rp = pts.map((p) => rot(p[0], p[1]));
    const rx = rp.map((p) => p[0]), ry = rp.map((p) => p[1]);
    const spanX = Math.max(...rx) - Math.min(...rx), spanY = (Math.max(...ry) - Math.min(...ry)) * TILT + FIT_H * ZS;
    const s = Math.min(W * 0.92 / spanX, H * 0.9 / spanY);
    const ox = W / 2 - s * (Math.max(...rx) + Math.min(...rx)) / 2;
    const oy = H / 2 + s * ((Math.max(...ry) + Math.min(...ry)) / 2 * TILT) + s * FIT_H * ZS / 2;
    const P = (x, y, z = 0) => { const [a, b] = rot(x, y); return [ox + a * s, oy - b * s * TILT - z * s * ZS]; };
    const depth = (x, y) => rot(x, y)[1];              // larger = farther from the viewer
    const poly = (pts, fill, stroke, lw = 1) => {
      g.beginPath();
      pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.closePath();
      if (fill) { g.fillStyle = fill; g.fill(); }
      if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw * dpr; g.stroke(); }
    };

    // --- floor, with a 1 m grid clipped to the outline
    const floor = plan.map((p) => P(p[0], p[1]));
    poly(floor, "#1d242c", null);
    g.save();
    poly(floor, null, null);
    g.clip();
    g.strokeStyle = "#ffffff10";
    g.lineWidth = dpr;
    for (let x = Math.floor(Math.min(...xs)); x <= Math.max(...xs); x++) {
      const [a, b] = [P(x, Math.min(...ys)), P(x, Math.max(...ys))];
      g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
    }
    for (let y = Math.floor(Math.min(...ys)); y <= Math.max(...ys); y++) {
      const [a, b] = [P(Math.min(...xs), y), P(Math.max(...xs), y)];
      g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
    }
    g.restore();

    // --- walls: the far ones stand (cut down), the near ones are a low kerb drawn last
    const area = plan.reduce((acc, p, i) => { const q = plan[(i + 1) % plan.length]; return acc + p[0] * q[1] - q[0] * p[1]; }, 0);
    const walls = plan.map((p, i) => {
      const q = plan[(i + 1) % plan.length];
      const nx = area > 0 ? q[1] - p[1] : p[1] - q[1], ny = area > 0 ? p[0] - q[0] : q[0] - p[0];   // outward
      const far = rot(cx + nx, cy + ny)[1] > 0;
      return { p, q, far, d: depth((p[0] + q[0]) / 2, (p[1] + q[1]) / 2) };
    });
    const wall = (w, h, fill) => poly([P(w.p[0], w.p[1]), P(w.q[0], w.q[1]), P(w.q[0], w.q[1], h), P(w.p[0], w.p[1], h)],
      fill, "#3a4552");
    for (const w of walls.filter((w) => w.far).sort((a, b) => b.d - a.d)) wall(w, WALL_H, "#2a333dcc");

    // --- door: a gap marker on its wall
    const door = round.input.door;
    if (door) {
      let best = null;
      for (const w of walls) {
        const dx = w.q[0] - w.p[0], dy = w.q[1] - w.p[1], L = Math.hypot(dx, dy) || 1;
        const t = Math.max(0, Math.min(1, ((door[0] - w.p[0]) * dx + (door[1] - w.p[1]) * dy) / (L * L)));
        const dist = Math.hypot(w.p[0] + t * dx - door[0], w.p[1] + t * dy - door[1]);
        if (!best || dist < best.dist) best = { dist, ux: dx / L, uy: dy / L };
      }
      const a = P(door[0] - best.ux * 0.45, door[1] - best.uy * 0.45), b = P(door[0] + best.ux * 0.45, door[1] + best.uy * 0.45);
      g.strokeStyle = "#ffd24a";
      g.lineWidth = 5 * dpr;
      g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke();
      const m = P(door[0], door[1], 0.15);
      g.fillStyle = "#ffd24a";
      g.font = `${Math.round(11 * dpr)}px Segoe UI, system-ui, sans-serif`;
      g.textAlign = "center";
      g.fillText("door", m[0], m[1]);
    }

    // --- furniture, far to near
    const newest = objs.length ? objs[objs.length - 1] : null;
    const items = objs.map((o) => {
      const key = `${o.room || 0}-${o.i}`;
      if (!view.seen.has(key)) view.seen.set(key, now);
      const k = Math.min(1, (now - view.seen.get(key)) / DROP_MS);
      const ease = 1 - Math.pow(1 - k, 3);
      const [x0, y0, x1, y1] = footprint(o);
      return { o, x0, y0, x1, y1, lift: (1 - ease) * 1.5, alpha: 0.25 + 0.75 * ease, d: depth(o.x, o.y) };
    }).sort((a, b) => b.d - a.d);

    const label = [];
    for (const it of items) {
      const { o, x0, y0, x1, y1, lift } = it;
      const z0 = lift, z1 = lift + Math.max(0.05, o.h);
      const h = hue(o.cls), bad = o.issues && o.issues.length;
      g.globalAlpha = it.alpha;
      if (lift <= 0.01) {   // shadow once landed
        poly([P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)], "#0007", null);
      }
      // the two side faces that face the viewer (normals pointing towards smaller depth)
      const faces = [
        { a: [x0, y0], b: [x1, y0], n: [0, -1] }, { a: [x1, y0], b: [x1, y1], n: [1, 0] },
        { a: [x1, y1], b: [x0, y1], n: [0, 1] }, { a: [x0, y1], b: [x0, y0], n: [-1, 0] },
      ];
      for (const f of faces) {
        if (rot(cx + f.n[0], cy + f.n[1])[1] >= 0) continue;
        const shade = f.n[0] !== 0 ? 34 : 42;
        poly([P(...f.a, z0), P(...f.b, z0), P(...f.b, z1), P(...f.a, z1)], `hsl(${h} 45% ${shade}%)`,
          bad ? BAD : "#0006", bad ? 3 : 1);
      }
      const top = [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)];
      poly(top, `hsl(${h} 55% 60%)`, bad ? BAD : "#0005", bad ? 3.5 : 1);
      // front marker: a short bar along the front edge of the top face
      const f = { 0: [[x0, y1], [x1, y1]], 180: [[x0, y0], [x1, y0]], 90: [[x1, y0], [x1, y1]], 270: [[x0, y0], [x0, y1]] }[o.rot] || null;
      if (f) {
        const inset = (p) => [p[0] + (o.x - p[0]) * 0.12, p[1] + (o.y - p[1]) * 0.12];
        g.strokeStyle = `hsl(${h} 70% 85%)`;
        g.lineWidth = 2 * dpr;
        g.beginPath(); g.moveTo(...P(...inset(f[0]), z1)); g.lineTo(...P(...inset(f[1]), z1)); g.stroke();
      }
      g.globalAlpha = 1;
      label.push({ o, at: P(o.x, o.y, z1 + 0.05), bad, newest: o === newest });
    }

    for (const w of walls.filter((w) => !w.far)) wall(w, 0.06, "#2a333d");
    poly(floor, null, "#55606d", 1.5);

    // --- labels last, so no box hides one: placement order + name
    // (near labels first, so the object in front keeps its label in place; others step aside)
    g.textAlign = "center";
    const taken = [];
    const hits = (r) => taken.some((t) => r[0] < t[2] && t[0] < r[2] && r[1] < t[3] && t[1] < r[3]);
    for (const l of label.reverse()) {
      const txt = `${l.o.i + 1} ${l.o.cls.replace(/_/g, " ")}${mark(l.o)}`;
      g.font = `${l.newest ? 700 : 500} ${Math.round((l.newest ? 13 : 11) * dpr)}px Segoe UI, system-ui, sans-serif`;
      const wpx = g.measureText(txt).width + 8 * dpr, hpx = 17 * dpr;
      let [x, y] = l.at;
      for (let k = 0; k < 6 && hits([x - wpx / 2, y - 13 * dpr, x + wpx / 2, y - 13 * dpr + hpx]); k++) y -= hpx + 2 * dpr;
      taken.push([x - wpx / 2, y - 13 * dpr, x + wpx / 2, y - 13 * dpr + hpx]);
      g.fillStyle = l.bad ? "#5a1515e6" : "#0b0d10cc";
      g.fillRect(x - wpx / 2, y - 13 * dpr, wpx, hpx);
      g.fillStyle = l.bad ? "#ffb3b3" : l.newest ? "#ffffff" : "#c9d1db";
      g.fillText(txt, x, y);
    }
    return objs;
  }

  // "8 objects · 1 flagged (overlap)"
  function tally(objs) {
    const flags = objs.flatMap((o) => o.issues || []);
    const bad = objs.filter((o) => o.issues && o.issues.length).length;
    const kinds = [...new Set(flags)];   // outside the walls / overlapping an earlier object / blocking the door
    const fixed = objs.filter((o) => o.fixing && !(o.issues && o.issues.length)).length;
    return `${objs.length} object${objs.length === 1 ? "" : "s"}` + (bad ? ` · ${bad} flagged (${kinds.join(", ")})` : " · no problems")
      + (fixed ? ` · ${fixed} fixed by the model` : "");
  }

  return { attach, draw, visible, tally, at, mark };
})();
// The 3D view (layout3d.js, a module, loads after this) replaces it when WebGL works.
window.Layout = window.Layout || window.LayoutCanvas;
