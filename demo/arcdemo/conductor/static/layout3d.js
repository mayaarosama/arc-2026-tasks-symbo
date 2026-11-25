// Task 8 stage view in 3D: the floor plan as a cut-away "dollhouse" with real furniture models
// (Kenney Furniture Kit, CC0, static/assets/furniture/) dropping in as the model places them.
// Same interface as the canvas view in layout.js (attach / draw / visible / tally), which stays as
// the fallback when WebGL is unavailable. three.js r160 is vendored in static/vendor/three/.
//   world: metres, x east, y north (the worker's convention, backends/layout.py)
//   three: X = x, Y = up, Z = -y  (north is -Z; the camera looks from the south-east)
// Each base model is scaled to the object's exact w x d x h, so what you see is the footprint the
// room checks use. Models face +Z in their own frame; rot r turns that front to the compass bearing r.
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const WALL_H = 1.1, WALL_T = 0.08, DROP_MS = 550, MOVE_MS = 700;   // a fixed object slides over MOVE_MS
const BAD = 0xff5d5d;

// Furniture type -> model file, props set on top and rug, from the furniture catalogue
// (data/layout/furniture.json, served by the conductor; the same document the model's prompt is built
// from). Props: "at" = [along the front, towards the back] in metres from the centre; "size" = the box
// the prop is fitted into, w x d x h.
let MODELS = {}, ASSETS = new URL("/layout-assets/", location.href);
try {
  const cat = await (await fetch("/api/layout_catalog")).json();
  ASSETS = new URL(cat.assets_url, location.href);
  MODELS = cat.furniture;
} catch (e) {
  console.error("layout3d: furniture catalogue not loaded; every object is drawn as a box", e);
}

const loader = new GLTFLoader();
const cache = new Map();          // file -> Promise<normalised Object3D>: centred, sitting on y = 0
function model(file) {
  if (!cache.has(file)) {
    cache.set(file, loader.loadAsync(new URL(file, ASSETS).href).then((g) => {
      const root = g.scene;
      root.traverse((m) => { if (m.isMesh) { m.castShadow = true; m.receiveShadow = true; } });
      const box = new THREE.Box3().setFromObject(root);
      const c = box.getCenter(new THREE.Vector3());
      root.position.sub(new THREE.Vector3(c.x, box.min.y, c.z));
      const wrap = new THREE.Group();
      wrap.add(root);
      wrap.userData.size = box.getSize(new THREE.Vector3());
      return wrap;
    }));
  }
  return cache.get(file);
}
Object.values(MODELS).forEach((m) => { model(m.model); (m.props || []).forEach((p) => model(p.model)); });

function fitted(src, w, d, h, exact) {          // exact: stretch to w x d x h; else fit inside, keep shape
  const o = src.clone(true), s = src.userData.size;
  if (exact) o.scale.set(w / s.x, h / s.y, d / s.z);
  else { const k = Math.min(w / s.x, h / s.y, d / s.z); o.scale.setScalar(k); }
  return o;
}

async function furniture(o) {
  const spec = MODELS[o.cls];
  const g = new THREE.Group();
  if (!spec) {                                   // not in the catalogue: a plain box, never nothing
    console.warn(`layout3d: '${o.cls}' is not in the furniture catalogue; drawn as a box`);
    const m = new THREE.Mesh(new THREE.BoxGeometry(o.w, o.h, o.d), new THREE.MeshStandardMaterial({ color: 0x8899aa }));
    m.position.y = o.h / 2;
    g.add(m);
    return g;
  }
  g.add(fitted(await model(spec.model), o.w, o.d, o.h, true));
  for (const p of spec.props || []) {
    const prop = fitted(await model(p.model), p.size[0], p.size[1], p.size[2], false);
    prop.position.set(p.at[0], o.h, -p.at[1]);   // on top; "towards the back" is -Z in the model frame
    g.add(prop);
  }
  if (spec.rug) {
    const rug = new THREE.Mesh(new THREE.PlaneGeometry(spec.rug[0], spec.rug[1]),
      new THREE.MeshStandardMaterial({ color: 0x6b4f7a, roughness: 1 }));
    rug.rotation.x = -Math.PI / 2;
    rug.position.y = 0.006;
    rug.receiveShadow = true;
    g.add(rug);
  }
  return g;
}

function footprint(o) {
  const side = Math.round(o.rot / 90) % 2 === 1;
  const ex = side ? o.d : o.w, ey = side ? o.w : o.d;
  return [o.x - ex / 2, o.y - ey / 2, o.x + ex / 2, o.y + ey / 2];
}

// ------------------------------------------------------------------ one cell -------------
class View {
  constructor(outputs) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.canvas = this.renderer.domElement;
    this.canvas.className = "layout-canvas";
    this.labels = document.createElement("div");
    this.labels.className = "layout-labels";
    outputs.replaceChildren(this.canvas, this.labels);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x2a3440, 1.15));
    this.sun = new THREE.DirectionalLight(0xffffff, 1.9);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(1024, 1024);
    this.sun.shadow.bias = -0.0005;
    this.scene.add(this.sun, this.sun.target);
    this.room = new THREE.Group();
    this.scene.add(this.room);
    this.items = new Map();      // "room-i" -> {group, born, o, flag}
    this.roomKey = null;
    this.phase = Math.random() * 10;
    this.cam = { az: 0, el: 0, zoom: 1 };        // viewer's offsets: drag rotates, wheel zooms, double-click resets
    this.canvas.style.touchAction = "none";
    this.canvas.addEventListener("pointerdown", (e) => { this.canvas.setPointerCapture(e.pointerId); this.drag = [e.clientX, e.clientY]; });
    this.canvas.addEventListener("pointerup", () => { this.drag = null; });
    this.canvas.addEventListener("pointermove", (e) => {
      if (!this.drag) return;
      this.cam.az -= (e.clientX - this.drag[0]) * 0.4;
      this.cam.el = Math.max(-33, Math.min(50, this.cam.el + (e.clientY - this.drag[1]) * 0.3));
      this.drag = [e.clientX, e.clientY];
    });
    this.canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.cam.zoom = Math.max(0.3, Math.min(3, this.cam.zoom * Math.exp(e.deltaY * 0.001)));
    }, { passive: false });
    this.canvas.addEventListener("dblclick", () => { this.cam = { az: 0, el: 0, zoom: 1 }; });
  }

  buildRoom(input) {
    this.room.clear();
    this.walls = [];
    const plan = input.floor_plan;
    const shape = new THREE.Shape(plan.map(([x, y]) => new THREE.Vector2(x, y)));
    const floor = new THREE.Mesh(new THREE.ShapeGeometry(shape),
      new THREE.MeshStandardMaterial({ color: 0xb79a74, roughness: 0.9 }));     // warm wood
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.room.add(floor);
    const xs = plan.map((p) => p[0]), ys = plan.map((p) => p[1]);
    const area = plan.reduce((a, p, i) => { const q = plan[(i + 1) % plan.length]; return a + p[0] * q[1] - q[0] * p[1]; }, 0);
    const wallMat = new THREE.MeshStandardMaterial({ color: 0xe9e4dc, roughness: 0.95 });
    plan.forEach((p, i) => {
      const q = plan[(i + 1) % plan.length];
      const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy);
      const nx = (area > 0 ? dy : -dy) / len, ny = (area > 0 ? -dx : dx) / len;      // outward, world
      const m = new THREE.Mesh(new THREE.BoxGeometry(len + WALL_T, WALL_H, WALL_T), wallMat.clone());
      m.position.set((p[0] + q[0]) / 2 + nx * WALL_T / 2, WALL_H / 2, -((p[1] + q[1]) / 2 + ny * WALL_T / 2));
      m.rotation.y = Math.atan2(dy, dx);
      m.castShadow = true;
      m.receiveShadow = true;
      this.room.add(m);
      this.walls.push({ mesh: m, n: new THREE.Vector3(nx, 0, -ny), mid: m.position.clone() });
    });
    const door = input.door;
    if (door) {
      let best = null;
      plan.forEach((p, i) => {
        const q = plan[(i + 1) % plan.length], dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy) || 1;
        const t = Math.max(0, Math.min(1, ((door[0] - p[0]) * dx + (door[1] - p[1]) * dy) / (L * L)));
        const dist = Math.hypot(p[0] + t * dx - door[0], p[1] + t * dy - door[1]);
        if (!best || dist < best.dist) best = { dist, ang: Math.atan2(dy, dx) };
      });
      const d = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.02, 0.5), new THREE.MeshStandardMaterial({ color: 0xffd24a, emissive: 0x5a4500 }));
      d.position.set(door[0], 0.01, -door[1]);
      d.rotation.y = best.ang;
      this.room.add(d);
      this.door = new THREE.Vector3(door[0], 0.05, -door[1]);
    } else this.door = null;
    // fit: camera distance and the sun's shadow box to the room
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cy = (Math.min(...ys) + Math.max(...ys)) / 2;
    this.centre = new THREE.Vector3(cx, 0.4, -cy);
    this.radius = Math.hypot(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2 + 0.4;
    this.corners = plan.flatMap(([x, y]) => [new THREE.Vector3(x, 0, -y), new THREE.Vector3(x, WALL_H, -y)]);
    const sc = this.sun.shadow.camera;
    sc.left = sc.bottom = -this.radius * 1.3; sc.right = sc.top = this.radius * 1.3; sc.near = 0.5; sc.far = 40;
    sc.updateProjectionMatrix();
    this.sun.position.set(cx - 4, 9, -cy + 6);
    this.sun.target.position.copy(this.centre);
  }

  async add(key, o, from) {   // from: where a fixed object was, so it slides rather than drops in
    const rec = { group: new THREE.Group(), born: performance.now(), o, ready: false, pose: pose(o), from };
    this.items.set(key, rec);
    const g = await furniture(o);
    if (this.items.get(key) !== rec) return;     // the room moved on while it loaded
    g.rotation.y = Math.PI - (o.rot * Math.PI) / 180;
    rec.group.add(g);
    rec.group.position.set(o.x, 0, -o.y);
    if (o.issues && o.issues.length) {          // flagged: red footprint frame and a red floor patch
      const [x0, y0, x1, y1] = footprint(o);
      const box = new THREE.Box3(new THREE.Vector3(x0 - o.x, 0, -(y1 - o.y)), new THREE.Vector3(x1 - o.x, o.h + 0.02, -(y0 - o.y)));
      rec.group.add(new THREE.Box3Helper(box, BAD));
      const patch = new THREE.Mesh(new THREE.PlaneGeometry(x1 - x0, y1 - y0),
        new THREE.MeshBasicMaterial({ color: BAD, transparent: true, opacity: 0.35, depthWrite: false }));
      patch.rotation.x = -Math.PI / 2;
      patch.position.set((x0 + x1) / 2 - o.x, 0.012, -((y0 + y1) / 2 - o.y));
      rec.group.add(patch);
    }
    rec.born = performance.now();
    rec.ready = true;
    this.scene.add(rec.group);
  }

  draw(round, c) {
    const W = this.canvas.clientWidth, H = this.canvas.clientHeight;
    if (!W || !H) return [];
    if (this.renderer.domElement.width !== Math.round(W * this.renderer.getPixelRatio())
        || this.renderer.domElement.height !== Math.round(H * this.renderer.getPixelRatio())) {
      this.renderer.setSize(W, H, false);
      this.camera.aspect = W / H;
      this.camera.updateProjectionMatrix();
    }
    const key = JSON.stringify([round.input.floor_plan, round.input.door]);
    if (key !== this.roomKey) { this.buildRoom(round.input); this.roomKey = key; this.clearItems(); }

    const objs = visible(round, c);
    const want = new Set(objs.map((o) => `${o.room || 0}-${o.i}`));
    for (const [k, rec] of this.items) if (!want.has(k)) { this.scene.remove(rec.group); this.items.delete(k); }
    for (const o of objs) {
      const k = `${o.room || 0}-${o.i}`, rec = this.items.get(k);
      if (!rec) this.add(k, o);
      else if (rec.pose !== pose(o)) {        // the model fixed it: rebuild at the new pose, sliding from the old
        const from = rec.ready ? rec.group.position.clone().setY(0) : null;
        this.scene.remove(rec.group);
        this.add(k, o, from);
      }
    }

    // camera: low and only slightly turned, so the room spreads across the (wide, short) cell; slow sway
    const now = performance.now();
    const az = (25 + 8 * Math.sin(now / 8000 + this.phase) + this.cam.az) * Math.PI / 180, el = (38 + this.cam.el) * Math.PI / 180;
    const place = (dist) => {
      this.camera.position.set(this.centre.x + dist * Math.cos(el) * Math.sin(az), this.centre.y + dist * Math.sin(el),
        this.centre.z + dist * Math.cos(el) * Math.cos(az));
      this.camera.lookAt(this.centre);
      this.camera.updateMatrixWorld();
    };
    // frame the room tightly: project its corners (floor and wall tops) and scale the distance so
    // the widest of them lands at 92% of the cell; perspective makes it converge in two passes
    let dist = this.radius * 2.6;
    for (let pass = 0; pass < 2; pass++) {
      place(dist);
      let ext = 0;
      for (const p of this.corners) {
        const v = p.clone().project(this.camera);
        ext = Math.max(ext, Math.abs(v.x), Math.abs(v.y));
      }
      dist *= ext / 0.92;
    }
    place(dist * this.cam.zoom);
    // walls facing the camera drop to a kerb so they never hide the room
    for (const w of this.walls) {
      const toCam = this.camera.position.clone().sub(w.mid).setY(0).normalize();
      const near = w.n.dot(toCam) > 0.15;
      w.mesh.scale.y = near ? 0.08 : 1;
      w.mesh.position.y = (near ? 0.08 : 1) * WALL_H / 2;
    }
    // drop-in animation; a fixed object slides from where it was instead
    for (const rec of this.items.values()) {
      if (!rec.ready) continue;
      if (rec.from) {
        const k = Math.min(1, (now - rec.born) / MOVE_MS), ease = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
        rec.group.position.set(rec.from.x + (rec.o.x - rec.from.x) * ease, 0, rec.from.z + (-rec.o.y - rec.from.z) * ease);
        continue;
      }
      const k = Math.min(1, (now - rec.born) / DROP_MS), ease = 1 - Math.pow(1 - k, 3);
      rec.group.position.y = (1 - ease) * 1.6;
    }
    this.renderer.render(this.scene, this.camera);
    this.drawLabels(objs, W, H);
    return objs;
  }

  clearItems() { for (const rec of this.items.values()) this.scene.remove(rec.group); this.items.clear(); }

  drawLabels(objs, W, H) {
    const newest = objs.length ? objs[objs.length - 1] : null;
    const list = objs.map((o) => {
      const rec = this.items.get(`${o.room || 0}-${o.i}`);
      const lift = rec && rec.ready ? rec.group.position.y : 1.6;
      const p = new THREE.Vector3(o.x, o.h + lift + 0.12, -o.y).project(this.camera);
      return { o, x: (p.x + 1) / 2 * W, y: (1 - p.y) / 2 * H, z: p.z, bad: o.issues && o.issues.length, newest: o === newest };
    }).sort((a, b) => a.z - b.z);                 // nearest first keeps its place
    if (this.door) {
      const p = this.door.clone().project(this.camera);
      list.push({ door: true, x: (p.x + 1) / 2 * W, y: (1 - p.y) / 2 * H });
    }
    const taken = [], frag = document.createDocumentFragment();
    for (const l of list) {
      const el = document.createElement("span");
      el.className = "lbl" + (l.door ? " door" : "") + (l.bad ? " bad" : "") + (l.newest ? " new" : "");
      el.textContent = l.door ? "door" : `${l.o.i + 1} ${l.o.cls.replace(/_/g, " ")}${window.LayoutCanvas.mark(l.o)}`;
      const w = el.textContent.length * 7 + 10, h = 18;
      let y = l.y;
      for (let k = 0; k < 6 && taken.some((t) => l.x - w / 2 < t[2] && t[0] < l.x + w / 2 && y - h < t[3] && t[1] < y); k++) y -= h + 2;
      taken.push([l.x - w / 2, y - h, l.x + w / 2, y]);
      el.style.left = `${l.x}px`;
      el.style.top = `${y}px`;
      frag.appendChild(el);
    }
    this.labels.replaceChildren(frag);
  }
}

function visible(round, c) { return window.LayoutCanvas.visible(round, c); }   // one rule for both views

// what changes when the model fixes an object: where it stands, which way it faces, its flags
const pose = (o) => `${o.x}|${o.y}|${o.rot}|${(o.issues || []).join(",")}`;

function webgl() {
  try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); }
  catch (e) { return false; }
}

const Layout3D = {
  attach(outputs) { return new View(outputs); },
  draw(view, round, c) { return view.draw(round, c); },
  visible,
  tally: (objs) => window.LayoutCanvas.tally(objs),
  is3d: true,
};
if (webgl()) window.Layout = Layout3D;
window.Layout3D = Layout3D;
window.dispatchEvent(new Event("layout3d-ready"));
