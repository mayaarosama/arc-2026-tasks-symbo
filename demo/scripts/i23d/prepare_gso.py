"""Prepare the image-to-3D inputs from Google Scanned Objects (CC BY 4.0).

For each object in data/i23d/stage_objects.json:
  1. download the GSO model from Gazebo Fuel (cached in data/i23d/gso_cache/),
  2. extract it to data/i23d/truth/<id>/  (the scanned mesh = ground truth for scoring),
  3. render ONE fixed view, 512x512 RGBA with a transparent background -> data/i23d/inputs/<id>.png.
     A render, not a photo, so the input is byte-identical on every machine; real alpha, so
     TRELLIS/SF3D skip background removal.

    python -m scripts.i23d.prepare_gso                 # the three stage objects
    python -m scripts.i23d.prepare_gso --candidates    # also render every alternate into a contact sheet
                                                  # data/i23d/candidates.png, to pick objects from

Needs: trimesh, pyrender (requirements/gso.txt, two-step install; offscreen OpenGL works on a desktop; on headless Linux set
PYOPENGL_PLATFORM=egl). Run it once on the operator laptop and copy data/i23d/ to every machine.
"""
import argparse
import json
import logging
import math
import zipfile
from pathlib import Path

import numpy as np
import requests
import trimesh
from PIL import Image, ImageDraw

try:                     # requirements/gso.txt, installed separately (only this script needs it)
    import pyrender
except ImportError:
    pyrender = None

from scripts.common.settings import SCRIPTS

ROOT = Path(__file__).resolve().parents[2]       # demo/
DATA = ROOT / "data" / "i23d"
P = SCRIPTS.prepare_gso                        # config/scripts.yaml
FUEL = P.fuel_url
log = logging.getLogger("prepare_gso")

# Fixed camera, identical for every object (degrees above the horizon, around, field of view).
ELEVATION, AZIMUTH, FOV = P.camera.elevation, P.camera.azimuth, P.camera.fov
SIZE, SUPERSAMPLE = P.size, P.supersample


def fetch(name: str) -> Path:
    """Download + extract one GSO model; returns the extracted folder."""
    zpath = DATA / "gso_cache" / f"{name}.zip"
    if not zpath.exists():
        url = FUEL.format(name=name)
        log.info("downloading %s", url)
        r = requests.get(url, timeout=P.download_timeout_seconds)
        r.raise_for_status()
        zpath.parent.mkdir(parents=True, exist_ok=True)
        zpath.write_bytes(r.content)
    out = DATA / "gso" / name
    if not (out / "meshes" / "model.obj").exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return out


def load_upright(folder: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(folder / "meshes" / "model.obj", force="mesh", process=False)
    # GSO's model.mtl says "map_Kd texture.png" but the file ships in materials/textures/,
    # so trimesh finds no image. Attach it explicitly, or every input renders untextured white.
    tex = folder / "materials" / "textures" / "texture.png"
    if tex.exists() and getattr(mesh.visual, "uv", None) is not None:
        mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=Image.open(tex))
    else:
        log.warning("%s: no texture/UVs found, rendering untextured", folder.name)
    # GSO / Gazebo meshes are Z-up; the renderer's camera convention is Y-up.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh


def render(mesh: trimesh.Trimesh) -> Image.Image:
    if pyrender is None:
        raise SystemExit("pyrender is not installed: pip install -r requirements/gso.txt")
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.35, 0.35, 0.35])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    radius = float(np.linalg.norm(mesh.bounding_box.extents) / 2)
    dist = radius / math.sin(math.radians(FOV / 2)) * 1.05
    el, az = math.radians(ELEVATION), math.radians(AZIMUTH)
    eye = dist * np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
    pose = look_at(eye, np.zeros(3), np.array([0, 1.0, 0]))
    scene.add(pyrender.PerspectiveCamera(yfov=math.radians(FOV)), pose=pose)
    scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
    key = look_at(np.array([dist, dist, dist * 0.3]), np.zeros(3), np.array([0, 1.0, 0]))
    scene.add(pyrender.DirectionalLight(intensity=2.0), pose=key)
    s = SIZE * SUPERSAMPLE
    r = pyrender.OffscreenRenderer(s, s)
    try:
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        r.delete()
    return Image.fromarray(color, "RGBA").resize((SIZE, SIZE), Image.LANCZOS)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = s, u, -f, eye
    return m


def contact_sheet(items: list[tuple[str, Image.Image]], path: Path) -> None:
    cols, cell = 3, 320
    rows = math.ceil(len(items) / cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 30)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(items):
        x, y = (i % cols) * cell, (i // cols) * (cell + 30)
        bg = Image.new("RGBA", img.size, (235, 235, 235, 255))
        bg.alpha_composite(img)
        sheet.paste(bg.convert("RGB").resize((cell, cell)), (x, y))
        d.text((x + 6, y + cell + 6), label[:48], fill="black")
    sheet.save(path)
    log.info("contact sheet: %s", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    objects_path = DATA / "stage_objects.json"
    objects = json.loads(objects_path.read_text(encoding="utf-8"))
    sheet = []
    for obj in objects:
        folder = fetch(obj["gso"])
        img = render(load_upright(folder))
        inp = ROOT / obj["input"]
        inp.parent.mkdir(parents=True, exist_ok=True)
        img.save(inp)
        obj["truth"] = str((folder / "meshes" / "model.obj").relative_to(ROOT)).replace("\\", "/")
        log.info("%s %s -> %s (truth %s)", obj["id"], obj["gso"], obj["input"], obj["truth"])
        sheet.append((f'{obj["id"]} {obj["gso"]}', img))
        if a.candidates:
            for alt in obj.get("alternates", []):
                sheet.append((f'{obj["id"]} alt: {alt}', render(load_upright(fetch(alt)))))
    objects_path.write_text(json.dumps(objects, indent=2), encoding="utf-8")
    contact_sheet(sheet, DATA / ("candidates.png" if a.candidates else "stage_inputs.png"))


if __name__ == "__main__":
    main()
