"""Operator-uploaded images for image-to-3D: check, cut out, normalise (no model, no torch).

Every machine must get the identical input, so this runs once, on the conductor, at upload, and
the result (512x512 RGBA, object centred, filling 85% of the side) is what each worker receives.
That is the same form as the stage inputs (scripts/prepare_gso.py renders 512x512 RGBA).

Background removal: the U^2-Net model that rembg (and SF3D's own run.py) use by default,
run directly with onnxruntime, so the conductor needs no rembg / OpenCV stack. Same preprocessing
as rembg 2.0.57 (320x320, ImageNet normalisation, min-max scaled mask). The model file is fetched
once with  python scripts/download_weights.py --cutout  (weights/u2net/u2net.onnx, or $U2NET_HOME,
or rembg's own ~/.u2net). Without it, only images that already have a transparent background work.

The operator-facing spec is INPUT_SPEC (shown next to the upload box and in the README).
"""
from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]          # demo/

MAX_BYTES = 10 * 1024 * 1024    # file size
MIN_SIDE = 256                  # shortest side: below this there is too little detail to rebuild
GOOD_SIDE = 512                 # shortest side we recommend (SF3D's input is 512 px)
MAX_SIDE = 4096                 # longer images are scaled down first (not rejected)
OUT_SIDE = 512                  # what every machine receives
FILL = 0.85                     # the object's share of the output side (upstream SF3D run.py uses 0.85)
MIN_COVER, MAX_COVER = 0.03, 0.95   # share of the image that is the object, before cropping
EDGE_PX = 2                     # foreground within this many pixels of the border = cropped object

INPUT_SPEC = [
    ("File", "PNG, JPEG or WebP, up to 10 MB. Shortest side 512 px or more (256 px is the minimum)."),
    ("Content", "One object, whole and not cropped: nothing touching the edge of the picture."),
    ("Background", "Plain and in a different colour from the object. A PNG with a transparent background is best."),
    ("View", "Three-quarter view from slightly above (about 20° up, 35° round), like the stage examples."),
    ("Light", "Even and soft. No hard shadows, glare or strong reflections: they get painted into the texture."),
    ("Avoid", "Glass and see-through objects, mirrors, thin wires or hair, text-heavy surfaces, people, "
              "and scenes with several objects."),
]


class Rejected(ValueError):
    """The image cannot be used; the message says why, in words for the operator."""


@dataclass
class Prepared:
    png: bytes                          # 512x512 RGBA, what every machine receives
    original_size: tuple[int, int]
    cutout: str                         # "alpha" (its own transparency) or "u2net"
    cover: float                        # the object's share of the uploaded image
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ shared with the workers -----
def resize_foreground(img: Image.Image, ratio: float = FILL) -> Image.Image:
    """Crop to the alpha bbox and pad to a square where the object fills `ratio` of the side."""
    a = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(a > 0)
    if len(xs) == 0:
        return img
    crop = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    side = int(max(crop.size) / ratio)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return canvas


# ------------------------------------------------------------------ background removal ---------
def cutout_model_path() -> Optional[Path]:
    for d in (os.environ.get("U2NET_HOME"), ROOT / "weights" / "u2net", Path.home() / ".u2net"):
        if d and (Path(d) / "u2net.onnx").is_file():
            return Path(d) / "u2net.onnx"
    return None


_SESSION = None
_SESSION_LOCK = threading.Lock()


def _session():
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            path = cutout_model_path()
            if path is None:
                raise Rejected("This picture has no transparent background, and the cut-out model is not "
                               "installed on this laptop (run: python scripts/download_weights.py --cutout). "
                               "Use a PNG with a transparent background instead.")
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise Rejected("The cut-out needs onnxruntime: pip install -r requirements/conductor.txt") from e
            want = os.environ.get("ARC_CUTOUT_DEVICE", "auto").lower()   # auto | cpu | cuda
            providers = ["CPUExecutionProvider"]
            if want != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")             # only with onnxruntime-gpu installed
            try:
                _SESSION = ort.InferenceSession(str(path), providers=providers)
            except Exception:  # noqa: BLE001 — a CUDA/cuDNN mismatch must never stop an upload: use the CPU
                _SESSION = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return _SESSION


def cutout_device() -> str:
    """Where the cut-out runs: "cuda" or "cpu" (loads the model if it is not loaded yet)."""
    return "cuda" if "CUDAExecutionProvider" in _session().get_providers() else "cpu"


def u2net_mask(img: Image.Image) -> Image.Image:
    """An 'L' mask the size of img: 255 = object. rembg 2.0.57's U2netSession, without rembg."""
    sess = _session()
    small = np.asarray(img.convert("RGB").resize((320, 320), Image.LANCZOS), dtype=np.float32)
    small = small / max(float(small.max()), 1e-6)
    x = (small - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    pred = sess.run(None, {sess.get_inputs()[0].name: x})[0][0, 0]
    pred = (pred - pred.min()) / max(float(pred.max() - pred.min()), 1e-6)
    return Image.fromarray((pred * 255).astype(np.uint8), "L").resize(img.size, Image.LANCZOS)


# ------------------------------------------------------------------ the whole step -------------
def _has_real_alpha(img: Image.Image) -> bool:
    if "A" not in img.getbands():
        return False
    a = np.asarray(img.getchannel("A"))
    return float((a < 128).mean()) > 0.02       # at least some of the picture is see-through


def prepare(data: bytes) -> Prepared:
    """Check an uploaded file, cut the object out and normalise it. Raises Rejected with a reason."""
    if len(data) > MAX_BYTES:
        raise Rejected(f"The file is {len(data) / 2**20:.1f} MB; the limit is {MAX_BYTES // 2**20} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001 — any decoder failure is the same answer for the operator
        raise Rejected("This file is not a picture that can be read (use PNG, JPEG or WebP).") from e
    if img.format not in ("PNG", "JPEG", "WEBP", "MPO"):
        raise Rejected(f"{img.format or 'This'} files are not accepted: use PNG, JPEG or WebP.")
    img = ImageOps.exif_transpose(img)            # phone photos: stand them the right way up
    size = img.size
    warnings: list[str] = []
    if min(size) < MIN_SIDE:
        raise Rejected(f"The picture is {size[0]} × {size[1]} px; it needs at least {MIN_SIDE} px on the "
                       f"shortest side ({GOOD_SIDE} or more is better).")
    if min(size) < GOOD_SIDE:
        warnings.append(f"Small picture ({size[0]} × {size[1]} px): the 3D model will be less detailed "
                        f"than with {GOOD_SIDE} px or more.")
    if max(size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

    if _has_real_alpha(img):
        rgba, how = img.convert("RGBA"), "alpha"
    else:
        rgb = img.convert("RGB")
        rgba = rgb.copy()
        rgba.putalpha(u2net_mask(rgb))
        how = "u2net"

    a = np.asarray(rgba.getchannel("A"))
    solid = a >= 128
    cover = float(solid.mean())
    if cover < MIN_COVER:
        raise Rejected("No object was found: the cut-out left almost nothing. Use a picture where the "
                       "object is large and stands out from a plain background.")
    if cover > MAX_COVER:
        raise Rejected("The background could not be separated from the object (almost all of the picture "
                       "was kept). Use a plain background in a different colour.")
    e = EDGE_PX
    if solid[:e].any() or solid[-e:].any() or solid[:, :e].any() or solid[:, -e:].any():
        raise Rejected("The object touches the edge of the picture, so part of it is probably cut off. "
                       "Use a picture with space all round the object.")
    if cover < 0.08:
        warnings.append("The object is small in the picture; it is enlarged, so it may look soft.")

    rgba.putalpha(Image.fromarray(np.where(a < 8, 0, a).astype(np.uint8), "L"))   # drop faint halo
    out = resize_foreground(rgba).resize((OUT_SIDE, OUT_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Prepared(png=buf.getvalue(), original_size=size, cutout=how, cover=round(cover, 3), warnings=warnings)
