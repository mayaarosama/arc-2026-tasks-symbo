"""Rescale a picture so it meets the upload requirements for image-to-3D (arcdemo/imageprep.py).

    python scripts/fit_upload.py IMG_2089.jpeg            # writes IMG_2089-fit.png next to it
    python scripts/fit_upload.py IMG_2089.jpeg -o out.png

Stands a phone photo the right way up, scales it so the shortest side is GOOD_SIDE (512 px) and the
longest at most MAX_SIDE, and saves it as PNG (keeps transparency) under MAX_BYTES.
Rescaling cannot fix content: the object must still be whole, on a plain background.
"""
import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arcdemo.imageprep import GOOD_SIDE, MAX_SIDE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("image", type=Path)
ap.add_argument("-o", "--out", type=Path)
args = ap.parse_args()
out = args.out or args.image.with_name(args.image.stem + "-fit.png")

img = ImageOps.exif_transpose(Image.open(args.image))
img = img.convert("RGBA" if img.has_transparency_data else "RGB")
scale = min(GOOD_SIDE / min(img.size), MAX_SIDE / max(img.size))
if scale > 1:
    print(f"warning: upscaling {img.size} by {scale:.2f}; a larger original gives a more detailed model")
img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
img.save(out, "PNG", optimize=True)   # at most 512 x 4096 RGBA: always under MAX_BYTES
print(f"{out}: {img.width} x {img.height} px, {out.stat().st_size / 2**20:.2f} MB")
