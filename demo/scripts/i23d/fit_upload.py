"""Rescale a picture so it meets the upload requirements for image-to-3D (arcdemo/imageprep.py).

    python -m scripts.i23d.fit_upload IMG_2089.jpeg            # writes IMG_2089-fit.png next to it
    python -m scripts.i23d.fit_upload IMG_2089.jpeg -o out.png

Stands a phone photo the right way up, scales it so the shortest side is GOOD_SIDE (512 px) and the
longest at most MAX_SIDE, and saves it as PNG (keeps transparency) under MAX_BYTES.
Rescaling cannot fix content: the object must still be whole, on a plain background.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageOps

from arcdemo.imageprep import GOOD_SIDE, MAX_SIDE


def fit(src: Path, out: Path) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(src))
    img = img.convert("RGBA" if img.has_transparency_data else "RGB")
    scale = min(GOOD_SIDE / min(img.size), MAX_SIDE / max(img.size))
    if scale > 1:
        print(f"warning: upscaling {img.size} by {scale:.2f}; a larger original gives a more detailed model")
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    img.save(out, "PNG", optimize=True)   # at most 512 x 4096 RGBA: always under MAX_BYTES
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()
    out = args.out or args.image.with_name(args.image.stem + "-fit.png")
    img = fit(args.image, out)
    print(f"{out}: {img.width} x {img.height} px, {out.stat().st_size / 2**20:.2f} MB")


if __name__ == "__main__":
    main()
