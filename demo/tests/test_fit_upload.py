"""scripts/fit_upload.py: its output meets the upload requirements (arcdemo/imageprep.py)."""
import subprocess
import sys
from pathlib import Path

from PIL import Image

from arcdemo import imageprep as P
from tests.test_imageprep import object_on

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fit_upload.py"


def fit(tmp_path, img: Image.Image, name: str, **save) -> Image.Image:
    src, out = tmp_path / name, tmp_path / "out.png"
    img.save(src, **save)
    subprocess.run([sys.executable, str(SCRIPT), str(src), "-o", str(out)], check=True)
    return Image.open(out)


def test_small_picture_is_scaled_up_and_accepted_without_the_size_warning(tmp_path):
    out = fit(tmp_path, object_on(size=(400, 300), box=(120, 70, 280, 230)), "small.png")
    assert out.size == (683, 512) and out.mode == "RGBA"
    assert not any("Small picture" in w for w in P.prepare((tmp_path / "out.png").read_bytes()).warnings)


def test_large_photo_is_scaled_down_and_stood_upright(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6                                            # phone held upright: rotate 90°
    out = fit(tmp_path, object_on(size=(4000, 3000), mode="RGB"), "big.jpg", exif=exif)
    assert out.size == (512, 683) and out.mode == "RGB"


def test_palette_transparency_is_kept(tmp_path):
    out = fit(tmp_path, object_on().convert("P"), "pal.png")
    assert out.mode == "RGBA" and out.getchannel("A").getextrema() == (0, 255)
