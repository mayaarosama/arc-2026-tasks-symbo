"""Operator uploads for image-to-3D: checks, cut-out and normalisation (arcdemo/imageprep.py)."""
import io

import numpy as np
import pytest
from PIL import Image

from arcdemo import imageprep as P


def png(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def object_on(size=(800, 600), box=(250, 150, 550, 450), bg=(0, 0, 0, 0), mode="RGBA") -> Image.Image:
    """A red 'object' (an ellipse) on a background; transparent background by default."""
    from PIL import ImageDraw
    img = Image.new("RGBA", size, bg)
    ImageDraw.Draw(img).ellipse(box, fill=(200, 40, 40, 255))
    return img.convert(mode)


def test_transparent_png_is_used_as_is_and_normalised():
    r = P.prepare(png(object_on()))
    out = Image.open(io.BytesIO(r.png))
    assert (r.cutout, out.size, out.mode) == ("alpha", (512, 512), "RGBA")
    a = np.asarray(out)[..., 3] > 0
    ys, xs = np.nonzero(a)
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    assert abs(side / 512 - P.FILL) < 0.02                       # the object fills 85% of the side
    assert abs((xs.min() + xs.max()) / 2 - 255.5) < 2 and abs((ys.min() + ys.max()) / 2 - 255.5) < 2
    assert r.warnings == [] and r.original_size == (800, 600)


@pytest.mark.parametrize("data, why", [
    (b"not a picture at all", "not a picture"),
    (png(object_on(size=(200, 200), box=(50, 50, 150, 150))), "at least 256"),
    (png(object_on(box=(0, 150, 300, 450))), "touches the edge"),
    (png(Image.new("RGBA", (600, 600), (0, 0, 0, 0))), "No object"),
    (png(object_on(), "GIF"), "not accepted"),
])
def test_unusable_pictures_are_rejected_with_a_reason(data, why):
    with pytest.raises(P.Rejected, match=why):
        P.prepare(data)


def test_too_big_a_file_is_rejected_before_decoding():
    with pytest.raises(P.Rejected, match="limit is 10 MB"):
        P.prepare(b"\0" * (P.MAX_BYTES + 1))


def test_small_picture_is_accepted_with_a_warning():
    r = P.prepare(png(object_on(size=(400, 300), box=(120, 70, 280, 230))))
    assert any("Small picture" in w for w in r.warnings)


def test_a_photo_without_transparency_needs_the_cutout_model(monkeypatch):
    monkeypatch.setattr(P, "cutout_model_path", lambda: None)
    monkeypatch.setattr(P, "_SESSION", None)
    jpeg = png(object_on(bg=(210, 210, 215, 255), mode="RGB"), "JPEG")
    with pytest.raises(P.Rejected, match="download_weights.py --cutout"):
        P.prepare(jpeg)


@pytest.mark.skipif(P.cutout_model_path() is None, reason="cut-out model not downloaded (download_weights.py --cutout)")
def test_a_photo_on_a_plain_background_is_cut_out():
    pytest.importorskip("onnxruntime")
    src = Image.open("data/i23d/inputs/i23d-03.png").convert("RGBA").resize((1024, 1024))
    bg = Image.new("RGBA", src.size, (200, 200, 205, 255))
    bg.alpha_composite(src)
    r = P.prepare(png(bg.convert("RGB"), "JPEG"))
    assert r.cutout == "u2net" and Image.open(io.BytesIO(r.png)).size == (512, 512)
    mask = np.asarray(P.u2net_mask(bg.convert("RGB"))) >= 128
    truth = np.asarray(src.getchannel("A")) >= 128
    assert (mask & truth).sum() / (mask | truth).sum() > 0.95   # matches the true outline
