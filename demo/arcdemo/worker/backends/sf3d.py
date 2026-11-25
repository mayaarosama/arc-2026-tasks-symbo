"""Stable Fast 3D — the CPU tier and GPU fallback.

SF3D is a single feed-forward pass (no diffusion sampling) and upstream supports CPU natively
(SF3D_USE_CPU=1 forces it). That makes a *live* CPU 3D cell plausible — measure it in rehearsal
before committing the CPU column of segment 3 to replay.

Install: clone github.com/Stability-AI/stable-fast-3d, `pip install -r requirements.txt`
(builds its texture_baker / uv_unwrapper extensions), and put the repo on PYTHONPATH.
API checked against sf3d/system.py on 2026-11-18:
  SF3D.from_pretrained(name_or_path, config_name, weight_name)
  run_image(image | [images], bake_resolution, remesh="none", vertex_count=-1) -> (mesh | [mesh], dict)
run_image requires RGBA with the object already cut out; our renders are RGBA, and we re-centre
the foreground ourselves (upstream run.py uses rembg + resize_foreground, ratio 0.85).
"""
from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

from PIL import Image

try:                     # the i23d venv only: the worker imports this module when an i23d model loads
    import torch
except ImportError:
    torch = None

from ...defaults import DEFAULTS
from ...imageprep import resize_foreground  # shared with the conductor's upload step
from ...schemas import JobRequest
from ..metrics import cuda_sync
from .base import Backend, JobContext


class StableFast3D(Backend):
    task = "i23d"

    def load(self) -> None:
        dev = self.devices[0]
        if dev == "cpu":
            os.environ["SF3D_USE_CPU"] = "1"
        # Deferred imports (the one exception to imports-at-the-top here): sf3d may not be importable
        # until its third_party clone is put on sys.path below.
        try:
            import sf3d  # noqa: F401 — on PYTHONPATH already (setup_gpu_machine.sh sets it)
        except ModuleNotFoundError:   # a worker started by hand: use the repo's own clone
            clone = Path(__file__).resolve().parents[3] / "third_party" / "stable-fast-3d"
            if not (clone / "sf3d").is_dir():
                raise ModuleNotFoundError(f"No module named 'sf3d': not on PYTHONPATH and no clone at {clone}")
            sys.path.insert(0, str(clone))
        from sf3d.system import SF3D

        model = SF3D.from_pretrained(self.repo, config_name="config.yaml",
                                     weight_name="model.safetensors")
        model.to(dev)
        model.eval()
        self.model = model

    def describe(self, req: JobRequest) -> dict:
        dev = self.devices[0]
        return {"texture_size": req.texture_size or DEFAULTS.i23d.sf3d.texture_size,
                "remesh": "none",
                "precision": "bf16 autocast" if dev.startswith("cuda") else "fp32"}

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        dev = self.devices[0]
        img = resize_foreground(Image.open(req.image_path).convert("RGBA"))
        s = self.describe(req)
        ctx.result.progress.total = 1
        ctx.progress(stage="generating")
        autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if dev.startswith("cuda") else contextlib.nullcontext())
        with torch.no_grad(), autocast:
            mesh, _ = self.model.run_image([img], bake_resolution=s["texture_size"],
                                              remesh=s["remesh"], vertex_count=-1)
        cuda_sync(self.devices)
        ctx.result.generation_seconds = round(ctx.clock.now(), 3)
        mesh = mesh[0] if isinstance(mesh, list) else mesh

        ctx.progress(stage="export")
        t_exp = ctx.clock.now()
        path = ctx.output_path(".glb")
        mesh.export(str(path), include_normals=True)
        ctx.result.export_seconds = round(ctx.clock.now() - t_exp, 3)
        ctx.record_output(path)
