"""Mock backend: realistic-looking timing and outputs with no GPU and no weights.

Used for (1) building and testing the stage UI on a laptop, (2) the failure drill (kill a mock
worker mid-run and watch the REPLAY badge engage), (3) operator practice.
Mock timings are invented and must never be shown as results — the worker reports mock: true
in /health and the conductor stamps every mock cell with a MOCK badge.
"""
from __future__ import annotations

import math
import time

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from ...schemas import JobRequest
from .base import Backend, JobContext

# seconds per denoising step for one 1024² image at workstation speed (mock_speed = 1.0)
_T2I_STEP = {"sd35-large": 1.25, "sd35-large-turbo": 1.25, "sd15": 0.15}
_T2I_STEP_DEFAULT = 0.5
# seconds per 3D generation at workstation speed
_I23D_SECONDS = {"trellis2-4b": 30.0, "trellis-image-large": 25.0, "sf3d": 2.0}
# GB of accelerator memory each model needs, to simulate out-of-memory on the 16 GB card
_NEEDS_GB = {"trellis2-4b": 24, "sd35-large": 28, "sd35-medium": 17}
_MACHINE_GB = {"dgx": 141, "workstation": 16, "cpu": 16}
_CUDA_ONLY = {"trellis2-4b", "trellis-image-large"}


class MockBackend(Backend):
    def load(self) -> None:
        machine = self.cfg["machine"]
        if self.key in _CUDA_ONLY and machine == "cpu":
            raise RuntimeError(f"{self.key} requires CUDA extensions; no CPU path")
        if _NEEDS_GB.get(self.key, 0) > _MACHINE_GB.get(machine, 999):
            time.sleep(1.5)  # real OOMs happen part-way through loading
            raise MemoryError(f"mock: CUDA out of memory loading {self.key} "
                              f"({_NEEDS_GB[self.key]} GB needed, {_MACHINE_GB[machine]} GB available)")
        time.sleep(0.3)

    @property
    def speed(self) -> float:
        return float(self.cfg.get("mock_speed", 1.0))

    def describe(self, req: JobRequest) -> dict:
        return {"mock": True, "devices": len(self.devices)}

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        {"t2i": self._t2i, "i23d": self._i23d}[req.task](req, ctx)

    # ---- t2i: devices work in parallel, images per device in sequence --------------------
    def _t2i(self, req: JobRequest, ctx: JobContext) -> None:
        n = max(1, req.batch)
        ndev = len(self.devices)
        per_step = (_T2I_STEP.get(self.key, _T2I_STEP_DEFAULT) * self.speed
                    * (req.width * req.height) / (1024 * 1024))
        ctx.result.progress.total = n
        for r in range(math.ceil(n / ndev)):
            idxs = range(r * ndev, min(n, (r + 1) * ndev))
            for step in range(req.steps):
                if ctx.should_stop():
                    return
                time.sleep(per_step)
                ctx.progress(step=step + 1, steps=req.steps, stage="denoising")
            for i in idxs:
                ctx.save_image(i, _placeholder_image(req, i, self.cfg["machine"], self.key))

    # ---- i23d ------------------------------------------------------------------------------
    def _i23d(self, req: JobRequest, ctx: JobContext) -> None:

        total = _I23D_SECONDS.get(self.key, 10.0) * self.speed
        if self.key == "trellis2-4b" and req.resolution:
            total *= {512: 0.2, 1024: 1.0, 1536: 3.5}.get(req.resolution, 1.0)
        ctx.result.progress.total = 1
        for stage, frac in (("sparse structure", 0.3), ("shape", 0.4), ("texture", 0.3)):
            ctx.progress(stage=stage)
            end = ctx.clock.now() + total * frac
            while ctx.clock.now() < end:
                if ctx.cancel.is_set():
                    return
                time.sleep(0.05)
        ctx.result.generation_seconds = round(ctx.clock.now(), 3)
        ctx.progress(stage="export")
        t_exp = ctx.clock.now()
        detail = {"trellis2-4b": 5, "trellis-image-large": 3, "sf3d": 2}.get(self.key, 2)
        body = trimesh.creation.icosphere(subdivisions=detail, radius=0.35)
        body.apply_scale([1.0, 0.8, 1.0])
        handle = trimesh.creation.torus(major_radius=0.18, minor_radius=0.04)
        handle.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [1, 0, 0]))
        handle.apply_translation([0.38, 0, 0])
        mesh = trimesh.util.concatenate([body, handle])
        colour = {"dgx": [118, 185, 0, 255], "workstation": [74, 168, 255, 255],
                  "cpu": [240, 166, 64, 255]}.get(self.cfg["machine"], [200, 200, 200, 255])
        mesh.visual.vertex_colors = colour
        time.sleep(0.2 * self.speed)
        path = ctx.output_path(".glb")
        mesh.export(path)
        ctx.result.export_seconds = round(ctx.clock.now() - t_exp, 3)
        ctx.record_output(path)


def _placeholder_image(req: JobRequest, i: int, machine: str, model: str) -> Image.Image:
    side = 512
    rng = np.random.default_rng(req.seed + i)
    base = rng.integers(40, 200, size=3)
    y = np.linspace(0, 1, side)[:, None, None]
    x = np.linspace(0, 1, side)[None, :, None]
    arr = (base * (0.5 + 0.5 * y) + 55 * x).clip(0, 255).astype(np.uint8)
    arr = np.broadcast_to(arr, (side, side, 3)).copy()
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    text = f"MOCK · {machine}\n{model}\n{req.width}x{req.height} · {req.steps} steps\nseed {req.seed + i}"
    d.multiline_text((20, 20), text, fill=(255, 255, 255), font=font, spacing=6)
    return img
