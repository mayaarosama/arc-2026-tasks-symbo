"""Image-to-3D with TRELLIS.2-4B (DGX tier) and TRELLIS v1 image-large (workstation tier).

Both are installed from their GitHub repos with setup.sh (compiled CUDA extensions, Linux only);
the worker's i23d env must have the repo on PYTHONPATH. APIs below were checked against the
public READMEs / pipeline source on 2026-11-18.

Budget is *soft* for 3D: the samplers cannot be interrupted mid-stage, so a run that crosses
the budget finishes and is reported as budget_exceeded with its true time. Mesh export is
timed separately (export_seconds), as the spec requires.

Inputs are RGBA renders with a real alpha channel, so both pipelines skip background removal
(TRELLIS.2 checks for non-trivial alpha before calling RMBG-2.0). RMBG-2.0 is still *loaded*
by TRELLIS.2's from_pretrained — see the licensing note in the plan.
"""
from __future__ import annotations

import os

from PIL import Image

from ...defaults import DEFAULTS
from ...schemas import JobRequest
from ..metrics import cuda_sync
from .base import Backend, JobContext

_T2_PIPELINE_TYPE = {512: "512", 1024: "1024_cascade", 1536: "1536_cascade"}   # TRELLIS.2's own names
T2 = DEFAULTS.i23d.trellis2             # defaults when the job does not set them: config/defaults.yaml i23d
T1 = DEFAULTS.i23d.trellis1


def _open_rgba(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGBA":
        raise ValueError(f"{path}: expected an RGBA render with alpha (got {img.mode}); "
                         "re-run scripts/i23d/prepare_gso.py")
    return img


class Trellis2(Backend):
    task = "i23d"

    def load(self) -> None:
        # Deferred imports (an exception to imports-at-the-top): TRELLIS reads these environment
        # variables when it is imported, and Trellis1 / Trellis2 need different ones.
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        pipe = Trellis2ImageTo3DPipeline.from_pretrained(self.repo)
        # low_vram=True (upstream default) shuttles sub-models between CPU and GPU, which costs
        # time. Turn it off where memory is plentiful (DGX) via model_overrides in the worker config.
        if "low_vram" in self.spec:
            pipe.low_vram = bool(self.spec["low_vram"])
        pipe.cuda()
        self.model = pipe

    def describe(self, req: JobRequest) -> dict:
        res = req.resolution or T2.resolution
        return {"pipeline_type": _T2_PIPELINE_TYPE.get(res, _T2_PIPELINE_TYPE[T2.resolution]),
                "low_vram": getattr(self.model, "low_vram", None),
                "texture_size": req.texture_size or T2.texture_size, "decimation": req.decimation or T2.decimation}

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        import o_voxel  # deferred: part of the TRELLIS.2 install, imported after load() set its environment

        img = _open_rgba(req.image_path)
        s = self.describe(req)
        ctx.result.progress.total = 1
        ctx.progress(stage=f"generating ({s['pipeline_type']})")
        mesh = self.model.run(img, seed=req.seed, pipeline_type=s["pipeline_type"])[0]
        cuda_sync(self.devices)
        ctx.result.generation_seconds = round(ctx.clock.now(), 3)
        if ctx.cancel.is_set():
            return

        ctx.progress(stage="export")
        t_exp = ctx.clock.now()
        mesh.simplify(T2.simplify_faces)  # nvdiffrast limit, per the upstream example
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=s["decimation"],
            texture_size=s["texture_size"],
            remesh=True, remesh_band=1, remesh_project=0, verbose=False)
        path = ctx.output_path(".glb")
        glb.export(str(path), extension_webp=True)
        ctx.result.export_seconds = round(ctx.clock.now() - t_exp, 3)
        ctx.record_output(path)


class Trellis1(Backend):
    task = "i23d"

    def load(self) -> None:
        # Deferred import: TRELLIS v1 reads SPCONV_ALGO when it is imported.
        os.environ.setdefault("SPCONV_ALGO", "native")
        from trellis.pipelines import TrellisImageTo3DPipeline

        pipe = TrellisImageTo3DPipeline.from_pretrained(self.repo)
        pipe.cuda()
        self.model = pipe

    def describe(self, req: JobRequest) -> dict:
        return {"ss_steps": req.ss_steps or T1.ss_steps, "slat_steps": req.slat_steps or T1.slat_steps,
                "texture_size": req.texture_size or T1.texture_size, "simplify": T1.simplify}

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        from trellis.utils import postprocessing_utils  # deferred: see load()

        img = _open_rgba(req.image_path)
        s = self.describe(req)
        ctx.result.progress.total = 1
        ctx.progress(stage="generating")
        out = self.model.run(
            img, seed=req.seed, formats=["gaussian", "mesh"],
            sparse_structure_sampler_params={"steps": s["ss_steps"], "cfg_strength": T1.ss_cfg},
            slat_sampler_params={"steps": s["slat_steps"], "cfg_strength": T1.slat_cfg})
        cuda_sync(self.devices)
        ctx.result.generation_seconds = round(ctx.clock.now(), 3)
        if ctx.cancel.is_set():
            return

        ctx.progress(stage="export")
        t_exp = ctx.clock.now()
        glb = postprocessing_utils.to_glb(out["gaussian"][0], out["mesh"][0],
                                          simplify=s["simplify"], texture_size=s["texture_size"])
        path = ctx.output_path(".glb")
        glb.export(str(path))
        ctx.result.export_seconds = round(ctx.clock.now() - t_exp, 3)
        ctx.record_output(path)
