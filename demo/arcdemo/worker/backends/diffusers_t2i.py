"""Text-to-image through diffusers: SD 3.5 (Large / Large Turbo / Medium) and SD 1.5.

Decisions that protect the comparison:
- Seeds come from a CPU torch.Generator, one per image (seed + i). A CUDA generator and a CPU
  generator produce *different* noise for the same seed, so without this the "same seed on
  every machine" claim is false and iso-quality images would not match across machines.
  Image i is therefore identical in content whichever GPU, or how many GPUs, produced it.
- Multi-GPU = one pipeline per device, images assigned round-robin. A single image is never
  split across GPUs.
- No CPU offload, no attention slicing: both cost time and would distort the comparison.
  If memory forces a change, change the *model variant* (models.yaml) so it is labelled.
- load() ends with a 2-step warm-up generation, so one-time GPU init is load time, not job time.
- Budget: a step callback aborts the in-flight image the moment the budget is hit; that
  image is discarded (partial images are never returned).
"""
from __future__ import annotations

import threading

try:                     # the t2i venv only: the worker imports this module when a t2i model loads
    import torch
    from diffusers import AutoPipelineForText2Image, StableDiffusion3Pipeline
    from transformers import BitsAndBytesConfig, T5EncoderModel
except ImportError:
    torch = AutoPipelineForText2Image = StableDiffusion3Pipeline = BitsAndBytesConfig = T5EncoderModel = None

from ...schemas import JobRequest
from .base import Backend, JobContext, resolve_dtype


class _BudgetAbort(Exception):
    pass


class DiffusersT2I(Backend):
    task = "t2i"

    def load(self) -> None:
        if StableDiffusion3Pipeline is None:
            raise ModuleNotFoundError("diffusers / torch are not installed in this venv (requirements/t2i.txt)")
        # Built in a helper, never held in a local here: a frame that a library keeps alive (a
        # captured stack) would otherwise keep the weights after unload.
        self.model = self._build()          # [(device, pipeline), ...]

    def _build(self) -> list:
        pipes = []
        self._dtypes: list[str] = []
        is_sd3 = "stable-diffusion-3" in self.repo
        for dev in self.devices:
            dtype = resolve_dtype(self.spec, self.cfg, dev)
            kw: dict = {"torch_dtype": dtype}
            if is_sd3:
                t5 = self.spec.get("t5", "full")
                if t5 == "none":
                    kw.update(text_encoder_3=None, tokenizer_3=None)
                elif t5 == "int8":
                    if not dev.startswith("cuda"):
                        raise RuntimeError("t5: int8 needs CUDA (bitsandbytes); use t5: none on CPU")
                    kw["text_encoder_3"] = T5EncoderModel.from_pretrained(
                        self.repo, subfolder="text_encoder_3", torch_dtype=dtype,
                        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                        device_map={"": dev})
                pipe = StableDiffusion3Pipeline.from_pretrained(self.repo, **kw)
            else:   # SD 1.5 keeps its own safety checker: a free guard for the live prompt
                pipe = AutoPipelineForText2Image.from_pretrained(self.repo, **kw)
            pipe.to(dev)   # an int8 T5 is already placed; diffusers skips moving it
            pipe.set_progress_bar_config(disable=True)
            # Warm-up off the clock: the first call pays CUDA/cuBLAS init and kernel selection
            # (90 s on a GTX 1050 Ti, 3 s after). Without this the first timed job is penalised.
            with torch.inference_mode():
                pipe(prompt="warm-up", num_inference_steps=2, generator=torch.Generator("cpu").manual_seed(0))
            pipes.append((dev, pipe))
            self._dtypes.append(str(dtype).replace("torch.", ""))
        return pipes

    def describe(self, req: JobRequest) -> dict:
        return {
            "precision": self._dtypes[0],
            "t5": self.spec.get("t5", "n/a"),
            "devices": len(self.model),
            "micro_batch": int(self.cfg.get("micro_batch", 1)),
            "offload": "none",
            "seeding": "cpu generator, seed+i per image",
        }

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        if not req.prompt:
            raise ValueError("t2i job without a prompt")
        n = max(1, req.batch)
        mb = max(1, int(self.cfg.get("micro_batch", 1)))
        pipes = self.model
        ctx.result.progress.total = n
        ctx.progress(step=0, steps=req.steps, stage="denoising")
        assignments = [list(range(k, n, len(pipes))) for k in range(len(pipes))]
        errors: list[BaseException] = []

        def work(k: int) -> None:
            dev, pipe = pipes[k]
            idxs = assignments[k]
            try:
                with torch.inference_mode():
                    for j in range(0, len(idxs), mb):
                        if ctx.should_stop():
                            return
                        chunk = idxs[j:j + mb]
                        gens = [torch.Generator("cpu").manual_seed(req.seed + i) for i in chunk]

                        def on_step(p, step, timestep, cb_kwargs):
                            if k == 0:
                                ctx.progress(step=step + 1)
                            if ctx.should_stop():
                                raise _BudgetAbort()
                            return cb_kwargs

                        out = pipe(prompt=[req.prompt] * len(chunk),
                                   num_inference_steps=req.steps, guidance_scale=req.guidance,
                                   width=req.width, height=req.height, generator=gens,
                                   callback_on_step_end=on_step)
                        for i, img in zip(chunk, out.images):
                            ctx.save_image(i, img)
            except _BudgetAbort:
                return
            except BaseException as e:  # surfaced to the job runner below
                errors.append(e)

        threads = [threading.Thread(target=work, args=(k,), daemon=True) for k in range(len(pipes))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
