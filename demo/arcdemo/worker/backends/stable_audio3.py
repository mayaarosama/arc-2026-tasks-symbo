"""Task 3 — music: Stable Audio 3.0 (Small-Music on any machine, Medium on CUDA only).

Library: github.com/Stability-AI/stable-audio-3 (cloned to third_party/stable-audio-3,
`pip install --no-deps -e third_party/stable-audio-3` plus requirements/music.txt).
API checked against stable_audio_3/model.py at commit 790793b (2026-11-18):
  StableAudioModel.from_pretrained(name, device=None, model_half=True)   name: "small-music" | "medium" | ...
  generate(prompt=, duration=, steps=8, cfg_scale=1.0, seed=-1, inpaint_audio=(sr, wav),
           inpaint_mask_start_seconds=) -> float32 tensor (batch, channels, samples), clamped to [-1, 1]
Weights resolve through hf_hub_download, so HF_HUB_OFFLINE=1 works once the repo is cached; the
T5Gemma text encoder ships inside the same repo (t5gemma-b-b-ul2/), no extra download.

Small runs on CPU and on any CUDA GPU without flash-attn (the library falls back to PyTorch
attention). Medium needs flash-attn 2; without it the output is static noise (upstream README).
Instrumental only: none of these models sing.

Precision: upstream picks fp16 whenever CUDA is *present*, even for a CPU model, so we always pass
it explicitly. dtype auto = fp16 on CUDA, fp32 on CPU; set dtype: fp32 for Pascal cards (GTX 10xx),
whose fp16 throughput is a small fraction of fp32.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

try:                     # the music venv only: the worker imports this module when a music model loads
    import soundfile as sf
    import torch
    from stable_audio_3 import StableAudioModel
except ImportError:
    sf = torch = StableAudioModel = None

from ...schemas import JobRequest
from ..metrics import cuda_sync
from .base import Backend, JobContext

DEFAULT_STEPS = 8          # the post-trained (ARC) checkpoints are tuned for 8 steps
SAMPLE_RATE = 44100


class StableAudio3(Backend):
    task = "music"

    @property
    def _device(self) -> str:
        return self.devices[0]

    @property
    def _half(self) -> bool:
        return self._device.startswith("cuda") and self.spec.get("dtype", "auto") in ("auto", "fp16")

    def load(self) -> None:
        if StableAudioModel is None:
            raise ModuleNotFoundError("stable_audio_3 is not installed in this venv (requirements/music.txt)")
        self.model = StableAudioModel.from_pretrained(self.spec["sa3_name"], device=self._device,
                                                      model_half=self._half)
        self.sample_rate = int(self.model.model_config.get("sample_rate", SAMPLE_RATE))

    def _steps(self, req: JobRequest) -> int:
        # JobRequest.steps defaults to the t2i value (28); only honour it when a setting gave it.
        return req.steps if "steps" in req.model_fields_set else DEFAULT_STEPS

    def describe(self, req: JobRequest) -> dict:
        return {"length_s": req.length_s, "steps": self._steps(req), "mode": req.mode or "text",
                "repeat_until_budget": req.repeat_until_budget,
                "precision": "fp16" if self._half else "fp32"}

    def _generate_track(self, req: JobRequest, seed: int, path: Path) -> None:
        length = float(req.length_s or 30)
        kwargs = {}
        if req.mode == "continuation" and req.audio_path:
            data, sr = sf.read(req.audio_path, dtype="float32", always_2d=True)   # (samples, ch)
            clip = torch.from_numpy(data.T.copy())
            kwargs = {"inpaint_audio": (sr, clip), "inpaint_mask_start_seconds": data.shape[0] / sr}
        audio = self.model.generate(prompt=req.prompt or "", duration=length, steps=self._steps(req),
                                    seed=seed, **kwargs)
        cuda_sync(self.devices)
        write_wav(path, audio[0].cpu().numpy(), self.sample_rate)

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        """Same throughput loop as the stub: whole tracks until the budget; a late track is discarded."""
        i = 0
        while not ctx.should_stop():
            path = ctx.output_path(f"-{i}.wav")
            ctx.progress(stage=f"track {i + 1}")
            self._generate_track(req, req.seed + i, path)
            if ctx.budget_hit():
                path.unlink(missing_ok=True)
                return
            ctx.record_output(path)
            i += 1
            if not req.repeat_until_budget or req.budget_seconds is None:
                return


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """audio: (channels, samples) float in [-1, 1]. Peak-normalised like the upstream example, 16-bit PCM."""
    peak = float(np.abs(audio).max()) or 1.0
    pcm = (np.clip(audio / peak, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[0])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.T.tobytes())
