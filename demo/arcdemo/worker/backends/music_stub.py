"""Task 3 (reserve) — music placeholder: what mock workers run for every music model.

The real model is Stable Audio 3.0 (backends/stable_audio3.py). This stub keeps the same contract
and throughput loop ("how many complete tracks in the budget") but writes a seeded tone, so UI
and failure drills need no weights. Partial tracks are discarded.
"""
from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

from ...schemas import JobRequest
from .base import Backend, JobContext


class MusicStub(Backend):
    task = "music"

    def load(self) -> None:
        pass

    def describe(self, req: JobRequest) -> dict:
        return {"stub": True, "length_s": req.length_s, "repeat_until_budget": req.repeat_until_budget}

    def _generate_track(self, req: JobRequest, seed: int, path: Path) -> None:
        if self.cfg.get("mock"):
            # Simulated generation time, scaled per machine like the other mock tasks: a 30 s track
            # takes 3 s at workstation speed (mock_speed 1.0), 0.45 s on the DGX mock, 36 s on the CPU mock.
            time.sleep(0.1 * (req.length_s or 10) * float(self.cfg.get("mock_speed", 1.0)))
        write_tone(path, seconds=min(req.length_s or 10, 5), seed=seed)

    def generate(self, req: JobRequest, ctx: JobContext) -> None:
        i = 0
        while not ctx.should_stop():
            path = ctx.output_path(f"-{i}.wav")
            ctx.progress(stage=f"track {i + 1}")
            self._generate_track(req, req.seed + i, path)
            if ctx.budget_hit():          # finished after the bell: discard, do not count
                path.unlink(missing_ok=True)
                return
            ctx.record_output(path)
            i += 1
            if not req.repeat_until_budget or req.budget_seconds is None:
                return


def write_tone(path, seconds: float = 5, sr: int = 44100, seed: int = 0) -> None:
    """Placeholder track: a short seeded melody, 44.1 kHz stereo WAV."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    notes = rng.choice([220, 247, 262, 294, 330, 392, 440], size=int(seconds * 2) + 1)
    f = np.repeat(notes, sr // 2)[: len(t)]
    sig = 0.2 * np.sin(2 * np.pi * f * t) * np.exp(-3 * (t % 0.5))
    data = (np.stack([sig, sig], 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())
