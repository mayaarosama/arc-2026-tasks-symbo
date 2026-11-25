"""What the worker does differently per task, in show order (Task 1 t2i, 2 i23d, 3 music, 4 llm, 8 layout).

app.py stays task-neutral and looks each rule up here:
  requested(req)          outputs the job must produce (0 = a throughput run: as many as fit in the budget)
  settings_echo(req)      the request settings written into the result and the run log
  prepare_inputs(req, d)  inputs written to disk before the clock starts (i23d: an uploaded picture)
  MOCK_BACKEND            the stand-in a mock worker runs for the task's models
  OWN_THREADS             tasks whose engine sets its own CPU threads (llama-server), not torch
"""
from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from typing import Callable

from ..schemas import JobRequest

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MAX_UPLOAD_BYTES = 20 * 2**20            # an uploaded input picture, already normalised by the conductor
RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")   # a run_id becomes a file name


# ------------------------------------------------------------------ Task 1: text to image
def _t2i_settings(req: JobRequest) -> dict:
    return {"width": req.width, "height": req.height, "steps": req.steps,
            "guidance": req.guidance, "batch": req.batch, "seed": req.seed}


# ------------------------------------------------------------------ Task 2: image to 3D
def _i23d_settings(req: JobRequest) -> dict:
    return {k: v for k, v in {"resolution": req.resolution, "texture_size": req.texture_size,
                              "seed": req.seed}.items() if v is not None}


def save_upload(b64: str, results_dir: Path, run_id: str) -> Path:
    """An uploaded input image (already checked and normalised by the conductor) -> a PNG file."""
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("run_id may hold only letters, digits, - and _")
    try:
        data = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("image_b64 is not valid base64") from e
    if len(data) > MAX_UPLOAD_BYTES or not data.startswith(PNG_MAGIC):
        raise ValueError(f"image_b64 must be a PNG of at most {MAX_UPLOAD_BYTES // 2**20} MB")
    d = results_dir / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{run_id}.png"
    path.write_bytes(data)
    return path


def _i23d_inputs(req: JobRequest, results_dir: Path) -> None:
    if req.image_b64:              # an operator upload: write it out before the clock starts
        req.image_path = str(save_upload(req.image_b64, results_dir, req.run_id))
        req.image_b64 = None


# ------------------------------------------------------------------ Task 3: music
def _music_settings(req: JobRequest) -> dict:
    return {"length_s": req.length_s, "seed": req.seed, "repeat_until_budget": req.repeat_until_budget}


# ------------------------------------------------------------------ Task 4: language model
def _streams_requested(req: JobRequest) -> int:
    return 0 if req.repeat_until_budget else max(1, req.streams)


def _llm_settings(req: JobRequest) -> dict:
    return {"streams": req.streams, "max_tokens": req.max_tokens, "temperature": req.temperature,
            "seed": req.seed, "repeat_until_budget": req.repeat_until_budget}


# ------------------------------------------------------------------ Task 8: layout
def _layout_settings(req: JobRequest) -> dict:
    return {"room_type": req.room_type, "max_objects": req.max_objects, "streams": req.streams,
            "max_tokens": req.max_tokens, "temperature": req.temperature, "seed": req.seed,
            "repeat_until_budget": req.repeat_until_budget,
            **({"repairs": req.repairs, "review_rounds": req.review_rounds}
               if req.repairs or req.review_rounds else {}),
            **({"layout_prompt": req.layout_prompt} if req.layout_prompt else {})}


# ------------------------------------------------------------------ the tables app.py reads
REQUESTED: dict[str, Callable[[JobRequest], int]] = {
    "t2i": lambda req: max(1, req.batch),
    "i23d": lambda req: 1,
    "music": lambda req: 0,                  # whole tracks until the budget
    "llm": _streams_requested,
    "layout": _streams_requested,
}
SETTINGS: dict[str, Callable[[JobRequest], dict]] = {
    "t2i": _t2i_settings, "i23d": _i23d_settings, "music": _music_settings,
    "llm": _llm_settings, "layout": _layout_settings,
}
INPUTS: dict[str, Callable[[JobRequest, Path], None]] = {"i23d": _i23d_inputs}
MOCK_BACKEND = {"t2i": "mock", "i23d": "mock", "music": "music_stub", "llm": "llm_mock", "layout": "layout_mock"}
OWN_THREADS = {"llm", "layout"}


def requested(req: JobRequest) -> int:
    return REQUESTED[req.task](req)


def settings_echo(req: JobRequest) -> dict:
    return SETTINGS[req.task](req)


def prepare_inputs(req: JobRequest, results_dir: Path) -> None:
    if req.task in INPUTS:
        INPUTS[req.task](req, results_dir)
