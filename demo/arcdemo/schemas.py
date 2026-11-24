"""Wire format shared by workers and the conductor.

Deviation from the task specs, on purpose: the specs describe one blocking POST per run.
Here a job is submitted, returns immediately, and is polled. That lets the stage show
progress (step k of n, image 3 of 8), lets the operator cancel, and means a venue network
blip during a 60 s run costs one poll rather than the whole result. The result record carries
every field named in the spec responses (completed, first_output_seconds, …).

JobRequest rejects unknown fields, so a typo in a demo.yaml setting fails the job with a 422
naming the field instead of being silently ignored.
"""
from __future__ import annotations

import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Task = Literal["t2i", "i23d", "music", "llm", "layout"]
Status = Literal["queued", "loading", "running", "ok", "budget_exceeded",
                 "out_of_memory", "error", "cancelled"]
TERMINAL: set[str] = {"ok", "budget_exceeded", "out_of_memory", "error", "cancelled"}


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task: Task
    model: str                                  # key in config/models.yaml
    input_id: str
    protocol: str = "iso_time"
    budget_seconds: Optional[float] = None      # None = run to completion (iso-quality)
    seed: int = 0
    # t2i
    prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: int = 28
    guidance: float = 4.5
    batch: int = 1
    # i23d
    image_path: Optional[str] = None            # path relative to the worker's demo root
    resolution: Optional[int] = None
    texture_size: Optional[int] = None
    decimation: Optional[int] = None
    ss_steps: Optional[int] = None
    slat_steps: Optional[int] = None
    # music
    mode: Optional[str] = None
    audio_path: Optional[str] = None
    length_s: Optional[float] = None
    repeat_until_budget: bool = False           # music, llm: keep producing until the budget
    # llm (prompt and seed shared with t2i)
    streams: int = 1                            # concurrent replies (continuous batching on the server)
    max_tokens: int = 256                       # per reply
    temperature: float = 0.0                    # 0 = greedy: the same answer on every machine
    # layout (streams, max_tokens, temperature, repeat_until_budget shared with llm)
    room_type: Optional[str] = None             # bedroom | living_room | library
    floor_plan: Optional[list[list[float]]] = None   # outline, metres, [[x, y], ...]
    door: Optional[list[float]] = None          # [x, y] on the outline
    max_objects: Optional[int] = None


class Progress(BaseModel):
    step: int = 0
    steps: int = 0
    done: int = 0           # finished outputs
    total: int = 0          # requested outputs
    stage: str = ""         # free text, e.g. "sparse structure", "export"


class JobResult(BaseModel):
    run_id: str
    machine: str
    task: Task
    model: str
    model_repo: str = ""
    input_id: str
    protocol: str
    status: Status = "queued"
    message: str = ""
    outputs: list[str] = Field(default_factory=list)   # file names, fetch via GET /files/{name}
    output_times: list[float] = Field(default_factory=list)  # seconds from clock start, per output
    completed: int = 0
    requested: int = 0
    load_seconds: float = 0.0
    first_output_seconds: Optional[float] = None
    all_outputs_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None         # i23d: model time
    export_seconds: Optional[float] = None             # i23d: mesh export, reported separately
    elapsed_seconds: float = 0.0                       # clock so far (live while running)
    peak_memory_mb: Optional[float] = None             # torch max_memory_allocated (GPU) / peak RSS (CPU)
    peak_reserved_mb: Optional[float] = None           # GPU only: allocator reserved, closer to nvidia-smi
    tokens: int = 0                                    # llm: tokens generated inside the budget, all streams
    tokens_per_second: Optional[float] = None          # llm: tokens / time of the last counted token
    first_token_seconds: Optional[float] = None        # llm: time to first token
    text: str = ""                                     # llm: stream 0's current reply (what the stage shows)
    objects: list[dict[str, Any]] = Field(default_factory=list)  # layout: stream 0's placed objects, with times
    progress: Progress = Field(default_factory=Progress)
    settings: dict[str, Any] = Field(default_factory=dict)  # echo of what actually ran, incl. precision/offload


class LoadRequest(BaseModel):
    model: str


class HealthInfo(BaseModel):
    machine: str
    tasks: list[str]
    mock: bool
    busy: bool
    loaded: list[str]
    loading: Optional[str] = None
    env_hash: str
    available: Optional[list[str]] = None   # this worker's models whose weights are on disk
