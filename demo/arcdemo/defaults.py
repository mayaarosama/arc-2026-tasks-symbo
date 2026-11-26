"""Runtime defaults from config/defaults.yaml: the one declared home of every tunable value.

    from arcdemo.defaults import DEFAULTS
    DEFAULTS.conductor.poll_every_seconds

The schema below is the expected format: every section forbids unknown keys and checks types, so a
typo in defaults.yaml stops the process at import with the key's name (pydantic's message).
Set ARC_DEFAULTS to read another file (tests, a one-off experiment).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[1]           # demo/
DEFAULTS_PATH = Path(os.environ.get("ARC_DEFAULTS") or ROOT / "config" / "defaults.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Conductor(_Section):
    host: str
    port: int
    config: str
    session_dir: str
    replay_dir: str
    timeout_grace_seconds: float
    unreachable_after_seconds: float
    token_env: str
    http_timeout_seconds: float
    http_connect_seconds: float
    health_every_seconds: float
    health_timeout_seconds: float
    unload_timeout_seconds: float
    prepare_timeout_seconds: float
    prepare_poll_seconds: float
    busy_retry_seconds: float
    busy_retry_every_seconds: float
    poll_every_seconds: float
    poll_timeout_seconds: float
    time_up_slack_seconds: float
    file_timeout_seconds: float
    reject_text_chars: int
    error_text_chars: int


class Stage(_Section):
    poll_ms: int
    operator_poll_ms: int
    views: list[str]
    spare_prompts: str


class Worker(_Section):
    host: str
    port: int
    results_dir: str
    job_history: int
    preload_warning_seconds: float
    error_text_chars: int
    disk_recheck_seconds: float
    env_hash_exclude: list[str]
    env_report_packages: list[str]
    max_upload_mb: int
    torch_compile_threads: int


class Moderation(_Section):
    max_prompt_chars: int


class Upload(_Section):
    max_mb: int
    min_side: int
    good_side: int
    max_side: int
    out_side: int
    fill: float
    min_cover: float
    max_cover: float
    edge_px: int
    small_cover_warning: float
    transparent_share: float
    label_chars: int
    extensions: list[str]
    rules: str


class Trellis2(_Section):
    resolution: int
    texture_size: int
    decimation: int
    simplify_faces: int


class Trellis1(_Section):
    ss_steps: int
    slat_steps: int
    ss_cfg: float
    slat_cfg: float
    texture_size: int
    simplify: float


class SF3D(_Section):
    texture_size: int


class ImageTo3D(_Section):
    upload: Upload
    trellis2: Trellis2
    trellis1: Trellis1
    sf3d: SF3D


class Music(_Section):
    steps: int
    sample_rate: int
    length_s: float


class Llm(_Section):
    server_binary: str
    slots: int
    ctx_per_slot: int
    gpu_layers: int
    read_timeout_seconds: float
    connect_timeout_seconds: float
    load_timeout_seconds: float
    stop_timeout_seconds: float
    warmup_prompt: str
    warmup_tokens: int


class LayoutChecks(_Section):
    door_clearance: float
    overlap_share: float
    wall_tolerance: float


class LayoutCosts(_Section):
    flag: float
    remove: float
    must_have: float


class Layout(_Section):
    room_type: str
    max_objects: int
    ctx_per_slot: int
    repair_tokens: int
    review_tokens: int
    max_edits: int
    idle_review_rounds: int
    layoutgpt_examples: int
    exemplars: str
    mask_cells: int
    mask_metres: float
    checks: LayoutChecks
    costs: LayoutCosts
    must_have: dict[str, list[str]]


class Mock(_Section):
    t2i_step_seconds: dict[str, float]
    t2i_step_default: float
    i23d_seconds: dict[str, float]
    i23d_seconds_default: float
    trellis2_resolution_factor: dict[int, float]
    needs_gb: dict[str, float]
    machine_gb: dict[str, float]
    cuda_only: list[str]
    colours: dict[str, list[int]]
    token_delay_seconds: float


class Defaults(_Section):
    conductor: Conductor
    stage: Stage
    worker: Worker
    moderation: Moderation
    i23d: ImageTo3D
    music: Music
    llm: Llm
    layout: Layout
    mock: Mock


def load(path: Path = DEFAULTS_PATH) -> Defaults:
    """Read and validate a defaults file. Raises pydantic.ValidationError naming every bad key."""
    return Defaults.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


DEFAULTS = load()
