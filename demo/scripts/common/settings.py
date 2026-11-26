"""The scripts' declared inputs: config/scripts.yaml (validated here) and config/versions.env.

    from scripts.common.settings import SCRIPTS, VERSIONS
    SCRIPTS.rehearse.max_wait_seconds        VERSIONS["LLAMA_TAG"]
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

DEMO = Path(__file__).resolve().parents[2]
SCRIPTS_PATH = DEMO / "config" / "scripts.yaml"
VERSIONS_PATH = DEMO / "config" / "versions.env"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DownloadWeights(_Section):
    fallbacks: dict[str, list[str]]          # "<machine>/<task>" -> models.yaml keys
    ignore: dict[str, list[str]]             # repo -> file patterns
    cutout_url: str
    cutout_bytes: int
    cutout_path: str


class Rehearse(_Section):
    max_wait_seconds: float


class RunMockDemo(_Section):
    workers: dict[str, int]
    conductor_config: str


class VerifyOffline(_Section):
    jobs: dict[str, dict]
    cutout_image: str
    cutout_min_iou: float


class BuildRehearsalPrompts(_Section):
    challenges: list[str]
    per_challenge: int
    seed: int
    first_prompt_seed: int
    min_words: int
    max_words: int
    out: str


class Camera(_Section):
    elevation: float
    azimuth: float
    fov: float


class PrepareGso(_Section):
    fuel_url: str
    download_timeout_seconds: float
    camera: Camera
    size: int
    supersample: int


class SampleLayout(_Section):
    out_dir: str


class PlanDownloads(_Section):
    vram_gb: dict[str, float]
    gguf_overhead_gb: float
    vram_slack_gb: float


class CheckCpu(_Section):
    min_ram_gb: float
    min_disk_gb: float


class FetchLlamaCpp(_Section):
    assets: dict[str, tuple[str, int]]       # platform -> (file name with {tag}, size in bytes)


class Scripts(_Section):
    download_weights: DownloadWeights
    rehearse: Rehearse
    run_mock_demo: RunMockDemo
    verify_offline: VerifyOffline
    build_rehearsal_prompts: BuildRehearsalPrompts
    prepare_gso: PrepareGso
    sample_layout: SampleLayout
    plan_downloads: PlanDownloads
    check_cpu: CheckCpu
    fetch_llama_cpp: FetchLlamaCpp


def read_env(path: Path = VERSIONS_PATH) -> dict[str, str]:
    """KEY=value lines (the same file bash sources); comments and blank lines skipped, quotes dropped."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip().strip("'\"")
    return out


SCRIPTS = Scripts.model_validate(yaml.safe_load(SCRIPTS_PATH.read_text(encoding="utf-8")))
VERSIONS = read_env()
