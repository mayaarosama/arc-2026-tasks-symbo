"""Task 3 (reserve) — text to music: whole tracks until the budget (no segment in demo.yaml yet)."""
from __future__ import annotations

from typing import Any

from .base import TaskHandler


class TextToMusic(TaskHandler):
    task = "music"

    def job_fields(self, cond, item: dict, settings: dict) -> dict[str, Any]:
        return {"prompt": item.get("prompt"), "mode": item.get("mode"), "audio_path": item.get("audio"),
                "length_s": settings.get("length_s", item.get("length_s"))}
