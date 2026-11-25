"""Task 4 — language model race: one prompt, streamed replies, tokens per second."""
from __future__ import annotations

from typing import Any, Optional

from ...prompts import llm_system
from .base import TaskHandler


class LanguageModel(TaskHandler):
    task = "llm"
    hard_budget = True

    def job_fields(self, cond, item: dict, settings: dict) -> dict[str, Any]:
        return {"prompt": item["prompt"]}

    def prompt_text(self, cond, seg: dict, item: dict) -> Optional[str]:
        return f"=== system ===\n{llm_system()}\n\n=== user ===\n{item.get('prompt', '')}"
