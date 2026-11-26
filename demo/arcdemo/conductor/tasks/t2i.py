"""Task 1 — text to image: the live audience prompt (moderated) and the image batch."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from ..moderation import check_prompt
from .base import TaskHandler


class PromptBody(BaseModel):
    prompt: str


class TextToImage(TaskHandler):
    task = "t2i"
    hard_budget = True

    def prepare_input(self, item: dict, prompt: Optional[str], override: bool) -> None:
        """Input "live": the operator types the audience's prompt; it must pass the blocklist
        unless the operator overrides it (false positives happen: "shoot" in "photo shoot")."""
        if item["id"] != "live":
            return
        if not prompt or not prompt.strip():
            raise ValueError("live round needs a prompt")
        mod = check_prompt(prompt)
        if not mod["ok"] and not override:
            raise PermissionError(f"prompt blocked: {', '.join(mod['hits'])}")
        item["prompt"] = prompt.strip()

    def job_fields(self, cond, item: dict, settings: dict) -> dict[str, Any]:
        return {"prompt": item["prompt"]}

    def unfinished(self, res: dict) -> bool:
        """A partial batch is unfinished too: the full recording shows every image."""
        done = res.get("completed", 0)
        return done == 0 or done < (res.get("requested") or 0)

    def routes(self, app, cfg, get_cond) -> None:
        @app.post("/api/moderate")
        def moderate(body: PromptBody) -> dict:
            return check_prompt(body.prompt)
