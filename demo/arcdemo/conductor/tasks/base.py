"""What the conductor does differently per task: one TaskHandler subclass per task (t2i.py, i23d.py,
music.py, llm.py, layout.py). Conductor (core.py) and the web app (app.py) stay task-neutral and ask
the handler at each step that differs:

  conductor     on_start()         once, when the Conductor is made (i23d: offer the saved uploads)
  round start   prepare_input()   check / fill the input (t2i: the live prompt; layout: wall names)
                caption()          add to the segment's caption (i23d: no score for an uploaded picture)
  dispatch      job_fields()       the task's fields of the worker JobRequest
  follow        hard_budget        stop and fall back when a run is still going past budget + grace
  result        budget_note()      the note for a run that hit the budget
                unfinished()       whether a budget_exceeded run counts as "no result" for this task
                after_result()     extra work on a finished live cell (i23d: score against the scan)
  summary       summary()          extra per-cell fields for the summary view (layout: objects, flags)
  operator page prompt_text()      the exact text the model is sent (llm, layout)
  web app       routes()           the task's own /api routes and static mounts
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:                       # imports for type hints only: no runtime cycle with core.py
    from fastapi import FastAPI

    from ..config import DemoConfig
    from ..core import Conductor


class TaskHandler:
    task: str = ""
    hard_budget: bool = False           # the worker stops at the bell itself; past budget + grace = lost

    # ---- conductor ----------------------------------------------------------------------------
    def on_start(self, cond: "Conductor") -> None:
        """Once, when the Conductor is made."""

    # ---- round start ------------------------------------------------------------------------
    def prepare_input(self, item: dict, prompt: Optional[str], override: bool) -> None:
        """Check and complete the round's input in place. Raise ValueError / PermissionError to refuse."""

    def caption(self, item: dict, caption: str) -> str:
        return caption

    # ---- dispatch -----------------------------------------------------------------------------
    def job_fields(self, cond: "Conductor", item: dict, settings: dict) -> dict[str, Any]:
        """This task's fields of the JobRequest (the shared ones are filled by Conductor._job_request)."""
        return {}

    # ---- result ---------------------------------------------------------------------------------
    def budget_note(self, res: dict, budget: float) -> str:
        done, req = res.get("completed", 0), res.get("requested") or 0
        return f"{done} of {req} finished in {budget:g} s" if done else f"Did not finish in {budget:g} s"

    def unfinished(self, res: dict) -> bool:
        """A budget_exceeded run with this result shows the full recording instead (slow machines)."""
        return res.get("completed", 0) == 0

    async def after_result(self, cond: "Conductor", rnd: dict, mid: str) -> None:
        """Extra work once a live cell has its outputs."""

    # ---- summary ------------------------------------------------------------------------------
    def summary(self, cell: dict) -> dict[str, Any]:
        return {}

    # ---- operator page / web app --------------------------------------------------------------
    def prompt_text(self, cond: "Conductor", seg: dict, item: dict) -> Optional[str]:
        """The exact text the model receives, for the operator page; None when there is none to show."""
        return None

    def routes(self, app: "FastAPI", cfg: "DemoConfig", get_cond) -> None:
        """Register this task's own routes. get_cond() returns the running Conductor."""
