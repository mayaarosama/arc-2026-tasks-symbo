"""Per-task conductor behaviour, in show order: Task 1 t2i, Task 2 i23d, Task 3 music, Task 4 llm,
Task 8 layout. See base.py for the hooks; core.py and app.py call handler(task) at each one."""
from __future__ import annotations

from .base import TaskHandler
from .i23d import ImageTo3D
from .layout import SceneLayout
from .llm import LanguageModel
from .music import TextToMusic
from .t2i import TextToImage

HANDLERS: dict[str, TaskHandler] = {h.task: h for h in (TextToImage(), ImageTo3D(), TextToMusic(),
                                                         LanguageModel(), SceneLayout())}


def handler(task: str) -> TaskHandler:
    try:
        return HANDLERS[task]
    except KeyError:
        raise KeyError(f"no task '{task}' (known: {', '.join(HANDLERS)})") from None
