"""The model prompts, read from data/prompts/ (plain text, UTF-8). See data/README.md for each file.

Shared by the worker (what the model is sent) and the conductor (what the operator page shows), so
the two can never disagree. Files are read once and cached; edit a file and restart the process.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "data" / "prompts"


@lru_cache(maxsize=None)
def text(name: str) -> str:
    """data/prompts/<name>, without its final newline. Raises FileNotFoundError naming the file."""
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"prompt file {path} is missing")
    return path.read_text(encoding="utf-8").rstrip("\n")


@lru_cache(maxsize=None)
def json_doc(name: str):
    return json.loads(text(name))


def llm_system() -> str:
    """Task 4: the system message of every language-model race reply."""
    return text("llm_system.txt")
