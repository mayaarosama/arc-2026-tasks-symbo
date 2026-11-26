"""Pre-flight check for the live audience prompt.

SD 3.5 ships no safety checker, and this runs in front of press. The guard is layered:
  1. the operator types the prompt (the audience never types into the system directly);
  2. this blocklist check — a hit blocks the run unless the operator explicitly overrides
     (false positives happen: "shoot" in "photo shoot");
  3. if in doubt, pick from data/t2i/live_fallback_prompts.json instead.
The terms are in config/blocklist.txt (one per line, read on every check); the length limit is
moderation.max_prompt_chars in config/defaults.yaml.
"""
from __future__ import annotations

import re

from ..defaults import DEFAULTS
from .config import ROOT

BLOCKLIST = ROOT / "config" / "blocklist.txt"


def _terms() -> list[str]:
    """Every blocked term. A missing blocklist is an error: the guard must never run empty."""
    return [t.strip().lower() for t in BLOCKLIST.read_text(encoding="utf-8").splitlines()
            if t.strip() and not t.startswith("#")]


def check_prompt(prompt: str) -> dict:
    text = prompt.lower()
    hits = sorted({t for t in _terms() if re.search(r"\b" + re.escape(t), text)})
    limit = DEFAULTS.moderation.max_prompt_chars
    too_long = len(prompt) > limit
    return {"ok": not hits and not too_long, "hits": hits + ([f"too long (>{limit} chars)"] if too_long else [])}
