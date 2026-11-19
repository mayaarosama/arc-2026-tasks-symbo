"""Pre-flight check for the live audience prompt.

SD 3.5 ships no safety checker, and this runs in front of press. The guard is layered:
  1. the operator types the prompt (the audience never types into the system directly);
  2. this blocklist check — a hit blocks the run unless the operator explicitly overrides
     (false positives happen: "shoot" in "photo shoot");
  3. if in doubt, pick from data/t2i/live_fallback_prompts.json instead.
Extend config/blocklist.txt (one term per line) — it is read on every check.
"""
from __future__ import annotations

import re

from .config import ROOT

_BUILTIN = """
nude naked nsfw porn sex sexy erotic topless lingerie underwear bikini
gore gory blood bloody corpse dead body decapitat dismember mutilat suicide self-harm
kill murder shoot shooting gun rifle bomb terror weapon massacre war crime
nazi swastika kkk racist slur
child kid minor teen schoolgirl
trump biden putin netanyahu xi jinping president prime minister celebrity
logo trademark disney marvel pixar nintendo pokemon mickey
""".split()


def _terms() -> list[str]:
    extra = ROOT / "config" / "blocklist.txt"
    terms = list(_BUILTIN)
    if extra.exists():
        terms += [t.strip().lower() for t in extra.read_text(encoding="utf-8").splitlines()
                  if t.strip() and not t.startswith("#")]
    return terms


def check_prompt(prompt: str) -> dict:
    text = prompt.lower()
    hits = sorted({t for t in _terms() if re.search(r"\b" + re.escape(t), text)})
    too_long = len(prompt) > 300
    return {"ok": not hits and not too_long, "hits": hits + (["too long (>300 chars)"] if too_long else [])}
