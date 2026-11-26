# scripts/t2i/: Task 1 — text to image

Scripts run as `python -m scripts.t2i.<name>` from `demo/`; shell scripts by path. The full list of scripts, and what each folder is for, is in [../README.md](../README.md).

| File | Runs where | What |
| --- | --- | --- |
| `build_rehearsal_prompts.py` | once, anywhere | Picks the rehearsal prompts from PartiPrompts → `data/t2i/rehearsal_prompts.json` |
