# scripts/layout/: Task 8 — scene construction

Scripts run as `python -m scripts.layout.<name>` from `demo/`; shell scripts by path. The full list of scripts, and what each folder is for, is in [../README.md](../README.md).

| File | Runs where | What |
| --- | --- | --- |
| `build_layout_exemplars.py` | once, anywhere | Builds the LayoutGPT example bank → `data/layout/exemplars.json` (deterministic) |
| `sample_layout.py` | a machine with a layout worker config | The Task 8 review run, in-process: every room, the prompt, a summary and a review page |
| `compare_layout.py` | anywhere | Compares `sample_layout` output folders prompt against prompt (Markdown) |
