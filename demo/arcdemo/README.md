# arcdemo/: the demo's Python package

Two programs share one package:
- **`conductor/`** runs once, on the operator laptop or the workstation. It serves the operator and stage pages and dispatches each round.
- **`worker/`** runs once per task on every machine. It loads the model, runs it on its own clock, and reports.

The modules at the top level are shared by both. [docs/Call hierarchy.md](../docs/Call%20hierarchy.md)
draws what calls what, for each task.

```text
arcdemo/
├── __init__.py        sets HF_HUB_CACHE to demo/hf-cache/hub before anything imports huggingface_hub
├── defaults.py        loads and validates config/defaults.yaml (every timeout, limit and fallback)
├── schemas.py         the wire format: JobRequest (rejects unknown fields), JobResult, Progress, HealthInfo
├── prompts.py         reads the model prompts in data/prompts/ (worker and conductor share them)
├── logs.py            one logging setup for every process (console + file, ARC_LOG_LEVEL)
├── weights.py         which models' weights are on this disk (download script, worker /health)
│   ── Task 2 (image to 3D) ──
├── imageprep.py       checks, cuts out and normalises an operator's uploaded picture (U2-Net via onnxruntime)
├── scoring.py         Chamfer distance / F-score of a generated mesh against the GSO scan
│   ── Task 8 (scene construction) ──
├── layout_prompt.py   the furniture catalogue, the prompt, the grammar, the room checks, self-correction, LayoutGPT
├── conductor/         see conductor/README.md
└── worker/            see worker/README.md
```

Rules the code keeps (checked by `tests/test_code_rules.py` and the loaders):
- **Imports at the top of each file.** Optional per-venv libraries (torch, diffusers, ...) use a guarded `try: import` at the top. The few deferred imports say why, where they are.
- **No tunable values in code.** They live in `config/` (see [config/README.md](../config/README.md)) or `data/` (see [data/README.md](../data/README.md)). What stays in code is fixed by a model or an algorithm, for example ImageNet normalisation or the 24 rotations in `scoring.py`, and is named at the top of its module.
- **Organised by task.** Code that differs per task sits in one place per task, in show order (Task 1 t2i, 2 i23d, 3 music, 4 llm, 8 layout):
  - `conductor/tasks/` for the conductor;
  - `worker/tasks.py` for the worker;
  - one backend per model library in `worker/backends/`.
