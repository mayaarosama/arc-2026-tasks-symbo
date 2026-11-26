# config/: every setting the demo reads

All inputs and tunables live here or in `data/` (see [data/README.md](../data/README.md)); code holds
no tunable values. Each file below states its format, and the loaders check that format at startup.
A typo stops the process with a message naming the bad key; it never shows up later as a failed cell on stage.

| File | What it holds | Read by | Format and check |
|---|---|---|---|
| `demo.yaml` | **The show:** machines and their worker URLs, inputs per task, the settings matrix (`settings.<task>.<protocol>.all/<machine>`), `model_choices`, the segments in show order | conductor (`arcdemo/conductor/config.py`), rehearsal and download scripts | YAML. `DemoConfig._validate` checks segment inputs, worker URLs, settings keys (must be `JobRequest` fields), models, the `model_choices` task, and the layout room types |
| `demo-full.yaml` | The full model matrix (largest model per machine) | conductor, `--config config/demo-full.yaml` | `extends: demo.yaml`, a shallow merge (a top-level key replaces the parent's) |
| `demo-workstation.yaml` | The single-box show: this workstation's GPU and CPU as two machines | `scripts/gpu/start_workstation_show.sh` | `extends: demo.yaml` |
| `demo-mock.yaml` | Dry run with three mock workers on localhost | `python -m scripts.common.run_mock_demo`, tests | `extends: demo.yaml` |
| `models.yaml` | **The model catalogue:** one entry per loadable variant (task, backend, repo, file, dtype, label, ...) | conductor, workers, download scripts | YAML. `task` must be one of t2i, i23d, music, llm, layout; `backend` must be in `arcdemo/worker/backends/base.py` (`tests/test_data_and_static.py`) |
| `worker-<machine>.yaml` | **One machine's worker:** `machine`, `devices` per task, `preload`, `exclusive_models`, `llm_slots`, `llama_args`, `cpu_threads`, `model_overrides`, optional `token_env` | `python -m arcdemo.worker --config ...` | YAML. `machine` and `devices` are required; `model_overrides` keys must be in `models.yaml` (`tests/test_data_and_static.py`) |
| `worker-mock-*.yaml` | Mock workers (invented timings) for the dry run and the tests | tests, `run_mock_demo` | as above, with `mock: true` |
| `defaults.yaml` | **Runtime defaults:** every timeout, interval, limit and fallback the code uses, by section (`conductor`, `stage`, `worker`, `moderation`, `i23d`, `music`, `llm`, `layout`, `mock`) | `arcdemo/defaults.py` (once, at import) | YAML. The pydantic schema in `arcdemo/defaults.py` forbids unknown keys and checks types. `demo.yaml` and the worker configs win where they set the same thing |
| `blocklist.txt` | Blocked terms for the live audience prompt (Task 1) | `arcdemo/conductor/moderation.py`, on every check | One term per line, matched at the start of a word; `#` starts a comment |
| `scripts.yaml` | The Python scripts' inputs and tunables (download patterns, mock ports, verification jobs, PartiPrompts selection, GSO camera, VRAM figures ...) | `scripts/common/settings.py` | YAML. A pydantic schema per script section, with unknown keys forbidden |
| `versions.env` | **Pinned versions and sources** (torch, llama.cpp tag, SF3D / SA3 commits, CUDA runfile) and the worker and conductor **ports** | `scripts/gpu/*.sh` and `scripts/cpu/cpu_box_linux.sh` (sourced), `cpu_box_windows.ps1` and `scripts/common/settings.py` (parsed) | One `KEY=value` per line, no spaces; `#` comments. `tests/test_scripts.py` checks that every variable the shell scripts use is defined here |

Not here:
- **Secrets** are in `demo/config.txt` on the GPU machines and in `../config.txt` on the CPU box. Neither is in git; see [docs/Setup and run.md](../docs/Setup%20and%20run.md).
- **Locked package versions** are in `../locks/` and `../requirements/`.

## Which value wins

1. **Conductor:** `demo.yaml`, then `defaults.yaml conductor`.
2. **Worker:** its `worker-*.yaml` (for example `llm_slots`, `llm_ctx`, `layout_ctx`, `llama_server`), then the model's `models.yaml` entry (for example `slots`), then `defaults.yaml` (`llm`, `layout`, `worker`).
3. **A job:** the settings matrix in `demo.yaml`, then the operator's model pick (its `model_choices` settings), then the `JobRequest` field default.
