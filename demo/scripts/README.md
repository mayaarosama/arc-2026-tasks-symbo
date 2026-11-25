# scripts/: setup, downloads, checks, rehearsal and data builders

The Python scripts run as modules from `demo/`, so they import `arcdemo` with no path tricks:

```bash
python -m scripts.<folder>.<name> --help         # e.g. python -m scripts.common.rehearse --help
```

The shell scripts run by path. Every script reads its inputs and tunables from
[`config/scripts.yaml`](../config/scripts.yaml) (Python) or [`config/versions.env`](../config/versions.env) (pins, sources, ports).

```text
scripts/
├── common/   every machine and every task
├── t2i/      Task 1 — text to image
├── i23d/     Task 2 — image to 3D
├── layout/   Task 8 — scene construction
├── cpu/      the CPU box (Windows or Linux)
└── gpu/      the GPU machines (Linux): the workstation and the DGX
```

| Script | Runs where | What |
| --- | --- | --- |
| `common/download_weights.py` | every machine, each task's venv | Downloads every model a machine runs for a task (from the settings matrix, plus fallbacks); `--cutout` fetches the upload cut-out model |
| `common/verify_offline.py` | every machine | Acceptance check: each model loads and generates with the network off (`HF_HUB_OFFLINE=1`); `--cutout` checks the cut-out |
| `common/rehearse.py` | next to the conductor | Runs segments N times against the real workers, prints the timing table, saves the median run as the replay record; `--full` records slow machines with no time limit |
| `common/run_mock_demo.py` | anywhere | Three mock workers and the conductor: the dry run, with no GPU and no weights |
| `common/run_gpu_tests.sh` | the GPU workstation | The real-model tests (`pytest -m gpu`) in each task venv |
| `common/settings.py` | — | Loads and validates `config/scripts.yaml`; reads `config/versions.env` |
| `t2i/build_rehearsal_prompts.py` | once, anywhere | Picks the rehearsal prompts from PartiPrompts → `data/t2i/rehearsal_prompts.json` |
| `i23d/prepare_gso.py` | once, anywhere (needs `requirements/gso.txt`) | Downloads the Google Scanned Objects models and renders the stage inputs → `data/i23d/` |
| `i23d/fit_upload.py` | anywhere | Rescales a photo to meet the upload rules (upright, 512 px, PNG) |
| `layout/build_layout_exemplars.py` | once, anywhere | Builds the LayoutGPT example bank → `data/layout/exemplars.json` (deterministic) |
| `layout/sample_layout.py` | a machine with a layout worker config | The Task 8 review run, in-process: every room, the prompt, a summary and a review page |
| `layout/compare_layout.py` | anywhere | Compares `sample_layout` output folders prompt against prompt (Markdown) |
| `cpu/cpu_box_linux.sh`, `cpu/cpu_box_windows.ps1` | the CPU box | `setup` / `start` / `stop` / `status` / `conductor` (and `firewall` on Windows) |
| `cpu/check_cpu.py` | the CPU box, t2i venv | What the CPU offers, and the `cpu_threads` / `cpu_bf16` lines for `worker-cpu.yaml` |
| `cpu/fetch_llama_cpp.py` | the CPU box, llm venv | llama.cpp's prebuilt CPU build of the pinned release |
| `gpu/setup_gpu_machine.sh` | a GPU machine (Linux) | Every setup step, then the workers; report in `results/setup-report-latest.md` |
| `gpu/start_workstation_show.sh` | the workstation | GPU workers, CPU workers and the conductor on one box; `--stop` |
| `gpu/plan_downloads.py` | a GPU machine | Checks disk and GPU memory per model, then downloads what fits (called by `setup_gpu_machine.sh`) |

The Windows dev-laptop scripts (`rehearse_local_gpu.sh` and the `*-local*` configs) are archived
outside the repo, in `/home/master/mayar/archive/arc-2026-demo-laptop/`.
