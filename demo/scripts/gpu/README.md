# scripts/gpu/: the GPU machines (Linux): the workstation and the DGX

Scripts run as `python -m scripts.gpu.<name>` from `demo/`; shell scripts by path. The full list of scripts, and what each folder is for, is in [../README.md](../README.md).

| File | Runs where | What |
| --- | --- | --- |
| `setup_gpu_machine.sh` | a GPU machine (Linux) | Every setup step, then the workers; report in `results/setup-report-latest.md` |
| `start_workstation_show.sh` | the workstation | GPU workers, CPU workers and the conductor on one box; `--stop` |
| `plan_downloads.py` | a GPU machine | Checks disk and GPU memory per model, then downloads what fits (called by `setup_gpu_machine.sh`) |
