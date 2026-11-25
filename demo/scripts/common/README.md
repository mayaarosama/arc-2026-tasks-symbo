# scripts/common/: every machine and every task

Scripts run as `python -m scripts.common.<name>` from `demo/`; shell scripts by path. The full list of scripts, and what each folder is for, is in [../README.md](../README.md).

| File | Runs where | What |
| --- | --- | --- |
| `download_weights.py` | every machine, each task's venv | Downloads every model a machine runs for a task (from the settings matrix, plus fallbacks); `--cutout` fetches the upload cut-out model |
| `verify_offline.py` | every machine | Acceptance check: each model loads and generates with the network off (`HF_HUB_OFFLINE=1`); `--cutout` checks the cut-out |
| `rehearse.py` | next to the conductor | Runs segments N times against the real workers, prints the timing table, saves the median run as the replay record; `--full` records slow machines with no time limit |
| `run_mock_demo.py` | anywhere | Three mock workers and the conductor: the dry run, with no GPU and no weights |
| `run_gpu_tests.sh` | the GPU workstation | The real-model tests (`pytest -m gpu`) in each task venv |
| `settings.py` | — | Loads and validates `config/scripts.yaml`; reads `config/versions.env` |
