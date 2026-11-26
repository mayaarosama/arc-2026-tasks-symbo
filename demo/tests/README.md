# tests/: the test suite

```bash
cd demo
.venv/bin/python -m pytest                 # everything except the real-model tests: no GPU, no weights
.venv/bin/python -m pytest -m "not slow"   # skip the tests that start worker processes
scripts/common/run_gpu_tests.sh            # real models: pytest -m gpu in each task venv (this workstation)
```

The results and findings are in [docs/Test report.md](../docs/Test%20report.md).

| File | Covers |
| --- | --- |
| `conftest.py` | Fixtures:<br>• `stack`: three mock worker processes and a `DemoConfig` for them<br>• `offline_cfg`: every worker unreachable<br>• `worker_client`: a mock worker in-process<br>• `fake_server`: a stand-in llama-server<br>Also the `gpu` marker rule |
| `helpers.py` | Shared helpers: `make_context`, `mock_worker_cfg`, `wait_job`, `free_port`, `FakeLlama` |
| `test_code_rules.py` | House rules: imports at the top of every file (the documented exceptions are listed in it) |
| `test_shared_modules.py` | Moderation, logging, schemas, weights-on-disk, replay table, metrics, `JobContext`, OOM detection, the backend registry, `config/defaults.yaml` validation, prompt files |
| `test_config.py` | The shipped `demo*.yaml` files are valid; a typo in a setting is reported once |
| `test_data_and_static.py` | Every config and data file parses and has its fields; named files exist; the pages load files that exist; the import-map order; the mock outputs |
| `test_conductor_unit.py` | The conductor with no workers: config merge and validation, model picks, the job request per task, fallback to replay, uploads, prepare, every `/api` route |
| `test_conductor_mock.py` | End to end with three mock workers: live rounds, dead worker, stop, full recordings, LLM and layout rounds, uploads, prepare |
| `test_worker_unit.py` | Worker config, per-task settings echo, upload guards, `/health` `/env` `/files` `/load` `/unload`, the token (`token_env`) |
| `test_worker_mock.py` | The worker job lifecycle on the mock backend: batches, budget, out of memory, busy, LLM tokens, layout repair and review |
| `test_backends_unit.py` | The real backends with stand-in libraries: argument wiring, seeds, budget, export, llama-server start / stop / usage count, layout warm-up |
| `test_llm_stop.py` | Stop and the bell close stalled llama-server streams at once |
| `test_stable_audio3.py` | The Stable Audio 3 backend with real tensors (skipped without torch: run it in `.venv-music`) |
| `test_layout.py` | Task 8: walls, footprints, checks, grammar, LayoutGPT prompt, example bank, catalogue |
| `test_imageprep.py`, `test_fit_upload.py` | Task 2 uploads: checks, cut-out, normalisation; the `fit_upload` script |
| `test_scoring.py` | Task 2 scoring: rotations, Chamfer / F-score |
| `test_scripts.py` | Every script starts (`--help`); the shell scripts parse; script helpers; `config/scripts.yaml` and `config/versions.env` |
| `test_gpu_real.py` | `-m gpu`: one tiny real job per task through the worker, offline (SD 1.5, SF3D, Stable Audio 3 Small, Qwen3.5 4B) |

Markers:
- `gpu` loads real weights. It is skipped unless the run has `-m gpu`; `run_gpu_tests.sh` passes it.
- `slow` marks tests that start worker processes.

The task venvs do not ship pytest. `run_gpu_tests.sh` installs it once into `demo/.pytest-overlay/` (gitignored) and
puts it on `PYTHONPATH`, so the venvs are never changed.
