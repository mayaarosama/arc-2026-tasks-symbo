# Test report

How the suite is organised and how to run it is in [tests/README.md](../tests/README.md).

- **Default suite:** `cd demo && .venv/bin/python -m pytest`. It runs with no GPU and no weights.
- **Real models:** `scripts/common/run_gpu_tests.sh` runs `pytest -m gpu` once in each task venv.

---

## Part 1 — Baseline, before the clean-up (branch `cleanup/review`, 2026-11-25)

Run on the RTX 5060 Ti workstation, against the code at `fa6c964`. Only tests were added at this stage.

### Results

| Suite | Passed | Failed | Skipped | Notes |
|---|---|---|---|---|
| Existing 9 files (before this session) | 91 | 0 | 1 | The 1 skip is all of `test_stable_audio3.py` (no torch in `.venv`) |
| Default suite, after adding the new tests | 247 | **4** | 12 | Plus 1 expected failure (music has no segment) |
| `test_stable_audio3.py` in `.venv-music` | 4 | 0 | 0 | Never runs in `.venv`; it passes in the music venv |
| `-m gpu` in `.venv-t2i` (SD 1.5) | 1 | 0 | 4 | 4 = the other tasks' cases |
| `-m gpu` in `.venv-i23d` (SF3D) | 1 | 0 | 4 | |
| `-m gpu` in `.venv-music` (Stable Audio 3 Small) | 1 | 0 | 4 | |
| `-m gpu` in `.venv-llm` (Qwen3.5 4B, llm and layout) | 2 | 0 | 3 | |

Skips in the default suite:
- **5 JavaScript syntax checks:** `node` is not installed.
- **1 PowerShell check:** `pwsh` is not installed.
- **5 `gpu` cases:** skipped by design; they run only with `-m gpu`.
- **1 module:** `test_stable_audio3.py`, because `.venv` has no torch.

### New coverage

| Area | Test file | Covers |
|---|---|---|
| Shared modules | `test_shared_modules.py` | Moderation, logging, wire schemas, weights-on-disk, replay table, `Clock` / `MemoryMeter`, `JobContext`, OOM detection, backend registry |
| Conductor | `test_conductor_unit.py` | Config merge and validation, model picks, `_job_request` for every task, `_final_seconds`, plan, fallback to replay, uploads, prepare, every `/api` route |
| Worker | `test_worker_unit.py` | Config loading, per-task settings echo, `save_upload` guards, `ModelManager` refusals, `/health` `/env` `/files` `/load` `/unload`, auth token |
| Real backends (stand-in libraries) | `test_backends_unit.py` | See the breakdown below |
| Scripts | `test_scripts.py` | `--help` of every Python script, `bash -n` of every shell script, the pure helpers, determinism of the generated data files |
| Data, config, UI | `test_data_and_static.py` | Every config and data file parses and has its expected fields, every referenced file exists, every file a page loads exists, the import-map order, the mock backends' outputs |
| Real models | `test_gpu_real.py` (`-m gpu`) | One tiny offline job per task through the real worker path |

What `test_backends_unit.py` checks for each real backend, using stand-in libraries:
- **`DiffusersT2I`:** CPU-generator seeds and round-robin across devices, the budget abort, the T5 options.
- **`Trellis2` / `Trellis1`:** pipeline type and sampler arguments, the RGBA check.
- **`StableFast3D`:** CPU mode and `run_image` arguments.
- **`StableAudio3`:** the throughput loop.
- **`LlamaServer`:**
  - `load` is tested with a fake binary, covering the arguments, the version check and the out-of-memory path;
  - the usage-count reconciliation;
  - slots and context size.
- **`LayoutServer`:** the warm-up sends one request per slot.

The existing fixtures moved to `tests/conftest.py` and `tests/helpers.py`, so every file shares them.

### Not working (baseline findings)

| # | Severity | What | Where | Test |
|---|---|---|---|---|
| 1 | Bug | The worker ignores `token_env`. It always reads `ARC_DEMO_TOKEN`, while the conductor reads the variable that `demo.yaml` names. Renaming the variable in `demo.yaml` breaks auth between conductor and workers. | `arcdemo/worker/app.py:318` | `test_worker_honours_the_configured_token_env` |
| 2 | Bug (data) | Music input `mus-03` (continuation) names `data/music/seed-piano-10s.wav`, which is not in the repo. A continuation job would fail on `sf.read`. | `data/music/stage_inputs.json` | `test_stage_inputs_have_the_required_fields_and_files[music]` |
| 3 | Bug (script) | `fit_upload.py` has no `if __name__ == "__main__"` guard: importing it parses `sys.argv` and runs. | `scripts/fit_upload.py:19` | `test_fit_upload_has_a_main_guard` |
| 4 | Docs drift | Commit `e956d49` moved the GPU scripts' default secrets file to `demo/config.txt`. Three places still say `../config.txt`: the header comment, the README ("next to demo/, never inside it") and the CPU box script. | `scripts/gpu/setup_gpu_machine.sh:24,30`, `start_workstation_show.sh:16`, `README.md:70` | `test_secrets_file_default_matches_its_own_comment` |
| 5 | Gap | Task 3 (music) has inputs, settings and worker URLs, but no segment in `demo.yaml` or `demo-full.yaml`, so the operator cannot run it. Intended ("reserve"), but not stated in the config. | `config/demo.yaml` | `test_every_task_with_inputs_has_a_segment` (xfail) |
| 6 | Gap | `test_stable_audio3.py` never runs in the documented test command, because `.venv` has no torch. The new stand-in tests in `test_backends_unit.py` cover the same backend without torch. | `tests/test_stable_audio3.py:11` | — |
| 7 | Stale | `demo-mock.yaml:2` points to `scripts/run_mock_demo.ps1`, which does not exist (it is `run_mock_demo.py`). | `config/demo-mock.yaml:2` | — |
| 8 | Stale | README says "34 tests". There were 91 before this session. | `README.md:63` | — |
| 9 | Design | The conductor imports the LLM system prompt from the worker backend (`worker.backends.llm.SYSTEM`), so the conductor depends on worker code. | `arcdemo/conductor/app.py:177` | — |
| 10 | Design | Leftover duplicate imports inside functions: `json` and `Path` are already imported at the top. | `arcdemo/layout_prompt.py:564-565` | — |
| 11 | Environment | Two `llama-server` processes (Qwen3.5 4B and 9B, about 6.8 GB VRAM) were running with no worker process as their parent. They are probably left over from a stopped show and hold VRAM until they are killed. Not touched by this session. | — | — |
