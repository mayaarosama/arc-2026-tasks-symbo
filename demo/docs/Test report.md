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

---

## Part 2 — After the clean-up (branch `cleanup/review`, 2026-11-25)

This run is on the same workstation, after every step of the clean-up. It covers 7 commits on the local branch `cleanup/review`, none pushed:

1. Tests and the baseline report
2. Archive of the dev-laptop files
3. Organisation by task
4. Imports at the top
5. No hard-coding
6. READMEs
7. Diagrams

### Results

| Check | Result |
|---|---|
| Default suite, `.venv/bin/python -m pytest` | **407 passed, 1 failed, 12 skipped, 1 expected failure** (94 s). The failure is open finding 2 below (data, not code) |
| `-m gpu` in every task venv (`scripts/common/run_gpu_tests.sh`) | **5 / 5 passed**: SD 1.5, SF3D, Stable Audio 3 Small, Qwen3.5 4B (llm and layout) on the RTX 5060 Ti |
| `tests/test_stable_audio3.py` in `.venv-music` | 4 / 4 passed |
| `python -m scripts.common.verify_offline` on `config/worker-workstation.yaml` | **PASS for t2i, i23d, music, llm, layout**, and for `--cutout` (outline match 0.990) |
| `python -m scripts.common.run_mock_demo` | Three mock workers and the conductor started; a Task 8 round ran on every machine; everything exited cleanly |
| Stage and operator pages, headless Firefox | Screenshots of Tasks 1, 2, 4 and 8, the summary and the operator page are **the same before and after** the JS reorganisation, and again after the pages moved to `/api/config` settings |
| `docs/Call hierarchy.md` | All 9 Mermaid diagrams render (checked with Mermaid 10.9.1 in Firefox) |
| House rules | `tests/test_code_rules.py`: imports at the top of all 78 Python files. `tests/test_data_and_static.py`: every relative link in all 78 Markdown files resolves |

The suite grew as the work went on:

| Stage | Tests |
|---|---|
| Before this session | 92 |
| Baseline (Part 1) | 264 |
| After the clean-up | 421 |

The new tests cover:
- the import rule;
- `config/defaults.yaml`, `config/scripts.yaml` and `config/versions.env`;
- the prompt files;
- documentation links.

### Baseline findings: status

| # | Finding | Status |
|---|---|---|
| 1 | The worker ignored `token_env` | **Fixed.** `worker/app.py` reads the worker config's `token_env`, else `config/defaults.yaml` `conductor.token_env`. The test now passes |
| 2 | `data/music/seed-piano-10s.wav` is missing | **Open.** The file has to be supplied, or input `mus-03` removed (see [Known issues.md](Known%20issues.md#open-problems)) |
| 3 | `fit_upload.py` ran at import | **Fixed.** It now has `fit()` and `main()` under a main guard |
| 4 | The secrets-file location disagreed between code and docs | **Fixed.** The GPU scripts' comment and `docs/Setup and run.md` say `demo/config.txt` (GPU) and `../config.txt` (CPU box). The redundant `/./` path segment is gone |
| 5 | Music has no segment | **Open, by design** (reserve task); kept as an expected failure |
| 6 | `test_stable_audio3.py` never runs in `.venv` | **Covered.** `tests/test_backends_unit.py` tests the backend without torch; `tests/README.md` says to run the torch tests in `.venv-music` |
| 7 | `demo-mock.yaml` pointed to `run_mock_demo.ps1` | **Fixed** |
| 8 | README said "34 tests" | **Fixed** (README rewritten) |
| 9 | The conductor imported the LLM prompt from worker code | **Fixed.** Both sides read `data/prompts/llm_system.txt` through `arcdemo/prompts.py` |
| 10 | Duplicate in-function imports | **Fixed,** with every other in-function import (Step 1) |
| 11 | Orphaned `llama-server` holding about 6.8 GB of VRAM | **Gone:** `start_workstation_show.sh` stops orphaned servers when it starts (Part 3) |

### Found during the clean-up

| What | Status |
|---|---|
| `test_conductor_mock.py::test_dead_worker_falls_back_with_a_reason` failed once under full-suite load. It passed in 6 later runs, alone and in the full suite | Flaky, timing-sensitive (three worker processes). Listed in Known issues |
| `scoring.py` imported scipy at the top, which the llm venv lacks. After the reorganisation, `sample_layout.py` (llm venv) reaches it through the conductor's task package | **Fixed:** a guarded import, like the other optional libraries |
| Moving `huggingface_hub` to the top of the download scripts would make it read `HF_HUB_CACHE` before `arcdemo` sets it, so downloads would go to `~/.cache` | **Avoided.** Those two imports stay deferred, with the reason written next to them and allow-listed in the rule test |
| Setting `HF_HUB_OFFLINE` inside `main()` is safe only if nothing imported `huggingface_hub` earlier | `verify_offline` and `sample_layout` now stop with a clear message if that ever happens |

### What changed in behaviour (on purpose)

- **Python scripts run as modules from `demo/`:** `python -m scripts.common.rehearse` instead of `python scripts/rehearse.py`. This applies to the shell scripts, the PowerShell script and every document. It is what lets every import sit at the top of the file with no `sys.path` edits.
- **The stage headline for a music cell that produced nothing** says "no result" instead of "no mesh"; "no mesh" was the Task 2 wording.
- **The pages start polling** once `/api/config` has answered, because the poll interval now comes from there.

### Where the plan was not followed exactly

| Plan | What was done, and why |
|---|---|
| `PROMPT_VERSION` becomes a hash of the prompt files | Kept as the label `v3.1`: `demo.yaml` and the run records name the prompt by it. The prompt texts moved to `data/prompts/`, and a before/after comparison of 18 generated prompts was byte-identical |
| A machine config may override a `defaults.yaml` section with `defaults:` | Not added. The worker configs already override the values that differ per machine (`llm_slots`, `llm_ctx`, `layout_ctx`, `llama_server`, `cpu_threads`); a second mechanism would duplicate them |
| JS checked with `node --check` | Node is not installed, and downloading it was blocked in this session. The JS was checked by comparing headless-Firefox screenshots before and after. `tests/test_data_and_static.py` runs `node --check` automatically wherever node exists |
| Scripts subfolders `common/`, `t2i/`, `i23d/`, `layout/`, `cpu/`, `gpu/` | Done. `scripts/common/settings.py` was added (the loader for `config/scripts.yaml` and `config/versions.env`) |

Values that stay in code, each named at the top of its module:
- **Fixed by a model:** U2-Net's 320 px input and ImageNet normalisation.
- **Fixed by an algorithm:** the 24 rotations and the ICP sample counts in `scoring.py`, and the designer rules in `build_layout_exemplars.py`.
- **The review's own measures:** `compare_layout.py`.
- **The served URL routes:** `conductor/urls.py`.
- **Room descriptions generated from geometry:** `describe_room`.

## Part 3 — Memory between tasks (2026-11-26)

Reported: memory is not cleared before the next task's models load. Measured on the workstation
show (`start_workstation_show.sh`: 5 GPU and 5 CPU workers), playing the segments in show order
through the conductor and reading each worker's PSS and VRAM after every Prepare and round.

**Prepare itself was right:** each machine's worker unloads its model before the next one loads.
The unloads did not free everything, for three reasons:

| What | Effect | Fix |
| --- | --- | --- |
| SF3D and TRELLIS import their library inside `load()`, and that library keeps the call stack. `load()`'s local `model` stayed alive in that kept frame after `unload()` | The GPU 3D worker kept **4.0 GB of VRAM** and the CPU 3D worker **4.5 GB of RAM** for the rest of the show | `load()` builds the model in a `_build()` helper and holds no local (`sf3d.py`, `trellis.py`; `diffusers_t2i.py` too, as a precaution). Test: `test_unload_frees_the_model_even_when_a_library_keeps_the_load_stack`, which fails on the old code |
| After a failed job or load (out of memory), the model was unloaded **inside** the `except` block, while its traceback still held the model, and memory was never freed again afterwards | VRAM stayed reserved in torch's cache, and RAM stayed with the process | The unload and the free now run after the `except` block (`worker/app.py`). Test: `test_out_of_memory_in_a_job_frees_the_model_after_its_traceback_is_gone` |
| Importing torch with diffusers starts torch's compile-worker pool: 16 idle processes per t2i worker | ~0.4 GB of real memory per t2i worker. RSS also counted these processes' shared pages 17 times, so the t2i workers looked 6 GB larger than they were | `TORCHINDUCTOR_COMPILE_THREADS=1`, set by the worker before torch is imported (`config/defaults.yaml` `worker.torch_compile_threads`) |

The worker now also checks each unload: if the model is still referenced afterwards, it logs
`unloaded <model>, but something still references it: its memory is NOT freed`.

After the fixes, the trace `seg0 → seg3 → seg6 → seg8 → seg8b → seg2 → seg3 → seg6`:

| Point | RAM available | VRAM used | The 3D workers after their unload |
| --- | --- | --- | --- |
| Start | 20.9 GB | 0.4 GB | — |
| `seg6` prepared, first time | 15.8 GB | 4.5 GB | GPU 2.2 GB RAM / 0.3 GB VRAM, CPU 0.7 GB (before the fix: 4.0 GB VRAM, 4.5 GB RAM) |
| `seg6` prepared, second time | 15.7 GB | 4.5 GB | GPU 2.0 GB / 0.3 GB, CPU 0.6 GB |

Nothing builds up from task to task: only the task on stage holds memory. The busiest point is a
3D round, with SF3D loaded on both machines at once (9.8 GB of RAM still available). No worker
logged the warning, and the default suite passes (one failure: the missing music clip, finding 1).
