# Known issues and debugging

Where to look when something goes wrong, the fixes for problems seen before, and the open problems.
Build problems on the GPU workstation (CUDA, torch wheels, compilers) are in the troubleshooting
table of [GPU workstation setup.md](GPU%20workstation%20setup.md#8-troubleshooting).

## Logs and where to look

Every log line has a time and a source. Worker lines carry the job's short `run_id`. The conductor logs
the same id when it dispatches the job, so one grep follows a cell end to end:
`grep 3f9c2a session/conductor.log results/worker-t2i.log`.

| Where | What |
| --- | --- |
| `session/conductor.log` | Rounds, dispatch (`run_id=`), FALLBACK reasons, REPLAY sources, worker UP/DOWN |
| `results/worker-<task>.log` (each machine) | Load times, clock start and settings, each output's time, final status and peak memory |
| `results/stdout-<task>.log` | A crash before logging starts (workers started by the setup scripts) |
| `results/llama-server-<model>.log` | llama-server's own output (Tasks 4 and 8): out-of-memory at load, CUDA errors |
| `session/runs.jsonl` | The measurement record: one JSON line per cell per round |
| `GET <worker>/jobs/<run_id>` | The full live state of a job. Also `GET /env` (library versions, last load error) and `GET /health` |
| Operator page → Machines | Reachability, loaded models, memory, env hash per machine (a mismatch shows in yellow) |
| Conductor start-up | `demo.yaml` and `defaults.yaml` are validated on load. Unknown keys, models, inputs or worker URLs stop it with a list of every problem |
| `ARC_LOG_LEVEL=DEBUG` | Also logs every failed poll (the default is INFO) |

## Uploads

**The upload box says the cut-out cannot run.** The device running the conductor is missing one or both
of the things the cut-out needs:

| Message | What's missing | Fix |
| --- | --- | --- |
| "The cut-out model is not installed here… (`python -m scripts.common.download_weights --cutout`)" | The model file `weights/u2net/u2net.onnx` | Download it |
| "The cut-out needs onnxruntime: `pip install -r requirements/conductor.txt`" | The `onnxruntime` package in the venv the conductor runs from | Install the conductor requirements |

The conductor setup fixes both in one go. It needs network access, and it runs on the device running the conductor:

```bash
scripts/cpu/cpu_box_linux.sh conductor                                         # Linux / the workstation
powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 conductor   # Windows
# or by hand:
.venv/bin/pip install -r requirements/conductor.txt
.venv/bin/python -m scripts.common.download_weights --cutout
.venv/bin/python -m scripts.common.verify_offline --cutout                     # must print PASS
```

Then check two things:
- **The right venv.** Install into the same venv the conductor runs from (`.venv`). Installing into the system Python gives the same error again.
- **Reload.** Restart the conductor, then **reload the operator page**. The "not installed" message is checked only when the page loads.

**A picture is refused.** The page says why. A picture is refused when:
- it is not PNG, JPEG or WebP;
- it is over 10 MB;
- its shortest side is under 256 px;
- the cut-out finds no object, or keeps almost the whole picture;
- the object touches the edge of the picture.

The limits are `i23d.upload` in `config/defaults.yaml`.

## Problems seen before

| Symptom | Cause | Fix |
| --- | --- | --- |
| A cell shows REPLAY with "worker unreachable" | The worker process is down, or the firewall or token is wrong | `curl http://<machine>:8101/health` from the conductor; use the same `ARC_DEMO_TOKEN` (and `token_env`) on both sides; `sudo ufw allow 8101:8105/tcp` |
| A worker loads, then `CUDA out of memory` | Another task's model is still on the GPU, or an orphaned `llama-server` holds VRAM | Press *Load this task's model* again (it unloads the others); check `nvidia-smi` and stop stray `python` / `llama-server` processes (`pkill -f third_party/llama.cpp/bin/llama-server`) |
| The Machines panel shows an env hash mismatch in yellow | Library versions differ between machines | Compare `GET <worker>/env` on both machines; freeze versions as in `locks/README.md` |
| A stage cell says "settings changed since rehearsal" | A setting in `demo.yaml` changed after the recording was made | Re-rehearse that cell (`python -m scripts.common.rehearse --segments <seg>`) |
| `verify_offline` FAIL naming a repo | That repo is missing from `hf-cache/hub` | Re-run `download_weights` for that task, with network |
| `verify_offline` stops with "huggingface_hub was imported before HF_HUB_OFFLINE was set" | A change made something import `huggingface_hub` at start-up | Keep `huggingface_hub` imports in the backends (loaded at model load) or deferred, as documented in `tests/test_code_rules.py` |
| The 3D view on stage is a flat box view | No WebGL, or the import map failed to load `three` | Use a browser with WebGL; `tests/test_data_and_static.py` checks the import map order and the vendored files |

## Open problems

From the [test report](Test%20report.md). Each one is still true at the time of writing.

| What | Where | Effect |
| --- | --- | --- |
| Music input `mus-03` (continuation) names `data/music/seed-piano-10s.wav`, which is not in the repo | `data/music/stage_inputs.json` | A continuation job would fail. Task 3 is the reserve task and has no segment, so nothing on stage uses it yet. Add the 10 s clip, or drop the input |
| Task 3 (music) has no segment in `demo.yaml` or `demo-full.yaml` | `config/demo.yaml` | The operator cannot run music. Intended (reserve task); add a segment to use it |
| `cpu_box_windows.ps1` is not executed by any test (no PowerShell on the workstation) | `scripts/cpu/` | Its versions.env parsing was reviewed, not run; run `cpu_box_windows.ps1 status` on the CPU box once |
| `tests/test_conductor_mock.py::test_dead_worker_falls_back_with_a_reason` failed once under full-suite load, and passes on its own | `tests/test_conductor_mock.py` | A timing-sensitive test (three worker processes); rerun if it fails alone in a run |
| Two `llama-server` processes with no worker as their parent held ~6.8 GB of VRAM on the workstation | — | Leftovers from a stopped show. `start_workstation_show.sh --stop` unloads models first; stop them by hand with the `pkill` line in [Setup and run.md](Setup%20and%20run.md) |
