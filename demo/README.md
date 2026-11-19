# ARC symposium hardware demo — runbook

One conductor on the operator laptop, one worker per task on each machine under test.
The plan, the review of the proposal and the open questions are in `../Implementation Plan.md`.

## Models

The show (`config/demo.yaml`) runs **one model per task, the same on every machine**. Each was
chosen because it runs on a CPU and on a CUDA GPU, so every cell of the stage can be live
(SF3D needs a GPU with at least 6 GB, which the workstation and the DGX both have):

| Task | Model (`models.yaml` key) | Hugging Face repo | Weights | Runs on |
| --- | --- | --- | --- | --- |
| 1 · text-to-image | SD 1.5 (`sd15`) | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 5.2 GB | CPU (fp32), any GPU from 4 GB (fp16) |
| 2 · image-to-3D | Stable Fast 3D (`sf3d`) | `stabilityai/stable-fast-3d` + `facebook/dinov2-large` + `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 5.6 GB | CPU, CUDA GPU with ≥ 6 GB VRAM (out of memory on a 4 GB card) |
| 3 · music | Stable Audio 3.0 Small (`sa3-small-music`) | `stabilityai/stable-audio-3-small-music` (text encoder included) | 3.5 GB | CPU, any CUDA GPU (no flash-attn needed) |

`stabilityai/*` repos are gated: the account behind `HF_TOKEN` must accept each licence on its
model page first. Machines differ only in settings (batch, steps, texture size), not in model.

### Full model matrix (other machines, later)

`config/demo-full.yaml` is the original plan: the largest model each machine can run. Use it in
place of `demo.yaml` once these weights are on each machine (download with
`python scripts/download_weights.py --config config/demo-full.yaml --machine <id> --task <task>`):

| Task | DGX H200 | Workstation (16 GB GPU) | CPU box |
| --- | --- | --- | --- |
| t2i | SD 3.5 Large `stabilityai/stable-diffusion-3.5-large` (38.8 GB) and SD 3.5 Medium | SD 3.5 Medium, T5 int8 `stabilityai/stable-diffusion-3.5-medium` (27.4 GB) | SD 3.5 Medium without T5, SD 1.5 fallback |
| i23d | TRELLIS.2 4B `microsoft/TRELLIS.2-4B` (16.2 GB) + `facebook/dinov3-vitl16-pretrain-lvd1689m` (1.2 GB) + `briaai/RMBG-2.0` (5.4 GB) | TRELLIS v1 `microsoft/TRELLIS-image-large` (3.3 GB), plus TRELLIS.2 for the OOM segment | Stable Fast 3D |
| music | Stable Audio 3.0 Medium `stabilityai/stable-audio-3-medium` (10.4 GB) | Stable Audio 3.0 Medium | Stable Audio 3.0 Small |

Notes: `facebook/dinov3-*` and `briaai/RMBG-2.0` are gated with *manual* approval (allow days);
RMBG-2.0 is CC BY-NC. TRELLIS and TRELLIS.2 are Linux-only (`requirements/i23d.txt`). Stable Audio
3.0 Medium needs flash-attn 2 on the GPU, or its output is static (`requirements/music.txt`).
Also switch each worker's `preload` to the full-matrix models named in its comment.

```text
 operator laptop                                 each machine (DGX, workstation, CPU box)
 ┌───────────────────────────────┐   job JSON    ┌──────────────────────────────────────────┐
 │ conductor  :8000              │ ────────────▶ │ worker t2i   :8101  (diffusers env)        │
 │  /stage     projector page    │ ◀──────────── │ worker i23d  :8102  (SF3D / TRELLIS env)   │
 │                               │               │ worker music :8103  (Stable Audio 3 env)   │
 │  /operator  controls          │  status+files │  model loaded at start, clock runs HERE   │
 │  replay/    rehearsal results │               └──────────────────────────────────────────┘
 │  session/   runs.jsonl, logs  │
 └───────────────────────────────┘
```

## 0. Try it on a laptop (no GPU)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements/conductor.txt   # Linux/mac: .venv/bin/pip
python scripts/run_mock_demo.py          # 3 mock workers + conductor
# operator: http://127.0.0.1:8000/operator     stage: http://127.0.0.1:8000/stage
python scripts/rehearse.py --config config/demo-mock.yaml --segments seg2,seg3 --repeat 1   # fills replay-mock/
python -m pytest                         # 18 tests (the 4 Stable Audio ones need torch): config validation, scoring, worker lifecycle, conductor + failure drill
```

Mock cells carry a **MOCK** badge; their timings are invented and are never results.

## Secrets

One file, `config.txt` next to `demo/` (never inside it, never in git, never in a shared zip):

```text
ARC_DEMO_TOKEN=<shared secret, same on every machine; empty = workers accept anyone>
HF_TOKEN=hf_...      (a bare hf_... line is accepted too)
```

Scripts read it at run time and never print a value. Set `ARC_DEMO_TOKEN` before the venue:
`python -c "import secrets; print(secrets.token_urlsafe(24))"`.

## 1a. CPU box: Windows or Linux (one script each, same actions)

```bash
scripts/cpu/cpu_box_linux.sh setup        # Linux: venvs, CPU torch, SF3D, Stable Audio 3, weights, offline check
scripts/cpu/cpu_box_linux.sh start | status | stop
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 setup     # Windows
powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 start     # also: status, stop
powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 firewall  # once, as Administrator
```

Needs Python 3.10/3.11 and git; plus `g++` (Linux) or Visual Studio Build Tools "Desktop development
with C++" (Windows), because Stable Fast 3D compiles two extensions. Pass `TORCH_VERSION=…` / `-TorchVersion`
to match the DGX lock. `setup` ends by printing `cpu_threads` / `cpu_bf16` for `config/worker-cpu.yaml`
and logs everything to `results/setup-<time>.log`.

## 1b. GPU machines (once)

Single-model show: the three envs below are all a GPU machine needs (CUDA GPU, Linux or Windows; ≥ 6 GB for SF3D).

```bash
export HF_TOKEN=<from config.txt>                       # account must have accepted every gated licence
python scripts/download_weights.py --machine <dgx|workstation> --task t2i     # SD 1.5
python scripts/download_weights.py --machine <dgx|workstation> --task i23d    # SF3D + its two encoders
python scripts/download_weights.py --machine <dgx|workstation> --task music   # Stable Audio 3 Small
# t2i env:   CUDA torch first, then  pip install -r requirements/t2i-gpu.txt
# i23d env:  CUDA torch, then clone third_party/stable-fast-3d and install as in scripts/cpu/ (SF3D);
#            for the full matrix, TRELLIS instead: header of requirements/i23d.txt (Linux only)
# music env: header of requirements/music.txt (torch 2.7.1 + third_party/stable-audio-3)
unset HF_TOKEN; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/verify_offline.py --config config/worker-<machine>.yaml --task t2i    # must print PASS
python scripts/verify_offline.py --config config/worker-<machine>.yaml --task i23d
python scripts/verify_offline.py --config config/worker-<machine>.yaml --task music
```

For the full matrix add `--config config/demo-full.yaml` to `download_weights.py` and
`--demo config/demo-full.yaml` to `verify_offline.py`.

Copy `data/` to every machine (inputs are referenced by relative path). Freeze versions: `locks/README.md`.

## 2. Run the workers (GPU machines; the CPU box uses `start` above)

```bash
export HF_HUB_OFFLINE=1 ARC_DEMO_TOKEN=<shared secret>
python -m arcdemo.worker --config config/worker-dgx.yaml --task t2i  --port 8101   # in the t2i env
python -m arcdemo.worker --config config/worker-dgx.yaml --task i23d --port 8102   # in the i23d env
python -m arcdemo.worker --config config/worker-dgx.yaml --task music --port 8103  # in the music env
curl http://localhost:8101/health
```

## 3. Conductor (operator laptop)

Edit the worker URLs in `config/demo.yaml`, then:

```bash
export ARC_DEMO_TOKEN=<same secret>
python -m arcdemo.conductor --config config/demo.yaml
```

Projector: `/stage` full-screen (F11). Operator: `/operator`.

## 4. Rehearsal → replay table

```bash
python scripts/rehearse.py --segments seg0,seg2,seg3,seg3q --repeat 3    # prints the timing table
python scripts/rehearse.py --segments seg1 --repeat 1 --max-wait 14400   # overnight: CPU iso-quality
python scripts/rehearse.py --segments seg0,seg2,seg3,seg5 --full --repeat 1 --max-wait 14400   # CPU full runs
```

The median run per cell becomes its replay record in `replay/`. Change a setting in `config/demo.yaml`
and the old record is flagged *settings changed since rehearsal* on stage — re-rehearse that cell.

`--full` runs only the machines with `show_full_recording: true` (the CPU), with no time limit, and
saves to `replay/.../<input>/full/`. On stage, when the CPU is down, its time is up, or it is stopped,
its cell shows that recording at once, badged RECORDED, with the time the run took to finish.
The operator page's *What will run* panel warns in yellow when a cell has no recording on file.

## Show flow (operator)

The operator page reads top to bottom: **Part of the show** → **What will run** (task, model, input,
expected output per machine, recordings on file) → **1 Load the model** → **2 Choose the input and
start** → **3 Running now**.

1. Before doors: start everything, pick Part 1, **Load the AI model on all machines**, **Start** once
   (warm-up, not logged in the summary).
2. Per part: **Load the AI model on all machines** (off the clock; on 16 GB machines it first unloads
   the other task), pick an input from **Example prompt**, **▶ Start on all machines**.
3. Part 2 (the stopwatch): when DGX and workstation are done, **■ Stop all machines** — the CPU shows
   its full recording and how long it took (or *still running* if none is on file).
4. A machine misbehaving: **■ Stop** or **Show recording** on its row. Every substitution is badged.
5. Part 5: choose *Type a new prompt…* (or a spare prompt), **Check the prompt is safe**, start.
   *More options* holds forced recordings and a different time limit.
6. **Summary chart** on the projector for the close.

## Debugging

Every log line has a time and a source; worker lines carry the job's short run_id, which the conductor
logs when it dispatches, so one grep follows a cell end to end.

| Where | What |
| --- | --- |
| `session/conductor.log` | rounds, dispatch (`run_id=`), FALLBACK reasons, REPLAY sources, worker UP/DOWN |
| `results/worker-<task>.log` (each machine) | load times, clock start + settings, each output time, final status + peak memory |
| `session/runs.jsonl` | the measurement record: one JSON line per cell per round |
| `GET <worker>/jobs/<run_id>` | full live state of a job; `GET /env` library versions; `GET /health` |
| operator page → Workers | reachability, loaded models, env hash per machine (mismatch warning in yellow) |
| conductor start-up | `demo.yaml` is validated on load: unknown setting keys, model keys, inputs or worker URLs stop it with a list of every problem |
| `ARC_LOG_LEVEL=DEBUG` | also logs every failed poll (default INFO) |

## Layout

| Path | Purpose |
| --- | --- |
| `arcdemo/schemas.py` | the job/result wire format shared by both sides |
| `arcdemo/worker/` | FastAPI worker, job runner, timing/memory measurement, model backends |
| `arcdemo/worker/backends/` | `diffusers_t2i` (SD 3.5 / 1.5), `trellis` (TRELLIS.2, v1), `sf3d`, `stable_audio3` (Small / Medium), `music_stub` + `mock` (mock workers) |
| `arcdemo/conductor/` | dispatcher + fallback (`core.py`), replay table, prompt check, web app, stage/operator pages |
| `arcdemo/scoring.py` | Chamfer / F-score against the GSO scan, axis-convention independent |
| `config/` | `demo.yaml` (single-model show), `demo-full.yaml` (largest model per machine), `models.yaml`, one worker config per machine |
| `data/` | stage prompts, rehearsal prompts (PartiPrompts), GSO inputs + ground truth, music inputs |
| `scripts/` | download, verify offline, prepare GSO, rehearse, mock dry-run |
| `scripts/cpu/` | CPU box: `cpu_box_linux.sh`, `cpu_box_windows.ps1` (setup / start / status / stop; workers :8101–8103), `check_cpu.py` |
| `third_party/` | `stable-fast-3d`, `stable-audio-3` (git clones; not pip packages) |
