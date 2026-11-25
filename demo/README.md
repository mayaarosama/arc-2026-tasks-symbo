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
| 4 · language model | Qwen3.5 4B Q4_K_M (`qwen35-4b`), served by llama.cpp | `bartowski/Qwen_Qwen3.5-4B-GGUF` (one file) | 2.7 GB | CPU, any CUDA GPU from 4 GB |
| 8 · scene construction | the same file (`qwen35-4b-layout`), placing furniture on a floor plan | same as task 4, no extra download | — | CPU, any CUDA GPU from 4 GB |

`stabilityai/*` repos are gated: the account behind `HF_TOKEN` must accept each licence on its
model page first. Machines differ only in settings (batch, steps, texture size), not in model.

### Full model matrix (other machines, later)

`config/demo-full.yaml` is the original plan: the largest model each machine can run. Use it in
place of `demo.yaml` once these weights are on each machine (download with
`python -m scripts.common.download_weights --config config/demo-full.yaml --machine <id> --task <task>`):

| Task | DGX H200 | Workstation (16 GB GPU) | CPU box |
| --- | --- | --- | --- |
| t2i | SD 3.5 Large `stabilityai/stable-diffusion-3.5-large` (38.8 GB) and SD 3.5 Medium | SD 3.5 Medium, T5 int8 `stabilityai/stable-diffusion-3.5-medium` (27.4 GB) | SD 3.5 Medium without T5, SD 1.5 fallback |
| i23d | TRELLIS.2 4B `microsoft/TRELLIS.2-4B` (16.2 GB) + `facebook/dinov3-vitl16-pretrain-lvd1689m` (1.2 GB) + `briaai/RMBG-2.0` (5.4 GB) | TRELLIS v1 `microsoft/TRELLIS-image-large` (3.3 GB), plus TRELLIS.2 for the OOM segment | Stable Fast 3D |
| music | Stable Audio 3.0 Medium `stabilityai/stable-audio-3-medium` (10.4 GB) | Stable Audio 3.0 Medium | Stable Audio 3.0 Small |
| llm | Qwen3.5 122B-A10B Q4_K_M (~75 GB, one H200) | Qwen3.5 9B Q4_K_M (~5.7 GB) | Qwen3.5 4B Q4_K_M |
| layout | the same files as llm (`*-layout` keys): seg9 becomes "bigger model, better room" | | |

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
 │                               │               │ worker llm   :8104  (llama-server)         │
 │                               │               │ worker layout:8105  (llama-server, llm env)│
 │  /operator  controls          │  status+files │  model loaded at start, clock runs HERE   │
 │  replay/    rehearsal results │               └──────────────────────────────────────────┘
 │  session/   runs.jsonl, logs  │
 └───────────────────────────────┘
```

## 0. Try it on a laptop (no GPU)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements/conductor.txt   # Linux/mac: .venv/bin/pip
python -m scripts.common.run_mock_demo          # 3 mock workers + conductor
# operator: http://127.0.0.1:8000/operator     stage: http://127.0.0.1:8000/stage
python -m scripts.common.rehearse --config config/demo-mock.yaml --segments seg2,seg3 --repeat 1   # fills replay-mock/
python -m pytest                         # 34 tests (the Stable Audio ones need torch): config validation, scoring, worker lifecycle, conductor + failure drill
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
scripts/cpu/cpu_box_linux.sh setup        # Linux: venvs, CPU torch, SF3D, Stable Audio 3, llama.cpp, weights, offline check
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

The CPU box runs all five workers: t2i :8101, i23d :8102, music :8103, llm :8104 and layout :8105 (the
language model and Task 8, sharing `.venv-llm`). `setup` installs llama.cpp's prebuilt CPU build of
the pinned release (b12608, `scripts/cpu/fetch_llama_cpp.py`, Windows or Linux x64) and downloads
Qwen3.5 4B (2.7 GB). The firewall action opens 8101–8105.

## 1b. GPU machines (once)

Single-model show: the envs below are all a GPU machine needs (CUDA GPU, Linux or Windows; ≥ 6 GB for SF3D).
Step-by-step for the RTX 5060 Ti workstation (Blackwell: `cu128` torch, llama.cpp built from source):
[`docs/GPU workstation setup.md`](docs/GPU%20workstation%20setup.md).

**Linux GPU machines: one command.** After cloning and writing `../config.txt` (see Secrets), this runs
every step below, then starts the five workers:

```bash
bash scripts/gpu/setup_gpu_machine.sh --machine workstation    # or --machine dgx
#   --show-only  only config/demo.yaml models    --no-start  set up and check only
#   --conductor  also start the conductor here   --no-pull   skip git pull
```

**Workstation alone, GPU + CPU on one box: one command** (after the setup above):

```bash
bash scripts/gpu/start_workstation_show.sh     # GPU :8101–8105, CPU :8201–8205, conductor :8000
#   --no-cpu  GPU workers + conductor only        --stop  stop all of it
```

Operator: `http://localhost:8000/operator`, stage: `http://localhost:8000/stage`. Logs in `results/`
and `results-cpu/`. Safe to re-run: it stops what it started, then starts everything again.

`--stop` unloads each model first, which frees GPU memory cleanly. To stop by hand instead (the same
`pkill` lines it runs):

```bash
pkill -f "arcdemo.conductor"                           # conductor
pkill -f "arcdemo.worker"                              # all workers, GPU and CPU
pkill -f "arcdemo.worker .*--port 8101( |$)"           # one worker, by its port
pkill -f "third_party/llama.cpp/bin/llama-server"      # leftover llama-server
pgrep -af "arcdemo|llama-server"                       # check: prints nothing when all are stopped
```

It sets up the venvs and llama.cpp. Before each download it checks free disk and GPU memory
(`scripts/gpu/plan_downloads.py`) and skips any model that doesn't fit. It then runs the offline
check for the show and the full matrix, and starts the workers. Every step reuses what is already
there, so re-running is safe. It stops early only for fixes that need you: sudo packages, the CUDA
12.8 toolkit, the secrets file, or a `git pull` conflict. Any other failure is recorded, the steps
that depend on it are skipped, and the run goes on. The report, `results/setup-report-latest.md`,
lists what needs your action and every step, model, offline check and worker that could not run.

The manual steps, for Windows or for running one step at a time:

```bash
export HF_TOKEN=<from config.txt>                       # account must have accepted every gated licence
python -m scripts.common.download_weights --machine <dgx|workstation> --task t2i     # SD 1.5
python -m scripts.common.download_weights --machine <dgx|workstation> --task i23d    # SF3D + its two encoders
python -m scripts.common.download_weights --machine <dgx|workstation> --task music   # Stable Audio 3 Small
python -m scripts.common.download_weights --machine <dgx|workstation> --task llm     # Qwen3.5 4B, Q4_K_M file only
# t2i env:   CUDA torch first, then  pip install -r requirements/t2i-gpu.txt
# i23d env:  CUDA torch, then clone third_party/stable-fast-3d and install as in scripts/cpu/ (SF3D);
#            for the full matrix, TRELLIS instead: header of requirements/i23d.txt (Linux only)
# music env: header of requirements/music.txt (torch 2.7.1 + third_party/stable-audio-3)
# llm env:   requirements/llm.txt (no torch) + the pinned llama.cpp release in third_party/llama.cpp/bin
unset HF_TOKEN; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task t2i    # must print PASS
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task i23d
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task music
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task llm
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task layout   # same weights as llm
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
python -m arcdemo.worker --config config/worker-dgx.yaml --task llm   --port 8104  # in the llm env
python -m arcdemo.worker --config config/worker-dgx.yaml --task layout --port 8105 # in the llm env
curl http://localhost:8101/health
```

## 3. Conductor (operator laptop)

Set it up once, with network: the conductor venv and the cut-out model for uploaded pictures (176 MB):

```bash
scripts/cpu/cpu_box_linux.sh conductor                                         # Linux / macOS
powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 conductor   # Windows
```

On a GPU machine, `setup_gpu_machine.sh --conductor` does the same. The cut-out runs on the CPU (about
1–2 s a picture). With `onnxruntime-gpu` installed in place of `onnxruntime` it uses the GPU, and it
falls back to the CPU if CUDA does not work; `ARC_CUTOUT_DEVICE=cpu` forces the CPU. Check it any
time with `python -m scripts.common.verify_offline --cutout`.

Edit the worker URLs in `config/demo.yaml`, then:

```bash
export ARC_DEMO_TOKEN=<same secret>
python -m arcdemo.conductor --config config/demo.yaml
```

Projector: `/stage` full-screen (F11). Operator: `/operator`.

## 4. Rehearsal → replay table

```bash
python -m scripts.common.rehearse --segments seg0,seg2,seg3,seg3q --repeat 3    # prints the timing table
python -m scripts.common.rehearse --segments seg1 --repeat 1 --max-wait 14400   # overnight: CPU iso-quality
python -m scripts.common.rehearse --segments seg0,seg2,seg3 --full --repeat 1 --max-wait 14400   # CPU full runs
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

1. Before doors: start everything, pick Part 1, **Load this task's model on all machines**, **Start** once
   (warm-up, not logged in the summary).
2. Per part: **Load this task's model on all machines** (off the clock). It first unloads every other task's model on every
   machine, so only this task is in memory (the worker table's *Memory* column shows the RAM coming
   back; `keep_other_tasks: true` on a machine in the config keeps them), then pick an input from **Example prompt**, **▶ Start on all machines**.
3. Part 2 (the stopwatch): when DGX and workstation are done, **■ Stop all machines** — the CPU shows
   its full recording and how long it took (or *still running* if none is on file).
4. A machine misbehaving: **■ Stop** or **Show recording** on its row. Every substitution is badged.
5. Part 5: choose *Type a new prompt…* (or a spare prompt), **Check the prompt is safe**, start.
   *More options* holds forced recordings and a different time limit.
6. **Summary chart** on the projector for the close.

## Image to 3D from your own picture

In an image-to-3D part, under **Example prompt**, choose **Or use a picture of your own** and pick a file.
The conductor checks the picture and cuts the object out of its background. It then centres the
object on a 512 × 512 transparent square, the same form as the stage examples. It does this once, on
the operator laptop, so every machine gets the identical file; the job carries it to the workers.
The operator page shows your picture next to *what every machine gets*. Check the cut-out, then
**▶ Start**. The picture stays in the list (*Uploaded pictures*) until the session folder is cleared.

What makes a good picture (also under *What makes a good picture* on the page):

| | Requirement | Why |
| --- | --- | --- |
| File | PNG, JPEG or WebP, up to 10 MB; shortest side 512 px or more (256 px minimum, with a warning below 512) | SF3D works from 512 px; larger pictures are scaled down |
| Content | One object, whole: nothing touching the edge of the picture | A cut-off part comes out flat; a touching object is refused |
| Background | Plain, in a different colour from the object. A PNG with a transparent background is used as it is | The cut-out (U²-Net, rembg's default model) fails on busy or same-coloured backgrounds |
| View | Three-quarter view from slightly above (about 20° up, 35° round) | Matches the stage examples and how SF3D was trained |
| Light | Even and soft; no hard shadows, glare or strong reflections | Light is baked into the texture: a shadow becomes dark paint |
| Avoid | Glass and see-through objects, mirrors, thin wires or hair, text-heavy surfaces, people, several objects | Known SF3D failure cases; people also raise consent questions |

To bring a picture to the size and format first (upright, shortest side 512 px, PNG under 10 MB):
`python -m scripts.i23d.fit_upload IMG_2089.jpeg` writes `IMG_2089-fit.png`. It cannot fix the content.

Refused with a reason on the page: files that are not PNG/JPEG/WebP or are over 10 MB, pictures
under 256 px, a cut-out that finds no object or keeps almost the whole picture, and an object
touching the edge.

An uploaded picture runs **live only**: there is no rehearsal recording (a failed machine shows its
error, not a REPLAY) and no scanned original, so no error score. The stage says so in the caption.
Background removal needs the model from `download_weights.py --cutout`. Without it, only PNGs that
already have a transparent background work, and the page says so.

## Task 8 (Furnish the room) in the demo interface

The language model places furniture on a floor plan, one line per object. The stage shows what the
model was asked, what it writes, and the room it builds. Background: `Task 8 — Scene construction.md`
and `docs/Task 8 — Layout model review.md`.

Assets: the 3D furniture is the Kenney Furniture Kit (CC0, 25 `.glb` files) in
`arcdemo/conductor/static/assets/furniture/`, served by the conductor, so there is nothing to download.
The rooms and the prompt's worked examples are in `data/layout/` (`stage_rooms.json`, `exemplars.json`).

**On this workstation with the real model** (uses the Task 4 weights and llama.cpp, see 1b):

```bash
python -m arcdemo.worker --config config/worker-workstation.yaml --task layout --port 8105   # llm env
python -m arcdemo.conductor --config config/demo-workstation.yaml
# operator: http://127.0.0.1:8000/operator    stage: http://127.0.0.1:8000/stage
```

The Windows dev-laptop configs used for the Task 8 review (`demo-local*.yaml`, `worker-local-*.yaml`,
`replay-local/`, `rehearse_local_gpu.sh`) are archived in `/home/master/mayar/archive/arc-2026-demo-laptop/`.

**At the show:** each machine runs the layout worker on port 8105 (section 2, `--task layout`, llm
env), and the parts are *Furnish the room* (seg8) and *Rooms in a minute* (seg9).
**Without a model:** `scripts/common/run_mock_demo.py`, where the mock workers serve layout too, with placeholder rooms.

1. Pick **Furnish the room**. *What will run* shows the floor plan with the wall names the model uses,
   the request in one sentence, and, under **Prompt**, exactly what the model is sent (rules, two
   worked examples, this room, and the grammar).
2. **Load this task's model on all machines**. Loading plus the prompt warm-up takes about 2 minutes on a CPU and is off the clock.
3. Choose the room under **Example prompt** (bedroom, L-shaped living room, library) and **▶ Start**.

What the projector shows:

- **Header:** the plan thumbnail (wall names, the door in yellow) and the request in words.
- **Each machine's cell, left:** the 3D room. Each piece of furniture drops in as the model writes its line, numbered in placement order. Flagged pieces get a red frame: outside the walls, overlapping, or blocking the door. In seg8 and seg9 they stay as the model wrote them. In seg8b *Furnish and fix* the model is told each problem and fixes it itself: the piece slides to its new place and turns green, marked ↻. The code never moves furniture.
- **Each machine's cell, right:** *What the model wrote*. The model's lines appear as they are written, numbered like the scene labels, in red with the reason when flagged.
- **Result:** time, number of objects and flags.

On the laptop CPU the first object appears after about 30 s and a room takes 55–75 s. The 3D view needs WebGL; without it the stage falls back to the box view.

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
| `arcdemo/worker/backends/` | `diffusers_t2i` (SD 3.5 / 1.5), `trellis` (TRELLIS.2, v1), `sf3d`, `stable_audio3` (Small / Medium), `llm` (llama-server; `LlmMock`), `layout` (task 8 on the same llama-server: prompt, grammar, room checks; `LayoutMock`), `music_stub` + `mock` (mock workers) |
| `arcdemo/conductor/` | dispatcher + fallback (`core.py`), replay table, prompt check, web app, stage/operator pages |
| `arcdemo/scoring.py` | Chamfer / F-score against the GSO scan, axis-convention independent |
| `config/` | `demo.yaml` (single-model show), `demo-full.yaml` (largest model per machine), `models.yaml`, one worker config per machine |
| `data/` | stage prompts, rehearsal prompts (PartiPrompts), GSO inputs + ground truth, music inputs, LLM prompts |
| `scripts/` | download, verify offline, prepare GSO, rehearse, mock dry-run |
| `scripts/gpu/` | GPU machines (Linux): `setup_gpu_machine.sh --machine workstation\|dgx` (every setup step, then the workers; report in `results/`), `start_workstation_show.sh` (GPU + CPU workers + conductor on one box), `plan_downloads.py` (disk + GPU-memory check, then download) |
| `scripts/cpu/` | CPU box: `cpu_box_linux.sh`, `cpu_box_windows.ps1` (setup / start / status / stop; workers :8101–8103), `check_cpu.py` |
| `third_party/` | `stable-fast-3d`, `stable-audio-3` (git clones; not pip packages) |
