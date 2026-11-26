# Setup and run

How to install each machine, start the workers and the conductor, rehearse, and run the show.
Every command runs from `demo/`. Scripts run as modules: `python -m scripts.<group>.<name>`.
For what goes wrong and where to look, see [Known issues.md](Known%20issues.md).

## 0. Try it without a GPU

```bash
python -m venv .venv && .venv/bin/pip install -r requirements/conductor.txt   # Windows: .venv/Scripts/pip
python -m scripts.common.run_mock_demo          # 3 mock workers + conductor
# operator: http://127.0.0.1:8000/operator     stage: http://127.0.0.1:8000/stage
python -m scripts.common.rehearse --config config/demo-mock.yaml --segments seg2,seg3 --repeat 1   # fills replay-mock/
.venv/bin/python -m pytest                       # the test suite: see tests/README.md
```

Mock cells carry a **MOCK** badge; their timings are invented and are never results.

## Secrets

A `config.txt` file, never in git and never in a shared zip:

```text
ARC_DEMO_TOKEN=<shared secret, same on every machine; empty = workers accept anyone>
HF_TOKEN=hf_...      (a bare hf_... line is accepted too)
```

Where it lives depends on the machine:
- **GPU machines:** `setup_gpu_machine.sh` and `start_workstation_show.sh` read `demo/config.txt`. It is gitignored. Override with `SECRETS=<path>`.
- **CPU box:** `cpu_box_linux.sh` and `cpu_box_windows.ps1` read `config.txt` next to `demo/`. Override with `SECRETS=<path>` or `-Secrets <path>`.

Scripts read it at run time and never print a value.
- **Create the token before the venue:** `python -c "import secrets; print(secrets.token_urlsafe(24))"`.
- **The token's variable** is `ARC_DEMO_TOKEN`. To use another name, set `token_env` in `demo.yaml` and the same `token_env` in every worker config.

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

Requirements:
- Python 3.10 or 3.11, and git.
- A C++ compiler for Stable Fast 3D's two extensions: `g++` on Linux, or Visual Studio Build Tools "Desktop development with C++" on Windows.

Notes on `setup`:
- To match the DGX lock, pass `TORCH_VERSION=…` (Linux) or `-TorchVersion` (Windows).
- It ends by printing `cpu_threads` / `cpu_bf16` for `config/worker-cpu.yaml`.
- It logs everything to `results/setup-<time>.log`.
- Pinned versions and ports come from `config/versions.env`.

The CPU box runs all five workers, on the ports in `config/versions.env` (`WORKER_PORTS`, 8101–8105):
- t2i, i23d and music each have their own venv.
- llm and layout share `.venv-llm`.

`setup` also:
- installs llama.cpp's prebuilt CPU build of the pinned release (`LLAMA_TAG`, via `python -m scripts.cpu.fetch_llama_cpp`);
- downloads Qwen3.5 4B (2.7 GB).

The `firewall` action opens the worker ports.

## 1b. GPU machines (once)

The single-model show needs a CUDA GPU (Linux or Windows) with at least 6 GB for SF3D, plus the venvs below.
The RTX 5060 Ti workstation (Blackwell: `cu128` torch, llama.cpp built from source) has a step-by-step guide:
[GPU workstation setup.md](GPU%20workstation%20setup.md).

**Linux GPU machines: one command.** Clone the repo and write `demo/config.txt` (see Secrets). This then runs
every step below and starts the five workers:

```bash
bash scripts/gpu/setup_gpu_machine.sh --machine workstation    # or --machine dgx
#   --show-only  only config/demo.yaml models    --no-start  set up and check only
#   --conductor  also start the conductor here
```

What it does, in order:
1. Sets up the venvs and llama.cpp.
2. Before each download, checks free disk and GPU memory (`python -m scripts.gpu.plan_downloads`), and skips any model that does not fit.
3. Runs the offline check for the show and the full matrix.
4. Starts the workers.

Every step reuses what is already there, so re-running it is safe.

It stops early only for fixes that need you:
- sudo packages;
- the CUDA 12.8 toolkit;
- the secrets file.

It never runs `git pull`, so it cannot overwrite your local changes; update the repo yourself when you want to.

Any other failure is recorded, the steps that depend on it are skipped, and the run goes on. The report,
`results/setup-report-latest.md`, lists what needs your action and every step, model, offline check
and worker that could not run.

**Workstation alone, GPU + CPU on one box: one command** (after the setup above):

```bash
bash scripts/gpu/start_workstation_show.sh     # GPU :8101–8105, CPU :8201–8205, conductor :8000
#   --no-cpu  GPU workers + conductor only        --stop  stop all of it
```

- **Pages:** operator `http://localhost:8000/operator`, stage `http://localhost:8000/stage`.
- **Logs:** `results/` and `results-cpu/`.
- **Re-running** is safe: it stops what it started, then starts everything again.

`--stop` unloads each model first, which frees GPU memory cleanly. To stop by hand instead, use the same
`pkill` lines it runs:

```bash
pkill -f "arcdemo.conductor"                           # conductor
pkill -f "arcdemo.worker"                              # all workers, GPU and CPU
pkill -f "arcdemo.worker .*--port 8101( |$)"           # one worker, by its port
pkill -f "third_party/llama.cpp/bin/llama-server"      # leftover llama-server
pgrep -af "arcdemo|llama-server"                       # check: prints nothing when all are stopped
```

**The manual steps**, for Windows or for running one step at a time:

```bash
export HF_TOKEN=<from config.txt>                       # account must have accepted every gated licence
python -m scripts.common.download_weights --machine <dgx|workstation> --task t2i     # SD 1.5
python -m scripts.common.download_weights --machine <dgx|workstation> --task i23d    # SF3D + its two encoders
python -m scripts.common.download_weights --machine <dgx|workstation> --task music   # Stable Audio 3 Small
python -m scripts.common.download_weights --machine <dgx|workstation> --task llm     # Qwen3.5 4B, Q4_K_M file only
# t2i env:   CUDA torch first, then  pip install -r requirements/t2i-gpu.txt
# i23d env:  CUDA torch, then clone third_party/stable-fast-3d and install as in scripts/cpu/ (SF3D);
#            for the full matrix, TRELLIS instead: header of requirements/i23d.txt (Linux only)
# music env: header of requirements/music.txt (torch + third_party/stable-audio-3, versions in config/versions.env)
# llm env:   requirements/llm.txt (no torch) + the pinned llama.cpp release in third_party/llama.cpp/bin
unset HF_TOKEN; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task t2i    # must print PASS
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task i23d
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task music
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task llm
python -m scripts.common.verify_offline --config config/worker-<machine>.yaml --task layout   # same weights as llm
```

Afterwards:
- Copy `data/` to every machine; inputs are referenced by relative path.
- Freeze versions as described in `locks/README.md`.

### Full model matrix (other machines, later)

`config/demo-full.yaml` is the original plan: the largest model each machine can run. Use it in place of
`demo.yaml` once these weights are on each machine:
- **Download:** add `--config config/demo-full.yaml` to `download_weights`.
- **Check:** add `--demo config/demo-full.yaml` to `verify_offline`.
- **Preload:** switch each worker's `preload` to the full-matrix models named in its comment.

| Task | DGX H200 | Workstation (16 GB GPU) | CPU box |
| --- | --- | --- | --- |
| t2i | SD 3.5 Large `stabilityai/stable-diffusion-3.5-large` (38.8 GB) and SD 3.5 Medium | SD 3.5 Medium, T5 int8 `stabilityai/stable-diffusion-3.5-medium` (27.4 GB) | SD 3.5 Medium without T5, SD 1.5 fallback |
| i23d | TRELLIS.2 4B `microsoft/TRELLIS.2-4B` (16.2 GB) + `facebook/dinov3-vitl16-pretrain-lvd1689m` (1.2 GB) + `briaai/RMBG-2.0` (5.4 GB) | TRELLIS v1 `microsoft/TRELLIS-image-large` (3.3 GB), plus TRELLIS.2 for the OOM segment | Stable Fast 3D |
| music | Stable Audio 3.0 Medium `stabilityai/stable-audio-3-medium` (10.4 GB) | Stable Audio 3.0 Medium | Stable Audio 3.0 Small |
| llm | Qwen3.5 122B-A10B Q4_K_M (~75 GB, one H200) | Qwen3.5 9B Q4_K_M (~5.7 GB) | Qwen3.5 4B Q4_K_M |
| layout | the same files as llm (`*-layout` keys): seg9 becomes "bigger model, better room" | | |

Licences and platform limits:
- **Gated repos:**
  - `stabilityai/*` repos are gated: the account behind `HF_TOKEN` must accept each licence on its model page first.
  - `facebook/dinov3-*` and `briaai/RMBG-2.0` are gated with *manual* approval, which can take days.
- **RMBG-2.0** is CC BY-NC.
- **TRELLIS and TRELLIS.2** are Linux-only (`requirements/i23d.txt`).
- **Stable Audio 3.0 Medium** needs flash-attn 2 on the GPU, or its output is static (`requirements/music.txt`).

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

## 3. Conductor

Set it up once, with network access. This installs the conductor venv and the cut-out model for uploaded pictures (176 MB):

```bash
scripts/cpu/cpu_box_linux.sh conductor                                         # Linux / macOS
powershell -ExecutionPolicy Bypass -File scripts\cpu\cpu_box_windows.ps1 conductor   # Windows
```

On a GPU machine, `setup_gpu_machine.sh --conductor` does the same.

The cut-out runs on the CPU, at about 1–2 s per picture:
- With `onnxruntime-gpu` installed instead of `onnxruntime`, it uses the GPU, and falls back to the CPU if CUDA does not work.
- `ARC_CUTOUT_DEVICE=cpu` forces the CPU.
- Check it any time with `python -m scripts.common.verify_offline --cutout`.

Edit the worker URLs in `config/demo.yaml`, then:

```bash
export ARC_DEMO_TOKEN=<same secret>
python -m arcdemo.conductor --config config/demo.yaml
```

Projector: `/stage`, full-screen (F11). Operator: `/operator`.

## 4. Rehearsal → replay table

```bash
python -m scripts.common.rehearse --segments seg0,seg2,seg3,seg3q --repeat 3    # prints the timing table
python -m scripts.common.rehearse --segments seg1 --repeat 1 --max-wait 14400   # overnight: CPU iso-quality
python -m scripts.common.rehearse --segments seg0,seg2,seg3 --full --repeat 1 --max-wait 14400   # CPU full runs
```

The median run per cell becomes its replay record in `replay/`. If you change a setting in `config/demo.yaml`,
the stage flags the old record *settings changed since rehearsal*; re-rehearse that cell.

`--full`:
- runs only the machines with `show_full_recording: true` (the CPU), with no time limit;
- saves to `replay/.../<input>/full/`.

On stage, when the CPU is down, its time is up, or it is stopped, its cell shows that recording at once.
It is badged RECORDED and shows the time the run took to finish.

The operator page's *What will run* panel warns in yellow when a cell has no recording on file.

## Show flow (operator)

The operator page reads top to bottom:
1. **Part of the show**
2. **What will run**: task, model, input, expected output per machine, and the recordings on file
3. **1 Load the model**
4. **2 Choose the input and start**
5. **3 Running now**

The steps of the show:

1. **Before doors:** start everything, pick Part 1, press **Load this task's model on all machines**, then **Start** once.
   This is the warm-up; it is not logged in the summary.
2. **For each part:**
   1. Press **Load this task's model on all machines** (off the clock).
      - It first unloads every other task's model on every machine, so only this task is in memory.
      - The worker table's *Memory* column shows the RAM coming back.
      - `keep_other_tasks: true` on a machine in the config keeps the other models loaded.
   2. Pick an input from **Example prompt**.
   3. Press **▶ Start on all machines**.
3. **Part 2 (the stopwatch):** when the DGX and the workstation are done, press **■ Stop all machines**.
   The CPU shows its full recording and how long it took, or *still running* if none is on file.
4. **A machine misbehaving:** press **■ Stop** or **Show recording** on its row. Every substitution is badged.
5. **Part 5:** choose *Type a new prompt…* (or a spare prompt), press **Check the prompt is safe**, then start.
   *More options* holds forced recordings and a different time limit.
6. **The close:** show the **Summary chart** on the projector.

## Image to 3D from your own picture

In an image-to-3D part, under **Example prompt**, choose **Or use a picture of your own** and pick a file.

What the conductor does with it:
- checks the picture;
- cuts the object out of its background;
- centres the object on a 512 × 512 transparent square, the same form as the stage examples.

It does this once, so every machine gets the identical file; the job carries it to the workers. The
operator page shows your picture next to *what every machine gets*. Check the cut-out, then press **▶ Start**.
The picture stays in the list (*Uploaded pictures*) until the session folder is cleared.

What makes a good picture. The same rules appear on the page under *What makes a good picture*; the
text is in `data/i23d/upload_rules.json` and the limits in `config/defaults.yaml` (`i23d.upload`).

| | Requirement | Why |
| --- | --- | --- |
| File | PNG, JPEG or WebP, up to 10 MB; shortest side 512 px or more (256 px minimum, with a warning below 512) | SF3D works from 512 px; larger pictures are scaled down |
| Content | One object, whole: nothing touching the edge of the picture | A cut-off part comes out flat; a touching object is refused |
| Background | Plain, in a different colour from the object. A PNG with a transparent background is used as it is | The cut-out (U²-Net, rembg's default model) fails on busy or same-coloured backgrounds |
| View | Three-quarter view from slightly above (about 20° up, 35° round) | Matches the stage examples and how SF3D was trained |
| Light | Even and soft; no hard shadows, glare or strong reflections | Light is baked into the texture: a shadow becomes dark paint |
| Avoid | Glass and see-through objects, mirrors, thin wires or hair, text-heavy surfaces, people, several objects | Known SF3D failure cases; people also raise consent questions |

To bring a picture to the right size and format first (upright, shortest side 512 px, PNG under 10 MB), run
`python -m scripts.i23d.fit_upload IMG_2089.jpeg`. It writes `IMG_2089-fit.png`. It cannot fix the content.

An uploaded picture runs **live only**:
- There is no rehearsal recording, so a failed machine shows its error, not a REPLAY.
- There is no scanned original, so there is no error score. The stage says so in the caption.

The reasons a picture is refused are listed in [Known issues.md](Known%20issues.md#uploads).

## Task 8 (Furnish the room) in the demo interface

The language model places furniture on a floor plan, one line per object. The stage shows what the
model was asked, what it writes, and the room it builds. Background: `Task 8 — Scene construction.md`
and [Task 8 — Layout model review.md](Task%208%20—%20Layout%20model%20review.md).

Where its inputs live:
- **Furniture models:** the Kenney Furniture Kit (CC0, 25 `.glb` files) in `arcdemo/conductor/static/assets/furniture/`, served by the conductor, so there is nothing to download.
- **Rooms and example bank:** `data/layout/` (`stage_rooms.json`, `exemplars.json`).
- **Prompt texts:** `data/prompts/layout_*.json|txt` and `layoutgpt.json`.

**On this workstation with the real model** (uses the Task 4 weights and llama.cpp, see 1b):

```bash
python -m arcdemo.worker --config config/worker-workstation.yaml --task layout --port 8105   # llm env
python -m arcdemo.conductor --config config/demo-workstation.yaml
# operator: http://127.0.0.1:8000/operator    stage: http://127.0.0.1:8000/stage
```

The Windows dev-laptop configs used for the Task 8 review (`demo-local*.yaml`, `worker-local-*.yaml`,
`replay-local/`, `rehearse_local_gpu.sh`) are archived in `/home/master/mayar/archive/arc-2026-demo-laptop/`.

- **At the show:** each machine runs the layout worker on port 8105 (section 2, `--task layout`, llm env). The parts are *Furnish the room* (seg8) and *Rooms in a minute* (seg9).
- **Without a model:** `python -m scripts.common.run_mock_demo`. The mock workers serve layout too, with placeholder rooms.

Running it:
1. Pick **Furnish the room**. *What will run* shows:
   - the floor plan with the wall names the model uses;
   - the request in one sentence;
   - under **Prompt**, exactly what the model is sent: rules, two worked examples, this room, and the grammar.
2. Press **Load this task's model on all machines**. Loading plus the prompt warm-up takes about 2 minutes on a CPU, off the clock.
3. Choose the room under **Example prompt** (bedroom, L-shaped living room, library), then press **▶ Start**.

What the projector shows:

- **Header:** the plan thumbnail (wall names, the door in yellow) and the request in words.
- **Each machine's cell, left: the 3D room.**
  - Each piece of furniture drops in as the model writes its line, numbered in placement order.
  - Flagged pieces get a red frame: outside the walls, overlapping, or blocking the door. In seg8 and seg9 they stay as the model wrote them.
  - In seg8b *Furnish and fix*, the model is told each problem and fixes it itself: the piece slides to its new place, turns green and is marked ↻. The code never moves furniture.
- **Each machine's cell, right: *What the model wrote*.** The model's lines appear as they are written, numbered like the scene labels, in red with the reason when flagged.
- **Result:** time, number of objects and flags.

On a laptop CPU the first object appears after about 30 s and a room takes 55–75 s. The 3D view needs
WebGL; without it the stage falls back to the box view.

### A room of your own

In a *Furnish the room* part, under **Example prompt**, open **Or draw a room of your own**:

1. Choose the room type: bedroom, living room, library, study or dining room. These are the room types in the furniture catalogue.
2. Type the corners in metres, in order round the room, for example `0,0  4,0  4,3  0,3`. x runs east and y runs north.
3. Optionally, type the door as `x,y` on a straight wall (or `none`), the number of pieces, and a name for the list.

A field left empty takes the value of the room chosen above, which is shown greyed in the field. The
room type follows the room chosen above until you pick one yourself. So to try the bedroom's plan as a
study, choose the bedroom, open the form, pick *study* and press **Add this room**.

The plan and the request appear under the form as you type, or the reason the room cannot run.
**Copy the selected room** fills the form from the room chosen above, so you can start from a stage room.
**Add this room** adds it under *Your rooms (live only)* and selects it. Then press **▶ Start**.

The conductor checks the room and may tidy it. It says what it changed:
- it drops a repeated corner, or a corner in the middle of a straight wall;
- it moves the room so that x and y start at 0, because the model writes no negative numbers;
- it moves a door that is within 0.25 m of a straight wall onto that wall, and at least 0.45 m from the wall's ends.

| | Rule |
| --- | --- |
| Corners | 3 to 12, in order round the room; walls must not cross |
| Size | 2 to 8 m each way (LayoutGPT's floor mask covers 8 m), at least 4 m² of floor, every wall at least 0.3 m |
| Walls | At least one wall must run north-south or east-west. Wall pieces stand only against those; angled walls are kept clear |
| Door | Optional; on a straight wall |
| Pieces | 1 to 14 |

The limits are `layout.custom` in `config/defaults.yaml`. The outline travels in the job, so the
workers need nothing new. A room stays in the list until the session folder is cleared
(`session_dir/uploads/rooms.json`).

A room of your own runs **live only**, like an uploaded picture. There is no rehearsal recording, so a
failed machine shows its error and not a REPLAY.
