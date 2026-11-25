# ARC symposium hardware demo

The same AI tasks run on three machines at once: a DGX H200, a 16 GB GPU workstation and a CPU-only
box. The projector shows the results side by side, so the audience sees what the hardware changes.

One **conductor** (operator laptop or the workstation) sends each job to one **worker** per task on
every machine. It follows the jobs, and shows the results on the projector page. Workers run the model and time it on
their own machine. When a machine fails, the stage falls back to that machine's rehearsal recording,
badged REPLAY.

```text
 conductor (:8000)                                each machine (DGX, workstation, CPU box)
 ┌───────────────────────────────┐   job JSON    ┌──────────────────────────────────────────┐
 │  /operator  controls          │ ────────────▶ │ worker t2i    :8101  (diffusers env)      │
 │  /stage     projector page    │ ◀──────────── │ worker i23d   :8102  (SF3D / TRELLIS env) │
 │  replay/    rehearsal results │  status+files │ worker music  :8103  (Stable Audio 3 env) │
 │  session/   runs.jsonl, logs  │               │ worker llm    :8104  (llama-server)       │
 └───────────────────────────────┘               │ worker layout :8105  (llama-server)       │
                                                 └──────────────────────────────────────────┘
```

## The tasks and their models

The show (`config/demo.yaml`) runs **one model per task, the same on every machine**. Machines differ
only in settings such as batch, steps and texture size. The full matrix (largest model per machine) is
`config/demo-full.yaml`.

| Task | What the audience sees | Model (`config/models.yaml` key) | Weights |
| --- | --- | --- | --- |
| 1 · text to image | images from one prompt, per machine, in a time limit or to completion | SD 1.5 (`sd15`) | 5.2 GB |
| 2 · image to 3D | a 3D model from one picture, scored against the scanned object | Stable Fast 3D (`sf3d`) | 5.6 GB |
| 3 · music (reserve) | tracks generated in a time limit | Stable Audio 3.0 Small (`sa3-small-music`) | 3.5 GB |
| 4 · language model race | the same reply streamed on every machine, tokens per second | Qwen3.5 4B Q4_K_M (`qwen35-4b`, llama.cpp) | 2.7 GB |
| 8 · scene construction | a model furnishing a floor plan one object at a time, in 3D | the same file (`qwen35-4b-layout`) | — |

## Try it without a GPU

```bash
python -m venv .venv && .venv/bin/pip install -r requirements/conductor.txt
python -m scripts.common.run_mock_demo      # 3 mock workers + conductor
# operator: http://127.0.0.1:8000/operator   stage: http://127.0.0.1:8000/stage
.venv/bin/python -m pytest                  # the test suite (no GPU, no weights)
```

Mock cells carry a **MOCK** badge: their timings are invented and are never results.

## Documentation

| Document | For |
| --- | --- |
| [docs/Setup and run.md](docs/Setup%20and%20run.md) | Installing each machine, running workers and the conductor, rehearsal, the show flow, uploads, Task 8 on stage |
| [docs/Known issues.md](docs/Known%20issues.md) | Logs and debugging, the upload box's error messages, open problems |
| [docs/Call hierarchy.md](docs/Call%20hierarchy.md) | Diagrams: what calls what, shared and for each task |
| [docs/Test report.md](docs/Test%20report.md) | The test suite, what it covers, what failed and why |
| [docs/GPU workstation setup.md](docs/GPU%20workstation%20setup.md) | The RTX 5060 Ti workstation, step by step, with its troubleshooting table |
| [docs/Task 8 — Layout model review.md](docs/Task%208%20—%20Layout%20model%20review.md) | How the Task 8 prompt was chosen |
| [tests/README.md](tests/README.md) | How the tests are organised and run (including real-model tests) |

## Folders

Each folder has a README describing its files.

| Path | What |
| --- | --- |
| [`arcdemo/`](arcdemo/README.md) | The Python package: the conductor, the workers, the model backends, and the shared code (wire format, prompts, scoring, defaults) |
| [`config/`](config/README.md) | Every setting the demo reads: the show, models, machines, runtime defaults, blocklist, script settings, pinned versions |
| [`data/`](data/README.md) | Stage inputs per task, prompts, GSO scans, furniture catalogue |
| [`scripts/`](scripts/README.md) | Setup, download, offline check, rehearsal, mock dry run, data builders; grouped by task and machine |
| [`tests/`](tests/README.md) | The test suite |
| `docs/` | The documents above, plus UI screenshots |
| `requirements/`, `locks/` | Package lists per venv and frozen versions (`locks/README.md`) |
| `third_party/`, `hf-cache/`, `weights/` | Model library clones, the Hugging Face cache, the cut-out model (not in git) |
| `session*/`, `results*/`, `replay*/` | Live outputs and logs, worker outputs, rehearsal recordings (`replay/` is the show's) |

Background and plans: `../Implementation Plan.md`, `../Proposal.md`, the `../Task *.md` specs, and
`Task 8 — Scene construction.md`.
