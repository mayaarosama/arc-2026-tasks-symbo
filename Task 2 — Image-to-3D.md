# Task 2 — Image-to-3D: implementation spec

One image in, one textured mesh out, on three machines, timed and scored against ground truth. This is the harder of the two tasks to run live and needs the most rehearsal.

## Models

| Model | Publisher | Licence | Reported VRAM | Status |
| --- | --- | --- | --- | --- |
| TRELLIS.2-4B (`microsoft/TRELLIS.2-4B`) | Microsoft Research | MIT | 24 GB | Current recommendation, DGX tier |
| TRELLIS v1 image-large (`microsoft/TRELLIS-image-large`) | Microsoft Research | MIT | 16 GB | Current recommendation, workstation tier |
| Stable Fast 3D (`stabilityai/stable-fast-3d`) | Stability AI | Stability AI Community | \~6 GB | CPU tier and fallback |
| Hunyuan3D-2.1 | Tencent | Tencent Community | Not published | Eligible candidate, leads on texture and PBR quality |
| Hunyuan3D-2mini | Tencent | Tencent Community | Not published | Eligible candidate, smaller tier of the same family |
| Pixal3D | Tencent ARC | MIT | Not published | Eligible candidate, newest open image-to-3D release |

The origin restriction has been lifted, so the Hunyuan3D family and Pixal3D are now in scope. Hunyuan3D offers a size ladder inside one family (2.1 and 2mini) and is reported to lead on texture quality, which makes it a direct competitor to the TRELLIS ladder above. Confirm the exact Hugging Face repository ids and the Tencent Community License terms before adopting it — the collection page itself publishes neither licence terms nor VRAM figures.

Confirm the exact TRELLIS v1 repository name before downloading; the figures above come from third-party surveys rather than publisher documentation, so treat all three as unverified until measured on our own hardware.

TRELLIS has heavier setup than a diffusers pipeline: custom CUDA extensions and rendering submodules that compile at install time. Budget a full day for first install on each GPU machine and do it well before the event. Some rendering submodules carry licences other than MIT, so run a dependency check if licensing is being presented on stage.

## Software requirements

| Component | Requirement | Notes |
| --- | --- | --- |
| Python | 3.10 or later | Same on all machines |
| PyTorch with CUDA | Version matched to the compiled extensions | TRELLIS is sensitive to this |
| TRELLIS repository and its submodules | From the official GitHub | Not a pip-only install |
| trimesh, pygltflib | Current | GLB export |
| A GLB viewer in the UI | `streamlit`, plus a three.js or model-viewer component | Must rotate the mesh live |
| FastAPI + uvicorn | All three | Worker service |

## Settings matrix

| Machine | Model | Output resolution | Texture | Expectation |
| --- | --- | --- | --- | --- |
| DGX H200 | TRELLIS.2-4B | 1536³ | PBR | Finishes in the budget |
| Workstation 16 GB | TRELLIS v1 image-large | mesh from Gaussians | baked | Finishes; visibly coarser |
| CPU 16 GB | Stable Fast 3D | low | albedo plus material params | Likely replay |

Under iso-quality all three attempt the DGX row. The workstation is expected to fail on memory at 24 GB, and that failure is a legitimate, showable result — the UI reports it as *out of memory*, not as an error.

## Data

Source images and ground-truth meshes from [Google Scanned Objects](https://arxiv.org/abs/2204.11918), Creative Commons licensed.

Preparation steps:

1. Pick three objects by difficulty: a simple chair, a patterned ceramic vessel, an object with a thin arm or handle.
2. Render one view of each at a fixed camera angle, 512×512, plain background, to use as the model input. Store the render, not a photograph, so the input is identical everywhere.
3. Keep the ground-truth mesh alongside each input for scoring.
4. Freeze all of this in a folder deployed to all three machines and to the scoring script.

```json
[
  {"id": "i23d-01", "tier": "easy", "source": "gso/<object_id>", "input": "inputs/i23d-01.png", "truth": "truth/i23d-01.obj"},
  {"id": "i23d-02", "tier": "medium", "source": "gso/<object_id>", "input": "inputs/i23d-02.png", "truth": "truth/i23d-02.obj"},
  {"id": "i23d-03", "tier": "hard", "source": "gso/<object_id>", "input": "inputs/i23d-03.png", "truth": "truth/i23d-03.obj"}
]
```

- [ ] Choose the three GSO object IDs and record them here.

### Operator uploads (live only)

The operator can also run a picture of their own (demo `README.md`, "Image to 3D from your own picture", for the input spec). The conductor checks it, cuts the object out with U²-Net (rembg's default model, run through onnxruntime) and normalises it to 512 × 512 RGBA, like the stage inputs, once, so every machine gets the identical file inside the job (`image_b64`). Uploads have no ground-truth mesh, so they are not scored, and no rehearsal recording, so a failed machine shows its error rather than a replay.

## Scoring against ground truth

This is what makes the task credible to an academic audience, and it runs after the mesh is produced, not on the clock.

1. Normalise both meshes to a unit bounding box and align them. Alignment is the fiddly part: use the same canonical orientation for every output, and verify by eye during rehearsal that a good mesh scores well.
2. Sample points uniformly from both surfaces.
3. Report Chamfer distance, lower is better, as the per-machine quality number.

Scoring runs on the operator laptop, not on the machines under test, so it never affects the timing. Precompute the scores for the rehearsed inputs; recompute live only if time allows.

## Worker contract

`POST /i23d/generate`

```json
{
  "run_id": "uuid",
  "input_id": "i23d-01",
  "image_path": "inputs/i23d-01.png",
  "resolution": 1536,
  "budget_seconds": 60
}
```

Response:

```json
{
  "run_id": "uuid",
  "machine": "dgx",
  "model": "microsoft/TRELLIS.2-4B",
  "mesh": "/results/uuid.glb",
  "load_seconds": 0.0,
  "generation_seconds": 17.2,
  "export_seconds": 2.1,
  "peak_memory_mb": 23100,
  "status": "ok"
}
```

Mesh export time is reported separately from generation. Export is largely CPU-bound and would otherwise flatter or penalise a machine for the wrong reason.

Status values: `ok`, `budget_exceeded`, `out_of_memory`, `error`. `out_of_memory` is shown to the audience as a result in its own right.

## How to run on each device

**DGX H200.** TRELLIS.2 at 1536³ is the headline capability. Reported timings are roughly 3 s, 17 s and 60 s at 512³, 1024³ and 1536³ on an H100, so 1536³ may sit near a 60 s budget; measure on our own hardware and pick the resolution that finishes with margin. Resolution, not GPU count, is the dial here — do not promise multi-GPU scaling for a single mesh without testing it.

**Workstation, 16 GB VRAM.** Runs TRELLIS v1, not TRELLIS.2. This boundary is the strongest single point in the demo: the DGX runs the current best open image-to-3D model and the workstation cannot load it. Attempt the TRELLIS.2 load once on stage during the iso-quality segment so the audience sees the memory failure.

**CPU box, 16 GB RAM.** Treat as replay by default. TRELLIS depends on CUDA extensions and will not run on CPU at all; Stable Fast 3D is the only candidate, and even it is likely to be impractically slow. Test during rehearsal and record whatever happens — including "did not complete" with the elapsed time at cutoff.

## Constraints and failure modes

| Constraint | Consequence | Handling |
| --- | --- | --- |
| TRELLIS needs compiled CUDA extensions | Long, brittle install; no CPU path | Install early, snapshot the environment |
| Rendering submodules carry mixed licences | Licensing claim on stage could be wrong | Dependency audit before the event |
| 24 GB requirement for TRELLIS.2 | Workstation cannot run it | Shown deliberately as the memory boundary |
| Mesh alignment for scoring | A good mesh can score badly if misaligned | Validate the scoring script against ground truth itself, which must score near zero |
| GLB viewing in Streamlit | Meshes may not display reliably in-browser | Test the viewer component early; fall back to a prerendered turntable video |
| Publisher VRAM and speed figures unverified | On-stage numbers could be wrong | Quote only our own measured numbers |

## Acceptance checks before the event

- [ ] TRELLIS.2 produces a mesh on the DGX inside the chosen budget.
- [ ] TRELLIS v1 produces a mesh on the workstation inside the budget.
- [ ] The workstation's TRELLIS.2 memory failure is reproducible and reported cleanly as `out_of_memory`.
- [ ] Scoring script returns near zero when given the ground-truth mesh as input.
- [ ] GLB meshes rotate smoothly in the UI on the presentation display.
- [ ] Replay cells populated for all three objects on all three machines, including the CPU non-completion.
