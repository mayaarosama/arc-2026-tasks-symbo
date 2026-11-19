# Task 1 — Text-to-image: implementation spec

One prompt in, one or more images out, on three machines, timed. This tab is written to be built from directly.

## Models

All weights from Hugging Face, downloaded and cached before the event. The SD 3.5 repositories are gated: sign in and accept the licence on the model page, then `huggingface-cli login`, before the first download. Do this weeks ahead, not at the venue.

| Tier | Hugging Face repo | Params | Role |
| --- | --- | --- | --- |
| Large | `stabilityai/stable-diffusion-3.5-large` | 8B | DGX |
| Medium | `stabilityai/stable-diffusion-3.5-medium` | 2.5B | Workstation, and CPU at reduced settings |
| Large Turbo | `stabilityai/stable-diffusion-3.5-large-turbo` | 8B | Optional 4-step speed variant |
| 1.5 (fallback) | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 0.9B | CPU, only if Medium cannot finish |

Confirm the SD 1.5 repository path before downloading — its canonical location moved out of the `stabilityai` organisation.

## Software requirements

| Component | Requirement | Notes |
| --- | --- | --- |
| Python | 3.10 or later | Same version on all three machines |
| PyTorch | CUDA build on DGX and workstation; CPU build on the CPU box | Pin one version across machines |
| diffusers | Current release supporting `StableDiffusion3Pipeline` | The pipeline class for SD 3.5 |
| transformers, accelerate, safetensors, sentencepiece | Current | Text encoders and loading |
| TensorRT | DGX and workstation only, optional | For the FP8 optimised build — second pass, not the baseline |
| FastAPI + uvicorn | All three | The worker service |

Pin every version in a lockfile and install identically on all three machines. A different diffusers version between machines is a silent source of output differences and would undermine the whole comparison.

## Settings matrix

These are the starting points. The rehearsal pass replaces them with what each machine actually finishes.

| Machine | Model | Resolution | Steps | Guidance | Batch | Precision |
| --- | --- | --- | --- | --- | --- | --- |
| DGX H200 | SD 3.5 Large | 1024×1024 | 40 | 4.5 | 8 | bf16 |
| Workstation 16 GB | SD 3.5 Medium | 1024×1024 | 28 | 4.5 | 1 | bf16 |
| CPU 16 GB | SD 3.5 Medium | 512×512 | 12 | 4.5 | 1 | fp32 |

Under iso-quality every machine runs the DGX row exactly, including batch size, and only the clock differs. Under iso-time each machine runs its own row.

Seed is fixed per prompt and identical across machines. Record it in the run log.

## Data

Three prompts, fixed, from the main proposal tab. Stored as a JSON file deployed to all three machines so no prompt is ever typed live:

```json
[
  {"id": "t2i-01", "seed": 1001, "prompt": "A sunlit modern interior with a glass coffee table, a brass floor lamp, and a bookshelf of five legible book spines"},
  {"id": "t2i-02", "seed": 1002, "prompt": "A dense city street at dusk in the rain, neon shop signs reflected in the puddles"},
  {"id": "t2i-03", "seed": 1003, "prompt": "A single ripe pomegranate split open on a marble countertop, water droplets on the skin"}
]
```

A fourth slot is reserved for the live audience prompt, with no seed fixed in advance.

## Worker contract

Each machine runs the same service. One endpoint for this task.

`POST /t2i/generate`

```json
{
  "run_id": "uuid",
  "input_id": "t2i-01",
  "prompt": "...",
  "seed": 1001,
  "width": 1024, "height": 1024,
  "steps": 28, "guidance": 4.5, "batch": 1,
  "budget_seconds": 60
}
```

Response:

```json
{
  "run_id": "uuid",
  "machine": "workstation",
  "model": "stabilityai/stable-diffusion-3.5-medium",
  "outputs": ["/results/uuid-0.png"],
  "completed": 1,
  "load_seconds": 0.0,
  "first_output_seconds": 18.4,
  "all_outputs_seconds": 18.4,
  "peak_memory_mb": 9840,
  "status": "ok"
}
```

Rules the worker must follow:

1. The model is loaded at service start, never per request. `load_seconds` is 0 for any request after warm-up.
2. The clock starts when generation starts and stops when the last image is written to disk.
3. On reaching `budget_seconds`, stop and return what is finished, with `status: "budget_exceeded"` and `completed` set to the count produced. Partial images are not returned.
4. Peak memory is read from `torch.cuda.max_memory_allocated()` on GPU machines and from process RSS on the CPU box. Reset the counter per request.
5. Any exception returns `status: "error"` with the message. The dispatcher then uses the replay table.

## How to run on each device

**DGX H200.** Pin the service to a device set, run bf16, and use the batch dimension rather than splitting one image across GPUs — a single image does not parallelise usefully. If multiple GPUs are available, the simplest credible setup is one pipeline per GPU, each generating part of the batch, with the clock stopping when the last one finishes. Confirm the GPU count first; this decision depends on it.

**Workstation, 16 GB VRAM.** SD 3.5 Medium at 1024² should fit comfortably. Use `enable_model_cpu_offload()` only if memory is tight — it costs time and would distort the comparison, so if it is needed, note it in the run log. Do not use attention slicing on the baseline run for the same reason.

**CPU box, 16 GB RAM.** Expect minutes, not seconds. Set thread count explicitly to the physical core count. Use fp32 unless bf16 is supported by the CPU. Memory is the risk, not only speed: SD 3.5 Medium plus its three text encoders in fp32 may not fit in 16 GB. Test this first — if it does not fit, drop to SD 1.5 at 512² and label it on screen as an earlier model generation.

**Optional TensorRT pass.** Build the engines on the machine they will run on, during preparation, never at the venue. Engine build takes a long time and is tied to the specific GPU. Run it as a separate labelled entry, not as a replacement for the baseline.

## Constraints and failure modes

| Constraint | Consequence | Handling |
| --- | --- | --- |
| SD 3.5 repos are gated | Download fails without an accepted licence and token | Accept and cache weeks ahead; verify the cache offline |
| Text encoders add several GB | CPU box may not fit SD 3.5 Medium in fp32 | Test early; SD 1.5 fallback |
| Different library versions between machines | Outputs differ for reasons unrelated to hardware | One lockfile, verified on all three |
| Same seed, different hardware | Outputs may still differ slightly at bit level | Expected and harmless; do not claim pixel-identical results |
| Cold first run | Unrepresentative timing | Warm-up run before the audience arrives |

## Acceptance checks before the event

- [ ] All weights present and loadable with networking disabled.
- [ ] Identical library versions confirmed on all three machines.
- [ ] Iso-quality run completes on DGX and workstation; CPU timing recorded even if it takes hours.
- [ ] Iso-time run produces a finished image on every machine at its own settings.
- [ ] Budget cutoff tested: a run that exceeds the budget returns cleanly rather than hanging.
- [ ] Replay cells populated for all three prompts on all three machines.
