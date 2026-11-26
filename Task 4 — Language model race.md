# Task 4 — Language model race: implementation spec

One prompt goes to every machine. Each streams its answer token by token, side by side. The measure is tokens per second, which is the number an NVIDIA audience already uses.

## Models

Qwen3.5 (Alibaba Qwen team, released March 2026). The whole family is Apache 2.0: no revenue threshold, no branding clause, no gating on Hugging Face. Weights are GGUF files served by llama.cpp's `llama-server`.

| Key (`models.yaml`) | Model | File | Size | Role |
| --- | --- | --- | --- | --- |
| `qwen35-4b` | Qwen3.5 4B | `bartowski/Qwen_Qwen3.5-4B-GGUF`, Q4_K_M | ~2.7 GB | **Show model, every machine** |
| `qwen3-4b-2507` | Qwen3 4B Instruct 2507 | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF`, Q4_K_M | ~2.5 GB | Fallback if Qwen3.5 misbehaves in the pinned llama.cpp build |
| `qwen35-9b` | Qwen3.5 9B | `bartowski/Qwen_Qwen3.5-9B-GGUF`, Q4_K_M | ~5.7 GB | Full matrix: workstation |
| `qwen35-122b-a10b` | Qwen3.5 122B-A10B (MoE, 10B active) | `bartowski/Qwen_Qwen3.5-122B-A10B-GGUF`, Q4_K_M | ~75 GB | Full matrix: DGX only |

Why this shape:

- **One model on every machine** follows the single-model show: the 4B at Q4_K_M runs on the CPU box, the 4 GB laptop GPU, the workstation and the DGX. It uses about 3 GB of disk.
- **One serving stack everywhere.** llama.cpp has equal CPU and CUDA paths, so the quantisation, sampler and seed are identical in every cell. vLLM would be faster on the DGX, but it has no equal CPU path, so it would change the software stack in the middle of the comparison.
- **Thinking is switched off** (`enable_thinking: false`). The stage shows answers, not reasoning traces.
- The `file` glob in `models.yaml` makes `download_weights.py` fetch only the one quantisation. A GGUF repo holds every quantisation, tens of GB in total.

## Protocols

| Segment | Protocol | Settings | What the audience sees |
| --- | --- | --- | --- |
| seg6 "Same answer, different time" | iso-quality | 1 stream, 256 tokens, temperature 0 | The same answer typing out at three speeds |
| seg7 "Tokens in a minute" | iso-time, 60 s | DGX 64 streams, workstation 8, CPU 1; 512 tokens per reply, repeated until the bell | Live token counters: total tokens and tok/s |

**The trap this design avoids.** One stream of a small model is limited by memory bandwidth, so on that the DGX and the workstation nearly tie, the same problem as SD 1.5 in Task 1. The DGX's real advantage is serving many users at once (continuous batching), so iso-time is a **throughput** run, like the music task. The caption says so: *each serving as many replies at once as it handles well*. Stream counts are TBC from rehearsal.

Greedy decoding on different hardware can still diverge after a few hundred tokens, because floating-point reductions run in a different order. If the iso-quality texts differ, say so rather than hide it.

## Data

`demo/data/llm/stage_prompts.json`: three fixed prompts.
- `llm-01` is short and factual, for the stopwatch.
- `llm-02` (a Cairo itinerary for an architecture student) and `llm-03` (a short story) are long-form, for throughput.

There is no live audience prompt for this task. Live output from a language model cannot be pre-screened, and the prompt blocklist only checks the input. If a live prompt is wanted, add `live` to seg7 and keep the operator in the loop.

## Worker

- Endpoint: `python -m arcdemo.worker --config config/worker-<machine>.yaml --task llm --port 8104`, in the llm env (`requirements/llm.txt`, which needs no torch).
- On load, the worker starts `llama-server` (off the clock), waits for `/health`, and sends one warm-up request.
- A job opens `streams` concurrent streaming chat requests. Tokens are counted on the job clock as they arrive, reconciled with the server's own usage count, and refused once the bell has rung. A reply cut off by the bell counts its tokens, but is not saved as an output.
- The result adds the fields `tokens`, `tokens_per_second`, `first_token_seconds` and `text` (stream 0's current reply). Peak memory is read from the server process: VRAM via `nvidia-smi` on GPU machines, RSS on CPU.
- Worker config keys:
  - `llama_server`: path to the binary.
  - `llm_slots`: must be at least the largest `streams` that machine runs.
  - `llm_ctx`: tokens per slot.
  - `llama_args`: extra flags.

## Before rehearsal

- [ ] Choose one llama.cpp release tag, install it on every machine (CUDA build on the GPUs, CPU build on the CPU box) and record the tag in `requirements/llm.txt`.
- [ ] Confirm that the Q4_K_M filenames resolve: `python -m scripts.common.download_weights --machine dgx --task llm --dry-run`, then download.
- [ ] `verify_offline.py --task llm` on each machine. If Qwen3.5 fails in the chosen build, switch the settings to `qwen3-4b-2507`.
- [ ] Rehearse seg7 to fix the stream count per machine: the largest count before tokens per second stops rising.
- [ ] The CPU-box setup scripts (`scripts/cpu/`) do not install the llm env yet; do it by hand or extend them.
