# arcdemo/worker/backends/: one class per model library

Each model in `config/models.yaml` names a `backend`. The registry `_BACKENDS` in `base.py` maps that
name to a class. Each class is imported only when its model loads, because each task's venv has only
its own libraries.

Every backend implements three methods:
- `load()` loads the model, off the clock;
- `generate(req, ctx)` runs it: it reads the budget and the cancel flag from `JobContext`, reports progress, and timestamps each output with `ctx.record_output`;
- `describe(req)` echoes the precision, offload and versions that actually ran.

| File | Task | Classes | Library |
| --- | --- | --- | --- |
| `base.py` | shared | `JobContext`, `Backend`, `free_memory`, `is_oom`, `resolve_dtype`, the registry, `backend_class` | — |
| `diffusers_t2i.py` | 1 · text to image | `DiffusersT2I`: SD 3.5 / SD 1.5. One pipeline per GPU, CPU-generator seeds, a 2-step warm-up, budget abort per step | diffusers, transformers, torch |
| `trellis.py` | 2 · image to 3D | `Trellis2` (TRELLIS.2-4B), `Trellis1` (TRELLIS image-large). Soft budget; export timed separately | TRELLIS repos (Linux, CUDA) |
| `sf3d.py` | 2 · image to 3D | `StableFast3D`: one feed-forward pass; CPU or CUDA | `third_party/stable-fast-3d` |
| `stable_audio3.py` | 3 · music | `StableAudio3`: whole tracks until the budget; a late track is discarded | `third_party/stable-audio-3` |
| `llm.py` | 4 · language model race | `LlamaServer`, which runs llama-server as a child process with concurrent streaming requests and closes stalled streams on Stop or the bell. Also `LlmBackend` (the shared loop), `run_streams` | llama.cpp's `llama-server` |
| `layout.py` | 8 · scene construction | `LayoutServer`: the Task 4 server with the layout prompt, grammar, prompt warm-up per slot, and self-correction (`_Room`: build, repair, review) | llama.cpp's `llama-server` |
| `mock.py` | 1, 2 (mock) | `MockBackend`: placeholder images and meshes at invented speeds; simulated out-of-memory | numpy, PIL, trimesh |
| `music_stub.py` | 3 (mock) | `MusicStub`: a seeded tone in the same throughput loop | numpy |
| `llm.py`, `layout.py` | 4, 8 (mock) | `LlmMock`, `LayoutMock`: canned text and wall-packed rooms at a simulated token rate | — |

Mock timings and sizes are in `config/defaults.yaml` `mock`. A mock worker reports `mock: true`, and the
stage badges every mock cell MOCK.
