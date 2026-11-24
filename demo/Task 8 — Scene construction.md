# Task 8 — Sequential 3D scene construction

Give the model a floor plan and watch it furnish the room, one object at a time. This is the architecture-facing task, and the only one in the whole proposal where the audience watches a model *build* rather than watches a result appear.

The field calls this **indoor scene synthesis** or **3D scene layout generation**. The "watch it build" quality is not a property of the task — it is a property of the architecture. Autoregressive models emit objects in sequence and can be watched. Diffusion models solve the whole room at once and cannot.

## Candidate models

| Model | Venue | Architecture | Sequential placement | Code licence | Verdict |
| --- | --- | --- | --- | --- | --- |
| [ATISS](https://github.com/nv-tlabs/ATISS) | NeurIPS 2021 | Autoregressive transformer | **Yes** | NVIDIA Source Code License | The reference choice |
| [DiffuScene](https://github.com/tangjiapeng/DiffuScene) | CVPR 2024 | Denoising diffusion | No | Mostly borrowed from ATISS, so inherits its terms | Better layouts, nothing to watch |
| [InstructScene](https://arxiv.org/abs/2402.04717) | ICLR 2024 | Semantic graph prior plus layout decoder | No | Peking University, confirm repo terms | Text-controllable, not sequential |
| [MiDiffusion](https://openaccess.thecvf.com/content/WACV2026/papers/Hu_Mixed_Diffusion_for_3D_Indoor_Scene_Synthesis_WACV_2026_paper.pdf) | WACV 2026 | Mixed discrete-continuous diffusion | No | Confirm | Current quality leader, not sequential |
| [LayoutGPT](https://arxiv.org/pdf/2305.15393) | NeurIPS 2023 | LLM in-context planning | **Yes** | Confirm repo terms | Sequential, and ties to the LLM ladder |
| [OptiScene](https://arxiv.org/abs/2506.07570v1) | 2025 | Open LLM fine-tuned for layout | **Yes** | Confirm | Purpose-built open LLM for this |

ATISS is the reference implementation and the one whose behaviour matches your description most exactly: it takes the room type and its floor plan as its only input, and predicts each object's bounding box attributes in sequence. DiffuScene produces better layouts but predicts the whole scene in one shot — there is no placement order to render.

ATISS also comes from NVIDIA's Toronto AI Lab, which is a convenient provenance at this event.

## Input and output

**Input.** A room type (bedroom, living room, library) plus a floor plan, supplied as a floor mask or boundary polygon. Optionally a partial scene, for the completion task.

**Output.** A set of placed objects, each with a class label, 3D bounding box, position and orientation — emitted one at a time. Rendering requires retrieving the matching furniture mesh per object from a model library and placing it at the predicted transform.

**What the audience sees.** An empty floor plan extrudes into walls. Then a bed appears, then a nightstand beside it, then a wardrobe against the far wall, then a lamp — each arriving in the model's own order, in real time, with no cuts.

## Data — the gating item

Every model in the table trains on **3D-FRONT**, and the furniture meshes needed to render the output come from **3D-FUTURE**. Both are Alibaba datasets distributed through the Tianchi platform.

| Dataset | Contents | Access |
| --- | --- | --- |
| [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) | 18,797 rooms furnished by 3D objects; 7,302 furniture objects with high-quality textures; professionally designed layouts | Signed licence agreement via Tianchi |
| [3D-FUTURE](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future) | 9,992 unique industrial 3D CAD furniture shapes with high-resolution textures; 20,240 photo-realistic synthetic images; an augmented release raises this to 16,563 models | Signed licence agreement via Tianchi |

**Access is not a download button.** 3D-FRONT requires signing a Data Sets Use License Agreement with Tao Bao (China) Software Co., Ltd., which grants a worldwide, revocable, non-exclusive licence and is conditional on complying with the Tianchi Platform Service Agreement. The dataset paper describes the data as freely available to the academic community and beyond, but the mechanism is still a signed agreement and a platform account.

- [ ] **Start the Tianchi registration and licence agreement immediately.** This gates the entire task and has a lead time we do not control. Nothing else here can be tested until it clears.

## Licensing

**ATISS code.** Released under the NVIDIA Source Code License for ATISS, with every source file carrying the same notice. This is not an open-source licence. NVIDIA's non-commercial licence family states that the work and any derivative works may only be used or intended for use non-commercially, defining "non-commercially" as for research or evaluation purposes only.

A symposium demonstration is plausibly research or evaluation, and we are not selling anything. But read the exact ATISS licence text rather than assuming it matches the NC template, and have an answer ready if someone in the audience asks — at an event where we are discussing licences for other models, this one is a conditional licence and should be described as such.

**Dataset licence applies to outputs too.** The furniture meshes rendered on screen come from 3D-FUTURE. Showing them is covered by the agreement; reusing the renders in marketing afterwards may not be. Decide in advance whether the press may photograph and publish this segment.

**DiffuScene** states that most of its code is borrowed from ATISS, so it carries the same constraint rather than escaping it.

## Running offline

Straightforward once the data is in hand. Weights and datasets are local files; inference needs no network. The install is an ordinary PyTorch environment, far lighter than TRELLIS.

The one exception is LayoutGPT in its published form, which calls a proprietary LLM API. That breaks the offline rule and must be replaced with a local model — see below.

## The hardware problem, and the fix

**The problem.** ATISS is a small transformer operating over object attributes, not pixels. Placing a full room takes milliseconds on any of the three machines. Run as specified, this task is a three-way tie and makes a poor contest — the same weakness as the upscaling task.

**The fix: drive placement with a local LLM.** LayoutGPT performs indoor scene synthesis on 3D-FRONT following ATISS's setup and achieves lower FID than ATISS on both bedrooms and living rooms. OptiScene is an open-source LLM fine-tuned for indoor layout generation on a purpose-built dataset. Either approach makes the model size the variable, and model size is exactly what your three machines differ on.

| Machine | LLM tier | Expected behaviour on stage |
| --- | --- | --- |
| DGX H200 | Qwen3-235B-A22B | Coherent layout, sensible adjacencies, placed quickly |
| Workstation 16 GB | Qwen3-8B or 14B | Workable layout, slower, occasional odd placement |
| CPU 16 GB | Qwen3-0.6B or 1.7B | Slow, and visibly worse — furniture in implausible positions |

This is the strongest version of the task, for three reasons. The quality difference is visible as furniture rather than as a number. It merges with the language-model race, saving a segment of stage time. And a badly placed wardrobe is funny, which is worth more on stage than a tokens-per-second counter.

**Cost of the fix:** the local-LLM layout path needs prompt engineering and an output parser, and is more build work than running ATISS as published. Budget for it, or run ATISS as a visual showpiece outside the timed contest.

## Constraints

| Constraint | Consequence | Handling |
| --- | --- | --- |
| 3D-FRONT and 3D-FUTURE require signed agreements | Nothing can be built or tested until access clears | Register now; treat as the critical path |
| ATISS licence is non-commercial | Must be described accurately if licences come up on stage | Read the exact text; prepare the answer |
| ATISS is small | Three-way tie as a speed contest | Use the LLM-driven variant, or run it untimed |
| Pretrained weights availability | The repo describes pretrained models; confirm they are actually downloadable | Verify before committing |
| Furniture retrieval and rendering | The model outputs boxes, not meshes; the visual depends on retrieval and a live 3D viewer | Reuse the GLB viewer built for the image-to-3D task |
| LayoutGPT calls a proprietary API as published | Breaks the offline rule | Substitute a local Qwen3 model |

## Acceptance checks

- [ ] Tianchi licence agreements signed and both datasets downloaded.
- [ ] ATISS runs offline and emits objects one at a time, with the sequence observable.
- [ ] Furniture meshes retrieve and render correctly in the viewer.
- [ ] A floor plan outside the training distribution produces a sane layout — test with a plan of our own.
- [ ] If using the LLM variant: all three Qwen3 tiers produce parseable layouts.
- [ ] Replay cells populated for every floor plan on every machine.

## Sources

- [ATISS paper](https://arxiv.org/pdf/2110.03675), [repository](https://github.com/nv-tlabs/ATISS) and [licence](https://github.com/nv-tlabs/ATISS/blob/master/LICENSE)
- [NVIDIA Source Code License-NC reference text](https://nvlabs.github.io/stylegan2/license.html)
- [DiffuScene paper](https://arxiv.org/pdf/2303.14207) and [repository](https://github.com/tangjiapeng/DiffuScene)
- [3D-FRONT paper](https://arxiv.org/abs/2011.09127) and [dataset page](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset); [3D-FRONT licence agreement](https://gw.alicdn.com/bao/uploaded/TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf)
- [3D-FUTURE dataset page](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)
- [LayoutGPT](https://arxiv.org/pdf/2305.15393); [OptiScene](https://arxiv.org/abs/2506.07570v1); [InstructScene](https://arxiv.org/abs/2402.04717)

## Implementation (LLM layout path)

Built on the Task 4 language-model stack. The ATISS path is not implemented; it remains an optional untimed showpiece if the Tianchi agreements ever clear.

**What runs.** `qwen35-4b-layout` in `config/models.yaml`: the same Qwen3.5 4B Q4_K_M file as Task 4, served by the same llama-server, in the same llm env. It needs no extra download. The layout worker is its own process on port 8105 (`--task layout`). It runs on the CPU box, a 4 GB laptop GPU, the workstation and the DGX.

**How it works** (`arcdemo/worker/backends/layout.py`):
- The prompt holds a system rule set, two hand-written example rooms (our own, not from 3D-FRONT), then the room type, floor outline, door and allowed objects.
- The model writes one object per line, in one of two forms (prompt v3, "wall-anchored"):
  - wall pieces (beds, nightstands, wardrobes, dressers, desks, sofas, TV stands, bookshelves, cabinets): `wardrobe wall=east at=1.70`. It names a wall from the room description and gives the centre's position along it. The worker computes x, y and facing from the wall and the object's fixed size.
  - free pieces (chairs, armchairs, tables, lamps, plants): `armchair x=2.10 y=3.00 facing=west`.

  Units are metres, the origin is the south-west corner, and sizes are fixed per object type. Earlier versions (v1, v2) asked for raw coordinates, size and rotation for everything, and the 4B model could not do that arithmetic. See `docs/Task 8 — Layout model review.md`.
- A GBNF grammar, sent per request, limits the output to those two forms, the room's vocabulary (wall pieces only in the wall form) and this room's wall names, so even a small model always produces parseable output. The grammar fixes syntax only. Which wall, where along it, and where free pieces stand are entirely the model's choices.
- The worked examples in the prompt are room types that are not on stage (a study, a dining room). With a bedroom example, the model reproduced it line for line for the stage bedroom.
- Each line is parsed when its newline arrives and timestamped on the job clock. The result carries `objects`, each with `t`, which is what the stage and replays animate.
- Prompt warm-up: the rules and the two example rooms (~1,140 tokens) are the same for every room. At load, off the clock, one warm-up request goes to each llama-server slot at once, so every slot caches that prefix (the cache is per slot, not shared). A room then reads only its own ~130 tokens. On the laptop CPU that cut a 3-object test room from 56 s to 29 s, and 4 rooms at once from 114 s to 29 s. Every run record says `prompt_warmed`. The worker config key `layout_warm_prompt: false` puts the full prompt read back on the clock.
- Every object is checked against the room: outside the walls, overlapping an earlier object, or within 0.8 m of the door. Problems are flagged (red outline and "!" on stage). The code never moves a piece of furniture.
- **Self-correction** (seg8b "Furnish and fix", protocol `iso_quality+fix`: `repairs: 2, review_rounds: 3`). The model fixes its own mistakes; the code only measures.
  - *Repair, as it builds.* When a line is flagged, the stream stops and the model is told exactly what is wrong ("it overlaps 3 sofa by 0.42 m²", "it comes within 0.35 m of the door; keep 0.80 m clear", "it sticks out of the room by 0.30 m"). It writes that line again; a grammar keeps it the same object. Up to 2 tries; the try with the lowest problem cost is kept. The room then continues from the kept lines (the reply so far is sent back as a prefilled assistant message).
  - *Review, when the room is written.* The numbered room, with every problem, goes back to the model, which answers with `move <n> wall=… at=…`, `move <n> x=… y=… facing=…`, `remove <n>` and `done`. An edit stays only if it lowers the room cost C; one that does not is undone and the model is told so in the next round. Up to 3 rounds; two rounds without a kept edit end it.
  - *Room cost C* (0 means nothing flagged): 1 per problem (so fewer problems always wins), plus metres outside the outline + metres inside the door clearance + m² of flagged overlap, + 0.5 per object removed (less than a problem, more than nothing, so emptying the room is never the cheapest fix) + 2 if the room's main piece is gone (bed, sofa, bookshelf). This is what makes it more than greedy: a left-to-right pass can never move an earlier piece, the review can.
  - *Shown honestly.* Each change is kept in the object's `history` with its time, so the stage, and replays, show the red piece first, then the slide to where the model put it, with "↻ fixed by the model" in the console. The first pass stays on record: the summary gives first-pass flags, final flags and fixes (`first_flagged`, `fixed` (now clean), `improved` (fewer problems, still flagged), `removed`, `fixes_tried`), and run records disclose `repairs` and `review_rounds`.
  - Without those settings (seg8, seg9) the worker sends exactly the same single request as before.

**Segments** (`config/demo.yaml`):
- seg8 "Furnish the room": iso-quality, greedy, no time limit. The same room is built on every machine and only the clock differs. Rooms: `lay-01` bedroom, `lay-02` L-shaped living room, `lay-03` library with an angled wall (our own plan).
- seg8b "Furnish and fix": seg8 with self-correction (below).
- seg8c "The published method": seg8 with LayoutGPT's prompt (CSS lines, the room given only as its size, 4 similar example rooms from `data/layout/exemplars.json`), for comparison. Results and what was adapted: `docs/Task 8 — Layout model review.md`, "LayoutGPT's prompt against ours". OptiScene has no public weights.
- seg9 "Rooms in a minute": iso-time, 60 s. Each machine furnishes many rooms at once (DGX 64, workstation 8, CPU 1; TBC), and the score is rooms finished. The stage shows stream 0's current room.
- `demo-full.yaml` turns seg9 into the model ladder: 122B-A10B on the DGX, 9B on the workstation, 4B on the CPU, one room each. Compare the rooms.

**Stage.** `static/layout3d.js` draws a 3D cut-away "dollhouse" with three.js. The walls facing the camera drop to a kerb, the floor is wood, the door is marked in yellow, and the camera sways slowly. Each object is a real furniture model scaled to its exact footprint, dropping in as it arrives, with a numbered label in placement order. Some carry props: a TV on the TV stand, a lamp on the nightstand, a monitor on the desk. Flagged objects get a red frame, a red floor patch and a red label.
- Models: the Kenney Furniture Kit 2.0, CC0 (public domain): 25 GLB files, 353 KB, in `static/assets/furniture/` with the licence. That means no 3D-FUTURE dependency and no licence question for press photos.
- three.js r160 is vendored in `static/vendor/three/`, so the stage works offline. The files are byte-identical to the npm package.
- `static/layout.js`, the 2D canvas view, remains as the fallback when WebGL is unavailable.

**Mock.** `LayoutMock` streams a wall-packed placeholder room at the Task 4 mock token rate. Tests: `tests/test_layout.py`, plus layout cases in `test_worker_mock.py` and `test_conductor_mock.py`.

### Still to do before rehearsal
- [ ] Run `verify_offline.py --task layout` on each machine with the pinned llama.cpp build. Done on the laptop CPU, 2026-11-21: llama.cpp b12608, grammar accepted, PASS. Not yet checked on a GPU: the laptop's driver (512.72) is too old for the CUDA 12.4 build. This confirms that llama-server accepts the `grammar` field on `/v1/chat/completions` and that Qwen3.5 4B produces sane rooms.
- [ ] Tune the prompt on the three stage rooms with the real 4B model. Two rounds done, 2026-11-21; see `docs/Task 8 — Layout model review.md`. Next: the v3 wall-anchored line format.
- [ ] Measure the CPU-box time per room. If it runs well over a minute, record a full run (`rehearse.py --full --segments seg8`).
- [ ] Fix seg9 stream counts from rehearsal. The DGX worker's `llm_slots: 64` covers both llm and layout. At about 1,000 prompt + 400 output tokens, the default `llm_ctx: 2048` per slot is enough.
- [ ] Extend the CPU-box setup scripts to install the llm env and start the layout worker (same gap as Task 4).
