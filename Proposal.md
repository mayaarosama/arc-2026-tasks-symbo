# Hardware Symposium — Session & Demo Proposal

Nov 17, 2026 · @mayar

## At a glance

The demo runs in about 40 minutes across three segments, on three machines, with two tasks. Every estimate below is a planning figure to be replaced with measured timings after the rehearsal pass.

| # | Segment | Task | Protocol | Estimate |
| --- | --- | --- | --- | --- |
| 0 | Machines introduced, one warm-up run | — | — | 4 min |
| 1 | The stopwatch | Text-to-image | Iso-quality | 7 min |
| 2 | Same minute, different results | Text-to-image | Iso-time, 3 prompts | 9 min |
| 3 | The same test in 3D | Image-to-3D | Iso-time, 3 objects | 13 min |
| 4 | Summary and a live audience prompt | Text-to-image | Iso-time, 1 prompt | 7 min |

Segment 3 is the longest because meshes have to be rotated on screen to be judged, and because the CPU cell is expected to run from the replay table rather than live.

### Flow

1. **Load and warm up** before the audience is in the room. All weights are already on disk; model load time is excluded from every clock.
2. **Segment 1 — iso-quality.** One fixed target image, identical settings on all three machines. They start together. The DGX finishes, the workstation finishes later, the CPU is still running when the segment ends. This establishes the speed claim before anything else is asked of the audience.
3. **Segment 2 — iso-time, image.** The clock is set to the rehearsed budget, 30 to 60 s. Each machine runs the largest model tier it can finish in that budget, labelled on screen. Three prompts, results side by side.
4. **Segment 3 — iso-time, 3D.** Same protocol, three objects of rising difficulty. Meshes rotate in the viewer. Geometric error against the ground-truth mesh is shown per machine.
5. **Segment 4 — summary and live prompt.** The timing chart for everything just run, then one prompt taken from the room and run live. The live prompt has no replay cell behind it, so it goes last and only if rehearsal shows a low failure rate.

Per-run timings for each machine and tier are not estimated here. They come out of the rehearsal matrix pass and then fix the budget for segments 2 to 4.

## Session objective

The session shows that hardware decides what a model can deliver in the time you have. Three machines run the same generative tasks from the same inputs; the audience watches the results arrive at different speeds and at different quality.

Three outcomes the session commits to:

1. A measured speed comparison across CPU, a single workstation GPU, and a DGX H200, on identical inputs.
2. Two ARC models shown working, as part of the same two-hour event.
3. One simple story for the press: same task, same minute, three very different results.

The demo is inference only. No training happens on stage, so every difference the audience sees comes from how fast each machine can run a model and how large a model it can hold.

The story arc: **start** — the same prompt goes to three machines; **contest** — a 30 to 60 second clock runs on all three; **result** — the outputs land side by side, and the gap is visible without explanation.

Open: the two ARC projects to be shown are still to be confirmed. Play-Learn Protect is not one of the demo tasks.

## How the comparison is built

At inference, hardware does not change output quality on its own. The same weights, the same seed and the same settings produce the same image on a CPU and on a DGX. Only the clock differs. Quality differences appear only when the hardware limits which model fits in memory, or how much work can finish in the time available.

The demo therefore runs two protocols, and labels which one is on screen.

|  | Iso-time | Iso-quality |
| --- | --- | --- |
| Rule | Every machine gets the same clock: 30 to 60 s | One fixed target output; each machine runs until it reaches it |
| Held constant | Wall-clock budget, prompt, seed | Model, resolution, steps, seed, batch size |
| Allowed to vary | Model tier, resolution, steps, batch size | Wall-clock time only |
| What the audience sees | Three different results, arriving together | The same result, arriving at wildly different times |
| Headline | Same minute, very different results | What takes the workstation an hour, the DGX does in a minute |

Iso-time is the only protocol that genuinely produces the quality difference the session is about, so it carries the headline. Iso-quality is the clean proof of speed, and the number the press will quote.

Two honesty requirements:

- Iso-time deliberately changes the model between machines. The screen must say so: *each machine ran the largest model it could finish in 60 seconds*. Unlabelled, it reads as a trick.
- A DGX does not make one small generation dramatically faster. Its advantage comes from larger models, higher resolution, and batches. The iso-time settings are chosen so that advantage is visible and real, not manufactured.

## Hardware under test

Three machines, on the same venue network, with all weights pre-downloaded. No internet dependency at run time.

| Machine | Compute | Memory for models | Role on stage |
| --- | --- | --- | --- |
| DGX H200 | Multi-GPU | To be confirmed | The ceiling: largest tier, highest resolution, batches |
| Workstation | 1 GPU | 16 GB VRAM | The realistic middle: what a well-equipped lab has |
| CPU box | CPU only | 16 GB RAM | The floor: what runs without a GPU at all |

The 16 GB VRAM ceiling on the workstation is the constraint that drives every model choice below. It is enough for mid-tier image models and the first-generation 3D model, and not enough for the top tier of either. That is exactly the boundary the demo needs.

Two items to confirm before rehearsal:

- [ ] DGX H200 configuration: number of GPUs and memory per GPU.
- [ ] Whether the CPU box is a separate machine or the workstation with its GPU disabled. A separate machine lets all three run at once; one machine in two modes means two passes.

## Model eligibility rules

Every model in the demo must pass all four tests:

1. **Open access** — weights published and downloadable, not API-only.
2. **Runs locally** — inference on our own hardware, offline.
3. **Licence permits the use** — licence text read and recorded, not assumed from a summary page.
4. **Family with size tiers** — at least two sizes or resolution tiers in the same family, so iso-time compares like with like.

Three consequences worth stating plainly, because they narrow the field:

**Origin is not a constraint.** Models of any national origin are eligible, so the field includes Tencent's Hunyuan3D family and Tencent ARC's Pixal3D for 3D, and ACE-Step for music. These are strong candidates and in some cases better than the alternatives, so the recommended ladders for tasks 2 and 3 are under review against them.

**Groq and OpenAI do not fit this demo.** Groq is a cloud inference provider on its own accelerators, so using it would both break the open-weights rule and put a competing hardware vendor on stage at an NVIDIA event. OpenAI publishes no downloadable image, 3D or music weights. Hugging Face is the right and sufficient source for all weights here. If you want Groq or OpenAI in the session for another reason, it needs to be a separate segment, not part of this comparison.

**Licence terms carry a revenue threshold.** Stability AI's Community License is free for research and non-commercial use, and free for commercial use by organisations under $1M annual revenue; above that, an enterprise licence is required. A conference demo is presentation, not a commercial product, but the threshold should be checked against your organisation before the event. Microsoft's MIT-licensed models carry no such clause.

On training data: Stable Audio 3.0 is stated by Stability AI to be trained entirely on licensed data. For the image and 3D models, training-data provenance is documented by the publishers but is not a licensed-data guarantee of the same kind. If "all data and models licensed" is a hard requirement for the stage, say so and the audio task becomes the strongest option on that ground alone.

## Task 1 — Text-to-image

Recommended as the anchor task: it is the only candidate that runs live on all three machines, and three images side by side are legible to a press photographer from the back of the room.

**Domain:** single-image generation from a text prompt, 1:1 aspect, no editing or post-processing, fixed seed.

### Candidate models

| Model | Publisher | Params | Reported memory | Licence | Verdict |
| --- | --- | --- | --- | --- | --- |
| [SD 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | Stability AI (UK) | 8B | 11–19 GB VRAM; \~11 GB with TensorRT FP8 | Stability AI Community | DGX tier |
| SD 3.5 Large Turbo | Stability AI (UK) | 8B | as Large | Stability AI Community | 4-step distilled; speed variant |
| [SD 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) | Stability AI (UK) | 2.5B | 9.9 GB VRAM, excluding text encoders | Stability AI Community | Workstation tier |
| SD 1.5 | Runway / CompVis / Stability AI | 0.9B | \~4 GB; CPU mode supported | CreativeML Open RAIL-M | CPU tier, different generation |
| FLUX.1 \[schnell\] | Black Forest Labs (Germany) | 12B | \~24 GB at full precision | Apache 2.0 | Eligible, but no size ladder |
| FLUX.1 \[dev\] | Black Forest Labs (Germany) | 12B | \~24 GB | FLUX.1 Non-Commercial | Non-commercial and non-production use only |

### Recommended family ladder

Stable Diffusion 3.5, all three variants sharing one MMDiT architecture, so iso-time compares tiers of one family rather than different designs. SD 1.5 is held as the CPU fallback if 3.5 Medium cannot finish a usable image in the budget — labelled on screen as an earlier generation, not a 3.5 tier.

| Machine | Model | Resolution | Steps | Batch |
| --- | --- | --- | --- | --- |
| DGX H200 | SD 3.5 Large | 1024² or higher | 40+ | 8 variants |
| Workstation 16 GB | SD 3.5 Medium | 1024² | 28–30 | 1–2 |
| CPU 16 GB | SD 3.5 Medium, or SD 1.5 | 512² | 10–15 | 1 |

The batch of 8 on the DGX is what makes the advantage obvious rather than marginal. One 1024² image is a weak showing for a DGX; eight finished variants in the same minute is not.

### An NVIDIA-specific angle worth adding

Stability AI and NVIDIA published TensorRT and FP8 optimised builds of the SD 3.5 family: up to 2.3x faster on Large, 1.7x on Medium, with VRAM reduced by about 40% ([Stability AI](https://stability.ai/news/stable-diffusion-35-models-optimized-with-tensorrt-deliver-2x-faster-performance-and-40-less-memory-on-nvidia-rtx-gpus)). Running the workstation twice — stock weights, then the TensorRT build — adds a fourth bar to the chart and makes the point that the software stack is part of the hardware story. It also means SD 3.5 Large may fit the 16 GB workstation under TensorRT, which is itself a good on-stage moment.

### Inputs

Prompts are drawn from published text-to-image benchmark prompt sets rather than written from scratch, so the selection is defensible to an academic audience. [DrawBench](https://arxiv.org/pdf/2205.11487) (200 prompts, 11 categories, from the Imagen paper) and [PartiPrompts](https://arxiv.org/pdf/2412.10604) (1,600 prompts) both target the categories where small models visibly fail: counting, spatial relations, text rendering, fine detail. The three chosen prompts are in the fixed input set below.

## Task 2 — Image-to-3D

Recommended as the second task. Image-to-3D is more reliable on stage than text-to-3D, which chains two models and so has two ways to fail. For an architecture or design audience, a mesh that can be rotated is the most persuasive thing in the session.

**Domain:** single input image to a textured 3D mesh, exported as GLB, displayed in a rotating viewer.

### Candidate models

| Model | Publisher | Params | Reported VRAM | Reported speed | Licence | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| [TRELLIS.2-4B](https://huggingface.co/microsoft/TRELLIS.2-4B) | Microsoft Research | 4B | 24 GB | 3 s / 17 s / 60 s at 512³ / 1024³ / 1536³ on an H100 | MIT | DGX tier |
| TRELLIS (v1) image-large | Microsoft Research | — | 16 GB | not published | MIT | Workstation tier |
| Stable Fast 3D | Stability AI (UK) | — | \~6 GB | 0.5 s on an H100, per the SF3D paper | Stability AI Community | Low tier / fallback |
| TripoSR | Stability AI with Tripo AI | — | low | fast | MIT | Co-developed with a Chinese company — your call |
| Hunyuan3D | Tencent | — | — | — | Tencent Community | Excluded by origin |
| Pixal3D | Tencent ARC | — | — | — | MIT | Excluded by origin |

### Recommended family ladder

TRELLIS, from v1 to TRELLIS.2-4B, with resolution as the dial. The 24 GB requirement for TRELLIS.2 against the workstation's 16 GB is the cleanest hardware boundary in the whole demo: the DGX can run the current best open image-to-3D model and the workstation genuinely cannot.

| Machine | Model | Output resolution | Expected behaviour |
| --- | --- | --- | --- |
| DGX H200 | TRELLIS.2-4B | 1536³, PBR textures | Finishes inside the budget; highest fidelity |
| Workstation 16 GB | TRELLIS v1 image-large | mesh from Gaussians | Finishes; visibly coarser geometry and texture |
| CPU 16 GB | Stable Fast 3D | low | Expect replay. Confirm in rehearsal whether any CPU run completes |

CPU image-to-3D is the most likely replay cell in the demo. That is not a weakness in the story — "this machine cannot do this task at all" is a legitimate and striking result, as long as the screen says the timing came from rehearsal.

### Inputs

Input images come from [Google Scanned Objects](https://arxiv.org/abs/2204.11918): over 1,000 3D-scanned household items released under a Creative Commons licence, and the standard benchmark set for single-image 3D reconstruction. Two advantages over photographing our own objects:

1. The licence is clean and the set is citable to an academic audience.
2. Every item has a ground-truth mesh, so the demo can show geometric error against the real object, not only which render looks nicer. That turns a subjective "which looks better" into a measured number, which is worth a great deal with this audience.

Three items are chosen by difficulty tier, listed in the fixed input set below.

## Task 3 (reserve) — Music generation

Built during preparation, held as the reserve. Audio is serial: three ten-second clips on three machines is about ninety seconds of listening per prompt, and the audience cannot compare them at a glance the way they can with images. It is the best task on licensing grounds and the weakest on stage legibility.

**Domain:** text-to-music, and music-to-music by continuation or inpainting from a fixed seed clip. Instrumental only — none of the eligible models produce vocals or lyrics.

### Candidate models

| Model | Publisher | Params | Max length | Licence | Verdict |
| --- | --- | --- | --- | --- | --- |
| [Stable Audio 3.0 Medium](https://huggingface.co/collections/stabilityai/stable-audio-3) | Stability AI (UK) | 1.4B | 6 min 20 s | Stability AI Community | DGX / workstation tier |
| Stable Audio 3.0 Small | Stability AI (UK) | 459M | 2 min | Stability AI Community | CPU tier; runs on CPU |
| Stable Audio 3.0 Small SFX | Stability AI (UK) | 459M | short effects | Stability AI Community | Sound effects variant |
| Stable Audio 3.0 Large | Stability AI (UK) | 2.7B | 6 min 20 s | API and enterprise only | Not open weights — excluded |
| MusicGen small / medium / large | Meta | 300M / 1.5B / 3.3B | short | Code MIT, weights CC-BY-NC | Cleanest size ladder, but non-commercial weights |

### Recommended family ladder

Stable Audio 3.0, Small on the CPU and Medium on the GPUs. Two reasons it beats MusicGen despite MusicGen's neater parameter ladder: Stability states the models are trained entirely on licensed data, and the licence permits commercial use below the revenue threshold rather than forbidding it outright. Inpainting and continuation are built in, so one family covers both the text-to-music and music-to-music tasks from your original list.

### One honest problem with this task

Stability reports 0.44 s inference for Small and 1.31 s for Medium on an H200. At those speeds a 30 to 60 second budget is meaningless on any GPU — everything finishes immediately, and the iso-time protocol collapses. The comparison becomes CPU against GPU, with the DGX and the workstation nearly tied unless the demo generates many tracks at once or very long tracks.

If the audio task is used, it should be framed as the **batch** demonstration: how many complete tracks each machine produces in one minute. That is a real DGX advantage and an easy number to report.

### Inputs

Prompts and a fixed 10-second seed clip for the continuation task are listed in the fixed input set below. The seed clip must be something we own or that is openly licensed — not a commercial recording.

## Fixed input set

The same inputs on every machine, every run, with a fixed seed. Chosen for traits that small models and low step counts visibly fail: legible text, reflections, counted objects, thin structures, fine material detail.

### Text-to-image

| # | Prompt | What it exposes |
| --- | --- | --- |
| 1 | A sunlit modern interior with a glass coffee table, a brass floor lamp, and a bookshelf of five legible book spines | Text rendering, counting, reflections |
| 2 | A dense city street at dusk in the rain, neon shop signs reflected in the puddles | Lighting, reflections, scene coherence |
| 3 | A single ripe pomegranate split open on a marble countertop, water droplets on the skin | Fine material detail, translucency |

Prompt 1 is the interior-design and architecture prompt, in case that audience is confirmed. All three are held at the same aspect ratio and seed.

### Image-to-3D

| # | Source object | Tier | What it exposes |
| --- | --- | --- | --- |
| 1 | A simple chair, plain background | Easy | Baseline — every machine should produce something |
| 2 | A patterned ceramic vessel | Medium | Texture fidelity and PBR material quality |
| 3 | An object with a thin arm or handle, such as a lamp or a mug | Hard | Thin geometry, where low tiers break down |

Selected from Google Scanned Objects so each has a ground-truth mesh for the error measurement.

### Music, if used

| # | Input | Task |
| --- | --- | --- |
| 1 | Slow oud melody over soft percussion, warm reverb | Text-to-music |
| 2 | Upbeat electronic track with a bright synth lead, 120 BPM | Text-to-music |
| 3 | A fixed 10 s piano phrase, openly licensed or our own | Music-to-music by continuation |

- [ ] Confirm the exact three GSO object IDs after a rehearsal pass.
- [ ] Source or record the 10 s seed clip for input 3.

## Measurement

Every run logs the same record, so the press gets numbers rather than impressions and the academics get a method they can question.

| Field | Why it is logged |
| --- | --- |
| Machine, model, tier | Identifies the cell |
| Protocol | iso-time or iso-quality |
| Resolution, steps, batch size, seed | Makes the run reproducible |
| Model load time | Separated from generation time; a cold load distorts the comparison |
| Time to first output | The number the audience watches |
| Time to all outputs | The batch number, where the DGX leads |
| Outputs completed in budget | The iso-time result |
| Peak memory used | Shows why a tier did not fit |
| LIVE or REPLAY | Honesty marker, shown on screen |

For the 3D task, one additional measure: geometric error against the Google Scanned Objects ground-truth mesh, reported per machine. This converts "which looks better" into a number and is the single most credible thing the demo can put in front of an academic audience.

Two rules that protect the comparison:

- **Model load time is excluded from the generation clock, and reported separately.** Pre-load all weights before the session. A 40-second load on the CPU box would otherwise be counted as slowness in the model.
- **Warm up every machine before the audience is in the room.** The first run after a cold start is unrepresentative on all three.

## Demo architecture

One Streamlit app on the operator's laptop, three identical worker services, one shared results store. Nothing on stage depends on the internet.

&#91;embedded content: demo architecture · one UI, three workers, one store\]

Each machine runs the same small worker service exposing one endpoint per task. The dispatcher sends the same job to all three at once and starts the clock; the UI polls the results store and fills each panel as it arrives, so a slow machine does not block a fast one. The replay table supplies a prerecorded output and its rehearsal timing whenever a cell times out or a worker is unreachable.

Why a worker service per machine rather than running everything from the laptop: the model must execute on the machine being measured, and the timing must be taken on that machine, not across the network. The network carries only the job request and the finished file.

UI panels run left to right in the same order as the machines. Each shows the live clock, the output as it lands, the model tier and settings used, and a LIVE or REPLAY badge. One summary bar underneath carries the headline number for the current protocol.

## Fallback and rehearsal

Every cell of the demo has a prerecorded output and a real rehearsal timing behind it. A cell is one combination of task, machine, input and protocol.

The replay table is keyed on task, machine, input ID, protocol and settings. When a live run times out, errors, or the worker is unreachable, the UI shows the prerecorded output and the rehearsal timing, badged **REPLAY**. A live run is badged **LIVE**.

Never substitute silently. The badge is what protects the session's credibility: if a journalist later discovers an unmarked substitution, the story becomes the substitution. Marked, it is simply a sensible precaution that nobody objects to.

### Rehearsal plan

1. **Full matrix pass.** Run every task on every machine with every input, under both protocols. Record timings, outputs, peak memory, and failures. This pass populates the replay table.
2. **Set the budget.** Use the recorded timings to choose between 30 s and 60 s. The right budget is the one where the three results differ most visibly.
3. **Set per-machine tiers.** Fix the model tier and settings each machine uses under iso-time, based on what it actually finished.
4. **Failure drill.** Unplug a worker mid-run and confirm the replay path engages cleanly and the badge changes.
5. **Dress rehearsal on the venue network.** Network conditions are the most common cause of live demo failure, and the only way to test them is in the room.

- [ ] Book venue access for the dress rehearsal, with all three machines in place.
- [ ] Decide who operates the UI on stage, and who narrates.

## Open decisions and risks

### Decisions needed from you

- [ ] Which two tasks. Recommendation: text-to-image and image-to-3D.
- [ ] Is TRELLIS acceptable under the origin rule, given Microsoft publishes the weights under MIT but the papers carry Tsinghua University co-authors? If not, image-to-3D has no strong eligible candidate and music becomes the second task.
- [ ] Does the Stability AI Community License revenue threshold apply to your organisation? If it does, the TRELLIS models (MIT) are unaffected but every Stability model needs a licence check.
- [ ] Audience composition, still to be confirmed. Architecture-heavy favours 3D; general press favours image.
- [ ] Which two ARC projects are shown, and how much time each needs.

### Risks

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| CPU cannot complete the 3D task at all | High | Replay cell, badged. Framed as a result, not a failure |
| Venue network drops a worker mid-run | Medium | Replay path, drilled in rehearsal |
| DGX advantage looks marginal on a single small generation | Medium | Batch of 8 and the top model tier are what make it visible |
| Iso-time reads as a rigged comparison | Medium | On-screen label stating that each machine ran the largest model it could finish |
| A model version is superseded before the event | Low | Freeze versions after rehearsal; weights are local, so nothing changes under us |

### One item to verify before locking models

Sources disagree on whether Stability AI has released a successor to Stable Diffusion 3.5. Several pages dated 2026 claim a Stable Diffusion 4 launched in April 2026; a source checked in September 2026 states plainly that no successor has been announced and that the pages claiming one cite no Stability announcement. This proposal assumes SD 3.5 is current. Confirm directly on Stability AI's own site before downloading weights — if an SD 4 family exists with open weights and size tiers, it is the better ladder.

## Sources

- [Stable Diffusion 3.5 with TensorRT and FP8 on NVIDIA RTX](https://stability.ai/news/stable-diffusion-35-models-optimized-with-tensorrt-deliver-2x-faster-performance-and-40-less-memory-on-nvidia-rtx-gpus) — Stability AI, speed and VRAM figures
- [SD 3.5 Large model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) and [SD 3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) — licence and gating
- [TRELLIS.2-4B model card](https://huggingface.co/microsoft/TRELLIS.2-4B) — MIT licence
- [Stable Audio 3.0 announcement](https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models) and [the Stable Audio 3 paper](https://arxiv.org/html/2605.17991) — model sizes, open-weight variants, licensed training data
- [Google Scanned Objects](https://arxiv.org/abs/2204.11918) — input images and ground-truth meshes
- [DrawBench, in the Imagen paper](https://arxiv.org/pdf/2205.11487) and [PartiPrompts, via EvalGIM](https://arxiv.org/pdf/2412.10604) — benchmark prompt sets

VRAM and speed figures for TRELLIS and Stable Fast 3D are as reported by third-party surveys of open 3D models, not by the publishers directly, and should be confirmed on our own hardware during rehearsal rather than quoted on stage.
