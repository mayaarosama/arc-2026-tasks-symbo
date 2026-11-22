# Additional candidate tasks

Four further tasks evaluated for stage impact. All four can be downloaded and run fully offline. Two have licence or hardware caveats that change what was assumed earlier.

## Summary

| Task | Model ladder | Publisher | Licence | Three-machine fit | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Live webcam restyling | SD-Turbo, SDXL-Turbo via StreamDiffusion | Stability AI; pipeline from Kodaira et al. | Stability AI Community (see caveat) | GPU live, CPU a slideshow | Strongest audience moment |
| Text-to-video | Wan 2.2 TI2V-5B then T2V-A14B | Alibaba (Wan team) | Apache 2.0 | DGX only at full spec | Biggest honest hardware gap |
| Language model race | Qwen3 0.6B to 235B-A22B | Alibaba Cloud | Apache 2.0 | All three, clean ladder | Cleanest licence and ladder |
| 4K upscaling | Real-ESRGAN x4plus | Tencent ARC Lab (Xintao Wang et al.) | BSD 3-Clause | All three, genuinely | Needs scaling up to be a contest |

## Task 4 — Live webcam restyling

**Input and output.** Live webcam frame in, restyled frame out, continuously. A text prompt sets the style and can be changed mid-stream. The measure is frames per second, not seconds per image.

**Models.** SD-Turbo (distilled from Stable Diffusion 2.1) and SDXL-Turbo, both single-step or few-step models, driven by the StreamDiffusion pipeline. StreamDiffusion is an ICCV 2025 pipeline-level system, Apache 2.0, with a ready-made real-time img2img demo that takes a live webcam feed or screen capture in a browser.

**Performance.** The StreamDiffusion authors report SD-Turbo at 93.9 fps img2img and 106.2 fps txt2img at one denoising step, measured on an RTX 4090 with a Core i9-13900K under Ubuntu 22.04. Those are publisher figures on one machine, not ours.

**Licence caveat — read this before downloading.** Sources conflict. SDXL Turbo was announced under a non-commercial research licence that explicitly said it was not intended for commercial use, and that text is still in Stability's GitHub `model_licenses` directory. The current Hugging Face `LICENSE.md` for `stabilityai/sdxl-turbo` is the Stability AI Community License with both research and commercial sections, and Stability's licence page lists SDXL Turbo under the Community tier alongside Stable Audio 3.0 and Stable Fast 3D. Treat it as the Community License, and re-read the Hugging Face copy on the day of download.

**Offline.** Fine. Weights are a few gigabytes and download once. StreamDiffusion installs from GitHub or pip, with an optional TensorRT extension built locally.

**Constraints.**

| Constraint | Handling |
| --- | --- |
| Needs a webcam feed into each machine, or one feed distributed to all three | Decide early: one camera fanned out is better theatre and harder to wire |
| TensorRT extension must be built per GPU | Build during preparation, never at the venue |
| CPU will be a slideshow, possibly under 1 fps | That is the demonstration, not a failure — badge it clearly |
| Live faces on a big screen | Get consent from whoever is in frame; use a volunteer, not the audience at large |

## Task 5 — Text-to-video

**Input and output.** Text prompt in, short video out, typically 5 seconds at 720p, 24 fps, written as MP4.

**Models.** Wan 2.2, Apache 2.0, from Alibaba's Wan team. Open checkpoints include `Wan-AI/Wan2.2-TI2V-5B`, `Wan-AI/Wan2.2-T2V-A14B` and `Wan-AI/Wan2.2-I2V-A14B`. The A14B models are a 27B mixture-of-experts with 14B active.

**Correction to the earlier hardware estimate.** Alibaba recommends **at least 80 GB VRAM** for single-GPU A14B inference and **at least 24 GB** (for example an RTX 4090) for TI2V-5B. The widely quoted 8 GB figure comes from ComfyUI's native offloading of the 5B workflow, not from the publisher. A third-party guide reports 14B at int8 fitting in about 16 GB. What this means for us:

| Machine | Realistic option | Confidence |
| --- | --- | --- |
| DGX H200 | T2V-A14B at 720p | High |
| Workstation 16 GB | TI2V-5B with ComfyUI offloading, or A14B int8 | Low — must be tested |
| CPU 16 GB | None | Replay cell |

**Version warning.** Only Wan 2.2 is Apache 2.0 with open weights. Wan 2.5, 2.6 and 2.7 are closed, API-only products. If a benchmark or article cites "Wan", confirm which version it means.

**Offline.** Weights from Hugging Face or ModelScope, downloaded ahead. No runtime network dependency.

**Constraint worth weighing.** Video is the most impressive output on the list and the only one with no CPU tier at all, so the three-way framing becomes two-way plus a replay. It is also the slowest to generate, which eats stage time.

## Task 6 — Language model race

**Input and output.** One prompt sent to all three machines, token streams rendered side by side. The measure is tokens per second and which model sizes each machine can hold at all.

**Recommended ladder: Qwen3, not Llama 3.1.** Qwen3 is released under Apache 2.0 across the entire family: dense models at 0.6B, 1.7B, 4B, 8B, 14B and 32B, plus mixture-of-experts models at 30B-A3B and 235B-A22B. No revenue threshold, no user-count clause, no branding requirement.

**Why not Llama 3.1.** The Llama 3.1 Community License imposes real obligations: you must prominently display "Built with Llama" on a related website, user interface, blog post, about page or product documentation; retain an attribution notice stating the model is licensed under the Llama 3.1 Community License; comply with Meta's Acceptable Use Policy; and include "Llama" at the start of the name of any model you train or fine-tune using Llama outputs. There is also a clause requiring a separate licence from Meta for licensees above 700 million monthly active users. None of this is prohibitive, but at an event where licensing may be discussed, Apache 2.0 with no obligations is a better position than a conditional licence with an on-screen branding requirement.

**Suggested per-machine assignment.**

| Machine | Model | Why |
| --- | --- | --- |
| DGX H200 | Qwen3-235B-A22B | Only the DGX can hold it — the visual point of the segment |
| Workstation 16 GB | Qwen3-8B or 14B quantised | Comfortable, fast |
| CPU 16 GB | Qwen3-0.6B or 1.7B | Runs, slowly, which is the contrast |

**Offline.** Weights from Hugging Face; serve with vLLM, llama.cpp or Ollama, all of which run without network once the model is local.

**Constraints.** Quantisation format must be identical across machines or the comparison is meaningless. Fix the prompt, the sampling parameters and the seed. Report tokens per second measured by the serving stack, not by eye.

## Task 7 — 4K upscaling

**Input and output.** A low-resolution image in, a 4x upscaled image out. For example 512×512 in, 2048×2048 out.

**Model.** Real-ESRGAN, BSD 3-Clause, from the ICCVW 2021 paper by Xintao Wang and collaborators at Tencent ARC Lab. The same codebase ships sibling checkpoints for anime imagery, anime video and a 2x general model. BSD 3-Clause requires only that you retain the copyright notice and disclaimer — the most permissive licence of any model across this whole project.

**The problem with this task.** Real-ESRGAN is tiny. The general x4v3 checkpoint is about 1.21M parameters and 4.66 MB. A single image upscales quickly on almost any hardware, so a one-image contest will look close to a tie and the demo will fall flat.

Three ways to make it a genuine contest, in order of preference:

1. **Upscale video.** Run it frame by frame over a short clip and measure frames per second. This restores a large, honest hardware gap.
2. **Batch.** Upscale 100 images and measure total time.
3. **Go very large.** Upscale to 8K with tiling, where memory and tile count start to matter.

**Offline.** Trivially. The weights are megabytes, not gigabytes.

**Constraints.** Large targets need tiled inference to avoid memory spikes; tile size must be identical across machines. Source images must be openly licensed — Google Scanned Objects renders already prepared for the 3D task would serve.

## Open questions

- [ ] Which of these four, if any, join the two confirmed tasks? Stage time is the limit, not build effort.
- [ ] Webcam task: one camera distributed to three machines, or three cameras?
- [ ] Video task: accept that it is two machines plus a replay, or drop it?
- [ ] Upscaling: switch to video upscaling to make it a real contest?

## Sources

- [SDXL Turbo announcement](https://stability.ai/news/stability-ai-sdxl-turbo) and [the Hugging Face licence](https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md); [Stability AI licence tiers](https://stability.ai/license)
- [StreamDiffusion paper](https://arxiv.org/abs/2312.12491) and [repository](https://github.com/cumulo-autumn/StreamDiffusion)
- [Wan 2.2 TI2V-5B model card](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B); [ComfyUI Wan 2.2 documentation](https://docs.comfy.org/tutorials/video/wan/wan2_2)
- [Qwen3 release blog](https://qwenlm.github.io/blog/qwen3/) and [the Qwen3 paper](https://arxiv.org/abs/2505.09388)
- [Llama 3.1 Community License](https://developer.meta.com/ai/llama3_1/license/)
- Real-ESRGAN licence and checkpoint details via [Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models/real_esrgan_general_x4v3) and model mirrors citing `xinntao/Real-ESRGAN`
