# data/: inputs and prompts

Everything the demo feeds to a model, and the text it shows about it. Paths are relative to `demo/`,
and every machine needs the same copy. The conductor serves this folder at `/media-input/data`.
`tests/test_data_and_static.py` checks each file's shape and that every path it names exists.

## Stage inputs, one file per task (named under `inputs:` in `config/demo.yaml`)

| File | Format (one JSON list, one object per input) |
| --- | --- |
| `t2i/stage_prompts.json` | `id`, `label`, `seed`, `prompt`, `exposes`. `id: live` is the audience prompt, typed by the operator |
| `i23d/stage_objects.json` | `id`, `label`, `seed`, `input` (the 512 × 512 RGBA render), `tier`, `gso`, `alternates` |
| `music/stage_inputs.json` | `id`, `label`, `seed`, `prompt`, `mode` (text or continuation), `length_s`, and `audio` for a continuation |
| `llm/stage_prompts.json` | `id`, `label`, `seed`, `prompt` |
| `layout/stage_rooms.json` | `id`, `label`, `seed`, `room_type`, `floor_plan` ([[x, y], ...] in metres), `door` ([x, y] on the outline), `max_objects` |

## Other inputs

| File | What |
| --- | --- |
| `t2i/live_fallback_prompts.json` | `{"prompts": [...]}`: vetted spare prompts offered on the operator page (path: `config/defaults.yaml` `stage.spare_prompts`) |
| `t2i/rehearsal_prompts.json` | Rehearsal prompts picked from PartiPrompts (`python -m scripts.t2i.build_rehearsal_prompts`) |
| `i23d/inputs/` | The rendered inputs (`python -m scripts.i23d.prepare_gso`, which downloads the GSO scans into the git-ignored `i23d/gso_cache/`); CC BY 4.0. Outputs are not scored against the scans |
| `i23d/candidates.png`, `i23d/stage_inputs.png` | Contact sheets of the candidates and of the chosen inputs |
| `i23d/upload_rules.json` | What an uploaded picture must be like, shown next to the upload box; `$max_mb`-style placeholders come from `config/defaults.yaml` `i23d.upload` |
| `layout/furniture.json` | The furniture catalogue: per class its size, `place` (wall or free) and 3D model file; per room type the classes it may hold; `assets_dir` |
| `layout/exemplars.json` | LayoutGPT's example bank: our own rooms (`python -m scripts.layout.build_layout_exemplars`) |

## Prompts (`prompts/`, read by `arcdemo/prompts.py`)

| File | Used by |
| --- | --- |
| `llm_system.txt` | Task 4: the system message of every reply (the worker sends it; the operator page shows it) |
| `layout_system.txt` | Task 8, prompt v3.1: the rules |
| `layout_examples.json` | Task 8, prompt v3.1: the two worked example rooms and their replies, one object per line |
| `layout_self_correction.json` | Task 8, *Furnish and fix*: what the model is told about its mistakes (repair, review, undone edits) |
| `layoutgpt.json` | Task 8, LayoutGPT's published prompt: system text, condition template, room names |

Templates use `$name` placeholders (Python `string.Template`). To change a prompt, edit the file and
restart the workers and the conductor. `tests/test_layout.py` checks that the worked examples are still valid rooms.
