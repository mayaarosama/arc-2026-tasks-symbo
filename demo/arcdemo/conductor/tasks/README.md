# arcdemo/conductor/tasks/: what the conductor does differently per task

`core.py` and `app.py` know no task by name. At each step that differs, they call the task's handler
(`handler(task)` in `__init__.py`). Every hook is defined in `base.py`, with the shared behaviour as
its default, so a task overrides only what it needs.

| File | Task | Overrides |
| --- | --- | --- |
| `base.py` | — | `TaskHandler`: the hooks and their defaults (see its docstring) |
| `t2i.py` | 1 · text to image | `prepare_input` (the moderated live prompt), `job_fields` (prompt), `unfinished` (a partial batch counts as unfinished), `hard_budget`, route `POST /api/moderate` |
| `i23d.py` | 2 · image to 3D | `on_start` (re-offer saved uploads), `caption` (no score for an upload), `job_fields` (image path, or the uploaded PNG as base64), `budget_note` (finished after the bell), `after_result` (score against the GSO scan), routes `GET /api/i23d/spec`, `POST /api/i23d/upload`. Also `add_upload` / `load_uploads` |
| `music.py` | 3 · music (reserve) | `job_fields` (prompt, mode, audio, length) |
| `llm.py` | 4 · language model race | `job_fields` (prompt), `prompt_text` (system + user, for the operator page), `hard_budget` |
| `layout.py` | 8 · scene construction | `prepare_input` (wall names and the request in words), `job_fields` (room, plan, door), `summary` (objects and flags), `prompt_text`, `hard_budget`, routes `GET /api/layout_catalog`, `GET /api/layout_view`, and the furniture models at `/layout-assets`. Also `layout_view`, `layout_prompt_text`, `layout_summary` |
| `__init__.py` | — | `HANDLERS` in show order, and `handler(task)` |

To add a task:
1. Write a `TaskHandler` subclass here.
2. Register it in `__init__.py`.
3. Add its rules to `arcdemo/worker/tasks.py` and a backend to `arcdemo/worker/backends/`.
4. Add its `STAGE_VIEW` / `OPERATOR_VIEW` / `TASK_TEXT` entries to the pages in `static/`.
