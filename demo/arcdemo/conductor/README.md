# arcdemo/conductor/: dispatch, fallback, and the operator and stage pages

```bash
python -m arcdemo.conductor --config config/demo.yaml     # host, port, config defaults: config/defaults.yaml conductor
```

A **round** is one segment × one input, sent to every machine at once. Each machine is a **cell**:
- cells are independent;
- a dead or slow machine falls back to its rehearsal recording, badged REPLAY or RECORDED;
- the other cells are not affected.

| File | What |
| --- | --- |
| `__main__.py` | Command line: reads the config, sets up logging, runs the web app (uvicorn) |
| `app.py` | The web app: static pages and media mounts, the shared `/api` routes (config, state, round, cancel, cell stop/replay, plan, prompt, model, prepare, view), then each task's own routes from `tasks/` |
| `core.py` | `Conductor`, task-neutral. In order: model picks, public state, health polling, prepare (load / unload across machines), rounds (start, finish, cancel, stop, switch to replay, plan), one cell (dispatch, poll, fetch outputs), replay and full recordings, the run log (`runs.jsonl`) and summary. It calls `tasks.handler(task)` wherever a task differs |
| `config.py` | `DemoConfig`: loads `demo.yaml` (with `extends`) and `models.yaml`, validates everything at start-up, resolves a cell's settings (the planned model or the operator's pick) |
| `tasks/` | One handler per task, in show order: see [tasks/README.md](tasks/README.md) |
| `moderation.py` | Task 1: the live prompt check against `config/blocklist.txt` |
| `replay.py` | The replay table: `replay/<task>/<protocol>/<machine>/<input>/[full/]record.json`, with a settings hash so stale records are flagged |
| `urls.py` | Where the web app serves each folder (`/media`, `/replay-media`, `/media-input/data`, `/layout-assets`) |
| `static/` | The operator and stage pages: see [static/README.md](static/README.md) |
