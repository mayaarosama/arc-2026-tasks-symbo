# arcdemo/worker/: runs one task's models on the machine under test

```bash
python -m arcdemo.worker --config config/worker-<machine>.yaml --task t2i --port 8101   # in the task's venv
```

There is one process per task, in that task's venv; a mock worker may serve several tasks at once.
A job is submitted, returns at once, and is polled. The clock starts when generation starts (the model
is already loaded) and stops when the last output file is written.

| File | What |
| --- | --- |
| `__main__.py` | Command line: loads the worker config, sets up logging (`results/worker-<task>.log`), runs the app |
| `app.py` | Shared, task-neutral parts:<br>• `load_worker_config` (the worker YAML + `models.yaml` + `model_overrides`)<br>• `ModelManager` (load / unload / one model at a time)<br>• `JobRunner` (one job at a time, a thread per job, the status at the end)<br>• the memory and environment reports<br>• the HTTP API: `/health`, `/env`, `/load`, `/unload`, `/jobs`, `/jobs/{id}`, `/jobs/{id}/cancel`, `/files/{name}`<br>All routes need `X-Demo-Token` when the token variable is set |
| `tasks.py` | What differs per task, in show order: outputs requested, the settings echoed in the result, inputs written before the clock (Task 2 uploads), the mock backend per task, the tasks whose engine sets its own CPU threads |
| `metrics.py` | `Clock` (synchronised with CUDA before each timestamp) and `MemoryMeter` (peak GPU memory, or sampled peak RSS on CPU) |
| `backends/` | One class per model library: see [backends/README.md](backends/README.md) |
