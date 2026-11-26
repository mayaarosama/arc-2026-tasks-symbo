# scripts/cpu/: the CPU box (Windows or Linux)

Scripts run as `python -m scripts.cpu.<name>` from `demo/`; shell scripts by path. The full list of scripts, and what each folder is for, is in [../README.md](../README.md).

| File | Runs where | What |
| --- | --- | --- |
| `cpu_box_linux.sh` | the CPU box | `setup` / `start` / `stop` / `status` / `conductor` (and `firewall` on Windows) |
| `cpu_box_windows.ps1` | the CPU box | `setup` / `start` / `stop` / `status` / `conductor` (and `firewall` on Windows) |
| `check_cpu.py` | the CPU box, t2i venv | What the CPU offers, and the `cpu_threads` / `cpu_bf16` lines for `worker-cpu.yaml` |
| `fetch_llama_cpp.py` | the CPU box, llm venv | llama.cpp's prebuilt CPU build of the pinned release |
