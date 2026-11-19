"""What the CPU box offers, and what to put in config/worker-cpu.yaml. Same on Windows and Linux.

    python scripts/cpu/check_cpu.py      (run with the t2i env's python: it needs torch + psutil)
"""
import platform
import shutil
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
GB = 1024 ** 3


def main() -> None:
    ram = psutil.virtual_memory()
    disk = shutil.disk_usage(ROOT)
    cores = psutil.cpu_count(logical=False)
    print(f"os            {platform.platform()}")
    print(f"python        {platform.python_version()}")
    print(f"cpu           {platform.processor() or platform.machine()}")
    print(f"cores         {cores} physical / {psutil.cpu_count()} logical")
    print(f"ram           {ram.total / GB:.1f} GB total, {ram.available / GB:.1f} GB free now")
    print(f"disk          {disk.free / GB:.0f} GB free at {ROOT}")
    try:
        import torch
        bf16 = bool(torch.ops.mkldnn._is_mkldnn_bf16_supported())   # oneDNN's own check (AVX512-BF16 / AMX)
        print(f"torch         {torch.__version__}, cpu capability {torch.backends.cpu.get_cpu_capability()}")
        print(f"bf16 on cpu   {'yes' if bf16 else 'no'}")
    except Exception as e:  # noqa: BLE001 — torch missing or too old: report, do not guess
        bf16 = None
        print(f"bf16 on cpu   unknown ({type(e).__name__}: {e})")

    print("\nworker-cpu.yaml:")
    print(f"  cpu_threads: {cores}")
    print(f"  cpu_bf16: {'true' if bf16 else 'false'}"
          + ("" if bf16 is not None else "   # could not detect; false is the safe choice"))
    if ram.total < 15 * GB:
        print("WARNING: under 16 GB RAM — SD 3.5 Medium (no T5, fp32) needs ~13 GB; plan on SD 1.5.")
    if disk.free < 40 * GB:
        print("WARNING: under 40 GB free — the CPU box's model cache needs ~30 GB.")


if __name__ == "__main__":
    main()
