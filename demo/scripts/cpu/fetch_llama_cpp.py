"""Fetch the pinned llama.cpp CPU release for the CPU box (Windows or Linux x64), offline afterwards.

    python -m scripts.cpu.fetch_llama_cpp            # -> third_party/llama.cpp/bin/llama-server(.exe)

The GPU machines build the same tag with CUDA (scripts/gpu/setup_gpu_machine.sh); the CPU box uses
llama.cpp's own prebuilt CPU build of that tag, so every machine runs the same release. Safe to
re-run: an installed binary of the right release is kept. Prints the installed version.
"""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from scripts.common.settings import SCRIPTS, VERSIONS

TAG = VERSIONS["LLAMA_TAG"]          # config/versions.env: the same release on every machine
ASSETS = {os_: (name.format(tag=TAG), size)      # release file -> its size in bytes (a truncated download
          for os_, (name, size) in SCRIPTS.fetch_llama_cpp.assets.items()}   # is refused); config/scripts.yaml
DEMO = Path(__file__).resolve().parents[2]
BIN = DEMO / "third_party" / "llama.cpp" / "bin"


def installed_version(exe: Path) -> str:
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main() -> int:
    if platform.machine().lower() not in ("amd64", "x86_64"):
        sys.exit(f"no prebuilt CPU release for {platform.machine()} here: build llama.cpp {TAG} from source")
    kind = "win" if sys.platform == "win32" else "linux"
    exe = BIN / ("llama-server.exe" if kind == "win" else "llama-server")
    if TAG[1:] in installed_version(exe):
        print(f"llama-server {TAG} already installed: {exe}")
        return 0
    name, size = ASSETS[kind]
    url = f"{VERSIONS['LLAMA_REPO']}/releases/download/{TAG}/{name}"
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / name
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, archive)
        if archive.stat().st_size != size:
            sys.exit(f"{name}: got {archive.stat().st_size} bytes, expected {size} (a download error page?)")
        unpacked = Path(tmp) / "x"
        if kind == "win":
            with zipfile.ZipFile(archive) as z:
                z.extractall(unpacked)
        else:
            with tarfile.open(archive) as t:
                if any(m.name.startswith("/") or ".." in Path(m.name).parts for m in t.getmembers()):
                    sys.exit(f"{name}: unexpected paths in the archive")
                try:
                    t.extractall(unpacked, filter="tar")  # keeps the .so symlinks and the exec bits
                except TypeError:                         # Python before 3.10.12 / 3.11.4: no filter argument
                    t.extractall(unpacked)
        server = next(unpacked.rglob(exe.name))
        if BIN.exists():
            shutil.rmtree(BIN)                            # another release: replace it whole
        shutil.copytree(server.parent, BIN, symlinks=True)
    version = installed_version(exe)
    if TAG[1:] not in version:
        sys.exit(f"llama-server did not report release {TAG}:\n{version}")
    print(f"installed {exe}\n{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
