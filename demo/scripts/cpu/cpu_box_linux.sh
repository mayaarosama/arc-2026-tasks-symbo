#!/usr/bin/env bash
# CPU box, Linux: set up, start, stop and check the five workers
# (t2i :8101, i23d :8102, music :8103, llm :8104, layout :8105; llm and layout share .venv-llm).
#
#   scripts/cpu/cpu_box_linux.sh setup      # venvs, CPU torch, SF3D, llama.cpp, weights, offline check (needs network)
#   scripts/cpu/cpu_box_linux.sh start      # all workers, offline, in the background
#   scripts/cpu/cpu_box_linux.sh status     # /health of each
#   scripts/cpu/cpu_box_linux.sh stop
#   scripts/cpu/cpu_box_linux.sh conductor  # the machine that runs the conductor (operator laptop): its venv
#                                           # and the cut-out model for uploaded pictures (needs network)
#
# Secrets come from a file (default: ../config.txt next to demo/), never from arguments:
#   ARC_DEMO_TOKEN=<shared secret, same on every machine>
#   HF_TOKEN=hf_...            (a bare line starting with hf_ is also accepted)
# Values are never printed. Optional env: PYTHON (default python3.11), TORCH_VERSION (match the
# DGX lock, e.g. 2.6.0), SECRETS (path), WORKER_CONFIG (default config/worker-cpu.yaml).
# Every command is echoed with "+"; setup output is also saved to results/setup-<time>.log.
set -euo pipefail

DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECRETS="${SECRETS:-$DEMO/../config.txt}"
WORKER_CONFIG="${WORKER_CONFIG:-config/worker-cpu.yaml}"
PYTHON="${PYTHON:-python3.11}"
SF3D_DIR="$DEMO/third_party/stable-fast-3d"
SA3_DIR="$DEMO/third_party/stable-audio-3"
SA3_TORCH="2.7.1"                 # stable-audio-3 pins torch + torchaudio to this version
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
PORTS=(t2i:8101 i23d:8102 music:8103 llm:8104 layout:8105)
cd "$DEMO"
mkdir -p results

venv_of() { case "$1" in llm|layout) echo .venv-llm ;; *) echo ".venv-$1" ;; esac; }   # layout = llm env
run()  { echo "+ $*"; "$@"; }
warn() { echo "WARNING: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }
vpy()  { if [ -x "$1/bin/python" ]; then echo "$1/bin/python"; else echo "$1/Scripts/python.exe"; fi; }  # Scripts/: Git Bash

load_secrets() {
    [ -f "$SECRETS" ] || die "secrets file not found: $SECRETS (set SECRETS=...)"
    local line key
    while IFS= read -r line || [ -n "$line" ]; do        # || [ -n ]: last line without newline
        line="${line%$'\r'}"                              # tolerate a file saved on Windows
        line="$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
        if [ -z "$line" ] || [ "${line:0:1}" = "#" ]; then continue; fi
        case "$line" in
            hf_*) export HF_TOKEN="$line" ;;
            *=*)  key="${line%%=*}"
                  case "$key" in
                      HF_TOKEN|ARC_DEMO_TOKEN) export "$key=${line#*=}" ;;
                      *) warn "secrets file: ignoring unknown key '$key'" ;;
                  esac ;;
            *)    warn "secrets file: ignoring a line that is neither KEY=VALUE nor an hf_ token" ;;
        esac
    done < "$SECRETS"
    echo "secrets: HF_TOKEN $([ -n "${HF_TOKEN:-}" ] && echo set || echo MISSING)," \
         "ARC_DEMO_TOKEN $([ -n "${ARC_DEMO_TOKEN:-}" ] && echo set || echo 'empty (workers accept anyone)')"
}

make_venv() {   # $1 = venv dir, $2.. = torch packages (default: torch[==TORCH_VERSION]).
    # CPU torch first, so nothing later pulls a CUDA wheel.
    local venv="$1" py; shift
    [ -d "$venv" ] || run "$PYTHON" -m venv "$venv"
    py="$(vpy "$venv")"
    [ $# -gt 0 ] || set -- "torch${TORCH_VERSION:+==$TORCH_VERSION}"
    run "$py" -m pip install -q --upgrade pip
    run "$py" -m pip install -q "$@" --index-url "$TORCH_INDEX"
    echo "torch in $venv: $("$py" -c 'import torch; print(torch.__version__)')"
}

setup() {
    exec > >(tee -a "results/setup-$(date +%Y%m%d-%H%M%S).log") 2>&1
    command -v "$PYTHON" >/dev/null || die "$PYTHON not found (set PYTHON=python3.10 or install 3.11)"
    "$PYTHON" -c 'import sys; assert sys.version_info[:2] in ((3, 10), (3, 11)), sys.version' \
        || die "need Python 3.10 or 3.11 for diffusers + SF3D"
    command -v git >/dev/null || die "git is required to fetch Stable Fast 3D"
    command -v g++ >/dev/null || die "g++ is required to build SF3D's texture_baker / uv_unwrapper"
    load_secrets
    [ -n "${HF_TOKEN:-}" ] || die "no Hugging Face token in $SECRETS (gated repos need it)"

    echo "== text-to-image env (.venv-t2i)"
    make_venv .venv-t2i
    run "$(vpy .venv-t2i)" -m pip install -q -r requirements/t2i.txt

    echo "== image-to-3D env (.venv-i23d): Stable Fast 3D"
    make_venv .venv-i23d
    [ -d "$SF3D_DIR" ] || run git clone https://github.com/Stability-AI/stable-fast-3d "$SF3D_DIR"
    echo "stable-fast-3d commit: $(git -C "$SF3D_DIR" rev-parse HEAD)"
    run "$(vpy .venv-i23d)" -m pip install -q -U setuptools==69.5.1 wheel
    # SF3D's requirements.txt installs ./texture_baker and ./uv_unwrapper by relative path
    (cd "$SF3D_DIR" && run "$(vpy "$DEMO/.venv-i23d")" -m pip install -q -r requirements.txt)
    run "$(vpy .venv-i23d)" -m pip install -q -r requirements/worker-base.txt
    # worker-base would pull huggingface_hub 1.x+, but SF3D pins transformers 4.42.3 (needs <1.0)
    run "$(vpy .venv-i23d)" -m pip install -q "huggingface_hub>=0.25,<1.0"

    echo "== music env (.venv-music): Stable Audio 3.0 Small"
    make_venv .venv-music "torch==$SA3_TORCH" "torchaudio==$SA3_TORCH"
    [ -d "$SA3_DIR" ] || run git clone https://github.com/Stability-AI/stable-audio-3 "$SA3_DIR"
    echo "stable-audio-3 commit: $(git -C "$SA3_DIR" rev-parse HEAD)"
    run "$(vpy .venv-music)" -m pip install -q -r requirements/music.txt
    run "$(vpy .venv-music)" -m pip install -q --no-deps -e "$SA3_DIR"

    echo "== language-model env (.venv-llm): no torch; the model runs in llama.cpp's llama-server (llm + layout)"
    [ -d .venv-llm ] || run "$PYTHON" -m venv .venv-llm
    run "$(vpy .venv-llm)" -m pip install -q --upgrade pip
    run "$(vpy .venv-llm)" -m pip install -q -r requirements/llm.txt
    run "$(vpy .venv-llm)" scripts/cpu/fetch_llama_cpp.py          # the pinned CPU release, third_party/llama.cpp/bin

    echo "== weights (HF_TOKEN from the secrets file)"
    run "$(vpy .venv-t2i)" scripts/download_weights.py --machine cpu --task t2i
    run "$(vpy .venv-i23d)" scripts/download_weights.py --machine cpu --task i23d
    run "$(vpy .venv-music)" scripts/download_weights.py --machine cpu --task music
    run "$(vpy .venv-llm)" scripts/download_weights.py --machine cpu --task llm      # Qwen3.5 4B, one GGUF file
    run "$(vpy .venv-llm)" scripts/download_weights.py --machine cpu --task layout   # the same file: already there

    echo "== this machine"
    run "$(vpy .venv-t2i)" scripts/cpu/check_cpu.py
    echo "== offline check (network use is now forbidden; a FAIL here would fail on stage)"
    unset HF_TOKEN
    run "$(vpy .venv-t2i)" scripts/verify_offline.py --config "$WORKER_CONFIG" --task t2i
    PYTHONPATH="$SF3D_DIR" run "$(vpy .venv-i23d)" scripts/verify_offline.py --config "$WORKER_CONFIG" --task i23d
    run "$(vpy .venv-music)" scripts/verify_offline.py --config "$WORKER_CONFIG" --task music
    run "$(vpy .venv-llm)" scripts/verify_offline.py --config "$WORKER_CONFIG" --task llm
    run "$(vpy .venv-llm)" scripts/verify_offline.py --config "$WORKER_CONFIG" --task layout
    echo "setup done. Firewall, if active:  sudo ufw allow 8101:8105/tcp"
}

conductor() {   # set up the conductor on this machine: its venv and the cut-out model for uploaded pictures
    exec > >(tee -a "results/setup-conductor-$(date +%Y%m%d-%H%M%S).log") 2>&1
    command -v "$PYTHON" >/dev/null || die "$PYTHON not found (set PYTHON=python3.10 or install 3.11)"
    [ -d .venv ] || run "$PYTHON" -m venv .venv
    run "$(vpy .venv)" -m pip install -q --upgrade pip
    run "$(vpy .venv)" -m pip install -q -r requirements/conductor.txt
    run "$(vpy .venv)" scripts/download_weights.py --cutout          # U2-Net, 176 MB -> weights/u2net/
    run "$(vpy .venv)" scripts/verify_offline.py --cutout            # runs on the CPU; on a GPU with onnxruntime-gpu
    echo "conductor ready:  $(vpy .venv) -m arcdemo.conductor --config config/demo.yaml"
}

health() {   # $1 = port; prints the /health JSON or fails
    curl -sf -m 2 ${ARC_DEMO_TOKEN:+-H "X-Demo-Token: $ARC_DEMO_TOKEN"} "http://127.0.0.1:$1/health"
}

start() {
    load_secrets
    unset HF_TOKEN                                   # not needed offline; keep it out of the workers
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SF3D_USE_CPU=1
    local entry task port venv pidf
    for entry in "${PORTS[@]}"; do
        task="${entry%%:*}"; port="${entry##*:}"; venv="$(venv_of "$task")"; pidf="results/worker-$task.pid"
        if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
            echo "$task already running (pid $(cat "$pidf"))"; continue
        fi
        [ -d "$venv" ] || die "$venv missing: run setup first"
        echo "+ $venv: python -m arcdemo.worker --config $WORKER_CONFIG --task $task --port $port"
        PYTHONPATH="$([ "$task" = i23d ] && echo "$SF3D_DIR")" \
            nohup "$(vpy "$venv")" -m arcdemo.worker --config "$WORKER_CONFIG" --task "$task" --port "$port" \
            > "results/stdout-$task.log" 2>&1 &
        echo $! > "$pidf"
    done
    for entry in "${PORTS[@]}"; do                   # weights preload in the background; /health answers first
        task="${entry%%:*}"; port="${entry##*:}"
        for _ in $(seq 60); do health "$port" >/dev/null && break; sleep 1; done
        health "$port" >/dev/null || die "$task worker did not answer on :$port — see results/stdout-$task.log"
    done
    status
}

stop() {
    local entry task pidf
    for entry in "${PORTS[@]}"; do
        task="${entry%%:*}"; pidf="results/worker-$task.pid"
        if [ -f "$pidf" ]; then run kill "$(cat "$pidf")" || true; rm -f "$pidf"; else echo "$task: not running"; fi
    done
}

status() {
    load_secrets
    local entry
    for entry in "${PORTS[@]}"; do
        echo "${entry%%:*} :${entry##*:}  $(health "${entry##*:}" || echo DOWN)"
    done
    echo "logs: results/worker-<task>.log (worker), results/stdout-<task>.log (crashes before logging starts)"
}

case "${1:-}" in
    setup|start|stop|status|conductor) "$1" ;;
    *) sed -n '2,17p' "$0"; exit 2 ;;
esac
