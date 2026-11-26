#!/usr/bin/env bash
# The single-model show on the workstation alone: GPU workers :8101-8105, CPU workers :8201-8205 and
# the conductor (config/demo-workstation.yaml) on this box. Run after setup_gpu_machine.sh. Safe to re-run:
# it stops what it started before and starts everything again.
#   scripts/gpu/start_workstation_show.sh            GPU + CPU workers + conductor
#   scripts/gpu/start_workstation_show.sh --no-cpu   GPU workers + conductor only
#   scripts/gpu/start_workstation_show.sh --stop     stop all of it
# Fixes built in:
#   - .venv-music with a torch that cannot run on this GPU (cu126 on sm_120) is reinstalled from cu128
#   - the i23d workers get PYTHONPATH=third_party/stable-fast-3d (SF3D is a clone, not a pip package)
#   - every worker gets the token, offline mode and its own venv; CPU workers get CUDA_VISIBLE_DEVICES=""
# Logs: results/stdout-<task>.log (GPU), results-cpu/stdout-<task>.log (CPU), results/stdout-conductor.log
set -u
cd "$(dirname "$0")/../.." || exit 1            # demo/
DEMO="$PWD"
SECRETS="${SECRETS:-$DEMO/config.txt}"
# shellcheck source=../../config/versions.env
. "$DEMO/config/versions.env"        # pinned versions and ports (one file for every machine script)
CPU=1 STOP=0
for a in "$@"; do
    case "$a" in
        --no-cpu) CPU=0 ;;
        --stop) STOP=1 ;;
        *) echo "unknown option: $a"; exit 2 ;;
    esac
done

TASKS="t2i i23d music llm layout"
declare -A VENV=([t2i]=.venv-t2i [i23d]=.venv-i23d [music]=.venv-music [llm]=.venv-llm [layout]=.venv-llm)
ports_of() {   # "t2i:8101,i23d:8102,..." -> declare -A NAME=([t2i]=8101 ...)
    local -n _out=$1; local kv
    for kv in ${2//,/ }; do _out[${kv%%:*}]=${kv##*:}; done
}
declare -A GPU_PORT=(); ports_of GPU_PORT "$WORKER_PORTS"
declare -A CPU_PORT=(); ports_of CPU_PORT "$CPU_WORKER_PORTS"
CU=$TORCH_INDEX_CUDA TORCH=$TORCH_VERSION

# ---- secrets and offline mode -------------------------------------------------------------
[ -f "$SECRETS" ] || { echo "secrets file not found: $SECRETS"; exit 1; }
ARC_DEMO_TOKEN="$(grep '^ARC_DEMO_TOKEN=' "$SECRETS" | head -n1 | cut -d= -f2- | tr -d '\r')"
[ -n "$ARC_DEMO_TOKEN" ] || { echo "ARC_DEMO_TOKEN missing in $SECRETS"; exit 1; }
export ARC_DEMO_TOKEN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN SF3D_USE_CPU

health() { curl -sf -m 3 -H "X-Demo-Token: $ARC_DEMO_TOKEN" "http://127.0.0.1:$1/health"; }

stop_port() {   # port: unload first (stops llama-server cleanly), then end the worker
    curl -s -m 5 -X POST -H "X-Demo-Token: $ARC_DEMO_TOKEN" -H "Content-Type: application/json" \
        -d '{}' "http://127.0.0.1:$1/unload" >/dev/null 2>&1
    pkill -f -- "arcdemo.worker .*--port $1( |$)" 2>/dev/null
}

stop_all() {
    pkill -f -- "arcdemo.conductor" 2>/dev/null
    for t in $TASKS; do stop_port "${GPU_PORT[$t]}"; stop_port "${CPU_PORT[$t]}"; done
    sleep 3
    pkill -f -- "third_party/llama.cpp/bin/llama-server" 2>/dev/null   # any orphan still holding memory
    sleep 1
}

stop_all
if [ $STOP = 1 ]; then echo "stopped workers and conductor"; exit 0; fi

# ---- fix 3: music torch must run a kernel on this GPU ---------------------------------------
probe="import torch; torch.ones(1, device='cuda').add_(1).item(); print(torch.__version__)"
if ! .venv-music/bin/python -c "$probe" >/dev/null 2>&1; then
    echo "music: torch cannot run on this GPU ($(.venv-music/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null)); reinstalling from cu128"
    .venv-music/bin/pip install --force-reinstall --no-deps "torch==$TORCH" "torchaudio==$TORCH" --index-url $CU \
        || { echo "music: reinstall failed (needs internet)"; exit 1; }
    .venv-music/bin/python -c "$probe" || { echo "music: torch still cannot run on the GPU"; exit 1; }
fi
echo "music torch: $(.venv-music/bin/python -c "$probe" 2>/dev/null)"

# ---- workers ------------------------------------------------------------------------------
start_worker() {   # task port config results_dir cpu(0|1)
    local task=$1 port=$2 cfg=$3 res=$4 cpu=$5
    mkdir -p "$res"
    (
        [ "$cpu" = 1 ] && export CUDA_VISIBLE_DEVICES=""                 # fix 1: CPU workers never touch the GPU
        [ "$task" = i23d ] && export PYTHONPATH="$DEMO/third_party/stable-fast-3d"   # fix 2
        nohup "${VENV[$task]}/bin/python" -m arcdemo.worker --config "$cfg" --task "$task" --port "$port" \
            > "$res/stdout-$task.log" 2>&1 &
        echo $! > "$res/worker-$task.pid"
    )
}

for t in $TASKS; do start_worker "$t" "${GPU_PORT[$t]}" config/worker-workstation.yaml results 0; done
if [ $CPU = 1 ]; then
    for t in $TASKS; do start_worker "$t" "${CPU_PORT[$t]}" config/worker-workstation-cpu.yaml results-cpu 1; done
fi

PORTS="${GPU_PORT[*]}"; [ $CPU = 1 ] && PORTS="$PORTS ${CPU_PORT[*]}"
echo "waiting for workers (up to 3 min)..."
for _ in $(seq 180); do
    down=0; for p in $PORTS; do health "$p" >/dev/null || down=1; done
    [ $down = 0 ] && break; sleep 1
done
bad=0
for p in $PORTS; do
    if h="$(health "$p")"; then echo "  :$p  $h"
    else bad=1; echo "  :$p  NOT ANSWERING"; fi
done
[ $bad = 1 ] && echo "a worker is down: see results/stdout-<task>.log or results-cpu/stdout-<task>.log"

# ---- conductor ----------------------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
    echo "creating the conductor venv (.venv)"
    "${PY:-python3.11}" -m venv .venv && .venv/bin/pip install -r requirements/conductor.txt || exit 1
fi
if [ ! -f weights/u2net/u2net.onnx ]; then   # uploaded pictures (image-to-3D) need it; the rest of the show does not
    .venv/bin/python -m scripts.common.download_weights --cutout         || echo "WARNING: no cut-out model: uploads work only as PNGs with a transparent background"
fi
CFG=config/demo-workstation.yaml
if [ $CPU = 0 ]; then   # GPU only: drop the CPU machine so the page does not show it as down
    CFG=config/demo-workstation-gpu.yaml
    printf '%s\n' "# Generated by start_workstation_show.sh --no-cpu: demo-workstation.yaml without the CPU machine." \
        "extends: demo-workstation.yaml" > "$CFG"
    .venv/bin/python - "$CFG" <<'EOF'
import sys, yaml
base = yaml.safe_load(open("config/demo-workstation.yaml", encoding="utf-8"))
gpu = [m for m in base["machines"] if m["id"] != "cpu"]
with open(sys.argv[1], "a", encoding="utf-8") as f:
    yaml.safe_dump({"machines": gpu}, f, sort_keys=False, allow_unicode=True)
EOF
fi
nohup .venv/bin/python -m arcdemo.conductor --config "$CFG" --host 0.0.0.0 --port $CONDUCTOR_PORT \
    > results/stdout-conductor.log 2>&1 &
echo $! > results/conductor.pid
for _ in $(seq 30); do curl -sf -m 3 -o /dev/null http://127.0.0.1:$CONDUCTOR_PORT/operator && break; sleep 1; done
if curl -sf -m 3 -o /dev/null http://127.0.0.1:$CONDUCTOR_PORT/operator; then
    echo "conductor ($CFG): http://localhost:$CONDUCTOR_PORT/operator   stage: http://localhost:$CONDUCTOR_PORT/stage"
else
    echo "conductor did not start; last lines of results/stdout-conductor.log:"; tail -n 15 results/stdout-conductor.log; exit 1
fi
