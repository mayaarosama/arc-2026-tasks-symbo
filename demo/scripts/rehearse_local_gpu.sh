#!/usr/bin/env bash
# Laptop GPU rehearsal (demo-local.yaml, machine laptop-gpu): normal replay records, 3 runs per input,
# median kept, real time limits. One worker at a time (4 GB VRAM). Waits for mains power first: on
# battery the GTX 1050 Ti is held at ~300 MHz and the timings would be meaningless.
#   bash scripts/rehearse_local_gpu.sh
set -u
cd "$(dirname "$0")/.."
CFG=config/demo-local.yaml
WCFG=config/worker-local-gpu.yaml
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_TOKEN SF3D_USE_CPU
PY=.venv/Scripts/python.exe

health_field() {   # port -> prints "up loaded1,loaded2 loading" or "down"
  curl -s -m 3 "http://127.0.0.1:$1/health" | $PY -c "
import json,sys
try: h=json.load(sys.stdin); print('up', ','.join(h.get('loaded') or []) or '-', h.get('loading') or '-')
except Exception: print('down')"
}

wait_up() { for i in $(seq 1 300); do [ "$(health_field $1 | cut -d' ' -f1)" = up ] && return 0; sleep 2; done; return 1; }

wait_model() {   # port model : wait until model is in the loaded list (not just loading)
  for i in $(seq 1 900); do
    read st loaded loading <<<"$(health_field $1)"
    case ",$loaded," in *",$2,"*) echo "[$(date +%T)] :$1 $2 loaded"; return 0;; esac
    if [ "$st" = up ] && [ "$loading" = "-" ] && [ $i -gt 5 ]; then
      echo "[$(date +%T)] :$1 $2 did not load: $(curl -s http://127.0.0.1:$1/env | head -c 300)"; return 1
    fi
    sleep 2
  done; return 1
}

stop_port() {   # kill the worker on this port: the venv launcher and the interpreter it starts
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'arcdemo.worker.*--port $1' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
  sleep 5
  [ "$(health_field $1)" = down ] && echo "[$(date +%T)] :$1 worker stopped" || echo "[$(date +%T)] :$1 STILL UP"
}

run_task() {   # venv task port model segments preload(yes|no) [pythonpath]
  local venv=$1 task=$2 port=$3 model=$4 segs=$5 preload=$6 pp=${7:-}
  if [ "$(health_field $port | cut -d' ' -f1)" = up ]; then
    echo "=================== [$(date +%T)] GPU $task: reusing the running worker"
  else
    echo "=================== [$(date +%T)] GPU $task: starting worker"
    PYTHONPATH="$pp" "$venv/Scripts/python.exe" -m arcdemo.worker --config $WCFG --task $task \
        --host 127.0.0.1 --port $port > "results-local/gpu/stdout-worker-$task.log" 2>&1 &
  fi
  if ! wait_up $port; then echo "[$(date +%T)] GPU $task worker never came up"; return; fi
  if [ "$preload" = yes ]; then
    curl -s -X POST -H "Content-Type: application/json" -d "{\"model\":\"$model\"}" "http://127.0.0.1:$port/load" >/dev/null
    wait_model $port $model || { stop_port $port; return; }
  fi
  until curl -s -m 3 "http://127.0.0.1:$port/health" | grep -q '"busy":false'; do sleep 2; done
  echo "[$(date +%T)] GPU $task: rehearse $segs"
  $PY scripts/rehearse.py --config $CFG --segments $segs --machines laptop-gpu --repeat 3 --max-wait 1800
  stop_port $port
}

echo "[$(date +%T)] waiting for mains power"
until [ "$(powershell -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -d '\r ')" = "2" ]; do sleep 10; done
echo "=================== [$(date +%T)] on mains power: starting GPU rehearsal"
sleep 20   # let the GPU clocks come back up

# Watchdog: if the laptop leaves mains power mid-run, stop the rehearsal before it saves anything
# (rehearse.py writes the replay records only at the very end). Throttled runs are not results.
(
  while true; do
    if [ "$(powershell -NoProfile -Command '(Get-CimInstance Win32_Battery).BatteryStatus' | tr -d '\r ')" != "2" ]; then
      echo "[$(date +%T)] POWER LOST: laptop is on battery — stopping the rehearsal, nothing from this run is saved"
      powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'rehearse.py' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
      kill $$ 2>/dev/null
      exit 0
    fi
    sleep 5
  done
) &
WATCHDOG=$!
trap 'kill $WATCHDOG 2>/dev/null' EXIT
mkdir -p results-local/gpu
run_task .venv-local     t2i   9201 sd15            local-t,local-q yes
run_task .venv-local     music 9221 sa3-small-music local-music     yes
run_task .venv-i23d-gpu  i23d  9211 sf3d            local-3d        no  "third_party/stable-fast-3d"
echo "=================== [$(date +%T)] ALL DONE"
find replay-local -path "*laptop-gpu*" -name record.json
