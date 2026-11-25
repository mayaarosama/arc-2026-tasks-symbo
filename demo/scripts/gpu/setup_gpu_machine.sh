#!/usr/bin/env bash
# GPU machine (workstation or DGX), Linux: run every setup step of the runbook, then start the workers.
#
#   scripts/gpu/setup_gpu_machine.sh --machine workstation
#   scripts/gpu/setup_gpu_machine.sh --machine dgx --show-only --no-start
#
# Steps: checks -> git pull -> venvs -> GPU check -> llama.cpp -> capacity check + downloads ->
# offline check -> start workers (-> conductor, with --conductor). Every step is safe to re-run:
# what is already there is reused, not rebuilt or downloaded again.
#
# It stops early ONLY for something you must fix by hand (system packages or the CUDA toolkit
# that need sudo, the secrets file, a git pull that conflicts). Any other failure is recorded, the
# steps that depend on it are skipped, and the run goes on. Either way it writes a report:
#   results/setup-<time>/report.md   (also copied to results/setup-report-latest.md)
#
# Options:
#   --machine workstation|dgx   required; selects config/worker-<machine>.yaml
#   --show-only                 only the show models (config/demo.yaml), not the full matrix
#   --no-start                  set up and check, but do not start the workers
#   --conductor                 also set up and start the conductor here (config/demo.yaml, CONDUCTOR_PORT),
#                               with the cut-out model for uploaded pictures (176 MB, runs on the CPU)
#   --no-pull                   skip git pull
#   --reserve-gb N              disk to keep free after downloads (default 10)
# Env: PY (default python3.11, else python3.10), SECRETS (default demo/config.txt, gitignored).
# Secrets are never printed. Downloads need network; the offline check and workers run offline.
set -uo pipefail

DEMO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO="$(cd "$DEMO/.." && pwd)"
SECRETS="${SECRETS:-$DEMO/config.txt}"
# shellcheck source=../../config/versions.env
. "$DEMO/config/versions.env"        # pinned versions, sources and ports (one file for every machine script)
CU=$TORCH_INDEX_CUDA
TORCH=$TORCH_VERSION
SF3D_DIR=third_party/stable-fast-3d
SA3_DIR=third_party/stable-audio-3
LLAMA_SRC=third_party/llama.cpp-src
LLAMA_BIN=third_party/llama.cpp/bin/llama-server
TASKS=(t2i i23d music llm layout)
ports_of() {   # "t2i:8101,i23d:8102,..." -> declare -A NAME=([t2i]=8101 ...)
    local -n _out=$1; local kv
    for kv in ${2//,/ }; do _out[${kv%%:*}]=${kv##*:}; done
}
declare -A PORT=(); ports_of PORT "$WORKER_PORTS"
declare -A VENV=([t2i]=.venv-t2i [i23d]=.venv-i23d [music]=.venv-music [llm]=.venv-llm [layout]=.venv-llm)
declare -A BROKEN=()

MACHINE="" SHOW_ONLY=0 START=1 CONDUCTOR=0 PULL=1 RESERVE=10 RUN=""
ORIG_ARGS=("$@")
while [ $# -gt 0 ]; do
    case "$1" in
        --machine) MACHINE="${2:-}"; shift ;;
        --show-only) SHOW_ONLY=1 ;;
        --no-start) START=0 ;;
        --conductor) CONDUCTOR=1 ;;
        --no-pull) PULL=0 ;;
        --reserve-gb) RESERVE="${2:-10}"; shift ;;
        --run-dir) RUN="${2:-}"; shift ;;        # internal: continue a run after git pull re-exec
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
case "$MACHINE" in workstation|dgx) ;; *) echo "give --machine workstation or --machine dgx" >&2; exit 2 ;; esac
WORKER_CONFIG="config/worker-$MACHINE.yaml"

cd "$DEMO" || exit 2
# A shell that started workers by hand (README) still has offline mode on, and the downloads need
# the network. The offline check and the workers turn it on again for themselves.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
if [ -z "$RUN" ]; then RUN="results/setup-$(date +%Y%m%d-%H%M%S)"; fi
mkdir -p "$RUN"
STEPS="$RUN/steps.tsv" REPORT="$RUN/report.md"
touch "$STEPS"
exec 3>&1 4>&2                        # 3/4 = the terminal, while a step's output goes to its log

say()    { echo "$*" >&3; }
record() { printf '%s\t%s\t%s\t%s\n' "$1" "${2:--}" "$3" "${4:--}" >> "$STEPS"; }   # step, task, result, detail
# (empty fields are written as "-": bash's read would merge two tabs in a row and shift the columns)
broken() { BROKEN[$1]="$2"; [ "$1" = llm ] && BROKEN[layout]="$2"; return 0; }
vpy()    { echo "$DEMO/$1/bin/python"; }
CUR=""

run_step() {   # name, task ("" = none), function [args]: output goes to the step's own log
    local name="$1" task="$2"; shift 2
    local log="$RUN/$(echo "$name" | tr -c 'A-Za-z0-9.-' '_').log" rc
    CUR="$name"
    say "== $name"
    if "$@" >>"$log" 2>&1; then
        record "$name" "$task" ok "log: $(basename "$log")"; say "   ok"; return 0
    else
        rc=$?
        record "$name" "$task" FAILED "exit $rc: $(grep -v '^\s*$' "$log" | tail -n1 | cut -c1-220 | tr '\t|' '  ') (log: $(basename "$log"))"
        say "   FAILED, see $log"; return 1
    fi
}

skip() {   # name, task, why
    record "$1" "$2" skipped "$3"; say "== $1"; say "   skipped: $3"
}

need_input() {   # stop: the first line is the reason, the rest is what to do
    record "${CUR:-start}" "" "NEEDS YOUR INPUT" "$*"
    say ""; say "STOPPED at \"${CUR:-start}\": this needs your input."
    local l; for l in "$@"; do say "  $l"; done
    write_report "stopped: needs your input"
    say ""; say "report: $DEMO/$REPORT"; say "Fix the above, then run the same command again: finished steps are skipped quickly."
    exit 2
}

# ------------------------------------------------------------------------------------------------
# Steps

preflight() {
    [ "$(uname -s)" = Linux ] || need_input "This script is for the Linux GPU machines (workstation, dgx)."
    command -v nvidia-smi >/dev/null || need_input "nvidia-smi not found: the NVIDIA driver is not installed."
    local q
    q="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits)" \
        || need_input "nvidia-smi failed: the driver is installed but not working (reboot after installing it?)."
    GPU_NAME="$(echo "$q" | head -n1 | cut -d, -f1 | sed 's/^ *//')"
    GPU_COUNT="$(echo "$q" | wc -l)"
    VRAM_GB="$(echo "$q" | awk -F, '{m=$2/1024; if (min=="" || m<min) min=m} END {printf "%.1f", min}')"
    CAP="$(echo "$q" | head -n1 | cut -d, -f3 | tr -d ' ')"          # e.g. 12.0 (5060 Ti), 9.0 (H200)
    CUDA_ARCH="${CAP/./}"
    echo "GPU: $GPU_COUNT x $GPU_NAME, $VRAM_GB GB each, compute capability $CAP"
    local missing=() t
    for t in git curl g++ make; do command -v "$t" >/dev/null || missing+=("$t"); done
    [ ${#missing[@]} -eq 0 ] || need_input "Missing system tools: ${missing[*]}." \
        "Install them: sudo apt update && sudo apt install -y build-essential git curl"
    PY="${PY:-}"
    if [ -z "$PY" ]; then
        if command -v python3.11 >/dev/null; then PY=python3.11; elif command -v python3.10 >/dev/null; then PY=python3.10; fi
    fi
    [ -n "$PY" ] && command -v "$PY" >/dev/null || need_input "Python 3.11 (or 3.10) not found." \
        "Ubuntu 24.04: sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt install -y python3.11 python3.11-venv python3.11-dev" \
        "Ubuntu 22.04: sudo apt install -y python3.10-venv python3.10-dev, then run with PY=python3.10"
    "$PY" -c 'import sys; assert sys.version_info[:2] in ((3, 10), (3, 11))' \
        || need_input "$PY is not Python 3.10 or 3.11 (diffusers and SF3D need one of those). Set PY=python3.11."
    "$PY" -c 'import ensurepip, venv' 2>/dev/null || need_input "$PY cannot create venvs." "Install: sudo apt install -y ${PY}-venv"
    "$PY" -c 'import sysconfig, os, sys; sys.exit(not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' \
        || need_input "Python headers for $PY are missing (needed to build SF3D's extensions)." "Install: sudo apt install -y ${PY}-dev"
    echo "python: $PY ($("$PY" --version 2>&1))"
    DISK_FREE="$(df -BG --output=avail "$DEMO" | tail -n1 | tr -dc 0-9)"
    echo "disk free on $DEMO: ${DISK_FREE} GB"
}

load_secrets() {
    [ -f "$SECRETS" ] || need_input "Secrets file not found: $SECRETS" \
        "Create it next to demo/ (section 2 of the Complete guide):" \
        "  ARC_DEMO_TOKEN=<shared secret>   and   HF_TOKEN=hf_...   then: chmod 600 $SECRETS"
    local line key
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        line="$(echo "$line" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
        if [ -z "$line" ] || [ "${line:0:1}" = "#" ]; then continue; fi
        case "$line" in
            hf_*) export HF_TOKEN="$line" ;;
            *=*)  key="${line%%=*}"
                  case "$key" in HF_TOKEN|ARC_DEMO_TOKEN) export "$key=${line#*=}" ;; esac ;;
        esac
    done < "$SECRETS"
    [ -n "${HF_TOKEN:-}" ] || need_input "No HF_TOKEN in $SECRETS: the gated models cannot be downloaded." \
        "Add a line HF_TOKEN=hf_... (a read token from huggingface.co/settings/tokens)."
    echo "secrets: HF_TOKEN set, ARC_DEMO_TOKEN $([ -n "${ARC_DEMO_TOKEN:-}" ] && echo set || echo 'EMPTY (workers accept anyone)')"
    [ -n "${ARC_DEMO_TOKEN:-}" ] || record "secrets" "" warning "ARC_DEMO_TOKEN is empty: workers will accept requests from anyone on the network"
}

update_repo() {
    local before after
    before="$(git -C "$REPO" rev-parse HEAD)"
    if ! git -C "$REPO" pull --ff-only; then
        exec 1>&3 2>&4
        git -C "$REPO" status --short | head -n 20 >&3
        need_input "git pull failed (local changes that would be overwritten, or a diverged branch; see above)." \
            "Keep your change and update: cd $REPO && git stash && git pull && git stash apply" \
            "Or skip updating: run again with --no-pull"
    fi
    after="$(git -C "$REPO" rev-parse HEAD)"
    if [ "$before" != "$after" ]; then
        echo "updated $before -> $after; restarting with the new version of this script"
        record "git pull" "" ok "updated to $(git -C "$REPO" rev-parse --short HEAD)"
        exec 1>&3 2>&4
        exec bash "$DEMO/scripts/gpu/setup_gpu_machine.sh" "${ORIG_ARGS[@]}" --no-pull --run-dir "$RUN"
    fi
    echo "already up to date at $(git -C "$REPO" rev-parse --short HEAD)"
}


# need_nvcc PURPOSE [torch|any]: find a CUDA compiler for one build. It only looks; it installs,
# changes and deletes nothing.
#   torch  12.8 or 12.9: SF3D's extensions must match the cu128 torch wheels (CUDA 13 cannot)
#   any    12.8, 12.9 or 13.x: llama.cpp does not link against torch
# Sets NVCC and CUDA_DIR, plus HOST_CC/HOST_CXX when nvcc needs an older GCC than the default
# (CUDA 12.x accepts GCC 14 at most; Ubuntu 26.04 ships a newer one). Empty = default compiler.
# Nothing is exported: each build passes these to its own subshell or cmake.
need_nvcc() {
    local purpose="$1" mode="${2:-torch}" c v found="" cands=()
    for c in "${CUDA_HOME:-}" /usr/local/cuda-12.9 /usr/local/cuda-12.8; do [ -n "$c" ] && cands+=("$c/bin/nvcc"); done
    if [ "$mode" = any ]; then
        cands+=(/usr/local/cuda/bin/nvcc)
        for c in $(ls -d /usr/local/cuda-13* 2>/dev/null | sort -rV); do cands+=("$c/bin/nvcc"); done
    fi
    command -v nvcc >/dev/null && cands+=("$(command -v nvcc)")
    NVCC="" CUDA_DIR="" HOST_CC="" HOST_CXX=""
    for c in "${cands[@]}"; do
        [ -x "$c" ] || continue
        v="$("$c" --version 2>/dev/null | grep -o 'release [0-9]*\.[0-9]*' | cut -d' ' -f2)"
        case "$mode:$v" in
            *:12.8|*:12.9|any:13.*) NVCC="$c"; break ;;
            *:?*) case " $found " in *" $v "*) ;; *) found="${found:+$found }$v" ;; esac ;;
        esac
    done
    if [ -z "$NVCC" ]; then
        local want="12.8 or 12.9"; [ "$mode" = any ] && want="12.8, 12.9 or 13.x"
        local here="not installed"; [ -n "$found" ] && here="version $found, not $want"
        if apt-cache policy cuda-toolkit-12-8 2>/dev/null | grep -q 'Candidate: [0-9]'; then
            need_input "The CUDA toolkit $want (nvcc) is needed to build $purpose; here it is $here." \
                "Install the toolkit only, not the driver (section 1 of the Complete guide):" \
                "  sudo apt install -y cuda-toolkit-12-8     (from NVIDIA's CUDA apt repository)" \
                "Then run this script again."
        fi
        need_input "The CUDA toolkit $want (nvcc) is needed to build $purpose; here it is $here." \
            "NVIDIA's apt repository for this Ubuntu has no cuda-toolkit-12-8, so install 12.8 from NVIDIA's" \
            "runfile, toolkit only. It goes into /usr/local/cuda-12.8 and leaves the driver, apt and any" \
            "CUDA 13 alone (section 1 of the Complete guide):" \
            "  ls -l /usr/local/cuda          # note where it points now, if it exists" \
            "  wget $CUDA_RUNFILE_URL" \
            "  sudo sh $CUDA_RUNFILE --silent --toolkit --toolkitpath=/usr/local/cuda-12.8 --no-man-page --override" \
            "  ls -l /usr/local/cuda          # if it changed and you want the old default: sudo ln -sfn <old target> /usr/local/cuda" \
            "  sudo apt install -y gcc-14 g++-14   # only if gcc --version is newer than 14; the default gcc stays" \
            "The downloaded .run file can be deleted afterwards. Then run this script again."
    fi
    CUDA_DIR="$(dirname "$(dirname "$NVCC")")"
    echo "nvcc $v at $CUDA_DIR (for $purpose)"
    NVCC_FIX=""
    case "$v" in 12.*) ;; *) return 0 ;; esac
    local g; g="$(gcc -dumpfullversion -dumpversion 2>/dev/null | cut -d. -f1)"
    if ! { [ -n "$g" ] && [ "$g" -le 14 ]; }; then
        for g in 14 13 12; do
            if command -v "g++-$g" >/dev/null && command -v "gcc-$g" >/dev/null; then
                HOST_CC="$(command -v "gcc-$g")" HOST_CXX="$(command -v "g++-$g")"
                echo "default GCC is newer than CUDA $v accepts; using GCC $g as its host compiler"
                break
            fi
        done
        [ -n "$HOST_CXX" ] || need_input "CUDA $v accepts GCC 14 at most; the default gcc here is $(gcc -dumpfullversion -dumpversion 2>/dev/null)." \
            "Install GCC 14 next to it (the default gcc stays the default):" \
            "  sudo apt install -y gcc-14 g++-14" \
            "Then run this script again: it uses gcc-14 for the CUDA builds only."
    fi
    cuda_glibc_fix "$v"
}

# cuda_glibc_fix VERSION: CUDA 12.8 declares sinpi, cospi and rsqrt (and their f versions) without
# noexcept; glibc 2.41 and later (Ubuntu 25.04+) declares them noexcept, and nvcc stops on every .cu
# file ("exception specification is incompatible with that of previous function"). The toolkit is
# not touched: a copy of its include dir with those six lines fixed goes into third_party/, and the
# builds put it first on nvcc's search path through NVCC_PREPEND_FLAGS (set in NVCC_FIX).
# A test compile decides whether the fix is needed and checks that it works.
cuda_glibc_fix() {
    local v="$1" t="$RUN/nvcc-probe" ccbin=() inc="$DEMO/third_party/cuda-$1-include-glibc" n
    mkdir -p "$t"
    printf '#include <cmath>\n__global__ void k(float *x) { x[0] = rsqrtf(x[0]) + sinpif(x[0]); }\n' > "$t/probe.cu"
    [ -n "$HOST_CXX" ] && ccbin=(-ccbin "$HOST_CXX")
    if "$NVCC" "${ccbin[@]}" -c "$t/probe.cu" -o "$t/probe.o" > "$t/plain.log" 2>&1; then return 0; fi
    if ! grep -q 'exception specification is incompatible' "$t/plain.log"; then
        echo "nvcc $v cannot compile a minimal .cu file:"; cat "$t/plain.log"; return 1
    fi
    echo "CUDA $v clashes with this glibc ($(ldd --version 2>/dev/null | head -n1 | grep -o '[0-9.]*$')): using a fixed copy of its headers"
    if [ ! -f "$inc/.fixed" ]; then
        rm -rf "$inc" && mkdir -p "$inc" && cp -rL "$CUDA_DIR/include/." "$inc/" || return 1
        sed -i -E 's/^(extern .*[^A-Za-z0-9_](sinpi|cospi|rsqrt)f?\((double|float) x\));/\1 noexcept (true);/' \
            "$inc/crt/math_functions.h"
        n="$(grep -cE '^extern .*[^A-Za-z0-9_](sinpi|cospi|rsqrt)f?\((double|float) x\) noexcept \(true\);' "$inc/crt/math_functions.h")"
        [ "$n" = 6 ] || { echo "expected to fix 6 declarations in crt/math_functions.h, fixed $n"; return 1; }
        touch "$inc/.fixed"
    fi
    if NVCC_PREPEND_FLAGS="-I$inc" "$NVCC" "${ccbin[@]}" -c "$t/probe.cu" -o "$t/probe.o" > "$t/fixed.log" 2>&1; then
        NVCC_FIX="-I$inc"; echo "fixed headers work: $inc"; return 0
    fi
    cat "$t/fixed.log"
    need_input "CUDA $v does not compile with this system's glibc, and the automatic header fix did not work (see the step log)." \
        "Either install CUDA 12.9, which supports this glibc, next to 12.8 (the script prefers 12.9):" \
        "  NVIDIA's cuda_12.9.1 runfile, as for 12.8: sudo sh <runfile> --silent --toolkit --toolkitpath=/usr/local/cuda-12.9 --no-man-page --override" \
        "or add 'noexcept (true)' before the ';' of the sinpi, sinpif, cospi, cospif, rsqrt and rsqrtf" \
        "declarations in $CUDA_DIR/include/crt/math_functions.h (sudo). Then run this script again."
}

mkvenv() {
    if [ -x "$1/bin/python" ]; then echo "$1 already exists, reusing it"; return 0; fi
    "$PY" -m venv "$1" && "$1/bin/pip" install -q --upgrade pip
}
clone() {   # url dir commit
    [ -d "$2/.git" ] || git clone "$1" "$2" || return 1
    git -C "$2" checkout -q "$3"
}

venv_t2i() {
    mkvenv .venv-t2i &&
    .venv-t2i/bin/pip install torch==$TORCH --index-url $CU &&
    .venv-t2i/bin/pip install -r requirements/t2i-gpu.txt
}
venv_i23d() {
    mkvenv .venv-i23d &&
    .venv-i23d/bin/pip install torch==$TORCH torchvision==$TORCHVISION_VERSION --index-url $CU &&
    clone $SF3D_REPO $SF3D_DIR $SF3D_COMMIT &&
    .venv-i23d/bin/pip install -U setuptools==69.5.1 wheel &&
    grep -v '^\./' $SF3D_DIR/requirements.txt > "$RUN/sf3d-req.txt" &&
    .venv-i23d/bin/pip install -r "$RUN/sf3d-req.txt" || return 1
    if .venv-i23d/bin/python -c "import texture_baker, uv_unwrapper" 2>/dev/null; then
        echo "SF3D extensions already built"
    else
        need_nvcc "SF3D's texture_baker and uv_unwrapper" torch || return 1
        (export CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH" TORCH_CUDA_ARCH_LIST="$CAP" USE_CUDA=1
         # torch passes $CC to nvcc as -ccbin; NVCC_CCBIN covers any nvcc call without it
         if [ -n "$HOST_CXX" ]; then export CC="$HOST_CC" CXX="$HOST_CXX" NVCC_CCBIN="$HOST_CXX"; fi
         if [ -n "$NVCC_FIX" ]; then export NVCC_PREPEND_FLAGS="$NVCC_FIX"; fi
         cd $SF3D_DIR && "$DEMO/.venv-i23d/bin/pip" install --no-build-isolation ./texture_baker ./uv_unwrapper) || return 1
    fi
    .venv-i23d/bin/pip install -r requirements/worker-base.txt &&
    .venv-i23d/bin/pip install "huggingface_hub>=0.25,<1.0"      # SF3D's transformers 4.42.3 needs hub < 1.0
}
venv_music() {
    mkvenv .venv-music &&
    .venv-music/bin/pip install torch==$TORCH torchaudio==$TORCH --index-url $CU &&
    clone $SA3_REPO $SA3_DIR $SA3_COMMIT &&
    .venv-music/bin/pip install -r requirements/music.txt &&
    .venv-music/bin/pip install --no-deps -e $SA3_DIR
}
venv_llm()       { mkvenv .venv-llm && .venv-llm/bin/pip install -r requirements/llm.txt; }
venv_conductor() { mkvenv .venv && .venv/bin/pip install -r requirements/conductor.txt; }
cutout_model()   { .venv/bin/python -m scripts.common.download_weights --cutout && .venv/bin/python -m scripts.common.verify_offline --cutout; }

gpu_check() {   # task: torch must run a kernel on the GPU; one automatic reinstall from cu128 if it cannot
    local py; py="$(vpy "${VENV[$1]}")"
    # is_available() is True even for a torch built without this card's arch (e.g. cu126 on sm_120),
    # so run a real kernel: that fails with "no kernel image is available" and triggers the reinstall
    local probe="import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'; torch.ones(1, device='cuda').add_(1).item(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
    "$py" -c "$probe" && return 0
    echo "retrying: reinstalling torch from the cu128 index"
    local pk="torch==$TORCH"
    [ "$1" = i23d ] && pk="$pk torchvision==$TORCHVISION_VERSION"
    [ "$1" = music ] && pk="$pk torchaudio==$TORCH"
    # shellcheck disable=SC2086
    "$(dirname "$py")/pip" install --force-reinstall --no-deps $pk --index-url $CU && "$py" -c "$probe"
}

llama_build() {
    if $LLAMA_BIN --version 2>&1 | grep -q "${LLAMA_TAG#b}" && $LLAMA_BIN --list-devices 2>&1 | grep -q CUDA0; then
        echo "llama-server $LLAMA_TAG with CUDA is already installed"; return 0
    fi
    command -v cmake >/dev/null || need_input "cmake not found (needed to build llama.cpp)." "Install: sudo apt install -y cmake"
    need_nvcc "llama.cpp" any || return 1
    local host=(); [ -n "$HOST_CXX" ] && host=(-DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX")
    [ -d $LLAMA_SRC/.git ] || git clone --depth 1 --branch $LLAMA_TAG $LLAMA_REPO $LLAMA_SRC || return 1
    if [ "$(git -C $LLAMA_SRC describe --tags 2>/dev/null)" != $LLAMA_TAG ]; then
        echo "$LLAMA_SRC is not release $LLAMA_TAG: delete it (rm -rf $DEMO/$LLAMA_SRC) and run again"; return 1
    fi
    # a configure that failed earlier leaves a cache that can keep the old compiler result
    [ -f $LLAMA_SRC/build/Makefile ] || [ -f $LLAMA_SRC/build/build.ninja ] || rm -rf $LLAMA_SRC/build
    (export CUDA_HOME="$CUDA_DIR" PATH="$CUDA_DIR/bin:$PATH"
     if [ -n "$NVCC_FIX" ]; then export NVCC_PREPEND_FLAGS="$NVCC_FIX"; fi
     cmake -S $LLAMA_SRC -B $LLAMA_SRC/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_COMPILER="$NVCC" "${host[@]}" \
        -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release &&
     cmake --build $LLAMA_SRC/build --target llama-server -j"$(nproc)") &&
    mkdir -p "$(dirname $LLAMA_BIN)" && cp $LLAMA_SRC/build/bin/llama-server $LLAMA_BIN &&
    $LLAMA_BIN --list-devices 2>&1 | tee /dev/stderr | grep -q CUDA0
}

downloads() {
    local py=.venv-llm
    [ -x "$py/bin/python" ] || py=.venv-t2i
    [ -x "$py/bin/python" ] || { echo "no venv with huggingface_hub to run the download step"; return 1; }
    local cfgs=(--config config/demo.yaml)
    [ $SHOW_ONLY = 1 ] || cfgs+=(--config config/demo-full.yaml)
    "$py/bin/python" -m scripts.gpu.plan_downloads --machine "$MACHINE" --vram-gb "$VRAM_GB" \
        --reserve-gb "$RESERVE" "${cfgs[@]}" --out-dir "$RUN"
}

verify_group() {   # config task model...: offline check of these models, one line per model in verify.tsv
    local cfg="$1" task="$2"; shift 2
    local args=() m py; for m in "$@"; do args+=(--model "$m"); done
    py="$(vpy "${VENV[$task]}")"
    local out="$RUN/verify-$(basename "$cfg" .yaml)-$task.out"
    ( unset HF_TOKEN SF3D_USE_CPU; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      [ "$task" = i23d ] && export PYTHONPATH="$DEMO/$SF3D_DIR"
      "$py" -m scripts.common.verify_offline --config "$WORKER_CONFIG" --demo "$cfg" --task "$task" "${args[@]}" ) > "$out" 2>&1
    local rc=$? line key res
    for m in "$@"; do
        line="$(grep -E "^(PASS|FAIL) +$m " "$out" | tail -n1)"
        if [ -z "$line" ]; then
            res=FAIL; line="did not run: $(grep -v '^\s*$' "$out" | tail -n1 | cut -c1-200)"
        else
            res="$(echo "$line" | awk '{print $1}')"; line="$(echo "$line" | cut -c7- | sed "s/^$m *//")"
        fi
        if [ "$res" = FAIL ] && echo "$line" | grep -q out_of_memory && \
           awk -F'\t' -v c="$cfg" -v t="$task" -v k="$m" '$1==c && $2==t && $3==k && $5=="deliberate_oom"' "$RUN/models.tsv" | grep -q .; then
            res="expected OOM"
        fi
        line="$(echo "$line" | tr '\t|' '  ' | tr -s ' ')"
        printf '%s\t%s\t%s\t%s\t%s\n' "$cfg" "$task" "$m" "$res" "${line:--}" >> "$RUN/verify.tsv"
        say "   $res  $(basename "$cfg") $task $m"
    done
    return $rc
}

verify_all() {
    : > "$RUN/verify.tsv"
    [ -f "$RUN/models.tsv" ] || { echo "no download plan: nothing to check"; return 1; }
    local cfg task keys
    for cfg in config/demo.yaml config/demo-full.yaml; do
        for task in "${TASKS[@]}"; do
            keys="$(awk -F'\t' -v c="$cfg" -v t="$task" '$1==c && $2==t && $4=="ready" {print $3}' "$RUN/models.tsv" | sort -u)"
            [ -n "$keys" ] || continue
            if [ -n "${BROKEN[$task]:-}" ]; then
                for k in $keys; do printf '%s\t%s\t%s\tnot checked\t%s\n' "$cfg" "$task" "$k" "${BROKEN[$task]}" >> "$RUN/verify.tsv"; done
                continue
            fi
            # shellcheck disable=SC2086
            verify_group "$cfg" "$task" $keys
        done
    done
    echo "$(awk -F'\t' '$4=="FAIL"' "$RUN/verify.tsv" | wc -l) model(s) failed; they are counted and listed one by one in the report"
}

health() { curl -sf -m 3 ${ARC_DEMO_TOKEN:+-H "X-Demo-Token: $ARC_DEMO_TOKEN"} "http://127.0.0.1:$1/health"; }

WAS_RUNNING=()
stop_workers() {   # free the GPU for the offline check: it loads every model in its own process
    local t pidf pid stopped=0
    for t in "${TASKS[@]}"; do
        pidf="results/worker-$t.pid"
        [ -f "$pidf" ] && pid="$(cat "$pidf")" && kill -0 "$pid" 2>/dev/null || continue
        # unload first, so llama-server is stopped cleanly instead of left holding VRAM
        curl -s -m 10 -X POST ${ARC_DEMO_TOKEN:+-H "X-Demo-Token: $ARC_DEMO_TOKEN"} -H "Content-Type: application/json" \
            -d '{}' "http://127.0.0.1:${PORT[$t]}/unload" >/dev/null 2>&1
        kill "$pid" 2>/dev/null
        local _; for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
        rm -f "$pidf"; WAS_RUNNING+=("$t"); stopped=$((stopped + 1))
        echo "stopped $t worker (pid $pid)"
    done
    [ $stopped -gt 0 ] || echo "no workers running"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
    return 0
}

start_worker() {   # task
    local task="$1" port="${PORT[$1]}" pidf="results/worker-$1.pid"
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && health "$port" >/dev/null; then
        echo "$task already running (pid $(cat "$pidf"))"; return 0
    fi
    ( unset HF_TOKEN SF3D_USE_CPU; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      [ "$task" = i23d ] && export PYTHONPATH="$DEMO/$SF3D_DIR"
      nohup "$(vpy "${VENV[$task]}")" -m arcdemo.worker --config "$WORKER_CONFIG" --task "$task" --port "$port" \
          > "results/stdout-$task.log" 2>&1 &
      echo $! > "$pidf" )
    local _; for _ in $(seq 120); do health "$port" >/dev/null && break; sleep 1; done
    health "$port" || { echo "no answer on :$port after 120 s; last lines of results/stdout-$task.log:"; tail -n 5 "results/stdout-$task.log"; return 1; }
}

start_conductor() {
    if curl -sf -m 3 -o /dev/null http://127.0.0.1:$CONDUCTOR_PORT/operator; then echo "conductor already running"; return 0; fi
    ( unset HF_TOKEN
      nohup .venv/bin/python -m arcdemo.conductor --config config/demo.yaml --host 0.0.0.0 --port $CONDUCTOR_PORT \
          > results/stdout-conductor.log 2>&1 &
      echo $! > results/conductor.pid )
    local _; for _ in $(seq 30); do curl -sf -m 3 -o /dev/null http://127.0.0.1:$CONDUCTOR_PORT/operator && return 0; sleep 1; done
    tail -n 5 results/stdout-conductor.log; return 1
}

# ------------------------------------------------------------------------------------------------
# Report

write_report() {   # overall result
    local result="$1" f
    {
        echo "# GPU machine setup report: $MACHINE"
        echo
        echo "$(date '+%Y-%m-%d %H:%M') · $(hostname) · result: **$result**"
        echo
        echo "GPU: ${GPU_COUNT:-?} × ${GPU_NAME:-unknown}, ${VRAM_GB:-?} GB each · disk free at start: ${DISK_FREE:-?} GB · repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)"
        echo
        echo "## Needs your action"
        echo
        local any=0
        while IFS=$'\t' read -r s t r d; do
            [ "$r" = "NEEDS YOUR INPUT" ] && { echo "- **$s:** $d"; any=1; }
        done < "$STEPS"
        if [ -f "$RUN/models.tsv" ]; then
            while IFS=$'\t' read -r c t k r o d; do
                case "$r" in
                    needs_licence) echo "- **Licence:** $k ($t, $(basename "$c")): accept the licence on Hugging Face with the token's account, then run again. $d"; any=1 ;;
                    skipped_disk)  echo "- **Disk:** $k ($t, $(basename "$c")) not downloaded: $d"; any=1 ;;
                esac
            done < "$RUN/models.tsv"
        fi
        [ $any = 1 ] || echo "Nothing."
        echo
        echo "## Could not be run"
        echo
        any=0
        while IFS=$'\t' read -r s t r d; do
            [ "$t" = - ] && t=""
            [ "$r" = FAILED ] && { echo "- **$s**${t:+ ($t)}: $d"; any=1; }
            [ "$r" = skipped ] && { echo "- **$s**${t:+ ($t)}: skipped, $d"; any=1; }
        done < "$STEPS"
        if [ -f "$RUN/models.tsv" ]; then
            while IFS=$'\t' read -r c t k r o d; do
                case "$r" in ready|needs_licence|skipped_disk) ;;
                    *) echo "- **Model $k** ($t, $(basename "$c")): $r. $d"; any=1 ;; esac
            done < "$RUN/models.tsv"
        fi
        if [ -f "$RUN/verify.tsv" ]; then
            while IFS=$'\t' read -r c t k r d; do
                case "$r" in PASS|"expected OOM") ;;
                    *) echo "- **Offline check $k** ($t, $(basename "$c")): $r. $d"; any=1 ;; esac
            done < "$RUN/verify.tsv"
        fi
        [ $any = 1 ] || echo "Nothing: every step, model and worker ran."
        echo
        echo "## Steps"
        echo
        echo "| Step | Task | Result | Detail |"
        echo "| --- | --- | --- | --- |"
        while IFS=$'\t' read -r s t r d; do echo "| $s | $t | $r | ${d//|//} |"; done < "$STEPS"
        echo
        echo "## Models (capacity check and download)"
        echo
        if [ -f "$RUN/downloads.md" ]; then cat "$RUN/downloads.md"; else echo "Not reached."; fi
        echo
        echo "## Offline check"
        echo
        if [ -s "$RUN/verify.tsv" ]; then
            echo "| Config | Task | Model | Result | Detail |"
            echo "| --- | --- | --- | --- | --- |"
            while IFS=$'\t' read -r c t k r d; do echo "| $(basename "$c") | $t | $k | $r | $d |"; done < "$RUN/verify.tsv"
        else echo "Not reached."; fi
        echo
        echo "Logs for every step: \`$RUN/\`. Worker logs: \`results/worker-<task>.log\`, \`results/stdout-<task>.log\`."
    } > "$REPORT"
    cp "$REPORT" results/setup-report-latest.md
}

# ------------------------------------------------------------------------------------------------

main() {
    say "GPU machine setup: $MACHINE   (logs and report: $DEMO/$RUN)"
    run_step "checks" "" preflight
    run_step "secrets" "" load_secrets
    if [ $PULL = 1 ]; then run_step "git pull" "" update_repo || true; fi

    run_step "venv t2i" t2i venv_t2i     || broken t2i "venv t2i failed"
    run_step "venv i23d" i23d venv_i23d  || broken i23d "venv i23d failed"
    run_step "venv music" music venv_music || broken music "venv music failed"
    run_step "venv llm" llm venv_llm     || broken llm "venv llm failed"
    if [ $CONDUCTOR = 1 ]; then run_step "venv conductor" "" venv_conductor || CONDUCTOR=0; fi
    if [ $CONDUCTOR = 1 ]; then run_step "cut-out model (uploads)" "" cutout_model || true; fi   # uploads need it, the show does not

    local t
    for t in t2i i23d music; do
        if [ -n "${BROKEN[$t]:-}" ]; then skip "GPU check $t" "$t" "${BROKEN[$t]}"
        else run_step "GPU check $t" "$t" gpu_check "$t" || broken "$t" "torch does not see the GPU"; fi
    done
    if [ -n "${BROKEN[llm]:-}" ]; then skip "llama.cpp" llm "${BROKEN[llm]}"
    else run_step "llama.cpp" llm llama_build || broken llm "llama.cpp build failed"; fi

    run_step "capacity check + downloads" "" downloads
    export -n HF_TOKEN; unset HF_TOKEN                       # offline from here on
    run_step "stop running workers" "" stop_workers
    run_step "offline check" "" verify_all || true

    # --no-start still brings back the workers this run stopped
    local restart=("${TASKS[@]}"); [ $START = 1 ] || restart=("${WAS_RUNNING[@]}")
    if [ ${#restart[@]} -gt 0 ]; then
        for t in "${restart[@]}"; do
            if [ -n "${BROKEN[$t]:-}" ]; then skip "start $t worker :${PORT[$t]}" "$t" "${BROKEN[$t]}"
            else run_step "start $t worker :${PORT[$t]}" "$t" start_worker "$t" || true; fi
        done
        if [ $START = 1 ] && [ $CONDUCTOR = 1 ]; then run_step "start conductor :$CONDUCTOR_PORT" "" start_conductor || true; fi
    fi

    local failed result
    failed="$(awk -F'\t' '$3=="FAILED"' "$STEPS" | wc -l)"
    failed=$(( failed + $( [ -f "$RUN/verify.tsv" ] && awk -F'\t' '$4=="FAIL"' "$RUN/verify.tsv" | wc -l || echo 0) ))
    failed=$(( failed + $( [ -f "$RUN/models.tsv" ] && awk -F'\t' '$4=="failed"' "$RUN/models.tsv" | wc -l || echo 0) ))
    if [ "$failed" -eq 0 ]; then result="completed"; else result="completed with $failed problem(s)"; fi
    write_report "$result"
    say ""; say "$result. Report: $DEMO/$REPORT"
    [ "$failed" -eq 0 ]
}

main; exit $?
