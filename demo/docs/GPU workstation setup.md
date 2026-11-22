# ARC 2026 Demo — GPU Workstation Setup (RTX 5060 Ti)

As of 2026-11-22. Mirrors the "Complete guide" tab of the shared setup doc.

## Overview

This guide takes the workstation from a clean Linux install to five running demo workers for the single-model show (`config/demo.yaml`). The conductor runs on the operator laptop. This machine only runs workers.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti, 16 GB (Blackwell, compute capability 12.0 / sm_120) |
| Driver / CUDA | 595.99.02 / CUDA 13.2 |
| OS | Linux with GNOME (Ubuntu 22.04 or 24.04 assumed) |
| Worker config | `config/worker-workstation.yaml` (`machine: workstation`) |

The five workers and the model each one serves:

| Task | Port | Python env | Model | Download |
| --- | --- | --- | --- | --- |
| 1 · text-to-image (`t2i`) | 8101 | `.venv-t2i` | SD 1.5 (`sd15`) | 5.2 GB |
| 2 · image-to-3D (`i23d`) | 8102 | `.venv-i23d` | Stable Fast 3D (`sf3d`) + DINOv2 + CLIP | 5.6 GB |
| 3 · music (`music`) | 8103 | `.venv-music` | Stable Audio 3.0 Small (`sa3-small-music`) | 3.5 GB |
| 4 · language model (`llm`) | 8104 | `.venv-llm` | Qwen3.5 4B Q4_K_M on llama.cpp | 2.7 GB |
| 8 · scene construction (`layout`) | 8105 | `.venv-llm` | the task 4 model file (`qwen35-4b-layout`) | none extra |

**Blackwell matters.** The RTX 5060 Ti only runs PyTorch builds compiled for CUDA 12.8 or newer (the `cu128` wheels). Older wheels fail with `no kernel image is available for execution on the device`. Every torch install below uses the `cu128` index. The CUDA 13.2 driver runs them.

**Automated run.** Once the repo is cloned and `config.txt` written (section 2), one script runs sections 3 to 7:

```bash
cd $DEMO
bash scripts/gpu/setup_gpu_machine.sh --machine workstation      # or: --machine dgx
```

It checks free disk and GPU memory before each model download, and skips a model that doesn't fit. It stops early only for something you must fix yourself: system packages or the CUDA toolkit (these need sudo), a missing `config.txt`, or a `git pull` that conflicts. Any other failure is recorded, the steps that depend on it are skipped, and the run goes on. The report is `results/setup-report-latest.md`: what needs your action, then every step, model, offline check and worker that could not run. Options: `--show-only` (show models only), `--no-start` (don't start the workers), `--conductor` (also start the conductor here). Running it again is safe.

## 1. Prerequisites

You need about 80 GB of free disk: 17 GB of weights, about 6 GB for each of the three torch envs, and the CUDA toolkit. The full model matrix adds about 85 GB of weights (section 5), so plan on 165 GB if you fetch it too. Do every step below with network access. The show itself runs offline.

**System packages.**

```bash
sudo apt update
sudo apt install -y git git-lfs build-essential cmake curl ffmpeg libsndfile1 libgl1 libglib2.0-0
```

**Python 3.11.** Diffusers and Stable Fast 3D need Python 3.10 or 3.11. Ubuntu 24.04 ships 3.12, so add 3.11 from the deadsnakes PPA. Ubuntu 22.04 can use its own 3.10 instead: set `PY=python3.10` in later steps.

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

**CUDA toolkit 12.8.** The driver (595.99, CUDA 13.2) is already installed: do not reinstall it. You also need the compiler (`nvcc`) for two builds: llama.cpp with CUDA, and SF3D's texture baker. Install the toolkit only, from NVIDIA's apt repo. Pick 12.8 or 12.9 so `nvcc` matches the `cu128` torch wheels. CUDA 13 will not build torch extensions for them.

```bash
# Ubuntu 24.04 (use ubuntu2204 in the URL on 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-toolkit-12-8            # toolkit only, keeps the 595 driver
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc && nvcc --version                 # release 12.8
```

**Ubuntu 26.04 and newer.** NVIDIA's apt repository for these releases has only CUDA 13, so `cuda-toolkit-12-8` is not there. Install 12.8 with NVIDIA's runfile, toolkit only. It copies files into `/usr/local/cuda-12.8` and does not touch the driver, apt or any CUDA 13 already installed. Apt does not track it, so remove it later with `sudo /usr/local/cuda-12.8/bin/cuda-uninstaller`. CUDA 12.8 accepts GCC 14 at most, so install GCC 14 next to the newer default compiler. `setup_gpu_machine.sh` uses it for the CUDA builds only, and plain `gcc` stays the default. The script also accepts CUDA 13 for llama.cpp, which does not link against torch. Only SF3D needs 12.x.

```bash
ls -l /usr/local/cuda                              # note where it points now, if it exists
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
sudo sh cuda_12.8.1_570.124.06_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-12.8 --no-man-page --override
/usr/local/cuda-12.8/bin/nvcc --version            # release 12.8
ls -l /usr/local/cuda                              # if it changed: sudo ln -sfn <old target> /usr/local/cuda
sudo apt install -y gcc-14 g++-14                  # only if gcc --version is newer than 14
```

Do not add `CUDA_HOME=/usr/local/cuda-12.8` to `~/.bashrc` on these releases. The script finds `/usr/local/cuda-12.8` on its own, and your shell keeps CUDA 13 as its default. Once the install works, you can delete the downloaded `.run` file.

**Hugging Face access.** Log in to the account whose token you will use. Accept the licence on each gated page:

- [stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d)
- [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)
- Full model matrix only: [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium), [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium), [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)

Create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Secrets.** Get the shared `ARC_DEMO_TOKEN` value from whoever runs the conductor. It must match on every machine.

## 2. Clone the repository

The repo is public, so no login is needed. Clone it to a path without spaces. Every later command runs from `demo/`.

```bash
cd ~
git clone https://github.com/mayaarosama/arc-2026-tasks-symbo.git arc2026
cd ~/arc2026/demo
export DEMO=~/arc2026/demo PY=python3.11
```

**Already cloned?** Skip the clone. Check for updates, pull them, then set the variables again:

```bash
cd ~/arc2026
git fetch && git status -sb      # "behind N" in the first line means the remote has new commits
git pull --ff-only
cd ~/arc2026/demo
export DEMO=~/arc2026/demo PY=python3.11
```

If the pull stops with "Your local changes to the following files would be overwritten" (for example `demo/config/demo.yaml` after pointing the workstation URLs at `127.0.0.1`), stash your change, pull, then apply it again. Do not run `git stash pop`: it conflicts on the same lines.

```bash
cd ~/arc2026
git stash
git pull --ff-only
sed -i 's#http://workstation.local:#http://127.0.0.1:#' demo/config/demo.yaml   # only if the conductor runs on this machine
```

Write the secrets file **next to** `demo/`, never inside it. It is git-ignored, and scripts never print its values.

```bash
cat > ~/arc2026/config.txt <<'EOF'
ARC_DEMO_TOKEN=<shared secret, same on every machine>
HF_TOKEN=hf_...
EOF
chmod 600 ~/arc2026/config.txt
```

Two git-ignored folders hold everything heavy: `third_party/` (cloned libraries and llama.cpp) and `hf-cache/hub` (all model weights, set by `arcdemo/__init__.py`). Neither comes with the clone.

**dgx variant (from the INSTRUCTION tab).** That tab works from `~/arc-2026-tasks-symbo/demo` and sets a `MACHINE` variable, so its commands read `config/worker-$MACHINE.yaml`:

```bash
cd ~/arc-2026-tasks-symbo/demo
MACHINE=dgx            # or: workstation
```

## 3. Python environments

Each task gets its own venv in `demo/`, because their libraries pin different versions. Torch goes in **first**, from the `cu128` index, so no later step pulls a wheel without Blackwell kernels. All three torch envs use torch 2.7.1, the version Stable Audio 3 pins.

```bash
cd $DEMO
CU=https://download.pytorch.org/whl/cu128
# Create a venv only if it is not there yet
mkvenv() {
  if [ -x "$1/bin/python" ]; then echo "$1 already exists, reusing it"; return 0; fi
  $PY -m venv "$1" && "$1/bin/pip" install -q --upgrade pip
}
# Clone a repo only if it is not there yet, then pin the commit
clone() {   # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" checkout -q "$3"
}
```

**Safe to re-run.** An existing venv or clone is reused, not recreated. The `pip install` lines then only add what is missing: pip skips anything already at the pinned version. To rebuild a venv from scratch, delete its folder first (`rm -rf .venv-t2i`).

**Text-to-image (`.venv-t2i`).**

```bash
mkvenv .venv-t2i
.venv-t2i/bin/pip install torch==2.7.1 --index-url $CU
.venv-t2i/bin/pip install -r requirements/t2i-gpu.txt
```

**Image-to-3D, Stable Fast 3D (`.venv-i23d`).** SF3D is a git clone, not a pip package. Its two C++/CUDA extensions are built against the torch already in the venv, so they are installed with `--no-build-isolation`. `TORCH_CUDA_ARCH_LIST=12.0` compiles them for the 5060 Ti.

```bash
mkvenv .venv-i23d
.venv-i23d/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url $CU
clone https://github.com/Stability-AI/stable-fast-3d third_party/stable-fast-3d ff21fc4   # the commit the laptop runs
.venv-i23d/bin/pip install -U setuptools==69.5.1 wheel
grep -v '^\./' third_party/stable-fast-3d/requirements.txt > /tmp/sf3d-req.txt
.venv-i23d/bin/pip install -r /tmp/sf3d-req.txt
export TORCH_CUDA_ARCH_LIST=12.0 USE_CUDA=1
# Build the two extensions only if they are not installed yet (the build takes minutes)
.venv-i23d/bin/python -c "import texture_baker, uv_unwrapper" 2>/dev/null && echo "SF3D extensions already built" || \
  (cd third_party/stable-fast-3d && $DEMO/.venv-i23d/bin/pip install --no-build-isolation ./texture_baker ./uv_unwrapper)
.venv-i23d/bin/pip install -r requirements/worker-base.txt
.venv-i23d/bin/pip install "huggingface_hub>=0.25,<1.0"   # SF3D's transformers 4.42.3 needs hub < 1.0
```

**Music, Stable Audio 3 (`.venv-music`).**

```bash
mkvenv .venv-music
.venv-music/bin/pip install torch==2.7.1 torchaudio==2.7.1 --index-url $CU
clone https://github.com/Stability-AI/stable-audio-3 third_party/stable-audio-3 790793b
.venv-music/bin/pip install -r requirements/music.txt
.venv-music/bin/pip install --no-deps -e third_party/stable-audio-3
```

**Language model and scene construction (`.venv-llm`).** No torch: the model runs inside `llama-server` (next section). Both the `llm` and `layout` workers use this env.

```bash
mkvenv .venv-llm
.venv-llm/bin/pip install -r requirements/llm.txt
```

**Check every torch env sees the GPU.** Each line must print `(12, 0)` and the card name. `False` or a kernel-image error means a non-`cu128` torch got in: reinstall torch in that venv from `$CU` with `--force-reinstall`.

```bash
for v in t2i i23d music; do
  .venv-$v/bin/python -c "import torch; print('$v', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))"
done
.venv-i23d/bin/python -c "import texture_baker, uv_unwrapper; print('SF3D extensions OK')"
```

## 4. llama.cpp with CUDA (tasks 4 and 8)

The `llm` and `layout` workers each start their own `llama-server`. The worker config expects it at `third_party/llama.cpp/bin/llama-server`. llama.cpp publishes no Linux CUDA binaries, so build release **b12608** from source: the same build number the laptops run. `CMAKE_CUDA_ARCHITECTURES=120` targets the 5060 Ti, and a static build lets the binary be copied anywhere.

```bash
cd $DEMO
LS=third_party/llama.cpp/bin/llama-server
SRC=third_party/llama.cpp-src
if $LS --version 2>&1 | grep -q 12608 && $LS --list-devices 2>&1 | grep -q CUDA0; then
  echo "llama-server b12608 with CUDA is already installed, skipping the build"
else
  # Clone only if the source is not there yet; stop if it is a different release
  [ -d $SRC/.git ] || git clone --depth 1 --branch b12608 https://github.com/ggml-org/llama.cpp $SRC
  if [ "$(git -C $SRC describe --tags 2>/dev/null)" != b12608 ]; then
    echo "$SRC is not release b12608: delete it (rm -rf $SRC) and run this block again"
  else
    cmake -S $SRC -B $SRC/build \
      -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
    cmake --build $SRC/build --target llama-server -j"$(nproc)"
    mkdir -p third_party/llama.cpp/bin
    cp $SRC/build/bin/llama-server third_party/llama.cpp/bin/
  fi
fi
```

The build takes 10 to 20 minutes. **Safe to re-run:** if `llama-server` is already installed at b12608 and sees the GPU, the whole build is skipped. If the source is already cloned, the clone is skipped, and cmake only recompiles what changed. Then check it:

```bash
third_party/llama.cpp/bin/llama-server --version        # version: ... (build 12608, ...)
third_party/llama.cpp/bin/llama-server --list-devices   # CUDA0: NVIDIA GeForce RTX 5060 Ti (16xxx MiB)
```

No extra flags are needed. Any `cuda:` device in the worker config makes the worker pass `--n-gpu-layers 999`, so the whole model sits on the GPU. The Vulkan settings in `worker-local-gpu-vulkan.yaml` are only for the laptop's old driver and do not apply here.

## 5. Download the model weights

`scripts/download_weights.py` reads a config and fetches every model the workstation runs for a task in it, plus that model's encoders. The commands below run it for every task in two configs: `config/demo.yaml` (the show) and `config/demo-full.yaml` (the full model matrix: the larger models that could not be downloaded on the laptop). Weights land in `demo/hf-cache/hub`. Run it with the token loaded from `config.txt`. Any worker env can run it; `.venv-llm` is the lightest.

```bash
cd $DEMO
set -a; . ../config.txt; set +a                        # loads HF_TOKEN and ARC_DEMO_TOKEN
CONFIGS="config/demo.yaml config/demo-full.yaml"       # the show, then the full model matrix
TASKS="t2i i23d music llm layout"
for c in $CONFIGS; do for t in $TASKS; do
  .venv-llm/bin/python scripts/download_weights.py --config $c --machine workstation --task $t --dry-run
done; done                                             # lists the repos first
for c in $CONFIGS; do for t in $TASKS; do
  .venv-llm/bin/python scripts/download_weights.py --config $c --machine workstation --task $t \
    || echo "!! $c / $t: see the FAILED lines above"
done; done
unset HF_TOKEN
```

**Safe to re-run.** Files already in `hf-cache/hub` are skipped, and an interrupted download resumes where it stopped. If the Hub can't be reached, a repo already on disk prints `cached` instead of failing. Only a repo that is neither on disk nor downloadable prints `FAILED`; the loop then goes on to the next task. To fetch only the show models, set `CONFIGS=config/demo.yaml`.

**The show (`config/demo.yaml`), about 17 GB:**

| Task | Repos fetched | Size |
| --- | --- | --- |
| t2i | `stable-diffusion-v1-5/stable-diffusion-v1-5` (fp32 folders only) | 5.2 GB |
| i23d | `stabilityai/stable-fast-3d`, `facebook/dinov2-large`, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 5.6 GB |
| music | `stabilityai/stable-audio-3-small-music` | 3.5 GB |
| llm | `bartowski/Qwen_Qwen3.5-4B-GGUF`, the `Q4_K_M` file only | 2.7 GB |
| layout | the llm file again, already on disk | 0 |

**Full model matrix (`config/demo-full.yaml`), about 85 GB more:**

| Task | Repos fetched | Size |
| --- | --- | --- |
| t2i | `stabilityai/stable-diffusion-3.5-medium` (SD 3.5 Medium, without T5 and with T5 int8) | 27.4 GB |
| i23d | `microsoft/TRELLIS-image-large`, `microsoft/TRELLIS.2-4B`, `facebook/dinov3-vitl16-pretrain-lvd1689m`, `briaai/RMBG-2.0` | 26.1 GB |
| music | `stabilityai/stable-audio-3-medium` | 10.4 GB |
| llm | `bartowski/Qwen_Qwen3.5-9B-GGUF`, the `Q4_K_M` file only (the 4B file is listed again, already on disk) | 6.2 GB |
| layout | `bartowski/Qwen3.8-27B-GGUF`, the `Q3_K_M` file only (Qwen3.8 27B; Q4_K_M is 17.4 GB and does not fit 16 GB) | 13.4 GB |

Downloading these does not change the show: the full matrix runs only when the conductor starts with `--config config/demo-full.yaml`. TRELLIS and TRELLIS.2 also need their own venv, which section 3 does not set up.

A `401` or `403` on a `stabilityai/*` repo (or on facebook/dinov3-* or briaai/RMBG-2.0) means the token's account has not accepted that model's licence (section 1).

**Faster alternative:** if the operator laptop already has `hf-cache/`, copy it over the LAN instead: `rsync -a <laptop>:<path>/demo/hf-cache/ $DEMO/hf-cache/`. Still run the offline check in section 6.

**dgx variant (from the INSTRUCTION tab).** That tab runs each task with its own venv's Python, reads the config with `$MACHINE`, and loads the tokens with carriage returns stripped (`tr -d '\r'`). The loop below covers all five tasks in both configs: about 18 GB for the show and about 177 GB more for the full matrix (SD 3.5 Large and Medium, TRELLIS.2, Stable Audio 3 Medium, and the 77.6 GB Qwen3.5 122B-A10B file). Weights land in `demo/hf-cache/hub`, and the same re-run rules apply.

```bash
set -a; source <(tr -d '\r' < ../config.txt); set +a

for c in config/demo.yaml config/demo-full.yaml; do
  .venv-t2i/bin/python   scripts/download_weights.py --config $c --machine $MACHINE --task t2i
  .venv-i23d/bin/python  scripts/download_weights.py --config $c --machine $MACHINE --task i23d
  .venv-music/bin/python scripts/download_weights.py --config $c --machine $MACHINE --task music
  .venv-t2i/bin/python   scripts/download_weights.py --config $c --machine $MACHINE --task llm     # any env works
  .venv-t2i/bin/python   scripts/download_weights.py --config $c --machine $MACHINE --task layout
done
```

## 6. Worker config and offline check

`config/worker-workstation.yaml` already fits this card. Every task is on `cuda:0`, and `exclusive_models: true` keeps one task's model in the 16 GB at a time. `llm_slots: 8` serves the 8 parallel streams the iso-time segments ask for. Change nothing unless the checks below fail.

| Key | Value | Why |
| --- | --- | --- |
| `devices` | `cuda:0` for t2i, i23d, music, llm, layout | one GPU |
| `exclusive_models` | `true` | loading a model unloads the other task's first |
| `preload` | `t2i: [sd15]`, the rest empty | the others load on the operator's *Load the model* |
| `llama_server` | `third_party/llama.cpp/bin/llama-server` | section 4's build |
| `llm_slots` | `8` | 2.7 GB model + 8 × 2048-token KV cache fits easily |

Now prove every worker loads with the network forbidden. This is what runs at the venue. Each command must end with `PASS`.

```bash
cd $DEMO
unset HF_TOKEN; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
W=config/worker-workstation.yaml
.venv-t2i/bin/python   scripts/verify_offline.py --config $W --task t2i
PYTHONPATH=third_party/stable-fast-3d .venv-i23d/bin/python scripts/verify_offline.py --config $W --task i23d
.venv-music/bin/python scripts/verify_offline.py --config $W --task music
.venv-llm/bin/python   scripts/verify_offline.py --config $W --task llm
.venv-llm/bin/python   scripts/verify_offline.py --config $W --task layout
```

Open the firewall for the conductor, if `ufw` is on: `sudo ufw allow 8101:8105/tcp`. Then confirm the operator laptop resolves `workstation.local`, the name `config/demo.yaml` uses. If not, install `avahi-daemon` or put the IP into `demo.yaml` on the laptop.

**dgx variant (from the INSTRUCTION tab).** That tab checks only t2i, i23d and music, against `config/worker-$MACHINE.yaml`. A `FAIL` here would also fail on stage, so fix it before going on.

```bash
unset HF_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.venv-t2i/bin/python scripts/verify_offline.py --config config/worker-$MACHINE.yaml --task t2i
PYTHONPATH=third_party/stable-fast-3d \
.venv-i23d/bin/python scripts/verify_offline.py --config config/worker-$MACHINE.yaml --task i23d
.venv-music/bin/python scripts/verify_offline.py --config config/worker-$MACHINE.yaml --task music
```

## 7. Run the complete demo

**On the workstation: start the five workers.** Each runs offline in its own env and logs to `results/`. Only SD 1.5 loads at start. The others load when the operator presses *Load the AI model on all machines*, which first unloads the previous task.

```bash
cd $DEMO
set -a; . ../config.txt; set +a; unset HF_TOKEN
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
W=config/worker-workstation.yaml
start() {  # $1 task  $2 port  $3 venv
  nohup .venv-$3/bin/python -m arcdemo.worker --config $W --task $1 --port $2 > results/stdout-$1.log 2>&1 &
}
mkdir -p results
start t2i    8101 t2i
PYTHONPATH=third_party/stable-fast-3d start i23d 8102 i23d
start music  8103 music
start llm    8104 llm
start layout 8105 llm
```

Check each answers. The token header is needed when `ARC_DEMO_TOKEN` is set.

```bash
for p in 8101 8102 8103 8104 8105; do
  echo ":$p $(curl -sf -m 3 -H "X-Demo-Token: $ARC_DEMO_TOKEN" http://127.0.0.1:$p/health || echo DOWN)"
done
```

To stop them all: `pkill -f arcdemo.worker`. This also ends the `llama-server` child processes.

**On the operator laptop: start the conductor.** `config/demo.yaml` points at `http://workstation.local:8101` to `:8105`. Either name this machine `workstation` (`sudo hostnamectl set-hostname workstation`, with `avahi-daemon` running) or edit the workstation URLs in the laptop's `demo.yaml` to this machine's IP.

```bash
export ARC_DEMO_TOKEN=<same secret>
python -m arcdemo.conductor --config config/demo.yaml
```

Open `http://<laptop>:8000/operator`. Its *Workers* panel should show the workstation's five workers up, with matching env hashes. Put `http://<laptop>:8000/stage` full-screen on the projector (F11).

**Run of show on this machine.**

| Part | Segment | Task | Workstation runs |
| --- | --- | --- | --- |
| Three machines, one task / The stopwatch / Same minute, different results / Your prompt | seg0, seg1, seg2, seg4 | t2i | SD 1.5, 512 × 512, 30 steps, batch 4 |
| The same test in 3D | seg3 | i23d | SF3D, 1024 texture |
| Music in a minute (reserve) | seg5 | music | Stable Audio 3 Small, 30 s clips |
| Same answer, different time / Tokens in a minute | seg6, seg7 | llm | Qwen3.5 4B, 1 then 8 streams |
| Furnish the room / Rooms in a minute | seg8, seg9 | layout | Qwen3.5 4B, 1 then 8 rooms |

Before doors: pick Part 1, load the model, and press **Start** once as the warm-up. Then rehearse the GPU cells so the replay table has records: `python scripts/rehearse.py --segments seg0,seg2,seg3,seg6,seg8 --repeat 3` on the laptop.

**dgx variant (from the INSTRUCTION tab).** That tab starts only three GPU workers (t2i 8101, i23d 8102, music 8103), against `config/worker-$MACHINE.yaml`, and also unsets `SF3D_USE_CPU`. Models keep loading in the background, so ask `/health` again after a minute and `loaded` should list them. If a port doesn't answer, look in `results/stdout-<task>.log`. To stop all three: `pkill -f arcdemo.worker`.

```bash
set -a; source <(tr -d '\r' < ../config.txt); set +a
unset HF_TOKEN SF3D_USE_CPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p results

nohup .venv-t2i/bin/python -m arcdemo.worker --config config/worker-$MACHINE.yaml --task t2i --port 8101 \
      > results/stdout-t2i.log 2>&1 &
PYTHONPATH=third_party/stable-fast-3d \
nohup .venv-i23d/bin/python -m arcdemo.worker --config config/worker-$MACHINE.yaml --task i23d --port 8102 \
      > results/stdout-i23d.log 2>&1 &
nohup .venv-music/bin/python -m arcdemo.worker --config config/worker-$MACHINE.yaml --task music --port 8103 \
      > results/stdout-music.log 2>&1 &

for p in 8101 8102 8103; do curl -s -H "X-Demo-Token: $ARC_DEMO_TOKEN" http://127.0.0.1:$p/health; echo; done
```

**Conductor on the GPU machine (from the INSTRUCTION tab).** If the conductor runs on this same GPU machine, first point its URLs at localhost (use `workstation.local` in the `sed` on the workstation), then start it:

```bash
sed -i 's#http://dgx.local:#http://127.0.0.1:#' config/demo.yaml     # workstation: use workstation.local

set -a; source <(tr -d '\r' < ../config.txt); set +a; unset HF_TOKEN
.venv/bin/python -m arcdemo.conductor --config config/demo.yaml --host 0.0.0.0 --port 8000
```

Then open the pages in a browser:

- **Operator controls:** `http://<gpu-machine-ip>:8000/operator`
- **Projector:** `http://<gpu-machine-ip>:8000/stage`

## 8. Troubleshooting

Worker logs are `results/worker-<task>.log`. A crash before logging starts lands in `results/stdout-<task>.log`. Each job's short `run_id` also appears in the conductor log, so one grep follows a cell end to end.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no kernel image is available for execution on the device`, or a `sm_120 is not compatible` warning | a torch wheel built before CUDA 12.8 | `pip install --force-reinstall torch==2.7.1 ... --index-url https://download.pytorch.org/whl/cu128` in that venv |
| `torch.cuda.is_available()` is `False` | CPU torch from PyPI got in | the same reinstall from `cu128` |
| SF3D build: `CUDA_HOME` not set, or `nvcc` version mismatch | toolkit missing, or CUDA 13 `nvcc` on the path | install `cuda-toolkit-12-8` (on Ubuntu 26.04+, the runfile in section 1), export `CUDA_HOME=/usr/local/cuda-12.8`, rebuild both extensions |
| SF3D or llama.cpp build: `unsupported GNU version! gcc versions later than 14 are not supported` | CUDA 12.8 with the newer default GCC of Ubuntu 26.04+ | `sudo apt install -y gcc-14 g++-14`, then rerun `setup_gpu_machine.sh`: it picks GCC 14 for the CUDA builds |
| `ImportError` from `transformers` in `.venv-i23d` | `huggingface_hub` 1.x pulled in by worker-base | `pip install "huggingface_hub>=0.25,<1.0"` |
| llama.cpp build fails at `ggml-cuda` | `nvcc` older than 12.8 has no sm_120 | check `nvcc --version`, then rerun the cmake configure |
| `llama-server --list-devices` shows no CUDA0 | built without `-DGGML_CUDA=ON` | delete `third_party/llama.cpp-src/build` and rebuild |
| `verify_offline.py` FAIL naming a repo | that repo is missing from `hf-cache/hub` | rerun section 5 for that task, with network |
| A worker loads, then `CUDA out of memory` | another task's model is still on the GPU | press *Load the AI model* again, which unloads it; check `nvidia-smi` for stray `python` or `llama-server` processes |
| Operator page shows the workstation DOWN | name, firewall or token | `curl http://workstation.local:8101/health` from the laptop; `sudo ufw allow 8101:8105/tcp`; the same `ARC_DEMO_TOKEN` on both sides |
| Workers panel shows an env hash mismatch in yellow | library versions differ from the other machines | compare `GET <worker>/env` on both machines; freeze per `locks/README.md` |

For more detail, set `ARC_LOG_LEVEL=DEBUG` before starting a worker.
