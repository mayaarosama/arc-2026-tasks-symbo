# ARC 2026 Demo — GPU Workstation Setup (RTX 5060 Ti)

As of 2026-11-22.

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

## 1. Prerequisites

You need about 80 GB of free disk: 17 GB of weights, about 6 GB for each of the three torch envs, and the CUDA toolkit. Do every step below with network access. The show itself runs offline.

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

**Hugging Face access.** Log in to the account whose token you will use. Accept the licence on each gated page:

- [stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d)
- [stabilityai/stable-audio-3-small-music](https://huggingface.co/stabilityai/stable-audio-3-small-music)

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

Write the secrets file **next to** `demo/`, never inside it. It is git-ignored, and scripts never print its values.

```bash
cat > ~/arc2026/config.txt <<'EOF'
ARC_DEMO_TOKEN=<shared secret, same on every machine>
HF_TOKEN=hf_...
EOF
chmod 600 ~/arc2026/config.txt
```

Two git-ignored folders hold everything heavy: `third_party/` (cloned libraries and llama.cpp) and `hf-cache/hub` (all model weights, set by `arcdemo/__init__.py`). Neither comes with the clone.

## 3. Python environments

Each task gets its own venv in `demo/`, because their libraries pin different versions. Torch goes in **first**, from the `cu128` index, so no later step pulls a wheel without Blackwell kernels. All three torch envs use torch 2.7.1, the version Stable Audio 3 pins.

```bash
cd $DEMO
CU=https://download.pytorch.org/whl/cu128
mkvenv() { $PY -m venv "$1" && "$1/bin/pip" install -q --upgrade pip; }
```

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
git clone https://github.com/Stability-AI/stable-fast-3d third_party/stable-fast-3d
git -C third_party/stable-fast-3d checkout ff21fc4          # the commit the laptop runs
.venv-i23d/bin/pip install -U setuptools==69.5.1 wheel
grep -v '^\./' third_party/stable-fast-3d/requirements.txt > /tmp/sf3d-req.txt
.venv-i23d/bin/pip install -r /tmp/sf3d-req.txt
export TORCH_CUDA_ARCH_LIST=12.0 USE_CUDA=1
(cd third_party/stable-fast-3d && $DEMO/.venv-i23d/bin/pip install --no-build-isolation ./texture_baker ./uv_unwrapper)
.venv-i23d/bin/pip install -r requirements/worker-base.txt
.venv-i23d/bin/pip install "huggingface_hub>=0.25,<1.0"   # SF3D's transformers 4.42.3 needs hub < 1.0
```

**Music, Stable Audio 3 (`.venv-music`).**

```bash
mkvenv .venv-music
.venv-music/bin/pip install torch==2.7.1 torchaudio==2.7.1 --index-url $CU
git clone https://github.com/Stability-AI/stable-audio-3 third_party/stable-audio-3
git -C third_party/stable-audio-3 checkout 790793b
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
git clone --depth 1 --branch b12608 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-src
cmake -S third_party/llama.cpp-src -B third_party/llama.cpp-src/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build third_party/llama.cpp-src/build --target llama-server -j"$(nproc)"
mkdir -p third_party/llama.cpp/bin
cp third_party/llama.cpp-src/build/bin/llama-server third_party/llama.cpp/bin/
```

The build takes 10 to 20 minutes. Then check it:

```bash
third_party/llama.cpp/bin/llama-server --version        # version: ... (build 12608, ...)
third_party/llama.cpp/bin/llama-server --list-devices   # CUDA0: NVIDIA GeForce RTX 5060 Ti (16xxx MiB)
```

No extra flags are needed. Any `cuda:` device in the worker config makes the worker pass `--n-gpu-layers 999`, so the whole model sits on the GPU. The Vulkan settings in `worker-local-gpu-vulkan.yaml` are only for the laptop's old driver and do not apply here.

## 5. Download the model weights

`scripts/download_weights.py` reads `config/demo.yaml` and fetches every model the workstation runs for a task, plus that model's encoders. Weights land in `demo/hf-cache/hub`. Run it with the token loaded from `config.txt`. Any worker env can run it; `.venv-llm` is the lightest.

```bash
cd $DEMO
set -a; . ../config.txt; set +a                        # loads HF_TOKEN and ARC_DEMO_TOKEN
for t in t2i i23d music llm layout; do
  .venv-llm/bin/python scripts/download_weights.py --machine workstation --task $t --dry-run
done                                                   # lists the repos first
for t in t2i i23d music llm layout; do
  .venv-llm/bin/python scripts/download_weights.py --machine workstation --task $t
done
unset HF_TOKEN
```

| Task | Repos fetched | Size |
| --- | --- | --- |
| t2i | `stable-diffusion-v1-5/stable-diffusion-v1-5` (fp32 folders only) | 5.2 GB |
| i23d | `stabilityai/stable-fast-3d`, `facebook/dinov2-large`, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | 5.6 GB |
| music | `stabilityai/stable-audio-3-small-music` | 3.5 GB |
| llm | `bartowski/Qwen_Qwen3.5-4B-GGUF`, the `Q4_K_M` file only | 2.7 GB |
| layout | the llm file again, already on disk | 0 |

A `401` or `403` on a `stabilityai/*` repo means the token's account has not accepted that model's licence (section 1).

**Faster alternative:** if the operator laptop already has `hf-cache/`, copy it over the LAN instead: `rsync -a <laptop>:<path>/demo/hf-cache/ $DEMO/hf-cache/`. Still run the offline check in section 6.

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

## 8. Troubleshooting

Worker logs are `results/worker-<task>.log`. A crash before logging starts lands in `results/stdout-<task>.log`. Each job's short `run_id` also appears in the conductor log, so one grep follows a cell end to end.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `no kernel image is available for execution on the device`, or a `sm_120 is not compatible` warning | a torch wheel built before CUDA 12.8 | `pip install --force-reinstall torch==2.7.1 ... --index-url https://download.pytorch.org/whl/cu128` in that venv |
| `torch.cuda.is_available()` is `False` | CPU torch from PyPI got in | the same reinstall from `cu128` |
| SF3D build: `CUDA_HOME` not set, or `nvcc` version mismatch | toolkit missing, or CUDA 13 `nvcc` on the path | install `cuda-toolkit-12-8`, export `CUDA_HOME=/usr/local/cuda-12.8`, rebuild both extensions |
| `ImportError` from `transformers` in `.venv-i23d` | `huggingface_hub` 1.x pulled in by worker-base | `pip install "huggingface_hub>=0.25,<1.0"` |
| llama.cpp build fails at `ggml-cuda` | `nvcc` older than 12.8 has no sm_120 | check `nvcc --version`, then rerun the cmake configure |
| `llama-server --list-devices` shows no CUDA0 | built without `-DGGML_CUDA=ON` | delete `third_party/llama.cpp-src/build` and rebuild |
| `verify_offline.py` FAIL naming a repo | that repo is missing from `hf-cache/hub` | rerun section 5 for that task, with network |
| A worker loads, then `CUDA out of memory` | another task's model is still on the GPU | press *Load the AI model* again, which unloads it; check `nvidia-smi` for stray `python` or `llama-server` processes |
| Operator page shows the workstation DOWN | name, firewall or token | `curl http://workstation.local:8101/health` from the laptop; `sudo ufw allow 8101:8105/tcp`; the same `ARC_DEMO_TOKEN` on both sides |
| Workers panel shows an env hash mismatch in yellow | library versions differ from the other machines | compare `GET <worker>/env` on both machines; freeze per `locks/README.md` |

For more detail, set `ARC_LOG_LEVEL=DEBUG` before starting a worker.
