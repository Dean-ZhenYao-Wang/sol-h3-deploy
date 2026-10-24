#!/bin/bash
# Sol-H3 一键部署（RTX 5090 / SM120，内存 ≥88GiB，磁盘 ≥80G）
# 用法: bash setup.sh [--skip-weights] [--skip-build] [--skip-comfyui]
#   --skip-weights  已有 /model 或已下载权重，跳过下载
#   --skip-build    已有构建好的 FP8 权重，跳过 FP8 构建（数小时）
#   --skip-comfyui  ComfyUI 已就绪，跳过安装
set -e

ROOT=${ROOT:-/root}
MODEL_DIR=${MODEL_DIR:-/model/ModelScope}
H3=${MODEL_DIR}/MiniMax/MiniMax-H3
CONFIG=${CONFIG:-/root/sol-h3-settings-final/config.json}

step() { echo -e "\n\033[1;36m==> $1\033[0m"; }

# ---------- 1. Python 环境 ----------
step "1/7 Python 环境（conda py312 + torch 2.13 cu132）"
if [ ! -d "$ROOT/h3_runtime_venv" ]; then
  conda create -y -n py312 python=3.12
  /usr/local/miniconda3/envs/py312/bin/python -m venv --system-site-packages "$ROOT/h3_runtime_venv"
fi
PY="$ROOT/h3_runtime_venv/bin/python"
$PY -m pip install -U pip
$PY -m pip install torch==2.13.0 triton==3.7.1 av scipy safetensors \
  transformers diffusers accelerate modelscope soundfile  # torch/triton 走 https://download.pytorch.org/whl/cu132

# ---------- 2. 编译 SageAttention 2.2 + Sol-Attn 0.5 ----------
step "2/7 编译 SageAttention 2.2.0（PyPI 只有 1.0.6，必须源码）"
apt-get update && apt-get install -y g++ aria2 git
if ! $PY -c "import sageattention" 2>/dev/null; then
  cd "$ROOT"
  git clone --depth 1 https://github.com/thu-ml/SageAttention
  cd SageAttention
  TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda-13.2 $PY -m pip install -e .
  cd "$ROOT" && rm -rf SageAttention
fi
step "2b/7 编译 Sol-Attn 0.5.0（NVlabs/Sana@9bfca5c4 的 techniques/sparse_backends）"
if ! $PY -c "import sol_attn" 2>/dev/null; then
  cd "$ROOT"
  git clone https://github.com/NVlabs/Sana sana-src && cd sana-src
  git checkout 9bfca5c4
  $PY -m pip install -e techniques/sparse_backends
  cd "$ROOT" && rm -rf sana-src
fi

# ---------- 3. 下载权重 ----------
if [ "$1" != "--skip-weights" ]; then
step "3/7 下载权重（约 75GB，/model 网络盘极快；无 /model 时走 ModelScope）"
  $PY -m pip install -U modelscope
  $PY -m modelscope download --model MiniMax/MiniMax-H3 --local_dir "$H3"
  $PY -m modelscope download --model videorebirth/hyperflow --local_dir "$ROOT/hyperflow-dl"
  $PY -m modelscope download --model Comfy-Org/MiniMax-H3 --local_dir "$MODEL_DIR/Comfy-Org/MiniMax-H3"
  $PY -m modelscope download --model Efficient-Large-Model/Sol-H3-VAE --local_dir "$ROOT/sol-h3-vae"
fi

# ---------- 4. 构建 FP8 ----------
if [ "$1" != "--skip-build" ]; then
step "4/7 构建 FP8 权重（数小时；顺序不能乱：不带 drop-adaln → 建表 → 合并 → 再 drop）"
  REPO="$ROOT/Sana/models/minimax_h3/Sol-H3-ComfyUI"
  cd "$REPO"
  # 4.1 主 DiT：BF16→FP8（保留 adaln_proj 供建表校验）
  SOL_ENGINE_CONFIG=$CONFIG $PY scripts/hyperflow/convert_fp8.py \
    "$H3/transformer" "$ROOT/hyperflow-dl/minimax_h3_hyperflow_8step_v1.0.safetensors" \
    "$ROOT/hyperflow/transformer_full"
  # 4.2 payload + 建表（t2va + fl2va 两个）
  SOL_ENGINE_CONFIG=$CONFIG $PY scripts/hyperflow/save_payload.py \
    "Sol-H3 one stage — RTX 5090" /tmp/payload_t2va.pt "A bird glides over a mountain lake."
  SOL_ENGINE_CONFIG=$CONFIG $PY scripts/hyperflow/save_payload.py \
    "Sol-H3 one stage — RTX 5090" /tmp/payload_fl2va.pt "A bird glides over a mountain lake." \
    "$ROOT/hyperflow/frame.png"
  SOL_ENGINE_CONFIG=$CONFIG $PY scripts/hyperflow/build_adaln_table.py \
    /tmp/payload_t2va.pt /tmp/payload_fl2va.pt /root/hyperflow/transformer_hf8/adaln_table.pt
  # 4.3 瘦身：drop adaln_proj（表已就位）
  SOL_ENGINE_CONFIG=$CONFIG $PY scripts/hyperflow/convert_fp8.py \
    "$H3/transformer" "$ROOT/hyperflow-dl/minimax_h3_hyperflow_8step_v1.0.safetensors" \
    "$ROOT/hyperflow/transformer_hf8" --drop-adaln
  # 4.4 参考图 DiT（fl2va 表行复用；本地 HTTP + aria2c 流式，66G 不落盘）
  (cd "$MODEL_DIR/MiniMax/MiniMax-H3" && python3 -m http.server 8990 --bind 127.0.0.1 &) 
  sleep 2
  SOL_ENGINE_CONFIG=$CONFIG $PY scripts/hyperflow/convert_ref.py \
    http://127.0.0.1:8990/transformer_ref \
    "$ROOT/hyperflow-dl/minimax_h3_hyperflow_8step_v1.0.safetensors" \
    "$ROOT/hyperflow/transformer_hf8/adaln_table.pt" "$ROOT/hyperflow/transformer_ref_hf8"
  pkill -f "http.server 8990" || true
  # 4.5 音频参考行（PyAV 重采样补丁已在引擎内）
  SOL_ENGINE_CONFIG=$CONFIG $PY scripts/hyperflow/../p0f_light.py 2>/dev/null || \
    echo "!! p0f_light.py 需放到引擎 scripts/ 同级；参照 docs/deploy-log.md 手动补音频 AdaLN 行"
fi

# ---------- 5. 生成配置 ----------
step "5/7 configure.py 生成 profile"
cd "$ROOT/Sana/models/minimax_h3/Sol-H3-ComfyUI"
$PY scripts/configure.py \
  --reuse-paths "$ROOT/sol-h3-settings/manifest.json" \
  --dense-lora "$MODEL_DIR/FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/dense-datafree/adapter_model.safetensors" \
  --hyperflow-weights "$ROOT/hyperflow-dl/minimax_h3_hyperflow_8step_v1.0.safetensors" \
  --hyperflow-deps "$ROOT/Sana/models/minimax_h3/HyperFlow" \
  --hyperflow-fp8-dir "$ROOT/hyperflow/transformer_hf8" \
  --hyperflow-adaln-table "$ROOT/hyperflow/transformer_hf8/adaln_table.pt" \
  --hyperflow-ref-fp8-dir "$ROOT/hyperflow/transformer_ref_hf8" \
  --hyperflow-fast-vae-weights "$ROOT/sol-h3-vae/fast_h3_vae_noncausal_24l.safetensors" \
  --fast-vae-code "$ROOT/sol-h3-vae/code" \
  --output-dir "$ROOT/sol-h3-settings-final" --gpu 0

# ---------- 6. ComfyUI ----------
if [ "$1" != "--skip-comfyui" ]; then
step "6/7 ComfyUI + 节点包 + 模型软链"
  cd "$ROOT"
  git clone https://github.com/comfyanonymous/ComfyUI
  cd ComfyUI && /usr/local/miniconda3/envs/py312/bin/python -m venv venv
  ./venv/bin/pip install -r requirements.txt
  bash "$ROOT/solh3-deploy/setup/install-custom-nodes.sh"
  bash "$ROOT/solh3-deploy/setup/link-models.sh"
  ln -sfn "$ROOT/Sana/models/minimax_h3/Sol-H3-ComfyUI" "$ROOT/ComfyUI/custom_nodes/Sol-H3-ComfyUI"
fi

step "7/7 完成！启动: bash scripts/launch_comfy.sh → http://<IP>:8188"
