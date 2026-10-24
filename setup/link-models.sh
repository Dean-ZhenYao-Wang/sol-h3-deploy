#!/bin/bash
# 重建 ComfyUI 模型目录软链（指向 ModelScope 下载的官方 comfyui 格式权重）
# 用法: bash link-models.sh [ModelScope目录，默认 /model/ModelScope] [ComfyUI目录，默认 /root/ComfyUI]
set -e
MS=${1:-/model/ModelScope}
COMFY=${2:-/root/ComfyUI}
SRC="$MS/Comfy-Org/MiniMax-H3"
[ -d "$SRC" ] || { echo "!! $SRC 不存在，先下载 Comfy-Org/MiniMax-H3"; exit 1; }

cd "$COMFY/models"
for d in diffusion_models text_encoders vae embeddings loras model_patches audio_encoders; do
  [ -d "$SRC/$d" ] || continue
  mkdir -p "$d"
  ln -sfn "$SRC/$d" "$d/minimax-h3-comfyorg"
done
echo "OK: linked $(ls -d */minimax-h3-comfyorg 2>/dev/null | wc -l) model dirs"
