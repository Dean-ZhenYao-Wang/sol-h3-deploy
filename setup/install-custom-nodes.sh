#!/bin/bash
# 安装 obvpm 导演台全家桶 + Sol-H3 引擎插件链接
# 用法: bash install-custom-nodes.sh [ComfyUI目录，默认 /root/ComfyUI]
set -e
COMFY=${1:-/root/ComfyUI}
cd "$COMFY/custom_nodes"

PACKS=(
  "https://github.com/chanon/comfyui-obvpm-timeline"       # 导演台时间线（Extend/Prepend/Bridge/Loop）
  "https://github.com/chanon/comfyui-obvpm"                # obvpm 基础包（依赖）
  "https://github.com/kijai/ComfyUI-KJNodes"               # 示例工作流依赖
  "https://github.com/rgthree/rgthree-comfy"               # 示例工作流依赖
  "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler"  # 潜空间放大
  "https://github.com/wuli-art/MiniMax-H3-Turbo-ComfyUI"   # Turbo LoRA 工作流
  "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3"  # Spectrum 采样
)
for url in "${PACKS[@]}"; do
  name=$(basename "$url")
  [ -d "$name" ] || git clone --depth 1 -q "$url"
done
echo "OK: ${#PACKS[@]} packs installed"
