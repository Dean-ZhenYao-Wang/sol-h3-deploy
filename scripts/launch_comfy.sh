#!/bin/bash
export SOL_ENGINE_CONFIG=/root/sol-h3-settings-final/config.json
export TORCHINDUCTOR_CACHE_DIR=/root/hyperflow/inductor_cache
cd /root/ComfyUI
exec /usr/local/miniconda3/envs/py312/bin/python main.py --listen 0.0.0.0 --port 8188
