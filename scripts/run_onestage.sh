#!/bin/bash
export SOL_ENGINE_CONFIG=/root/sol-h3-settings-final/config.json
export TORCHINDUCTOR_CACHE_DIR=/root/hyperflow/inductor_cache
cd /root/Sana/models/minimax_h3/Sol-H3-ComfyUI
/usr/local/miniconda3/envs/py312/bin/python scripts/run_local.py \
  --profile "Sol-H3 one stage — RTX 5090" \
  --prompt "A golden retriever running through a sunflower field, cinematic lighting" \
  --seed 42 \
  --output /root/onestage_test
echo "ONESTAGE_EXIT=$?"
