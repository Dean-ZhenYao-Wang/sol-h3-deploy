#!/bin/bash
set -e
# 本地 HTTP 服务指向 /model 的 MiniMax-H3（只读），aria2c 走 localhost
nohup python3 -m http.server 8990 --bind 127.0.0.1 --directory /model/ModelScope/MiniMax/MiniMax-H3 \
  > /root/hyperflow/ref_http.log 2>&1 &
HTTP_PID=$!
sleep 2
curl -s -o /dev/null -w "http_check=%{http_code}\n" "http://127.0.0.1:8990/transformer_ref/config.json"
export SOL_ENGINE_CONFIG=/root/sol-h3-settings-final/config.json
export PYTHONPATH=/root/Sana/models/minimax_h3/HyperFlow
cd /root/Sana/models/minimax_h3/Sol-H3-ComfyUI
/root/h3_runtime_venv/bin/python scripts/hyperflow/convert_ref.py \
  http://127.0.0.1:8990/transformer_ref \
  /root/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors \
  /root/hyperflow/transformer_hf8/adaln_table.pt \
  /root/hyperflow/transformer_ref_hf8
echo "REF_CONVERT_EXIT=$?"
kill $HTTP_PID 2>/dev/null || true
