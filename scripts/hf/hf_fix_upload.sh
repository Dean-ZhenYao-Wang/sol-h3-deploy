#!/bin/bash
# HF repo fix: correct subdirectory upload + root junk cleanup + verify
LOG=/root/hf_fix_upload.log
{
echo "=== step 0: move .bak-audio out of upload dirs ==="
mkdir -p /root/hyperflow/table-backups
mv /root/hyperflow/transformer_hf8/*.bak-audio /root/hyperflow/table-backups/ 2>/dev/null
mv /root/hyperflow/transformer_ref_hf8/*.bak-audio /root/hyperflow/table-backups/ 2>/dev/null
ls /root/hyperflow/transformer_hf8/ | wc -l
ls /root/hyperflow/transformer_ref_hf8/ | wc -l

echo "=== step 1: upload transformer_hf8 -> transformer_hf8/ ==="
date
hf upload ZhenYaoWang/sol-h3-fp8-5090 /root/hyperflow/transformer_hf8 transformer_hf8
echo "STEP1_DONE"

echo "=== step 2: upload transformer_ref_hf8 -> transformer_ref_hf8/ ==="
date
hf upload ZhenYaoWang/sol-h3-fp8-5090 /root/hyperflow/transformer_ref_hf8 transformer_ref_hf8
echo "STEP2_DONE"

echo "=== step 3: delete flattened root junk ==="
/usr/local/miniconda3/envs/py312/bin/python /root/hf_delete_root.py

echo "=== step 4: verify structure ==="
/usr/local/miniconda3/envs/py312/bin/python /root/hf_verify.py

echo "ALL_DONE"
date
} 2>&1
