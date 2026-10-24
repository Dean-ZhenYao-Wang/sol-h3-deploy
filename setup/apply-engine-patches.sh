#!/bin/bash
# 把本仓库对 Sol-H3 引擎的扩展打到引擎目录上（备份 → 打补丁 → 校验）。
#
# 用法:
#   bash apply-engine-patches.sh [引擎目录]
#   bash apply-engine-patches.sh --revert [引擎目录]
#
# 引擎目录默认 /root/Sana/models/minimax_h3/Sol-H3-ComfyUI
# 补丁在 setup/patches/*.patch，编号递增，按文件名顺序应用。
# 幂等：已经打过的补丁会自动跳过（反向试打能过就说明已在目标上）。
# --revert：反向还原回上游版本（同样先备份）。
#
# 为什么需要它：引擎本体来自上游 NVlabs/Sana，本仓库的扩展（续接、参考素材口、
# 保存位置、条件冻结等）是对引擎源码的改动。只装本仓库而没打这些补丁，引擎就会
# 退回上游行为——Studio 的「续接来源」下拉、节点图的参考素材口都会消失。
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="$HERE/patches"
MODE=apply
ENGINE=""
for arg in "$@"; do
  case "$arg" in
    --revert) MODE=revert ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) ENGINE="$arg" ;;
  esac
done
ENGINE="${ENGINE:-${SOL_H3_ENGINE:-/root/Sana/models/minimax_h3/Sol-H3-ComfyUI}}"

if [ ! -d "$ENGINE" ]; then
  echo "!! 引擎目录不存在: $ENGINE"
  echo "   先取一份上游引擎: bash $HERE/get-engine.sh"
  exit 1
fi

PY="${SOL_H3_PY:-/root/h3_runtime_venv/bin/python}"
[ -x "$PY" ] || PY="$(command -v python3 || echo python3)"
STAMP="$(date +%Y%m%d-%H%M%S)"
shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.patch)
if [ ${#PATCHES[@]} -eq 0 ]; then
  echo "!! $PATCH_DIR 下没有补丁文件"
  exit 1
fi

CHANGED="$(mktemp)"
trap 'rm -f "$CHANGED"' EXIT

for patch_file in "${PATCHES[@]}"; do
  name="$(basename "$patch_file")"
  echo "==> $name"
  # 补丁会碰到的文件（+++ b/<路径>）
  files="$(grep -E '^\+\+\+ b/' "$patch_file" | awk '{print $2}' | sed 's|^b/||' || true)"
  # 1) 先备份：出问题可以按名字退回去
  for rel in $files; do
    [ -n "$rel" ] || continue
    if [ -f "$ENGINE/$rel" ]; then
      cp -a "$ENGINE/$rel" "$ENGINE/$rel.bak-$STAMP"
      echo "   备份 $rel -> $rel.bak-$STAMP"
    fi
  done
  # 2) 打补丁（先试打，不落盘）
  if [ "$MODE" = revert ]; then
    if patch -p1 -R -d "$ENGINE" --dry-run < "$patch_file" >/dev/null 2>&1; then
      patch -p1 -R -d "$ENGINE" < "$patch_file" >/dev/null
      echo "   已还原为上游版本"
      for rel in $files; do
        [ -n "$rel" ] && echo "$rel" >> "$CHANGED"
      done
    else
      echo "   无需还原（补丁不在目标上）"
    fi
    continue
  fi
  if patch -p1 -d "$ENGINE" --dry-run --forward < "$patch_file" >/dev/null 2>&1; then
    patch -p1 -d "$ENGINE" --forward < "$patch_file" >/dev/null
    echo "   已应用"
    for rel in $files; do
      [ -n "$rel" ] && echo "$rel" >> "$CHANGED"
    done
  elif patch -p1 -R -d "$ENGINE" --dry-run < "$patch_file" >/dev/null 2>&1; then
    echo "   已经打过，跳过"
    for rel in $files; do
      [ -n "$rel" ] && echo "$rel" >> "$CHANGED"
    done
  else
    echo "!! 补丁不适用: $name"
    echo "   目标文件与上游不一致（上游可能已更新）。两条出路："
    echo "   1) 直接用本仓库 engine/Sol-H3-ComfyUI 覆盖 $ENGINE"
    echo "   2) 手工把改动并进去（备份在 $ENGINE/*.bak-$STAMP）"
    exit 1
  fi
done

# 3) 校验：语法 + 关键标记
echo "==> 校验"
status=0
while read -r rel; do
  [ -n "$rel" ] || continue
  [ -f "$ENGINE/$rel" ] || continue   # 还原后新增文件会消失，跳过
  case "$rel" in
    *.py)
      if ! "$PY" -m py_compile "$ENGINE/$rel" >/dev/null 2>&1; then
        echo "!! 语法错误: $rel"; status=1
      fi ;;
    *.js)
      if command -v node >/dev/null 2>&1; then
        tmp_js="$(mktemp --suffix=.mjs)"
        cp "$ENGINE/$rel" "$tmp_js"
        if ! node --check "$tmp_js" >/dev/null 2>&1; then
          echo "!! 语法错误: $rel"; status=1
        fi
        rm -f "$tmp_js"
      fi ;;
  esac
done < "$CHANGED"

marker() {
  # 模式决定期望状态：打完补丁标记该在，还原后该不在。
  if [ "$MODE" = revert ]; then
    if grep -q -- "$2" "$ENGINE/$1" 2>/dev/null; then
      echo "!! 标记仍在（还原不干净）: $1"; status=1
    fi
  elif ! grep -q -- "$2" "$ENGINE/$1" 2>/dev/null; then
    echo "!! 关键标记缺失: $1（找 '$2'）"; status=1
  fi
}
marker sol_h3_engine/extend.py "def freeze_references"
marker sol_h3_engine/hyperflow_worker.py "freeze_references"
marker sol_h3_engine/comfy_nodes.py "drop_scratch_copies"
marker web/studio.js "/solh3/sessions"

if [ "$status" -ne 0 ]; then
  echo "!! 校验未通过"
  exit 1
fi
echo "OK: 引擎扩展已就位（备份 $ENGINE/*.bak-$STAMP）"
