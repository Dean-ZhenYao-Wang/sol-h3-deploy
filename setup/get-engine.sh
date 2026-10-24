#!/bin/bash
# 取一份上游 Sol-H3 引擎。
#
# 用法: bash get-engine.sh [Sana 目录，默认 /root/Sana]
#
# 上游 NVlabs/Sana 里带两样我们都要的东西：
#   models/minimax_h3/Sol-H3-ComfyUI  引擎本体（ComfyUI 插件）
#   models/minimax_h3/HyperFlow       运行时依赖 overlay（configure.py 需要它）
#
# ⚠️ 引擎在 **sol-engine 分支**上，main 分支里没有 Sol-H3：不指定分支会克隆出一个
#    83MB 的仓库，看起来成功，其实没有引擎目录（静默失败，最坑的一种）。
#
# 引擎到手后，用 setup/apply-engine-patches.sh 把本仓库的扩展打上去。
set -euo pipefail

ROOT="${1:-/root/Sana}"
ENGINE="$ROOT/models/minimax_h3/Sol-H3-ComfyUI"
OVERLAY="$ROOT/models/minimax_h3/HyperFlow"
REPO_URL="https://github.com/NVlabs/Sana.git"
BRANCH="${SOL_H3_BRANCH:-sol-engine}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d "$ENGINE" ] && [ -d "$OVERLAY" ]; then
  echo "OK: 引擎已就位 $ENGINE"
  echo "    下一步: bash $HERE/apply-engine-patches.sh"
  exit 0
fi

if [ ! -d "$ROOT/.git" ]; then
  echo "==> 克隆上游 $REPO_URL 分支 $BRANCH -> $ROOT（浅克隆，~80MB）"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT"
else
  echo "==> $ROOT 已是 git 仓库，补取 $BRANCH 分支的缺失目录"
  cd "$ROOT"
  git fetch --depth 1 origin "$BRANCH"
  git checkout -q FETCH_HEAD -- models/minimax_h3
fi

[ -d "$ENGINE" ] || {
  echo "!! 引擎目录缺失: $ENGINE"
  echo "   上游把 Sol-H3 引擎放在 $BRANCH 分支（main 分支里没有）"
  exit 1
}
[ -d "$OVERLAY" ] || { echo "!! HyperFlow overlay 缺失: $OVERLAY"; exit 1; }
echo "OK: 引擎已就位 $ENGINE"
echo "    下一步: bash $HERE/apply-engine-patches.sh"
