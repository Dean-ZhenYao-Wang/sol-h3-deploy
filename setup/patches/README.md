# 引擎扩展补丁

上游引擎来自 [NVlabs/Sana](https://github.com/NVlabs/Sana) 的 `models/minimax_h3/Sol-H3-ComfyUI`。
本仓库对引擎的扩展（续接 Extend、参考素材原生插槽、保存目录/文件名、条件冻结成产物、
作品库对齐、Studio 续接来源选择器……）以**补丁**形式放在这里，由
`setup/apply-engine-patches.sh` 打到引擎目录上。

| 文件 | 说明 |
|---|---|
| `0001-sol-h3-engine-extensions.patch` | 本仓库全部引擎改动（相对上游 commit，路径相对引擎目录） |
| `orig/hyperflow_worker.py.orig` | 上游原始 `hyperflow_worker.py`（对照 / 回退用） |

> **上游分支**：Sol-H3 引擎在 NVlabs/Sana 的 **`sol-engine`** 分支上，`main` 分支里没有它
> （不指定分支克隆会得到一个 83MB 的空仓库，**看起来成功其实没有引擎**）。
> 本补丁相对上游 commit **`41866cd`**（sol-engine 分支尖端）生成。

## 用法

```bash
bash setup/get-engine.sh                      # 取上游引擎（还没取的话）
bash setup/apply-engine-patches.sh            # 备份 → 打补丁 → 校验
bash setup/apply-engine-patches.sh --revert   # 反向还原回上游版本
```

脚本是**幂等**的：补丁已经打过就跳过，不会重复打；每个被改动的文件都会先备份成
`<文件>.bak-<时间戳>`。

## 补丁包含什么

| 文件 | 改动 |
|---|---|
| `sol_h3_engine/hyperflow_worker.py` | 续接的尾窗钉住、ref2va 续接、音频行偏移修正、条件冻结 |
| `sol_h3_engine/extend.py` | **新增文件**：masked 续接的帧/潜空间算术与 session sidecar |
| `sol_h3_engine/comfy_nodes.py` | `output_dir`/`filename` 保存控制、IMAGE/AUDIO 原生插槽、参考继承与指纹校验 |
| `sol_h3_engine/comfy_routes.py` | 作品库按 take 收录、新增 `GET /solh3/sessions` |
| `sol_h3_engine/bridge.py` | 常驻引擎进程管理 |
| `web/studio.js` / `web/studio.css` | Studio 界面：续接来源下拉、保存位置、Extend 模式的参考上传区 |
| `tests/test_layout.py` / `tests/test_studio.py` | 配套测试 |
| `scripts/hyperflow/convert_ref.py` | 构建 FP8 参考图 DiT 时的行偏移（**必须在 FP8 构建前打上**） |

## 补丁不适用怎么办

上游更新后上下文可能对不上，脚本会**报错停下**（不会硬打、不会留下半打的状态）。
两条出路：

1. 直接用本仓库 `engine/Sol-H3-ComfyUI/` 覆盖引擎目录 —— 那份就是打过补丁的完整引擎（4.7M）
2. 手工把改动并进去（脚本已在 `引擎目录/*.bak-<时间戳>` 留了备份）

## 重新生成补丁

改动落在 `/root/Sana` 这个 Sana 克隆的工作区里（未提交），所以：

```bash
cd /root/Sana
git add -N models/minimax_h3/Sol-H3-ComfyUI/sol_h3_engine/extend.py
git diff --relative=models/minimax_h3/Sol-H3-ComfyUI -- models/minimax_h3/Sol-H3-ComfyUI \
  > /root/solh3-deploy/setup/patches/0001-sol-h3-engine-extensions.patch
git reset -q -- models/minimax_h3/Sol-H3-ComfyUI/sol_h3_engine/extend.py
```

上游原始文件：`git show HEAD:models/minimax_h3/Sol-H3-ComfyUI/sol_h3_engine/hyperflow_worker.py`
