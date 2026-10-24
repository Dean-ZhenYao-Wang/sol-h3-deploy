# Sol-H3 云服务器使用手册

> RTX 5090 32GB · NVlabs Sol-H3（MiniMax-H3 视频生成加速引擎）· 最后更新 2026-10-25

## 这是什么

这台机器部署了 Sol-H3 引擎的 **one-stage 加速管线**（8 步 HyperFlow LoRA + FP8 + Sol-Attn + SageAttention + fast-VAE），通过 ComfyUI 提供视频生成服务。核心指标：**5 秒 768p 带音频视频，热态约 66 秒/条**（原版 50 步 BF16 需 989 秒）。

## 快速开始

```bash
# 1. 启动（已配置好，直接跑）
bash /root/launch_comfy.sh

# 2. 浏览器打开
http://<服务器IP>:8188

# 3. 第一条请求会等 2-3 分钟（模型加载+编译缓存），之后每条约 66 秒
```

停止：在运行终端按 `Ctrl+C`（或 `pkill -f "main.py"`）。

## 功能总览

ComfyUI 里有三个使用入口：

| 入口 | 用途 | 速度 |
|---|---|---|
| **Sol-H3 Studio**（默认界面） | 日常创作：文生视频 / 首帧 / 参考图 / 续接 | 快（66s/条） |
| **Nodes 节点图** | 批处理、接入 ComfyUI 生态（Sol Engine 分类下 3 个节点） | 快 |
| **Templates 模板浏览器** | 官方 MiniMax-H3 工作流（跑官方核，功能最全但慢） | 慢（分钟级/条） |
> **关于其它 pipeline 的 Reference 模式**：Studio 里除 One stage 外的 profile（HyperFlow 8 步 / H3 FP8 / H3 BF16）都不支持参考图模式——这不是没配置，是硬件约束：ref2va 需要额外加载 `transformer_ref`（参考图 DiT），BF16 系两个 DiT 共 128GB 超过本机 88GiB 内存上限，H3 FP8 profile 更是加载即 OOM。只有 One stage（FP8 换入换出架构）能同时容纳两个 DiT。**其它 pipeline 要用 ref2va（含视频/音频参考的全集）请走官方模板栈**：Templates 浏览器 → MiniMax-H3 官方模板，用 Comfy-Org 的 int8/fp8 量化 checkpoint 跑官方核（约 2-4 分钟/条，参考类型不限）。


---

## 一、Studio 使用（推荐日常）

### 四种模式

| 模式 | 输入 | 说明 |
|---|---|---|
| **Text** | 提示词 | 文生视频，最快路径 |
| **First frame** | 1 张图 | 图片动画化 |
| **Reference** | **图片区最多 9 张 + 音频区最多 3 段** | 角色/风格/音色参考 |
| **Extend（续接）** | 上一条生成 | 尾窗潜空间作上下文继续生成，接缝连续 |

### Reference 模式的双上传区（图片与音频分开）

界面分成两个独立上传区，**上传顺序即编号**：

- **图片区**：拖入或点击上传，缩略图左上角有 `图1…图N` 徽标（最多 9 张）
- **音频区**：仅在 Reference 模式出现，拖入或点击上传，标签 `音1…音3`（最多 3 段）

提示词写法：按编号指代，例如「图1的短发女孩沿着河堤走，音1的音色说着话，整体像音2的雨天氛围」。引擎会把每张图/每段音频按其编号作为对应参考送入模型——图片进 Qwen 视觉上下文，音频只进音频 VAE（不占视觉上下文）。

音频使用要点：
- 支持 mp3/wav/m4a 等常见格式、任意采样率（内部自动重采样到 32kHz）
- **音频参考必须搭配至少 1 张图片**（官方约束：音频不能单独作为参考）
- 人声参考会显著影响生成语音的音色；音乐参考影响配乐风格

### Extend 续接模式（Studio）

1. 先用任意模式生成一条片段——每次生成都会自动落下 `session.session.safetensors`（潜空间交接文件）
2. 切到 **Extend** 模式，顶部提示词框写新内容（描述接下来发生什么），点 Generate
3. **续接窗口**chip 行选择钉入多少帧：39（默认）/ 90 为共享音视频网格（推荐）；56/22/5 合法但接缝机制不同
4. **续接来源**下拉框选要接着的那一段：默认是刚生成的那条，也可以选任意历史片段（**可以从链条中段续接**，与节点图里「把某个 session 输出口接到 Extend」完全等价）。下拉数据来自 `GET /solh3/sessions`
5. 续接走 latent 级上下文（非像素重编码），59-66 秒/条，可无限链式

### 保存位置（两种模式都能指定）

Studio 界面新增「**保存位置**」区块（两种模式都生效），节点图里对应两个输入框：

| 字段 | 含义 | 留空时 |
|---|---|---|
| **目录**（`output_dir`） | 相对 ComfyUI 输出目录的子目录，可写中文，如 `我的项目/2026-10` | `SolH3/<模式>`（如 `SolH3/text`、`SolH3/extend`） |
| **文件名**（`filename`） | 不带扩展名，如 `河边日落` | `solh3_<模式>_<年月日-时分秒>_s<种子>`，如 `solh3_text_20261025-051233_s42` |

要点：

- 成片直接落在你指定的位置，例如 `output/我的项目/河边日落.mp4`；同名自动加 `_1`/`_2` 后缀，**不会静默覆盖**
- 目录必须是**相对路径**且在 ComfyUI 输出目录内（不能写 `..` 或绝对路径）——因为 ComfyUI 的预览、下载、作品库都按输出目录寻址
- 引擎内部的工作文件仍在 `output/sol_h3/<uuid>/request-NNNNN/`（session 潜变量、worker 日志都留在那里），成片被移出到可读路径

---

## 二、节点图使用（Nodes）

### 打开方式

ComfyUI 菜单 → **Workflow → Browse Templates → Sol-H3**，或直接拖入 `/root/Sana/models/minimax_h3/Sol-H3-ComfyUI/example_workflows/Sol-H3-Studio.json`。

### 三个节点（分类：Sol Engine）

#### 1. SolH3Generate —— 生成/参考节点

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| profile | 下拉 | — | 选 `Sol-H3 one stage — RTX 5090`（唯一可用加速 profile） |
| prompt | 多行文本 | 示例文本 | 提示词；ref2va 可按编号写「图1…」「音1…」 |
| seed | 整数 | 42 | 随机种子，固定可复现 |
| keep_loaded | 布尔 | True | True=跑完保留模型（下条免加载，**推荐**）；False=每次重新加载 |
| mode | 下拉 | text | text / first_frame / reference |
| image | 文本 | 空 | 图片文件名，逗号分隔（reference ≤9，顺序=编号；first_frame 恰好 1 张） |
| audio | 文本 | 空 | 音频文件名，逗号分隔（仅 reference，≤3，顺序=编号） |
| backend | 下拉 | local | 保持 local |
| resolution | 下拉 | 768p | 768p（66s）/ 480p（22s） |
| output_dir | 文本 | 空 | 保存目录（相对 ComfyUI 输出目录）。空 = `SolH3/<mode>` |
| filename | 文本 | 空 | 保存文件名（不含扩展名）。空 = `solh3_<mode>_<时间>_s<seed>` |

输出：`video_with_audio`（VIDEO，接预览/SaveVideo）、`timing_json`（STRING，计时明细）、`session`（STRING，**第三个输出口**——每次生成都会落 `session.session.safetensors`，可直接接到 SolH3Extend 的 `.session` 输入，或留空由 Extend 自动续最新一条）。

#### 2. SolH3Extend —— 续接节点

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| profile | 下拉 | — | 同上 |
| prompt | 多行文本 | 示例文本 | 描述接下来发生什么 |
| seed | 整数 | 43 | 换条时建议 +1 避免与上条相同 |
| session | 文本 | 空 | **接上一条 Generate 或 Extend 节点的 session 输出口（第三个输出）**，或直接粘 `.session.safetensors` 文件路径 |
| window_frames | 整数 | 39 | 上一条尾部钉入上下文的帧数。推荐 39 或 90（共享音视频网格）；56/22/5 合法；范围 5–124 |
| keep_loaded / resolution | — | 同 Generate | — |
| output_dir | 文本 | 空 | 保存目录（相对 ComfyUI 输出目录）。空 = `SolH3/extend` |
| filename | 文本 | 空 | 保存文件名（不含扩展名）。空 = `solh3_extend_<时间>_s<seed>` |

输出：`video_with_audio`、`timing_json`、**`session`**（链条上下一个 Extend 的输入）。

#### 3. SolH3Release —— 释放节点

输入 `after`（VIDEO，接在链尾任何 VIDEO 输出后面）：VIDEO 原样透传，同时**关闭引擎进程**释放内存/显存。跑完长链后想彻底释放时接在末端。

### 标准连线图（文本版）

**单条生成：**
```
[SolH3Generate] --video_with_audio--> [Preview Video / Save Video]
```

**链式续接（整条链都是 Sol-H3 在跑）：**
```
[SolH3Generate] --video--> [Save Video]
       |--session(STRING)--> [SolH3Extend].session
                                |--video--> [Save Video]
[SolH3Extend#1] --session--> [SolH3Extend#2].session   （想续多长接多长）
                                ...
[最后一个节点] --video--> [SolH3Release].after          （可选：链尾释放引擎）
```

要点：
- session 线是 STRING 类型（橙色），必须接到 `.session` 输入口——这就是潜空间传递的通道，**断开会话线就等于丢弃接缝连续性**
- 每条 Extend 的 prompt 只写“新发生的内容”，上一条的画面由上下文保证连续
- 节点图里发请求后看进度：节点下方有阶段提示（queued→loading→conditioning→denoising→decoding→saving）

---

## 三、与 obvpm 导演台搭配

obvpm（节点菜单搜 `obvpm`）是官方核生态的导演台：时间线、Extend/Prepend/Bridge/Loop、`.mctx.safetensors` 潜空间 sidecar。

### 分工原则

| 环节 | 用什么 | 原因 |
|---|---|---|
| **全部生成与续接** | **Sol-H3**（Studio 或 SolH3 节点链） | latent 级续接 + 66s/条，全程不离开加速引擎 |
| 时间线组织、排序、剪切、预览、导出 | obvpm Timeline | 它是个编辑器，任何 mp4 都能拖进去 |
| 外部片段接入 | obvpm From Frames → 官方核续接 | Sol 的 session 格式与 obvpm 的 mctx 不互通（见下） |

### 正确搭配姿势

1. 在 Sol-H3 Studio / 节点图里生成第一条，用 Extend 续接出所有片段（成片落在你指定的保存位置，默认 `output/SolH3/<模式>/`；引擎工作文件在 `/root/ComfyUI/output/sol_h3/`）
2. 把成片 mp4 拖进 obvpm Timeline：排序、裁剪、预览整体节奏、导出成片
3. 对不满意的中间段：回 Sol-H3 用 Extend 重生成那一段（latent 连续性保持），再拖回时间线
4. 长链质量漂移时：在 Sol 里新起一条 seed 不同的生成，用 Extend 从旧链结尾接过去（相当于 obvpm 的 Bridge 思路）

### 必须知道的限制

- **obvpm 的 latent 级续接只认它自己保存的 `.mctx.safetensors` sidecar**（哈希校验）。Sol-H3 的片段带的是 `.session.safetensors`（Sol 自有格式），obvpm 读不到 → 会退化成**像素路由**（From Frames 重新编码，接缝略软）
- 所以：**要 latent 级无缝续接，就必须全程用 Sol-H3 的 Extend**；obvpm 在这里当“剪辑台”而不是“生成器”
- 什么时候切到官方核 + obvpm 全家桶：不在乎 66s vs 分钟级的精修场景、或必须要 obvpm 的 Bridge/Loop 完整算法时

---

## 目录速查

| 路径 | 内容 |
|---|---|
| `/root/launch_comfy.sh` | **启动脚本**（唯一入口） |
| `/root/README.md` | 本文件 |
| `/root/sol-h3-settings-final/config.json` | 引擎配置（profile 定义，SOL_ENGINE_CONFIG 指向它） |
| `/root/hyperflow/transformer_hf8/` | 主 DiT FP8 权重（20G）+ AdaLN 查表 |
| `/root/hyperflow/transformer_ref_hf8/` | 参考图 DiT FP8 权重（20G）+ AdaLN 查表 |
| `/root/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors` | HyperFlow LoRA（运行时加载，勿删） |
| `/root/hyperflow/inductor_cache/` | 编译缓存（删了会多等几分钟，别删） |
| `/root/sol-h3-vae/` | fast-VAE 解码器权重（6.1G） |
| `/root/Sana/models/minimax_h3/Sol-H3-ComfyUI/` | **引擎源码**（sol_h3_engine/ 是 ComfyUI 插件；example_workflows/ 有示例） |
| `/root/ComfyUI/` | ComfyUI 本体（v0.42）+ custom_nodes/（obvpm 等 7 个包） |
| `/root/ComfyUI/output/SolH3/<模式>/` | **成片**（默认保存位置；`output_dir`/`filename` 可改，Studio 作品库收录这里） |
| `/root/ComfyUI/output/sol_h3/<uuid>/request-NNNNN/` | 引擎工作目录：session 潜变量（`session.session.safetensors`）、worker 日志、原始 MP4 |
| `/root/p0f_light.py` | AdaLN 补表工具（见排障） |
| `/model/ModelScope/MiniMax/MiniMax-H3/` | 原始 BF16 权重（只读网络盘，勿占本地空间拷贝） |

## 日常运维

```bash
# 看 GPU
nvidia-smi

# 看生成日志（worker 崩溃时在这里找原因）
ls -t /root/ComfyUI/output/sol_h3/*/worker.log | head -1 | xargs tail -50

# 完全重启（改了引擎代码后必须）
pkill -f "main.py"; sleep 3; bash /root/launch_comfy.sh
```

## 排障

| 症状 | 处理 |
|---|---|
| 启动后第一条特别慢（2-3 分钟） | 正常，模型加载+编译缓存命中 |
| 显存不足报错 | 确认没有第二个生成任务在跑；`nvidia-smi` 看占用 |
| `AdaLN table has no entry for the pair set ...` | 调度/LoRA 被改动导致查表缺行。运行 `cd /root && /root/h3_runtime_venv/bin/python -u /root/p0f_light.py` 补行（约 20 分钟，逐位自检自动做） |
| worker 崩溃/请求卡住 | 看 worker.log；不行就 `pkill -f "sol_h3_engine/worker.py"` 后重发请求 |
| 换机器/重装后缺权重 | FP8 权重可用 `/root/Sana/models/minimax_h3/Sol-H3-ComfyUI/scripts/hyperflow/convert_fp8.py` 从 `/model` 的 BF16 重新构建（见当日部署日志） |
| Reference 模式上传区只剩图片区 | 检查选的是不是 Reference 模式（音频区仅该模式显示）；浏览器 Ctrl+F5 强制刷新 |

## 重要约束（硬件天花板）

- **内存上限 88GiB**：不要同时常驻两个全量 BF16 DiT；大构建任务放本地盘、中间文件及时清
- 磁盘 79G，FP8 权重占 44G——重装或构建前先 `df -h`
- `/model` 是只读网络盘（40T），**读取极快但不要往里写**；构建类脚本的中间产物一律写 `/root`
- GPU 被占时新请求会排队，不要并发发多条生成

## 改动过的引擎（相对官方 Sol-H3）

以下为自定义扩展，升级引擎源码时注意保留（源码旁有 `.bak-audio` 备份可对照）：

1. **音频参考支持**：bridge.py / hyperflow_worker.py / comfy_nodes.py / web/studio.js——Reference 模式 3 段音频（PyAV 重采样 + audio VAE 上卡 + AdaLN 音频行）
2. **图片/音频分区上传**：web/studio.js——Reference 模式图片区（图1…图N）与音频区（音1…音3）独立，编号即上传顺序
3. **参考图上限 4→9**：web/studio.js 的 `limit()`
4. **续接（Extend）**：引擎 + SolH3Extend 节点 + Studio Extend 模式（P1/P2，已验证）
5. **AdaLN 表音频扩展行**：transformer_hf8 与 transformer_ref_hf8 的 adaln_table.pt 各 +8 个音频参考变体行（p0f_light.py 生成）
6. **保存位置可控**：comfy_nodes.py 的 `output_dir` / `filename`（Generate 与 Extend 都有）——成片移到可读路径，默认名 `solh3_<模式>_<时间>_s<种子>`
7. **Studio 续接来源选择器**：web/studio.js + `GET /solh3/sessions`——Studio 可选任意历史片段续接（与节点图端口接法对齐）
