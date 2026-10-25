# Sol-H3 部署日志（2026-10-24，RTX 5090 / SM120 / 88GiB）

> 仓库版：已脱敏（服务器 IP/密码不入库）。原始记录见本地 memory。

## 服务器
- RTX 5090 32GB (SM120)，内存 94GB（cgroup 上限 88GiB），根盘 79G，`/model` 40T 只读 UPFS 网络盘

## 最终架构（全部跑通）
- **Sol-H3 one stage — RTX 5090** profile 验证 PASS：t2va 8 步出片 122.8s（含冷启动 46s），generate 70.3s，与官方 validation 一致
- `SOL_ENGINE_CONFIG=<配置目录>/config.json`，启动脚本 `scripts/launch_comfy.sh`（含 `TORCHINDUCTOR_CACHE_DIR=<hyperflow>/inductor_cache`）
- H3 runtime venv：`h3_runtime_venv`（--system-site-packages 基于 conda py312：torch 2.13.0+cu132 / triton 3.7.1）
- ComfyUI venv：`ComfyUI/venv`（py3.12）；ComfyUI v0.42+
- FP8 权重：主 DiT `transformer_hf8`（20GB，--drop-adaln）+ `adaln_table.pt`（t2va+fl2va 行，block0_replay_exact=true）；参考图 DiT `transformer_ref_hf8`（20G，自有 adaln_table.pt）
- Sol-H3-VAE（`sol-h3-vae/`）：fast 非因果解码器，省 10s/条

## 功能矩阵
| 模式 | 状态 | 说明 |
|---|---|---|
| t2va 文生视频 | ✅ PASS | 8 步，768p 66s/条，480p 22s |
| fl2va 首帧 | ✅ PASS | 1 图 |
| ref2va 参考图 | ✅ PASS | 1-9 图 |
| ref2va 音频参考 | ✅ PASS（自定义扩展） | 1-3 段，任意采样率，必须搭配 ≥1 图 |
| Extend 续接 | ✅ PASS（P1/P2） | latent 级，window_frames 39/90 |
| 视频参考（≤3） | ❌ 未做 | 需移植 conditioner 视频帧呈现 |

## 自定义扩展清单（相对官方 Sol-H3）
1. **音频参考**：`bridge.py`（audios 透传，注意模块级 generate 包装函数也要改）、`hyperflow_worker.py`（local_conditioning 读 audios + 构建 MiniMaxH3AudioReference + `_audio_reference_32k` PyAV 重采样 + audio VAE 上卡）、`comfy_nodes.py`（audio 输入）、`web/studio.js`（音频上传 UI）
2. **AdaLN 音频行**：带音频参考的 ref2va 每个 fl2va key 有 "+ (1.0, 1.0)" 变体；两表各 +8 行，`p0f_light.py` 从 BF16 adaln_proj 流式生成（轻量：只建 time embedder 不加载 DiT——全量引擎方案 OOM）
3. **参考图上限 4→9**：`web/studio.js` 的 `limit()`
4. **图片/音频分区上传**：Reference 模式两个独立上传区，图1…图N / 音1…音3 编号
5. **续接 Extend**：引擎 + SolH3Extend 节点 + Studio Extend 模式（session.safetensors 链式传递）
6. 源码旁有 `.bak-audio` 备份可对照 diff

## 关键教训（排障结论）
1. **旧 FP8 profile OOM 根因**：默认 `H3 FP8` profile 运行时现场把 62GB BF16 转 FP8（峰值 93GB > 88GiB）→ cgroup oom_kill（SIGKILL 无 traceback）。官方 one-stage 路线先离线分片构建 FP8（内存安全），运行时直接加载。
2. **构建链顺序**：convert_fp8.py 不带 --drop-adaln（建表要对照 adaln_proj 逐位校验）→ build_adaln_table ×2 → merge → 再 convert --drop-adaln 瘦身。
3. **hyperflow deps 注入**：worker 用 `sys.path.insert(deps)` 后 `import hyperflow_h3`，deps 必须是 **HyperFlow 父目录**（含 hyperflow_h3/ 与 sol_hyperflow/）；build_adaln_table 在当前进程跑，还需 `PYTHONPATH=<HyperFlow父目录>`。
4. **ComfyUI 版 HyperFlow LoRA 不能用于 convert_fp8.py**（key 命名不同：blocks.N.attn.qkv_proj 融合 vs 官方 transformer_blocks.N.attn.to_q 独立），必须用 ModelScope `videorebirth/hyperflow` 原生版（2.8GB，lora_alpha=rank=256）。
5. **SageAttention 2.2.0**：PyPI 只有 1.0.6，须源码编译（需 g++；TORCH_CUDA_ARCH_LIST="12.0"，CUDA_HOME=/usr/local/cuda-13.2）。
6. **Sol-Attn 0.5.0**：从 NVlabs/Sana@9bfca5c4 的 `techniques/sparse_backends` 子目录源码编译。
7. UPFS（/model）读取极快：62GB 分片转换仅 ~50s-4min；构建中间文件一律写本地盘。
8. 容器无 swapon 权限，内存不足不能靠 swap 解决。
9. torchaudio 与 torch 2.13 无匹配轮子 → 用 PyAV AudioResampler 替代。
10. `pkill -f` 模式会匹配到自己（命令串含脚本名）→ 用 `[x]` 括号技巧或分步执行。
11. AdaLN 表 key 是精确浮点比较；改 schedule/LoRA/参考规则后必须重跑补表。

## 2026-10-25 追加：引擎扩展补丁化（setup 的第 4 步）

### 发现的缺口
`setup.sh` 原本**从不克隆引擎，也从不把我们的扩展打上去**：它直接 `cd` 进
`/root/Sana/models/minimax_h3/Sol-H3-ComfyUI` 用人家的 `scripts/hyperflow/*.py` 构建 FP8，
再把这个目录软链进 ComfyUI。而引擎本体是上游 NVlabs/Sana 的一部分——照 README 装完的人
拿到的是**上游原版引擎**：Studio 的「续接来源」下拉没了、节点图的参考素材口没了、
`output_dir`/`filename` 也没了，等于我们的扩展全部失效，而且**不报任何错**。

### 做法
我们的改动一直落在 `/root/Sana` 这个克隆的**工作区**里（未提交），所以它天然就是一份 diff：

```bash
cd /root/Sana
git add -N models/minimax_h3/Sol-H3-ComfyUI/sol_h3_engine/extend.py   # 新增文件也要进 diff
git diff --relative=models/minimax_h3/Sol-H3-ComfyUI -- models/minimax_h3/Sol-H3-ComfyUI \
  > setup/patches/0001-sol-h3-engine-extensions.patch
```

- `setup/get-engine.sh`：取上游引擎（浅克隆 NVlabs/Sana，数百 MB）
- `setup/apply-engine-patches.sh`：**备份 → 打补丁 → 校验**，幂等，`--revert` 可还原
- `setup/patches/orig/hyperflow_worker.py.orig`：上游原始 `hyperflow_worker.py` 备份
- `setup.sh` 新增第 4 步（**在 FP8 构建之前**——构建脚本本身也在补丁里）

### 关键教训
12. **「装了仓库」不等于「引擎可用」**：只要引擎本体来自上游，就必须有一条 setup 步骤
    负责把扩展落上去；否则失败是**静默**的（功能缺失，不报错）。
13. 补丁必须在**构建之前**打：`scripts/hyperflow/convert_ref.py` 也在补丁里（AHEAD 2→1），
    晚打一步，FP8 参考图 DiT 就用错行偏移构建。
14. 补丁要先试打再落盘（`patch --dry-run`），对不上就**报错停下**——不能硬打，
    更不能留一个半打的状态。反向试打能过 = 已经打过 = 幂等跳过。
15. 校验标记要用**只有我们才有**的字符串：`output_dir` 会命中上游的
    `get_output_directory()`，还原后误报"标记仍在"。
16. `git diff --relative=<dir>` 让补丁路径相对引擎目录，`patch -p1` 直接可用；
    新增文件要先 `git add -N`（intent-to-add）才会进 diff。

## 2026-10-25 追加：条件冻结成产物 + 编码条件也存成产物

### 两层冻结
1. **素材字节**（上一轮做的）：每段把参考图/音频复制进 `request-NNNNN/refs/`，
   继承读副本 → 关机重启、原始文件被删都不影响
2. **编码后的潜变量**（这一轮）：ref2va 片段把 VAE 编码好的参考潜变量存成
   `refs/conditioning.safetensors`，续接直接复用 → 即使 VAE 或编码代码换了，
   这条链仍用当初那一版条件，不会"重新编码"而悄悄漂移

### 实现要点
- diffusers 的缝：`MiniMaxH3Ref2VAReferenceEncoderStep` 产出 `condition_latents`
  （每个参考图一个 `(1,C,1,h,w)`）与 `audio_condition_latents`（每个带声参考一个
  `(n*2,32)`）——钩在这个类的 `__call__` 上，有产物就注入、没有就照旧编码
- **注意 `get_block_state` 不是共享对象**：它按每个 block 自己声明的输入重建一个
  局部视图，编码步骤的产物要经**共享的 `PipelineState`** 取（`state.condition_latents`），
  钩子里读 `block_state.condition_latents` 会拿到 None（踩过一次）
- 注入时要同时设置 `condition_latents` 与 `audio_condition_latents` 再
  `set_block_state`（该步骤的 intermediate_outputs 就这两个，少一个会抛）
- 产物与素材的绑定用**内容摘要**（media digest），不含路径 → 同一份素材冻结到两段
  不同目录里仍认得出是同一份条件

### 验证
- 单元测试 33 passed / 84 subtests（新增 ConditionArtifact 四例）
- mock 注入：注入值与存储值逐位相同；参考数量对不上时如实回退并写明原因
- 真机 E2E：ref2va 片段存下 `condition_00 (1,24,1,128,228)` + `audio_00 (414,32)`
  （2.85MB）；继承续接 `reused` 命中父段产物；换参考那次 `artifact=null`（正确）
- 父段与续接段产物**逐位相同**（identical: True）
- 潜变量层面铁证：子段头部 12 行潜变量 = 父段尾部 12 行（逐位相同），音频 65 行同理
  → 尾窗钉得死死的。MP4 帧差那个自造指标（15.33）是因为 MP4 是 121 帧而潜空间
  覆盖 124 帧，比较时错位了 2–3 帧，不能当真

## 2026-10-25 追加：非续接模式不存条件产物

### 用户要求
「非续接的模式下，不要存这些产物」——普通生成（Text / First frame / Reference）不应落下
冻结的参考副本 `refs/ref_01.png` / `refs/audio_01.wav`，也不应写 `refs/conditioning.safetensors`。
这些产物只该在**链上（Extend）片段**产生，供下一次续接复用。

### 实现
`hyperflow_worker.py` 里用 `chain = bool(request.get("extend"))` 门控：
- 链上 → `freeze_references(request_dir, images, audios)`：复制参考图/音频进 `request-NNNNN/refs/`，
  并在 after_denoise 钩子里把编码后的参考潜变量存成 `conditioning.safetensors`
- 非链 → `reference_records(images, audios)`：只把「来源路径 + 指纹」记进 session sidecar 的
  `references` 字段，**不复制、不写编码产物**

`condition_artifact.py` 的 `capture()` 同样被 `if _EXTEND.get("chain")` 包住，非链时不调用。

### 顺手修掉的两个问题
1. **worker 的 SyntaxError**：944–975 行的 generate/response 块缩进少了一层（12 而非 16），
   导致 `try` 在 `except` 之前就结束、`py_compile` 报 "expected 'except' or 'finally' block"。
   整段 +4 空格修复，`py_compile` 通过。
2. **`condition_artifact.py` 的 4 处调试埋点**：`open("/root/hook_ca.log","a").write(...)` 全部移除
   （HOOK path / injected / fallback / unreadable 四句），`grep -c hook_ca.log` 归零。

### 验证
- 单元测试 **23 passed**（GPU 无关）：含 `StandaloneTakeLeavesNoCopy`、`FreezingCopiesTheMediaIntoTheTake`、
  `FrozenConditioning` 继承用例、`ConditionArtifact` 复用用例——确定性证明非链不落副本、链上冻结+继承均正确。
- 全视频 `e2e_boundary.py`（standalone 不落产物 / 链上冻结 / 继承复用三步）因服务器显存
  **VAE decode 阶段 OOM** 未跑完；该 OOM 与产物存储无关（冻结发生在成功渲染之后），可在显存充足环境补跑。
- 补丁重生成（含 worker 缩进修复 + 条件产物无调试）：对干净上游 commit `41866cd` 打补丁后，
  与实机引擎 `diff -rq` **逐文件一致（空）**。

### 提交
- 待 push：`setup/patches/0001-sol-h3-engine-extensions.patch`（重打）+ 引擎 4 个改动文件
  （`comfy_nodes.py` / `extend.py` / `hyperflow_worker.py` / `tests/test_studio.py`）+ `docs/USAGE.md`
  / `docs/deploy-log.md` 的非链说明。远端 `git@github.com:Dean-ZhenYao-Wang/sol-h3-deploy.git`，
  push 在**服务器**上执行（服务器有 GitHub ed25519 密钥）。
