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
