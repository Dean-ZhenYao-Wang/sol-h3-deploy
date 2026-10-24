# sol-h3-deploy

> **[English](README_EN.md)** · [中文](#sol-h3-deploy)

NVlabs Sol-H3（MiniMax-H3）在单卡 RTX 5090 上的完整部署：8 步加速出片（768p 66s/条，480p 22s），支持文生视频/首帧/参考图（9图+3音频）/latent 级续接，附 obvpm 导演台搭配方案。

## 仓库结构

```
sol-h3-deploy/
├── README.md                    ← 本文件
├── README_EN.md                 ← English version
├── setup.sh                     ← 一键部署（环境→编译→权重→FP8构建→配置→ComfyUI）
├── LICENSE                      ← MIT
├── .gitignore
├── engine/
│   └── Sol-H3-ComfyUI/          ← 修改过的引擎源码（含自定义扩展，.bak-audio 备份可 diff）
├── config/
│   ├── sol-h3-settings/         ← configure 的 manifest（reuse-paths 输入）
│   └── sol-h3-settings-final/   ← 最终 profile 配置（SOL_ENGINE_CONFIG 指向它）
├── scripts/
│   ├── launch_comfy.sh          ← ComfyUI 启动器
│   ├── run_onestage.sh          ← CLI 测试（t2va）
│   ├── run_ref.sh               ← ref2va 转换辅助
│   └── p0f_light.py             ← AdaLN 音频参考行补表工具
├── setup/
│   ├── install-custom-nodes.sh  ← 7 个第三方节点包（obvpm 全家桶等）
│   └── link-models.sh           ← ComfyUI 模型软链重建
└── docs/
    ├── USAGE.md                 ← 完整使用手册（模式/节点连线/参数表/obvpm 搭配/排障）
    └── deploy-log.md            ← 部署日志与全部关键教训
```

## 快速开始

```bash
git clone https://github.com/Dean-ZhenYao-Wang/sol-h3-deploy.git && cd sol-h3-deploy
# 一键部署
bash setup.sh
# 或分步：先装环境/节点，权重与构建另行准备时
bash setup.sh --skip-weights --skip-build --skip-comfyui
```

**权重不在本仓库**（51G，GitHub 装不下）：由 `setup.sh` 从 ModelScope/HuggingFace 下载官方权重并现场构建 FP8；想跳过构建可直接下载预构建 FP8（见下）。

### 分步耗时（setup.sh 全流程约 3-4 小时）

| 步骤 | 内容 | 耗时 |
|---|---|---|
| 1 | Python 环境（conda py312 + torch 2.13.0+cu132 等） | ~10 分钟 |
| 2 | 源码编译 SageAttention 2.2 + Sol-Attn 0.5（SM120） | ~20-40 分钟 |
| 3 | 下载官方权重（75G，ModelScope） | 10MB/s ≈ 2 小时（机房内网更快） |
| 4 | 构建 FP8（主 DiT + 参考图 DiT + AdaLN 查表，内存安全的分片流式路线） | ~2-2.5 小时 |
| 5 | configure.py 生成 profile | ~1 分钟 |
| 6 | ComfyUI + 7 个节点包 + 模型软链 | ~10 分钟 |

**跳过构建的快速路线**（用 HF 预构建 FP8，45G @10MB/s ≈ 75 分钟 + 步骤 1/2/5/6 ≈ 1 小时）→ **总计约 2 小时**：

```bash
hf download ZhenYaoWang/sol-h3-fp8-5090 --local-dir hyperflow
bash setup.sh --skip-weights --skip-build --skip-comfyui
```

## 预构建 FP8 权重（可选加速）

`setup.sh --skip-build` 时，把以下目录放到对应位置即可（总计约 51G）：

| 目录 | 内容 | 获取 |
|---|---|---|
| `hyperflow/transformer_hf8/` | 主 DiT FP8 + adaln_table.pt（含音频行） | HuggingFace [`ZhenYaoWang/sol-h3-fp8-5090`](https://huggingface.co/ZhenYaoWang/sol-h3-fp8-5090) |
| `hyperflow/transformer_ref_hf8/` | 参考图 DiT FP8 + adaln_table.pt | 同上 |
| `hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors` | HyperFlow LoRA | ModelScope `videorebirth/hyperflow` |
| `sol-h3-vae/` | fast-VAE 权重 + code | ModelScope `Efficient-Large-Model/Sol-H3-VAE` |

上传预构建权重到 HuggingFace（40G，数小时，建议 tmux 内执行防断线）：

```bash
hf auth login                  # token: https://huggingface.co/settings/tokens（Write 权限）
hf upload ZhenYaoWang/sol-h3-fp8-5090 /root/hyperflow/transformer_hf8 .
hf upload ZhenYaoWang/sol-h3-fp8-5090 /root/hyperflow/transformer_ref_hf8 .
```

使用者下载预构建权重（配合 `setup.sh --skip-build`）：

```bash
hf download ZhenYaoWang/sol-h3-fp8-5090 --local-dir hyperflow
```

## 硬件要求

- **GPU**：RTX 5090 32GB（SM120；Sol-Attn/SageAttention 按 12.0 编译）
- **内存**：≥88GiB 可用（FP8 构建期峰值 ~93GB 会 OOM，务必走离线构建路线）
- **磁盘**：≥80G（权重 75G + FP8 51G + 环境 13G）
- **CUDA**：13.2（torch 2.13.0+cu132）

## 文档

- **[docs/USAGE.md](docs/USAGE.md)** —— 使用手册：Studio 四模式、Reference 双上传区（图1…图N / 音1…音3）、三个节点的参数表与连线图、obvpm 导演台搭配、排障表
- **[docs/deploy-log.md](docs/deploy-log.md)** —— 部署日志：架构、功能矩阵、11 条关键教训（OOM 根因、构建链顺序、LoRA 格式陷阱等）

## 已知限制

- 视频参考（ref2va ≤3 段）未实现（conditioner 视频帧呈现未移植）
- AdaLN 表按精确 (t, endpoint) 对匹配；改 LoRA/调度/参考规则后重跑 `scripts/p0f_light.py`
- obvpm 的 latent 续接只认它自己的 `.mctx.safetensors`；要 latent 级无缝续接请全程用 Sol-H3 Extend

## License

[MIT](LICENSE) © 2026 王振耀 (Dean-ZhenYao-Wang)
