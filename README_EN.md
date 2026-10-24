# sol-h3-deploy (English)

> **[中文说明](README.md)** · English

A complete single-GPU (RTX 5090) deployment of NVlabs Sol-H3 (MiniMax-H3): 8-step accelerated video generation (~66 s/clip at 768p, ~22 s at 480p), with text-to-video, first-frame, reference (9 images + 3 audios) and latent-grade continuation, plus a recipe for pairing with the obvpm director's-timeline node pack.

## Repository layout

```
sol-h3-deploy/
├── README.md / README_EN.md     ← This document (EN)
├── setup.sh                     ← One-command deploy (env → compile → weights → engine+patch → FP8 → config → ComfyUI)
├── LICENSE                      ← MIT
├── .gitignore
├── engine/
│   └── Sol-H3-ComfyUI/          ← Patched engine source (custom extensions; .bak-audio backups for diffing)
├── config/
│   ├── sol-h3-settings/         ← configure manifest (reuse-paths input)
│   └── sol-h3-settings-final/   ← Final profile config (SOL_ENGINE_CONFIG points here)
├── scripts/
│   ├── launch_comfy.sh          ← ComfyUI launcher
│   ├── run_onestage.sh          ← CLI smoke test (t2va)
│   ├── run_ref.sh               ← ref2va conversion helper
│   └── p0f_light.py             ← AdaLN audio-reference row rebuilder
├── setup/
│   ├── install-custom-nodes.sh  ← 7 third-party node packs (obvpm family etc.)
│   ├── get-engine.sh            ← fetch the upstream engine (Sol-H3-ComfyUI from NVlabs/Sana)
│   ├── apply-engine-patches.sh  ← back up + apply this repo's engine extension patches (idempotent, --revert)
│   └── patches/
│       ├── 0001-sol-h3-engine-extensions.patch  ← every engine change vs. upstream
│       ├── orig/hyperflow_worker.py.orig        ← pristine upstream hyperflow_worker.py
│       └── README.md
│   └── link-models.sh           ← Recreate ComfyUI model symlinks
└── docs/
    ├── USAGE.md                 ← Full manual (Chinese): modes, node wiring, params, obvpm pairing, troubleshooting
    └── deploy-log.md            ← Deploy log (Chinese): architecture, feature matrix, 11 hard-won lessons
```

## Quick start

```bash
git clone https://github.com/Dean-ZhenYao-Wang/sol-h3-deploy.git && cd sol-h3-deploy
# one-command deploy
bash setup.sh
# or staged: install env/nodes only, prepare weights separately
bash setup.sh --skip-weights --skip-build --skip-comfyui
```

**Weights are not in this repo** (51 GB — GitHub can't host them): `setup.sh` downloads official weights from ModelScope/HuggingFace and builds the FP8 checkpoints on the fly; prebuilt FP8 can be downloaded instead (below).

### Step timings (full `setup.sh` ≈ 3-4 h)

| Step | What | Time |
|---|---|---|
| 1 | Python env (conda py312 + torch 2.13.0+cu132, etc.) | ~10 min |
| 2 | Compile SageAttention 2.2 + Sol-Attn 0.5 from source (SM120) | ~20-40 min |
| 3 | Download official weights (75 GB, ModelScope) | ~2 h at 10 MB/s (much faster on intranet) |
| 4 | Fetch the upstream engine (NVlabs/Sana) + apply this repo's engine extension patches | ~3-5 min (few-hundred-MB clone) |
| 5 | Build FP8 (main DiT + ref DiT + AdaLN tables; memory-safe sharded streaming) | ~2-2.5 h |
| 6 | configure.py generates the profile | ~1 min |
| 7 | ComfyUI + 7 node packs + model symlinks | ~10 min |

**Fast path with prebuilt FP8** (45 GB from HF at 10 MB/s ≈ 75 min + steps 1/2/5/6 ≈ 1 h) → **~2 h total**:

```bash
hf download ZhenYaoWang/sol-h3-fp8-5090 --local-dir hyperflow
bash setup.sh --skip-weights --skip-build --skip-comfyui
```

### The engine comes from upstream; our extensions travel as patches

The engine itself is upstream [NVlabs/Sana](https://github.com/NVlabs/Sana)'s
`models/minimax_h3/Sol-H3-ComfyUI`. Everything we added to it (Extend continuation,
native reference image/audio sockets, `output_dir`/`filename`, conditions frozen as
artifacts, take-based library, Studio's "continue from" picker) ships as a **patch**
under `setup/patches/`, applied by `setup/apply-engine-patches.sh`:

```bash
bash setup/get-engine.sh                # fetch the upstream engine (if you don't have one)
bash setup/apply-engine-patches.sh      # back up → patch → verify (idempotent)
bash setup/apply-engine-patches.sh --revert   # restore the upstream version
```

`setup.sh` step 4 does both automatically, **before** the FP8 build (the build script
is part of the patch too). Every touched file is backed up as `<file>.bak-<timestamp>`.
If the patch no longer applies (upstream moved on) the script **stops with an error**
instead of forcing it; in that case just overwrite the engine directory with this
repo's `engine/Sol-H3-ComfyUI/` (the fully patched engine, 4.7 MB).

## Prebuilt FP8 weights (optional)

With `setup.sh --skip-build`, place these directories (≈51 GB total):

| Path | Contents | Source |
|---|---|---|
| `hyperflow/transformer_hf8/` | Main DiT FP8 + adaln_table.pt (incl. audio rows) | HuggingFace [`ZhenYaoWang/sol-h3-fp8-5090`](https://huggingface.co/ZhenYaoWang/sol-h3-fp8-5090) |
| `hyperflow/transformer_ref_hf8/` | Reference DiT FP8 + adaln_table.pt | same |
| `hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors` | HyperFlow LoRA | ModelScope `videorebirth/hyperflow` |
| `sol-h3-vae/` | fast-VAE weights + code | ModelScope `Efficient-Large-Model/Sol-H3-VAE` |

> **Warning**: the third argument of `hf upload` is the path *inside the repo* and must be a directory name. Using `.` flattens the local directory into the repo root — both DiT directories contain identically named files (config.json / 14 shards / adaln_table.pt), so the second upload silently overwrites the first.

Upload prebuilt weights to HuggingFace (40 GB, hours — run inside tmux):

```bash
hf auth login                  # token: https://huggingface.co/settings/tokens (Write scope)
hf upload ZhenYaoWang/sol-h3-fp8-5090 /root/hyperflow/transformer_hf8 transformer_hf8
hf upload ZhenYaoWang/sol-h3-fp8-5090 /root/hyperflow/transformer_ref_hf8 transformer_ref_hf8
```

Move local backups out of the upload directories first:

```bash
mkdir -p /root/hyperflow/table-backups
mv /root/hyperflow/transformer_hf8/*.bak-audio /root/hyperflow/table-backups/ 2>/dev/null
mv /root/hyperflow/transformer_ref_hf8/*.bak-audio /root/hyperflow/table-backups/ 2>/dev/null
```

Download prebuilt weights (pairs with `setup.sh --skip-build`):

```bash
hf download ZhenYaoWang/sol-h3-fp8-5090 --local-dir hyperflow
```

## Hardware requirements

- **GPU**: RTX 5090 32 GB (SM120; Sol-Attn/SageAttention compiled for 12.0)
- **RAM**: ≥88 GiB available (FP8 build peaks at ~93 GB and OOMs — always use the offline build route)
- **Disk**: ≥80 GB (75 GB weights + 51 GB FP8 + 13 GB envs)
- **CUDA**: 13.2 (torch 2.13.0+cu132)

## Docs

- **[docs/USAGE.md](docs/USAGE.md)** — full manual (Chinese): four Studio modes, split image/audio upload zones (图1…图N / 音1…音3), node parameter tables and wiring diagrams, obvpm pairing, troubleshooting
- **[docs/deploy-log.md](docs/deploy-log.md)** — deploy log (Chinese): architecture, feature matrix, 11 key lessons (OOM root cause, build-chain order, LoRA format traps, etc.)

## Known limitations

- Video references (ref2va ≤3) not implemented (conditioner video-frame presentation not ported)
- AdaLN tables match on exact (t, endpoint) pairs; after changing LoRA/schedule/reference rules rerun `scripts/p0f_light.py`
- obvpm's latent continuation only reads its own `.mctx.safetensors` sidecars; for latent-grade seamless continuation stay on Sol-H3 Extend end to end

## LoRA & AdaLN table maintenance

**The LoRA *is* the acceleration**: the HyperFlow 8-step LoRA (2.7 GB) distills 50-step sampling into 8 steps - 66 s/clip vs ~16 min/clip. Not optional; in daily use one LoRA serves forever with zero upkeep.

**AdaLN lookup tables**: the FP8 DiT drops 26 GB of `adaln_proj` weights and reads precomputed rows keyed by exact (timestep, endpoint) pair sets. Consequences:

| Change | Rebuild scope | Time |
|---|---|---|
| Nothing | Zero - weights + tables live in `/root/hyperflow/` permanently | - |
| New reference rules (e.g. video refs) | Only add new pair-set rows (`scripts/p0f_light.py` pattern) | ~20 min |
| Steps / schedule change | `build_adaln_table` + merge + `--drop-adaln` | ~1.5 h |
| Swap / add a LoRA | Full chain below | ~3-4 h |

Full rebuild chain for a new LoRA (same as `setup.sh` step 4, change `--hyperflow-weights`):

```bash
convert_fp8.py <BF16 transformer> <new LoRA> transformer_full          # keep adaln_proj
save_payload.py x2 -> build_adaln_table.py -> merge_adaln_tables.py    # temb from the new LoRA
convert_fp8.py --drop-adaln -> transformer_hf8
convert_ref.py -> transformer_ref_hf8
python scripts/p0f_light.py                                            # audio-reference rows
```

**Want a style/character LoRA on top?** Either merge it offline into the FP8 weights together with the HyperFlow LoRA (one rebuild, then zero runtime cost - recommended), or use the official ComfyUI node stack for occasional runtime LoRA loading (2-4 min/clip). The engine does **not** support runtime LoRA injection - offline merging is the premise for dropping `adaln_proj` safely.

## License

[MIT](LICENSE) © 2026 王振耀 (Dean-ZhenYao-Wang)
