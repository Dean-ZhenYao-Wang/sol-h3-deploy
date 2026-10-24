"""P0f (lightweight): add audio-variant AdaLN rows to both tables WITHOUT loading the DiT.

Replicates convert_ref.py's exact recipe: time embedder from the FP8 dir (LoRA merged via
TwoTimeEmbedder), temb per variant key, official AdaLN call per block with BF16 adaln_proj
streamed from /model. No transformer weights are ever loaded.
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

import os
os.environ.setdefault("SOL_ENGINE_CONFIG", "/root/sol-h3-settings-final/config.json")
sys.path.insert(0, "/root/Sana/models/minimax_h3/Sol-H3-ComfyUI")
sys.path.insert(0, "/root/Sana/models/minimax_h3/Sol-H3-ComfyUI/sol_h3_engine")
sys.path.insert(0, "/root/Sana/models/minimax_h3/HyperFlow")

from diffusers import MiniMaxH3Transformer3DModel  # noqa: E402
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AdaLayerNormModulation  # noqa: E402
from hyperflow_h3 import TwoTimeEmbedder, read_metadata  # noqa: E402

OUT = Path("/root/ref_audio_test")
BF16 = Path("/model/ModelScope/MiniMax/MiniMax-H3")
LORA = Path("/root/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors")
dev = torch.device("cuda")
AUDIO_TAIL = (1.0, 1.0)

lora = safe_open(str(LORA), "pt")


def add_variants(tag, fp8_dir, bf16_dir, table_path):
    fp8_dir, bf16_dir, table_path = Path(fp8_dir), Path(bf16_dir), Path(table_path)
    config = json.loads((fp8_dir / "config.json").read_text())
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(config)
    first_shard = sorted(set(json.loads(
        (fp8_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"].values()))[0]
    with safe_open(str(fp8_dir / first_shard), "pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys() if k.startswith("time_embedder.")}
    model.time_embedder.load_state_dict({k.split(".", 1)[1]: v for k, v in state.items()}, assign=True)
    metadata = read_metadata(LORA)
    wrapper = TwoTimeEmbedder.wrap(model, metadata.gate)
    with torch.no_grad():
        for prefix, module in (("time_embedder", wrapper.base), ("endpoint_time_embedder", wrapper.endpoint)):
            for lin in ("linear_1", "linear_2"):
                layer = getattr(module, lin)
                a = lora.get_tensor(f"transformer.{prefix}.{lin}.lora_A.weight").float()
                b = lora.get_tensor(f"transformer.{prefix}.{lin}.lora_B.weight").float()
                layer.weight.copy_((layer.weight.float() + metadata.lora_alpha / metadata.lora_rank * (b @ a))
                                   .to(layer.weight.dtype))
    embedder = wrapper.base
    wrapper_endpoint = wrapper
    time_proj = model.time_proj.to(dev)
    embedder.to(dev)
    time_proj.to(dev)

    bf16_index = json.loads((bf16_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    n_blocks = len(model.transformer_blocks)
    hidden = model.transformer_blocks[0].adaln_proj.hidden_size

    table = torch.load(table_path, map_location="cpu", weights_only=True)
    have = {tuple(map(tuple, k)) for k in table["keys"]}
    todo = []
    for k in table["keys"]:
        base = tuple(map(tuple, k))
        if len(base) == 3 and base[-1] == (0.9990000128746033, 0.9990000128746033):
            variant = base + (AUDIO_TAIL,)
            if variant not in have:
                todo.append(variant)
    print(f"[{tag}] variants to add: {len(todo)}", flush=True)
    if not todo:
        return
    shutil.copyfile(table_path, str(table_path) + ".bak-audio")
    handles = {}
    with torch.no_grad():
        for vi, variant in enumerate(todo):
            t = torch.tensor([p[0] for p in variant], dtype=torch.float32, device=dev)
            r = torch.tensor([p[1] for p in variant], dtype=torch.float32, device=dev)
            with wrapper_endpoint.endpoint_context(r):
                temb = wrapper(time_proj(t).to(next(wrapper.parameters()).dtype))
            rows, exact = [], True
            for i in range(n_blocks):
                module = MiniMaxH3AdaLayerNormModulation(temb.shape[-1], hidden)
                for leaf in ("weight", "bias"):
                    name = f"transformer_blocks.{i}.adaln_proj.linear.{leaf}"
                    f = bf16_index[name]
                    if f not in handles:
                        handles[f] = safe_open(str(bf16_dir / f), "pt")
                    getattr(module.linear, leaf).data = handles[f].get_tensor(name).to(dev)
                out_i = torch.cat(module(temb), dim=-1).reshape(temb.shape[0], -1)
                rows.append(out_i.cpu())
                if i == 0 and vi == 0:
                    again = torch.nn.functional.linear(torch.nn.functional.silu(temb).to(module.linear.weight.dtype),
                                                       module.linear.weight, module.linear.bias)
                    exact = torch.equal(again, out_i)
                del module
            table["keys"].append([list(p) for p in variant])
            for i in range(n_blocks):
                table["tables"][i] = torch.cat([table["tables"][i], rows[i]], dim=0)
            torch.save(table, table_path)
            print(f"[{tag}] merged {vi + 1}/{len(todo)} exact={exact}", flush=True)


add_variants("ref", "/root/hyperflow/transformer_ref_hf8", BF16 / "transformer_ref",
             "/root/hyperflow/transformer_ref_hf8/adaln_table.pt")
add_variants("main", "/root/hyperflow/transformer_hf8", BF16 / "transformer",
             "/root/hyperflow/transformer_hf8/adaln_table.pt")
print("P0F DONE", flush=True)
