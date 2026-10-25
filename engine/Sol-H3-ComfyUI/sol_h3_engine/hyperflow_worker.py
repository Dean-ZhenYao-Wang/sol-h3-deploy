"""HyperFlow-8 MiniMax-H3 on one 32 GB GPU, behind the same JSON-line protocol as worker.py.

Text conditioning reuses the verified NVFP4 Qwen session from the bundled runtime, in a
per-request subprocess (as the existing pipeline does), so the 62 GB BF16 Qwen3-VL is never
loaded. The DiT, VAEs and the official HyperFlow LoRA (PEFT, unmerged) stay resident in this
process: Diffusers >= 0.40 + hyperflow_h3, supplied by the profile's `hyperflow.deps` overlay.
The text-encoder blocks are removed exactly as in HyperFlow's official SplitPipeline; the DiT
streams one block group at a time; decode keeps the official fp16-autocast-over-fp32 recipe.

`t2va` and first-frame `fl2va` share one transformer partition, so one fl2va pipeline serves
both (a text request simply anchors no keyframes). First-frame conditioning runs locally: the
NVFP4 Qwen session presents the keyframe on the generation canvas exactly as the official
conditioner does (same stretch, same `"<Picture 1>: "` label and video-tagged vision block).
A request may instead carry a precomputed `conditioning` payload from the official conditioner.
"""
import json
import os
import sys
import time
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
RUNTIME = ENGINE / "runtimes" / "h3_fp8"
FPS = 24
OUTPUT_FRAMES = 121
CANVAS = {"768p": (768, 1344), "480p": (480, 864)}
PHASES = {}  # seconds per staged encode/decode step of the current request (filled by the wrappers below)


class PinnedModule:
    """Keep a module's weights in pinned host memory and swap it on/off the card by repointing tensors.

    The VAEs never change at inference, so "offload" is only a pointer swap back to the pinned copy, and
    "onload" is one pinned→device copy per tensor (~0.2 s for the 9.8 GB video VAE). That leaves the card
    free for the per-request NVFP4 Qwen process between requests.
    """

    def __init__(self, module):
        import torch
        self.module = module
        self.host = {}
        for name, t in list(module.named_parameters()) + list(module.named_buffers()):
            self.host[name] = t.data.cpu().pin_memory() if not t.data.is_pinned() else t.data
            t.data = self.host[name]
        self.torch = torch
        self.on_gpu = False

    def _tensors(self):
        return list(self.module.named_parameters()) + list(self.module.named_buffers())

    def to_gpu(self):
        if not self.on_gpu:
            for name, t in self._tensors():
                t.data = self.host[name].to("cuda", non_blocking=True)
            self.torch.cuda.synchronize()
            self.on_gpu = True

    def to_host(self):
        if self.on_gpu:
            for name, t in self._tensors():
                t.data = self.host[name]
            self.on_gpu = False


def conditioned_blocks(workflow):
    """HyperFlow blocks without the steps our NVFP4 Qwen conditioning replaces.

    The text encoder always goes; `before_encode` goes for t2va/fl2va (the keyframe canvas is prepared before Qwen)
    but stays for ref2va, where it is MiniMaxH3Ref2VASetupStep: the request plan and normalized references.
    """
    from hyperflow_h3 import hyperflow_blocks
    blocks = hyperflow_blocks(workflow)
    for name in ("text_encoder",) if workflow == "ref2va" else ("before_encode", "text_encoder"):
        if name in blocks.sub_blocks:
            blocks.sub_blocks.pop(name)
    return blocks


def dump_qkv(path, block_index=0, step=0):
    """Diagnostics: save the post-RoPE q/k/v that one block's dense attention receives at one step."""
    import torch
    import diffusers.models.attention_dispatch as dispatch
    from hyperflow_h3.blocks import current_step
    original = dispatch.dispatch_attention_fn
    state = {"layer": -1, "done": False}

    def wrapped(query, key, value, *args, **kwargs):
        if not state["done"] and current_step() == step and query.shape[1] > 4096:
            state["layer"] += 1
            if state["layer"] == block_index:
                torch.save({"q": query.detach().cpu(), "k": key.detach().cpu(), "v": value.detach().cpu()}, path)
                state["done"] = True
        return original(query, key, value, *args, **kwargs)

    import hyperflow_h3.sol_attn as sol
    sol.dispatch_attention_fn = wrapped


def pin_unmatched_group(model):
    """Keep diffusers' block-level "unmatched group" (everything that is not a streamed block: token-refiner
    leftovers, embedders, norm_out, heads, top-level parameters) on the card for good.

    diffusers builds that group with stream=None and non_blocking=False, so every DiT forward moved it with a
    blocking .to() both ways (~2 GB of synchronous device<->host copies per step on H3). Onloading it once and
    turning its onload_/offload_ into no-ops keeps the prefetch chain of the streamed blocks intact.
    """
    from diffusers.hooks.group_offloading import _GROUP_OFFLOADING
    from diffusers.hooks.hooks import HookRegistry
    hook = HookRegistry.check_if_exists_or_initialize(model).get_hook(_GROUP_OFFLOADING)
    group = hook.group
    if group.stream is not None:
        raise RuntimeError("expected diffusers' unmatched group (stream=None) on the top-level DiT")
    group.onload_()
    group.onload_ = lambda: None
    group.offload_ = lambda: None
    return sum(p.numel() * p.element_size() for m in group.modules for p in m.parameters()) + \
        sum(p.numel() * p.element_size() for p in group.parameters)



def _audio_reference_32k(path, vae_rate=32000):
    """MiniMaxH3AudioReference already resampled onto the audio VAE's rate.

    torchaudio is unavailable for this torch build, so decode with PyAV and
    resample with its own AudioResampler; the reference then carries no
    sample_rate (the VAE's own), skipping the official torchaudio resample.
    """
    import av
    import torch
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3AudioReference
    with av.open(str(path)) as container:
        stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="fltp", rate=vae_rate, layout=stream.layout)
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
    if not chunks:
        raise ValueError(f"No audio stream content in {path}")
    waveform = torch.from_numpy(chunks[0] if len(chunks) == 1 else _concat_planar(chunks)).float()
    return MiniMaxH3AudioReference(audio=waveform)


def _concat_planar(chunks):
    import numpy as np
    width = max(c.shape[1] for c in chunks[:-1]) + chunks[-1].shape[1]
    total = sum(c.shape[1] for c in chunks)
    out = np.zeros((chunks[0].shape[0], total), dtype=chunks[0].dtype)
    pos = 0
    for c in chunks:
        out[:, pos:pos + c.shape[1]] = c
        pos += c.shape[1]
    del width
    return out


# ---------------------------------------------------------------------------
# Extend (masked continuation): the four pipeline-block patches
# ---------------------------------------------------------------------------

# Process-global request state for the extend patches. One worker process serves
# one request at a time, so a module-level slot is enough. "window" is the
# context built by extend.build_extend(); "session_target"/"session_meta" make
# every take write its latents sidecar for the next extend in the chain.
_EXTEND = {"window": None, "session_target": None, "session_meta": None, "chain": False}

try:  # package import under ComfyUI, flat import in the worker process
    from . import extend as _extend
except ImportError:  # pragma: no cover - worker process
    import extend as _extend
try:
    from . import condition_artifact as _artifact
except ImportError:  # pragma: no cover - worker process
    import condition_artifact as _artifact


def install_extend_patches():
    """Wrap the four blocks a masked extend needs; idempotent, once per process.

    With ``_EXTEND["window"]`` set, the target's head rows are replaced by the
    source clip's tail window at the conditioning level, pinned there for every
    denoising step and re-imposed after each scheduler step -- ComfyUI core
    PR #15375's per-row mask semantics, ported to the Diffusers pipeline. After
    the loop the window latents are restored to their clean values (so the decode
    and the session sidecar carry the exact source latents) and the take's own
    session sidecar is written.
    """
    if getattr(install_extend_patches, "_solh3_extend", False):
        return
    import torch
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLatentsStep, patchify_video_latents)
    from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3AfterDenoiseStep
    from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopSchedulerStep
    from hyperflow_h3.blocks import HyperFlowSetTimestepsStep
    from hyperflow_h3.schedule import shift_sigmas, endpoints_from_sigmas

    # -- 0. the encoded conditioning artifact -----------------------------------
    # Lets a take continue on the reference latents its parent was built on,
    # instead of encoding the references again.
    _artifact.install_encoder_patch()

    # -- 1. prepare_latents: inject the window at cond strength ------------------
    original = MiniMaxH3PrepareLatentsStep.__call__

    def prepare_latents_extend(self, components, state, _original=original):
        result = _original(self, components, state)
        window = _EXTEND.get("window")
        if window is None:
            return result
        block_state = self.get_block_state(state)
        device = components._execution_device
        patch_size = components.patch_size
        rows_per_frame = ((block_state.latent_height // patch_size[1])
                          * (block_state.latent_width // patch_size[2]))
        window_video_rows = window["window_steps"] * rows_per_frame
        if block_state.latents.shape[0] < window_video_rows:
            raise RuntimeError("extend window (%d rows) exceeds the target's video rows (%d)"
                               % (window_video_rows, block_state.latents.shape[0]))
        # One noise draw from the request generator, after the stream draws: the
        # 0.001 component of the cond-strength value, exactly as the keyframe
        # conditioning noises its anchors.
        noise = randn_tensor(window["window_video"].shape, generator=block_state.generator,
                             device=device, dtype=torch.float32)
        noised = components.scheduler.scale_noise(
            window["window_video"].to(device, torch.float32), components.keyframe_noise_aug, noise)
        cond_rows = patchify_video_latents(noised, patch_size)
        window["cond_video_rows"] = cond_rows
        window["window_video_rows"] = window_video_rows
        audio = block_state.audio_latents
        steps = window["audio_steps"]
        total = block_state.num_audio_latents
        if audio.shape[0] < total + steps:
            raise RuntimeError("extend audio window (%d+%d steps) exceeds the target's audio rows (%d)"
                               % (total, steps, audio.shape[0]))
        ch0 = window["window_audio"][0].t().to(device, audio.dtype).contiguous()
        ch1 = window["window_audio"][1].t().to(device, audio.dtype).contiguous()
        window["audio_rows_ch0"], window["audio_rows_ch1"] = ch0, ch1
        block_state.latents[:window_video_rows] = cond_rows.to(block_state.latents.dtype)
        audio[:steps] = ch0
        audio[total:total + steps] = ch1
        self.set_block_state(state, block_state)
        return result

    MiniMaxH3PrepareLatentsStep.__call__ = prepare_latents_extend

    # -- 2. set_timesteps: pin the window rows at the conditioning level ---------
    original = HyperFlowSetTimestepsStep.__call__

    def set_timesteps_extend(self, components, state, _original=original):
        result = _original(self, components, state)
        window = _EXTEND.get("window")
        # The plan is an intermediate output: read it from the PipelineState, not
        # from get_block_state (which only carries this block's declared inputs).
        plan = state.get("row_timestep_plan")
        if plan:
            head = plan[0]
            _EXTEND["adaln_key"] = tuple(zip(head[0].float().tolist(), head[1].float().tolist()))
        if window is None or not plan or len(plan[0]) != 3:
            return result  # the official (passthrough) plan carries no endpoints
        block_state = self.get_block_state(state)
        device = components._execution_device
        video_sigmas = shift_sigmas(self.sigmas, float(components.scheduler.shift))
        audio_sigmas = shift_sigmas(self.sigmas, float(components.audio_scheduler.shift))
        video_endpoints = endpoints_from_sigmas(video_sigmas)
        audio_endpoints = endpoints_from_sigmas(audio_sigmas)
        video_timesteps = components.scheduler.timesteps.cpu()
        audio_timesteps = components.audio_scheduler.timesteps.cpu()
        patch_size = components.patch_size
        latent_height = state.get("latent_height")
        latent_width = state.get("latent_width")
        rows_per_frame = ((latent_height // patch_size[1]) * (latent_width // patch_size[2]))
        ncv = block_state.num_condition_video_rows
        nca = block_state.num_condition_audio_rows
        window_video_rows = window["window_steps"] * rows_per_frame
        audio_steps = window["audio_steps"]
        total_audio = state.get("num_audio_latents")
        # The window occupies the head of the GENERATED section of each stream:
        # its video rows follow any conditioning rows, its audio rows are the
        # channel-major head of each stereo channel.
        pin_video = block_state.video_indices[ncv:ncv + window_video_rows].tolist()
        pin_audio = (block_state.audio_indices[nca:nca + audio_steps].tolist()
                     + block_state.audio_indices[nca + total_audio:nca + total_audio + audio_steps].tolist())
        window["pin_video_rows"], window["pin_audio_rows"] = pin_video, pin_audio
        new_plan = []
        for step in range(len(plan)):
            triple = _extend.build_extend_row_pairs(
                video_indices=block_state.video_indices,
                audio_indices=block_state.audio_indices,
                num_condition_video_rows=ncv,
                num_condition_audio_rows=nca,
                num_text_tokens=block_state.text_indices.numel(),
                video_timestep=float(video_timesteps[step]),
                video_endpoint=float(video_endpoints[step]),
                audio_timestep=float(audio_timesteps[step]),
                audio_endpoint=float(audio_endpoints[step]),
                condition_video_timestep=max(float(video_timesteps[step]), components.keyframe_noise_aug),
                condition_audio_timestep=1.0,
                pin_video_rows=pin_video, pin_audio_rows=pin_audio)
            new_plan.append(tuple(tensor.to(device) for tensor in triple))
        state.set("row_timestep_plan", new_plan)
        head = new_plan[0]
        _EXTEND["adaln_key"] = tuple(zip(head[0].float().tolist(), head[1].float().tolist()))
        return result

    HyperFlowSetTimestepsStep.__call__ = set_timesteps_extend

    # -- 3. scheduler step: re-impose the window rows after every step -----------
    original = MiniMaxH3LoopSchedulerStep.__call__

    def scheduler_step_extend(self, components, block_state, i, t, _original=original):
        result = _original(self, components, block_state, i, t)
        window = _EXTEND.get("window")
        if window is None or window["cond_video_rows"] is None:
            return result
        ncv = block_state.num_condition_video_rows
        block_state.latents[ncv:ncv + window["window_video_rows"]] = \
            window["cond_video_rows"].to(block_state.latents)
        audio = block_state.audio_latents
        nca = block_state.num_condition_audio_rows
        steps = window["audio_steps"]
        # The reference audio rows (nca) sit in front of the generated ones, so the
        # generated rows per channel are what is left after them -- the same count
        # the row plan pins the window against (num_audio_latents).
        total = (audio.shape[0] - nca) // 2  # channel-major rows: T steps per stereo channel
        audio[nca:nca + steps] = window["audio_rows_ch0"].to(audio)
        audio[nca + total:nca + total + steps] = window["audio_rows_ch1"].to(audio)
        return result

    MiniMaxH3LoopSchedulerStep.__call__ = scheduler_step_extend

    # -- 4. after_denoise: clean window latents + the take's session sidecar -----
    original = MiniMaxH3AfterDenoiseStep.__call__

    def after_denoise_extend(self, components, state, _original=original):
        result = _original(self, components, state)
        window = _EXTEND.get("window")
        block_state = self.get_block_state(state)
        if window is not None:
            block_state.latents[:, :, :window["window_steps"]] = \
                window["window_video"].to(block_state.latents)
            block_state.audio_latents[:, :, :window["audio_steps"]] = \
                window["window_audio"].to(block_state.audio_latents)
        target = _EXTEND.get("session_target")
        if target:
            meta = dict(_EXTEND.get("session_meta") or {})
            meta.setdefault("raw_frames", _extend.pixel_frames(block_state.num_latent_frames))
            meta.setdefault("raw_steps", int(block_state.num_latent_frames))
            meta.setdefault("audio_steps_total", int(block_state.num_audio_latents))
            if window is not None:
                info = window["info"]
                meta.setdefault("pinned_head_frames", info["covered_frames"])
                meta.setdefault("window_frames", info["window_frames"])
                meta.setdefault("delivered_frames", info["delivered_frames"])
                meta.setdefault("handover_frames", info["handover_frames"])
                meta.setdefault("parent_session", window["session"])
                meta.setdefault("parent_id", window["parent_id"])
            else:
                meta.setdefault("pinned_head_frames", 0)
                meta.setdefault("delivered_frames", meta["raw_frames"])
            # The reference latents this take was built on, stored as an artifact
            # beside the take: the next take in the chain continues on exactly
            # these, not on a re-encoding of the same media. Only a take that is
            # part of a chain keeps one.
            record = None
            if _EXTEND.get("chain"):
                media = ""
                try:
                    frozen = json.loads(meta.get("references") or "[]")
                    media = _artifact.media_digest([r.get("path") for r in frozen if r.get("path")])
                except Exception:
                    media = ""
                record = _artifact.capture(state, meta.get("task"), Path(target).parent, media)
                if record:
                    meta["condition_artifact"] = json.dumps(record)
            written = _extend.write_session(target, block_state.latents,
                                            block_state.audio_latents, meta)
            if window is not None:
                window["session_written"] = written
        self.set_block_state(state, block_state)
        return result

    MiniMaxH3AfterDenoiseStep.__call__ = after_denoise_extend
    install_extend_patches._solh3_extend = True


class HyperFlow:
    def __init__(self, options, paths):
        overlay = options["hyperflow"]["deps"]
        sys.path.insert(0, str(Path(overlay).resolve(strict=True)))
        import torch
        from hyperflow_h3 import hyperflow_blocks, load_hyperflow_lora
        self.torch = torch
        self.paths = paths
        self.root = Path(options["work_dir"])
        self.started = time.monotonic()
        self.resident = bool(options["hyperflow"].get("resident"))
        model = paths["h3_model"]
        self.pipes = {}
        self.dit_pinned = {}
        transformer = vae = audio_vae = None
        workflows = list(options["hyperflow"].get("workflows", ["t2va"]))
        # t2va and fl2va read the same `transformer/` partition: build the fl2va pipeline once and
        # route text requests through it instead of holding a second 62 GB copy in host RAM.
        shared = "t2va" in workflows and "fl2va" in workflows and options["hyperflow"].get("share_fl2va_pipe", True)
        if shared:
            workflows.remove("t2va")
        self.options, self.model = options, model
        # swap_ref_dit: transformer_ref is built on the first reference request and swapped with `transformer`
        # (one FP8 DiT pinned in host RAM at a time: two exceed a 90 GB host).
        self.lazy_ref = "ref2va" in workflows and options["hyperflow"].get("swap_ref_dit", False)
        if self.lazy_ref:
            workflows.remove("ref2va")
        self.loaded = "main"
        for workflow in workflows:
            pipe = conditioned_blocks(workflow).init_pipeline(model, components_manager=None)
            name = "transformer_ref" if workflow == "ref2va" else "transformer"
            if options["hyperflow"].get("fp8_dir"):
                others = [n for n, spec in pipe._component_specs.items()
                          if spec.default_creation_method == "from_pretrained" and n not in ("transformer", "transformer_ref")]
                pipe.load_components(names=others, dtype=torch.bfloat16, pretrained_model_name_or_path=model,
                                     local_files_only=True)
                self.install_dit(pipe, workflow)
            else:
                pipe.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=model,
                                     local_files_only=True)
                load_hyperflow_lora(pipe, options["hyperflow"]["weights"])
                self.configure_dit(pipe, workflow, getattr(pipe, name))
            # Share the decoders between workflows instead of holding two copies in host RAM.
            vae = vae or pipe.vae
            audio_vae = audio_vae or pipe.audio_vae
            pipe.audio_vae.set_attention_backend("native")
            # Resident: the whole pipeline lives on the card, so every step resolves its execution
            # device to cuda. Streamed: decoders wait on the host until the DiT has streamed out.
            for part in ("vae", "audio_vae"):
                getattr(pipe, part).to("cuda" if self.resident else "cpu")
            self.pipes[workflow] = pipe
        if shared:
            # Text requests get the t2va blocks over the fl2va pipeline's components (same DiT, decoders,
            # scheduler objects): no second copy of the weights, and the official t2va layout instead of an
            # empty keyframe list through the fl2va one.
            source = self.pipes["fl2va"]
            pipe = conditioned_blocks("t2va").init_pipeline(model, components_manager=None)
            pipe.update_components(**{name: getattr(source, name) for name in pipe._component_specs
                                      if getattr(source, name, None) is not None})
            from hyperflow_h3 import read_metadata
            from hyperflow_h3.blocks import configure_hyperflow_blocks
            configure_hyperflow_blocks(pipe.blocks, read_metadata(Path(options["hyperflow"]["weights"])))
            self.pipes["t2va"] = pipe
        # Decoders live in pinned host memory and visit the card only for encode/decode (resident: always there).
        self.pinned = {} if self.resident else {"vae": PinnedModule(vae), "audio_vae": PinnedModule(audio_vae)}
        self._stage_decoders()
        if os.environ.get("SOLH3_DUMP_QKV"):
            dump_qkv(os.environ["SOLH3_DUMP_QKV"])
        if options["hyperflow"].get("fast_vae"):
            self.load_fast_vae(options["hyperflow"]["fast_vae"])
        install_extend_patches()
        self.load_s = time.monotonic() - self.started

    def install_dit(self, pipe, workflow):
        """Build the FP8 HyperFlow DiT for `workflow` (transformer, or transformer_ref for ref2va) into `pipe`."""
        from hyperflow_h3.blocks import configure_hyperflow_blocks
        from hyperflow_fp8 import build_fp8_transformer
        flow, ref = self.options["hyperflow"], workflow == "ref2va"
        # LoRA merged + W8A8 block linears (convert_fp8.py / convert_ref.py) and the exact AdaLN table.
        dit, metadata = build_fp8_transformer(self.model, flow["ref_fp8_dir" if ref else "fp8_dir"], flow["weights"],
                                              flow.get("ref_adaln_table" if ref else "adaln_table"))
        pipe.update_components(**{"transformer_ref" if ref else "transformer": dit})
        configure_hyperflow_blocks(pipe.blocks, metadata)
        self.configure_dit(pipe, workflow, dit)
        return dit

    def configure_dit(self, pipe, workflow, dit):
        torch, flow = self.torch, self.options["hyperflow"]
        # Dense attention (HyperFlow's dense steps/blocks + the token refiner): SDPA, or SageAttention2++
        # (INT8 QK, FP8 PV with fp16 accumulation; diffusers backend `_sage_qk_int8_pv_fp8_cuda`).
        dit.set_attention_backend({"sage2pp": "_sage_qk_int8_pv_fp8_cuda"}.get(flow.get("dense_attn"), "native"))
        if flow.get("sol_attn"):
            # HyperFlow's own validated Sol-Attn recipe: dense for steps 1-2 and blocks 0-1, tau 1.0, diag.
            from hyperflow_h3 import enable_sol_attention
            enable_sol_attention(pipe)
        if flow.get("compile"):
            # Regional compile: one graph per block kind; attention stays an opaque eager op (hyperflow_compile).
            from hyperflow_compile import install as install_compile
            install_compile(dit)
        if self.resident:
            dit.to("cuda")  # cards with room for the whole BF16 DiT (e.g. 80 GB)
        elif flow.get("dit_onload"):
            # Whole DiT on the card for denoising only (FP8 + AdaLN table ≈ 21 GB fits a 32 GB card with the
            # 33k-token activations), parked in pinned host memory while Qwen encodes and the VAE decodes.
            dit.to("cpu")
            self.dit_pinned[workflow] = PinnedModule(dit)
        else:
            dit.enable_group_offload(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                                     offload_type="block_level", num_blocks_per_group=1,
                                     use_stream=True, record_stream=True)
            if flow.get("pin_unmatched_group"):
                pin_unmatched_group(dit)

    def ensure_dit(self, task):
        """Swap the resident FP8 DiT when a request needs the other checkpoint partition (swap_ref_dit)."""
        if not self.lazy_ref:
            return
        want = "ref" if task == "ref2va" else "main"
        if want == self.loaded:
            return
        import gc
        torch = self.torch
        t0 = time.monotonic()
        main = [p for k, p in self.pipes.items() if k != "ref2va"]
        if want == "ref":
            for pipe in main:
                pipe.update_components(transformer=None)  # drop every reference to the main DiT first
            if "ref2va" not in self.pipes:
                pipe = conditioned_blocks("ref2va").init_pipeline(self.model, components_manager=None)
                source = main[0]
                pipe.update_components(**{n: getattr(source, n) for n in pipe._component_specs
                                          if n not in ("transformer", "transformer_ref") and getattr(source, n, None) is not None})
                pipe.audio_vae.set_attention_backend("native")
                self.pipes["ref2va"] = pipe
            gc.collect(); torch.cuda.empty_cache()
            self.install_dit(self.pipes["ref2va"], "ref2va")
        else:
            self.pipes["ref2va"].update_components(transformer_ref=None)
            gc.collect(); torch.cuda.empty_cache()
            dit = self.install_dit(main[0], "fl2va" if "fl2va" in self.pipes else "t2va")
            for pipe in main[1:]:
                pipe.update_components(transformer=dit)
        self.loaded = want
        PHASES["dit.swap"] = time.monotonic() - t0

    def _stage_decoders(self):
        torch = self.torch
        from diffusers.modular_pipelines.minimax_h3.decoders import (
            MiniMaxH3AudioDecodeStep, MiniMaxH3VideoDecodeStep)
        for cls in (MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep):
            if getattr(cls, "_solh3_staged", False):
                continue
            original = cls.__call__

            def staged(self, components, state, _original=original, _name=cls.__name__, _pinned=self.pinned,
                       _engine=self):
                # Denoising is over: give the card back from a resident DiT, then decode.
                for part in _engine.dit_pinned.values():
                    if part.on_gpu:
                        t0 = time.monotonic()
                        part.to_host()
                        torch.cuda.empty_cache()
                        PHASES["dit.to_host"] = PHASES.get("dit.to_host", 0.0) + time.monotonic() - t0
                dump = os.environ.get("SOLH3_DUMP_LATENTS")
                if dump and _name == "MiniMaxH3VideoDecodeStep":  # diagnostics: the exact latents the decoder sees
                    torch.save(self.get_block_state(state).latents.detach().cpu(), dump)
                dump_audio = os.environ.get("SOLH3_DUMP_AUDIO_LATENTS")
                if dump_audio and _name == "MiniMaxH3AudioDecodeStep":  # diagnostics: the audio latents likewise
                    torch.save(self.get_block_state(state).audio_latents.detach().cpu(), dump_audio)
                if _name == "MiniMaxH3VideoDecodeStep" and _engine.fast_vae is not None:
                    return _engine.fast_video_decode(self, components, state)
                t0 = time.monotonic()
                for part in _pinned.values():
                    part.to_gpu()
                torch.cuda.synchronize()
                PHASES[_name + ".to_cuda"] = PHASES.get(_name + ".to_cuda", 0.0) + time.monotonic() - t0
                t0 = time.monotonic()
                with torch.autocast("cuda", dtype=torch.float16):
                    out = _original(self, components, state)
                torch.cuda.synchronize()
                PHASES[_name] = PHASES.get(_name, 0.0) + time.monotonic() - t0
                return out
            cls.__call__ = staged
            cls._solh3_staged = True
        # Keyframe / reference encoders run before the DiT streams in: borrow the card for the
        # video VAE (official fp32 encode, no autocast) and give it back before denoising.
        from diffusers.modular_pipelines.minimax_h3 import encoders
        for cls in (encoders.MiniMaxH3KeyframeVaeEncoderStep, encoders.MiniMaxH3Ref2VAReferenceEncoderStep):
            if getattr(cls, "_solh3_staged", False):
                continue
            original = cls.__call__

            def encode_staged(self, components, state, _original=original, _pinned=self.pinned, _name=cls.__name__):
                t0 = time.monotonic()
                if "vae" in _pinned:
                    _pinned["vae"].to_gpu()
                if "audio_vae" in _pinned and not _pinned["audio_vae"].on_gpu:
                    _pinned["audio_vae"].to_gpu()  # ref2va reference soundtracks encode on the audio VAE too
                try:
                    return _original(self, components, state)
                finally:
                    torch.cuda.synchronize()
                    PHASES[_name] = PHASES.get(_name, 0.0) + time.monotonic() - t0
                    if "vae" in _pinned:
                        _pinned["vae"].to_host()
                    if "audio_vae" in _pinned and _pinned["audio_vae"].on_gpu:
                        _pinned["audio_vae"].to_host()
                        torch.cuda.empty_cache()
            cls.__call__ = encode_staged
            cls._solh3_staged = True

    fast_vae = None

    def load_fast_vae(self, config):
        """Distilled noncausal decoder (Efficient-Large-Model/Sol-H3-VAE): replaces only the video decode."""
        sys.path.insert(0, config["code"])
        from spark_decoder import SparkFastH3VAEDecoder
        self.fast_vae = SparkFastH3VAEDecoder(config["vae_config"], config["weights"], engine="fast", max_engines=1)
        self.fast_vae.model.to("cpu")
        self.fast_vae_pinned = PinnedModule(self.fast_vae.model)

    def fast_video_decode(self, step, components, state):
        """Same contract as MiniMaxH3VideoDecodeStep: normalized latents in, pixels in [0, 1] out (the student
        denormalizes internally, so the latents are handed over as-is), then the pipeline's own postprocess."""
        torch = self.torch
        block_state = step.get_block_state(state)
        t0 = time.monotonic()
        self.fast_vae_pinned.to_gpu()
        try:
            latents = block_state.latents  # (1, 24, T, H, W), normalized
            # Eager engine per request: a captured CUDA graph would keep the weight addresses of this visit, and
            # the weights go back to pinned host memory afterwards (the decoder alone peaks near 30 GB at 768p).
            from noncausal_fast3 import NonCausalFast3
            _, _, frames, height, width = latents.shape
            engine = NonCausalFast3(self.fast_vae.model, frames, height, width, cuda_graph=False)
            pixels = engine.decode(latents.transpose(1, 2).float().contiguous())
            del engine
            device = pixels.device
            pixel_mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
            pixel_std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
            video = (pixels.float() * pixel_std + pixel_mean).clamp(0, 1)
            block_state.videos = components.video_processor.postprocess_video(video, output_type=block_state.output_type)
            torch.cuda.synchronize()
        finally:
            self.fast_vae_pinned.to_host()
            torch.cuda.empty_cache()
        PHASES["FastVideoDecode"] = time.monotonic() - t0
        step.set_block_state(state, block_state)
        return components, state

    def generate(self, case, payload, out_dir, progress):
        torch = self.torch
        task = case.get("task", "t2va")
        swap_started = time.monotonic()
        self.ensure_dit(task)
        swap_s = time.monotonic() - swap_started
        pipe = self.pipes[task]
        dit = pipe.transformer_ref if task == "ref2va" else pipe.transformer
        steps = [0]

        def count_step(*_):
            # A forward pre-hook must return None: any other value replaces the module's inputs.
            steps[0] += 1
            progress("denoising", steps[0], 8)
        hook = dit.register_forward_pre_hook(count_step)
        # Everything the official conditioner left in the state (keyframes on the canvas, anchors,
        # resolved size, tags) goes to the remaining blocks unchanged, as in SplitPipeline.
        kwargs = {k: v for k, v in payload.items() if k not in ("task", "prompt", "input_spec",
                                                                "prepared_media", "external_anchor_used",
                                                                "input_conditioned")}
        kwargs["prompt_embeds"] = payload["prompt_embeds"].to("cuda", torch.bfloat16)
        kwargs["text_token_tags"] = payload["text_token_tags"].to(torch.long)
        kwargs.setdefault("height", 768)
        kwargs.setdefault("width", 1344)
        if task == "ref2va":
            from diffusers.modular_pipelines.minimax_h3.references import (MiniMaxH3ImageReference,
                                                                           MiniMaxH3AudioReference)
            references = [MiniMaxH3ImageReference.from_file(path) for path in kwargs.pop("reference_images")]
            for audio_path in kwargs.pop("reference_audios", None) or []:
                references.append(_audio_reference_32k(audio_path))
            kwargs["references"] = references
        kwargs.update(num_frames=124, output_type="pt", generator=torch.Generator().manual_seed(case["seed"]))
        PHASES.clear()
        if swap_s > 0.5:
            PHASES["dit.swap"] = swap_s
        parked = self.dit_pinned.get(task) or (self.dit_pinned.get("fl2va") if task == "t2va" else None)
        if parked is not None:
            t0 = time.monotonic()
            parked.to_gpu()  # pinned → card, one copy per request (decode parks it again)
            PHASES["dit.to_gpu"] = time.monotonic() - t0
        try:
            started = time.monotonic()
            result = pipe(**kwargs, output=["videos", "audio", "sampling_rate"])
            torch.cuda.synchronize()
        finally:
            hook.remove()
            t0 = time.monotonic()
            for part in self.pinned.values():
                part.to_host()
            torch.cuda.empty_cache()
            PHASES["decoders.to_cpu"] = time.monotonic() - t0
        # Which precomputed AdaLN conditioning variant this request's row plan hit.
        variant = None
        key = _EXTEND.get("adaln_key")
        if key is not None:
            blocks = getattr(dit, "transformer_blocks", None)
            starts = getattr(blocks[0].adaln_proj, "starts", None) if blocks else None
            if starts is not None:
                variant = ("hit" if tuple(key) in starts else "MISS") + ":%d" % len(key)
        if steps[0] != 8:
            raise RuntimeError("HyperFlow must run exactly 8 DiT forwards, got %d" % steps[0])
        progress("saving", 8, 8)
        video, audio = result["videos"][0], result["audio"][0]
        if not (bool(torch.isfinite(video).all()) and bool(torch.isfinite(audio).all())):
            raise RuntimeError("non-finite HyperFlow output")
        output = Path(out_dir) / "hyperflow.mp4"
        t0 = time.monotonic()
        write_mp4(output, video[:OUTPUT_FRAMES], audio, result["sampling_rate"])
        PHASES["write_mp4"] = time.monotonic() - t0
        # A masked extend trims its pinned head: the take delivers from the
        # handover frame on (obvpm's decode-edge rule), picture and sound together.
        # The full clip stays for seam inspection; the delivered one is what a
        # timeline chain plays.
        delivered = None
        window = _EXTEND.get("window")
        if window is not None:
            handover = int(window["info"]["handover_frames"])
            rate = int(result["sampling_rate"])
            cut = int(round(handover / float(FPS) * rate))
            # Tail-match the delivered sound to exactly the delivered frame count:
            # the engine writes 121 of 124 latent frames while the audio grid keeps
            # all 207 ticks, so a plain head cut would leave ~130 ms of surplus per
            # take -- which compounds down a chain (obvpm's trim rule).
            frames = OUTPUT_FRAMES - handover
            want = int(round(frames / float(FPS) * rate))
            delivered = Path(out_dir) / "hyperflow.delivered.mp4"
            t0 = time.monotonic()
            write_mp4(delivered, video[handover:OUTPUT_FRAMES], audio[:, cut:cut + want], rate)
            PHASES["write_delivered"] = time.monotonic() - t0
        return {"output": str(output), "delivered_output": str(delivered) if delivered else None,
                "generate_s": time.monotonic() - started, "forwards": steps[0],
                "adaln_variant": variant,
                "phases_s": {k: round(v, 2) for k, v in PHASES.items()}}


def write_mp4(path, video, audio, rate):
    import av
    with av.open(str(path), "w") as container:
        vs = container.add_stream("libx264", rate=FPS)
        vs.height, vs.width = video.shape[-2:]
        vs.pix_fmt = "yuv420p"
        vs.options = {"crf": "18", "preset": "veryfast"}  # medium cost ~10 s per 5 s clip at no visible gain
        au = container.add_stream("aac", rate=rate)
        au.layout = "stereo"
        for index, frame in enumerate(video):
            rgb = (frame.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
            packet_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            packet_frame.pts = index
            for packet in vs.encode(packet_frame):
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)
        samples = av.AudioFrame.from_ndarray(audio.float().cpu().numpy(), format="fltp", layout="stereo")
        samples.sample_rate, samples.pts = rate, 0
        for packet in au.encode(samples):
            container.mux(packet)
        for packet in au.encode():
            container.mux(packet)


def local_conditioning(case, request, request_dir, paths, config, Worker, qwen=None):
    """NVFP4 Qwen conditioning for a text, first-frame or reference request, plus the state the rest needs."""
    import torch
    sys.path.insert(0, str(RUNTIME))
    from runtime.qwen_ops.media import load_rgb, prepare_keyframe_image
    mode, images = request.get("mode", "text"), list(request.get("images") or [])
    audios = [a for a in (request.get("audios") or []) if a]
    # 16:9 canvases, both axes multiples of 32 (the VAE's 16 x the 2x2 patch).
    height, width = CANVAS[request.get("resolution", "768p")]
    qcase = dict(case)
    keyframes = []
    if mode == "first_frame":
        if len(images) != 1:
            raise ValueError("First frame mode needs exactly one image")
        # Same canvas and arithmetic as the official ResizeStep: the keyframe is stretched onto
        # the canvas, and those exact pixels feed both Qwen's vision block and the VAE encoder.
        qcase.update(task="fl2va", first_frame=images[0], keyframe_canvas=(height, width))
        keyframes = [prepare_keyframe_image(load_rgb(images[0]), height, width, stretch=True)]
    elif mode == "reference":
        if len(audios) > 3:
            raise ValueError("At most 3 audio references")
        if not 1 <= len(images) <= 9:
            raise ValueError("Reference mode needs 1 to 9 images")
        # Qwen sees each reference at the canvas's pixel budget (Sol-H3 Spark ref2va "match"), not at a 2048 short
        # edge: two 2048-edge images overflow the card beside the 15.7 GB NVFP4 encoder.
        qcase.update(task="ref2va", references=[{"type": "image", "path": path} for path in images],
                     reference_pixel_budget=height * width)
    elif mode != "text":
        raise ValueError("Unknown mode: %s" % mode)
    if qwen is None:  # one process per request (original lifetime)
        qwen = Worker("qwen", request_dir / "qwen-worker", paths, config)
        try:
            receipt = qwen.call(case=qcase, output_root=str(request_dir / "qwen"))
        finally:
            qwen.close()
    else:  # resident process: weights parked in pinned host RAM between requests
        torch.cuda.empty_cache()  # the DiT's cached blocks must not crowd the encoder
        qwen.call(op="onload")
        try:
            receipt = qwen.call(case=qcase, output_root=str(request_dir / "qwen"))
        finally:
            qwen.call(op="offload")
    payload = torch.load(Path(receipt["conditioning_path"]).parent / "conditioning.pt",
                         map_location="cpu", weights_only=True)
    out = {"task": qcase["task"], "prompt": case["prompt"],
           "prompt_embeds": payload["prompt_embeds"], "text_token_tags": payload["text_token_tags"],
           "height": height, "width": width}
    if keyframes:
        out.update(keyframes=keyframes, keyframe_anchors=("first",))
    if mode == "reference":
        out["reference_images"] = images
        out["reference_audios"] = audios
    return out, receipt["request_wall_s"]


def serve(options, write_json):
    """JSON-line loop; mirrors worker.py's request/response contract."""
    import signal
    sys.path.insert(0, str(RUNTIME))
    from runtime.config import load_paths, load_recipe
    from runtime.pipeline import Worker
    paths = load_paths(options["paths"])
    config = load_recipe()
    config["execution"] = {"offload": "cpu", "gpu_count": 1, "qwen_lifetime": "one_request",
                           "overlap_stages": False}
    root = Path(options["work_dir"])
    engine = None
    qwen = None  # resident Qwen worker (hyperflow.qwen_resident): loaded once, parked in pinned host RAM when idle
    stopped = False

    def stop(_sig, _frame):
        nonlocal stopped
        if not stopped:
            stopped = True
            raise KeyboardInterrupt("Closing this ComfyUI worker")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for line in sys.stdin:
            request = json.loads(line)
            if request["op"] == "close":
                break
            if request["op"] != "generate":
                raise ValueError("Unsupported operation")
            progress_path = request.get("progress")
            _EXTEND["window"] = None
            _EXTEND["session_target"] = None
            _EXTEND["session_meta"] = None

            def progress(stage, step=None, steps=None):
                if progress_path:
                    write_json(progress_path, {"stage": stage, "step": step, "steps": steps,
                                               "time": time.time()})
            try:
                started = time.monotonic_ns()
                case = {"case_id": request["case_id"], "prompt": request["prompt"],
                        "seed": request["seed"], "task": "t2va"}
                request_dir = root / case["case_id"]
                request_dir.mkdir(parents=True, exist_ok=False)
                cold = engine is None and request.get("op_detail") != "conditioning_only"
                if cold:
                    progress("loading")
                    engine = HyperFlow(options, paths)
                    if options["hyperflow"].get("qwen_resident"):
                        load_started = time.monotonic()
                        qwen = Worker("qwen", root / "qwen-resident", paths, config)
                        qwen.call(op="offload")
                        engine.load_s += time.monotonic() - load_started
                if request.get("conditioning"):
                    import torch
                    payload = torch.load(request["conditioning"], map_location="cpu", weights_only=False)
                    case["task"] = payload.get("task", "t2va")
                    qwen_s = qwen_call_s = 0.0
                else:
                    progress("conditioning")
                    qwen_started = time.monotonic()
                    payload, qwen_call_s = local_conditioning(case, request, request_dir, paths, config, Worker, qwen)
                    # Full Qwen wall time of this request (process start + load, or onload/offload), not only encode.
                    qwen_s = time.monotonic() - qwen_started
                    case["task"] = payload["task"]
                if request.get("op_detail") == "conditioning_only":
                    import torch
                    torch.save(payload, request_dir / "payload.pt")
                    write_json(request["response"], {"status": "PASS", "payload": str(request_dir / "payload.pt"),
                                                     "qwen_s": qwen_s, "task": case["task"]})
                    continue
                # Every take writes its latents sidecar, so the next extend in the
                # chain has a source; an `extend` request additionally pins one.
                if options["hyperflow"].get("session_sidecar", True):
                    _EXTEND["session_target"] = str(_extend.session_path_for(request_dir))
                    # A take that continues a chain freezes its reference media into
                    # its own directory and stores the encoded conditioning with it;
                    # a standalone take only records where its references came from
                    # and leaves no copy behind. The artifacts are for the chain,
                    # not for every take.
                    chain = bool(request.get("extend"))
                    _EXTEND["chain"] = chain
                    images = [str(p) for p in (request.get("images") or [])]
                    audios = [str(p) for p in (request.get("audios") or [])]
                    if chain:
                        records = _extend.freeze_references(request_dir, images, audios)
                    else:
                        records = _extend.reference_records(images, audios)
                    _EXTEND["session_meta"] = {
                        "self_id": case["case_id"], "prompt": case["prompt"],
                        "seed": str(case["seed"]), "profile": options["name"],
                        "resolution": str(request.get("resolution", "768p")),
                        "task": payload.get("task", case["task"]), "source_path": "",
                        # What this take was conditioned on: the frozen copies when it
                        # is part of a chain, the source files otherwise.
                        "references": json.dumps(records),
                    }
                extend_request = request.get("extend")
                if extend_request:
                    mode = request.get("mode", "text")
                    if mode not in ("text", "reference"):
                        raise ValueError("An extend request carries text or reference conditioning "
                                         "only (no other mode)")
                    if mode == "reference" and not request.get("images"):
                        raise ValueError("A reference extend needs at least one reference image")
                    window_ctx = _extend.build_extend(
                        extend_request["session"],
                        window_frames=int(extend_request.get("window_frames",
                                                             _extend.DEFAULT_WINDOW_FRAMES)),
                        session_target=_EXTEND["session_target"],
                        session_meta=_EXTEND["session_meta"])
                    height, width = int(payload["height"]), int(payload["width"])
                    latent_h, latent_w = window_ctx["window_video"].shape[-2:]
                    if (latent_h, latent_w) != (height // 16, width // 16):
                        raise ValueError(
                            "The session's latents are %dx%d but this request renders %dx%d; "
                            "an extend chain must keep one resolution"
                            % (latent_h * 16, latent_w * 16, height, width))
                    source_resolution = window_ctx["parent_meta"].get("resolution") or ""
                    if source_resolution and source_resolution != str(request.get("resolution", "768p")):
                        raise ValueError("The session was rendered at %s, this request at %s"
                                         % (source_resolution, request.get("resolution", "768p")))
                    _EXTEND["window"] = window_ctx
                    # A reference extend keeps the ref2va task: the reference DiT and the
                    # reference conditioning rows stay in play, and the pinned window rows
                    # ride at the same conditioning levels (0.999 visual, 1.0 audio), so the
                    # unique (timestep, endpoint) set -- and therefore the AdaLN table key --
                    # is the one the ref2va variants already carry. A text extend stays t2va.
                    if mode != "reference":
                        case["task"] = "t2va"
                _artifact.set_current(request.get("condition_artifact"))
                row = engine.generate(case, payload, request_dir, progress)
                e2e = (time.monotonic_ns() - started) / 1e9
                progress("done", 8, 8)
                window = _EXTEND.get("window")
                extend_report = {}
                if window is not None:
                    extend_report = {
                        "window_frames": window["info"]["window_frames"],
                        "window_steps": window["window_steps"],
                        "audio_steps": window["audio_steps"],
                        "delivered_frames": window["info"]["delivered_frames"],
                        "handover_frames": window["info"]["handover_frames"],
                        "parent_session": window["session"],
                        "parent_id": window["parent_id"],
                        "session": window.get("session_written"),
                    }
                    write_json(request["response"], {
                        "status": "PASS", "case_id": case["case_id"], "seed": case["seed"],
                        "task": case["task"], "output": row["output"], "e2e_s": e2e,
                        "delivered_output": row.get("delivered_output"),
                        "qwen_s": qwen_s, "qwen_encode_s": qwen_call_s if not request.get("conditioning") else 0.0,
                        "h3_phases_s": {"generate_s": row["generate_s"], **row.get("phases_s", {})},
                        "forwards": row["forwards"], "adaln_variant": row.get("adaln_variant"),
                        "session": (_EXTEND.get("session_target") if window is None
                                    else window.get("session_written")),
                        "extend": extend_report,
                        "condition_artifact": _artifact.current_state(),
                        "profile": options["name"], "precision": "hyperflow",
                        "decoder": "original", "single_stage": True, "cold_start": cold,
                        "startup_and_warmup_s": engine.load_s if cold else 0, "prompt": case["prompt"],
                        "timing": "request through completed MP4; model load reported separately"})
            except Exception as error:
                write_json(request["response"], {"status": "FAIL",
                                                 "error": "%s: %s" % (type(error).__name__, error)})
                raise
            finally:
                _EXTEND["window"] = None
                _EXTEND["session_target"] = None
                _EXTEND["session_meta"] = None
                _EXTEND["adaln_key"] = None
                _EXTEND["chain"] = False
                _artifact.set_current(None)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if qwen is not None:
            qwen.close()
