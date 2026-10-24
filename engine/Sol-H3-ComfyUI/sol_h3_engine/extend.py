# Copyright 2026 The Sol-H3 authors. Licensed under the Apache License, Version 2.0.
#
# `frames` arithmetic is ported from comfyui-obvpm-timeline (frames.py), Copyright
# the obvpm authors, MIT-style -- see THIRD_PARTY_NOTICES.md. The masked-continuation
# semantics follow ComfyUI core PR #15375 (per-row denoise mask) and obvpm-timeline's
# nodes_masked.py window rules.
"""Native video extend (continuation) for the Sol-H3 engine -- the masked route.

The previous clip's tail-window latents are written into the HEAD of the target's
generated rows and pinned at the conditioning noise level for every denoising step,
exactly as ComfyUI core's per-row denoise mask does it: a row held at mask 0 runs at
the conditioning timestep, its latent is injected at cond strength
(``0.999 * clean + 0.001 * noise`` -- the same footing a keyframe cond row stands on),
and it is re-imposed after every scheduler step. The model therefore generates the
continuation with the exact tail as context, and the window survives verbatim: no
re-render, no level step, audio kept as-is.

Grid arithmetic (video 17k+5 frames <-> 5k+2 latent steps, audio 40 steps/s) is a
single implementation ported from obvpm-timeline's frames.py, trap tests included.

Session sidecars (safetensors: video + audio latents + str metadata) carry a clip's
latents between takes, modelled on obvpm's mctx format.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# frame <-> latent arithmetic (ported from obvpm-timeline frames.py)
# ---------------------------------------------------------------------------

FPS = 24
AUDIO_LATENT_FPS = 40
FRAME_RESCALE = 5.0 / 3.0  # audio steps per video frame

FRAMES_PER_GROUP = 17
FRAME_BASE = 5
LATENTS_PER_GROUP = 5
LATENT_BASE = 2

# Frames covered by each latent step, indexed by step % 5. Every fifth step spans
# ONE frame, the rest span four.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

# Whole-step context windows, largest first (obvpm's WINDOW_LADDER without the
# 1-frame rung, which is not tail-sliceable from a latent).
WINDOW_LADDER = (124, 107, 90, 73, 56, 39, 22, 5)
DEFAULT_WINDOW_FRAMES = 39

SESSION_FORMAT = "solx-v1"
SESSION_SUFFIX = ".session.safetensors"


def frames_to_latents(frame_count):
    """Latent steps the VAE produces for a clip of frame_count frames (on-grid)."""
    if frame_count <= FRAME_BASE:
        return LATENT_BASE
    return ((frame_count - FRAME_BASE) // FRAMES_PER_GROUP) * LATENTS_PER_GROUP + LATENT_BASE


def latents_to_frames(latent_t):
    """Pixel frames of an on-grid clip with latent_t latent steps."""
    return FRAMES_PER_GROUP * ((latent_t - LATENT_BASE) // LATENTS_PER_GROUP) + FRAME_BASE


def frame_at_latent(k):
    """First pixel frame of latent step k, valid for ANY k >= 0 (off-grid safe)."""
    k = int(k)
    if k <= 0:
        return 0
    full, rem = divmod(k, len(FRAME_PER_TOKEN))
    return full * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:rem])


def pixel_frames(latent_t):
    """Pixel frames covered by latent_t steps from cycle phase 0."""
    return frame_at_latent(latent_t)


def steps_for_frames(n):
    """Latent steps covering EXACTLY n pixel frames from phase 0, else None."""
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def snap_window(n):
    """Largest ladder window <= n, or None when even 5 does not fit."""
    for g in WINDOW_LADDER:
        if g <= n:
            return g
    return None


def audio_total(frame_count):
    """Cumulative audio latent steps at a frame count. THE conversion."""
    return int(round(frame_count / float(FPS) * AUDIO_LATENT_FPS))


def tail_slice_start(total_steps, window_steps):
    """Start step of a tail slice, verified to sit at cycle phase 0."""
    if window_steps > total_steps:
        raise ValueError(
            "tail slice of %d steps from a %d step latent" % (window_steps, total_steps))
    start = total_steps - window_steps
    if start % LATENTS_PER_GROUP != 0:
        raise ValueError(
            "tail slice of %d steps from a %d step latent starts at cycle phase %d, "
            "not 0; the slice would be unsound"
            % (window_steps, total_steps, start % LATENTS_PER_GROUP))
    return start


def handover_frames(covered, steps, place="before", decode_edge=True):
    """Where responsibility for a hard-held window changes hands (frames, in-window).

    Ported from obvpm nodes_masked.handover_frames for the hard-hold profile
    (ramp 0, edge 0, deep 0): a before-pin hands over one latent step inside its
    edge when decode_edge is set -- H3's video VAE draws a file's last step with
    nothing after it, so the older clip's frames there are its decode edge while
    the take decoded the very same latents mid-file.
    """
    covered = int(covered)
    steps = int(steps)
    if place == "after":
        if decode_edge and steps > 2:
            return frame_at_latent(2)
        return 0
    if decode_edge and steps > 1:
        return frame_at_latent(steps - 1)
    return covered


# ---------------------------------------------------------------------------
# session sidecars
# ---------------------------------------------------------------------------

def session_path_for(request_dir):
    return Path(request_dir) / ("session" + SESSION_SUFFIX)


def write_session(path, video, audio, meta):
    """Atomic session write: tmp file + os.replace (MP4 first, then this)."""
    from safetensors.torch import save_file
    clean = {}
    for key, value in dict(meta or {}).items():
        if not isinstance(key, str):
            raise ValueError("session metadata keys must be str, got %r" % (key,))
        clean[key] = str(value)
    clean["format"] = SESSION_FORMAT
    tensors = {"video": video.detach().contiguous().cpu(),
               "audio": audio.detach().contiguous().cpu()}
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp), metadata=clean)
    os.replace(str(tmp), str(path))
    return str(path)


def load_session(path):
    """Read a session sidecar -> {"video", "audio", "meta"} with validation."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    path = Path(path)
    if not path.is_file():
        raise ValueError("session sidecar not found: %s" % path)
    with safe_open(str(path), framework="pt") as handle:
        meta = dict(handle.metadata() or {})
    if meta.get("format") != SESSION_FORMAT:
        raise ValueError(
            "%s is not a %s session sidecar (format=%r)" % (path, SESSION_FORMAT, meta.get("format")))
    tensors = load_file(str(path))
    video, audio = tensors["video"], tensors["audio"]
    if video.dim() != 5 or video.shape[0] != 1 or audio.dim() != 3 or audio.shape[0] != 2:
        raise ValueError(
            "%s holds unexpected latent shapes: video %s, audio %s"
            % (path, tuple(video.shape), tuple(audio.shape)))
    return {"video": video, "audio": audio, "meta": meta}


def slice_tail_window(session, window_frames=DEFAULT_WINDOW_FRAMES):
    """The tail window of a loaded session, on-grid and phase-0 verified.

    Returns (video_window (1, 24, steps, h, w), audio_window (2, 32, audio_steps), info).
    """
    video, audio, meta = session["video"], session["audio"], session["meta"]
    raw_steps = int(video.shape[2])
    raw_frames = int(meta.get("raw_frames") or latents_to_frames(raw_steps))
    if int(meta.get("raw_frames") or raw_frames) != raw_frames:
        raise ValueError("session raw_frames %s disagrees with the stored latent (%d frames)"
                         % (meta.get("raw_frames"), raw_frames))
    n = int(window_frames)
    if n > raw_frames:
        snapped = snap_window(raw_frames)
        raise ValueError(
            "window of %d frames exceeds the session's %d frames; nearest legal window is %s"
            % (n, raw_frames, snapped if snapped else "nothing"))
    steps = steps_for_frames(n)
    if steps is None:
        raise ValueError("%d frames is not a whole number of latent steps" % n)
    start = tail_slice_start(raw_steps, steps)
    if frame_at_latent(start) != start and False:  # pragma: no cover - documentation guard
        raise RuntimeError("step mapping disagreement")
    video_window = video[:, :, start:start + steps].clone()
    covered = pixel_frames(steps)
    if covered != n:
        raise RuntimeError("%d steps cover %d frames, expected %d" % (steps, covered, n))
    # Audio: end-aligned with the video window's END, on cumulative totals.
    total_audio = int(audio.shape[-1])
    want = audio_total(raw_frames) - audio_total(raw_frames - n)
    if want <= 0 or total_audio < want:
        raise ValueError(
            "session audio latent (%d steps) cannot supply a %d step window"
            % (total_audio, want))
    audio_window = audio[:, :, total_audio - want:].clone()
    handover = handover_frames(covered, steps, "before", decode_edge=True)
    info = {
        "window_frames": n, "window_steps": steps, "covered_frames": covered,
        "audio_steps": want, "source_start_step": start, "source_start_frame": frame_at_latent(start),
        "raw_frames": raw_frames, "raw_steps": raw_steps, "total_audio_steps": total_audio,
        "handover_frames": handover, "delivered_frames": raw_frames - handover,
        "source_path": meta.get("source_path", ""), "self_id": meta.get("self_id", ""),
    }
    return video_window, audio_window, info


def build_extend(session_path, window_frames=DEFAULT_WINDOW_FRAMES,
                 session_target=None, session_meta=None):
    """Load a session and slice its tail window into an extend context.

    The context is consumed by the engine's pipeline patches (see
    hyperflow_worker.install_extend_patches): the raw window latents, the geometry
    the patches derive their row indices from, and the session target the new
    take's own sidecar is written to.
    """
    session = load_session(session_path)
    video_window, audio_window, info = slice_tail_window(session, window_frames)
    meta = dict(session["meta"])
    context = {
        "window_video": video_window,       # (1, 24, W, h, w) clean
        "window_audio": audio_window,       # (2, 32, WA) clean
        "window_steps": info["window_steps"],
        "audio_steps": info["audio_steps"],
        "info": info,
        "parent_meta": meta,
        "session": str(session_path),
        "parent_id": meta.get("self_id", ""),
        "session_target": str(session_target) if session_target else None,
        "session_meta": dict(session_meta or {}),
        "session_written": None,
        # filled by the patches at run time:
        "cond_video_rows": None,             # (W * rows_per_frame, C * patch) at cond strength
        "audio_rows_ch0": None, "audio_rows_ch1": None,
        "pin_video_rows": None, "pin_audio_rows": None,
        "window_video_rows": None,
    }
    return context


# ---------------------------------------------------------------------------
# per-row (t, r) plan with the window pinned (port of build_row_time_pairs + pins)
# ---------------------------------------------------------------------------

def build_extend_row_pairs(
    *,
    video_indices,
    audio_indices,
    num_condition_video_rows,
    num_condition_audio_rows,
    num_text_tokens,
    video_timestep,
    video_endpoint,
    audio_timestep,
    audio_endpoint,
    condition_video_timestep,
    condition_audio_timestep=1.0,
    pin_video_rows=(),
    pin_audio_rows=(),
):
    """Assign a (t, r) pair to every row, with the pinned window rows at the
    conditioning level, and reduce to the transformer's (timestep, endpoint, indices).

    Faithful port of hyperflow_h3.schedule.build_row_time_pairs plus one rule: the
    extend window's video rows ride at the visual conditioning timestep (the same
    footing a keyframe cond row stands on) and its audio rows at t = r = 1.0 (the
    audio-reference level). The unique pair set therefore matches the fl2va
    ("image") or fl2va+audio-reference ("image_audio") conditioning variants, so
    the precomputed AdaLN tables cover an extend without regeneration.
    """
    import torch
    video_indices = video_indices.cpu()
    audio_indices = audio_indices.cpu()
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)

    timestep = torch.full((sequence_length,), float(video_timestep), dtype=torch.float32)
    endpoint = torch.full((sequence_length,), float(video_endpoint), dtype=torch.float32)

    condition_video = video_indices[:num_condition_video_rows]
    timestep[condition_video] = float(condition_video_timestep)
    endpoint[condition_video] = float(condition_video_timestep)

    generated_audio = audio_indices[num_condition_audio_rows:]
    timestep[generated_audio] = float(audio_timestep)
    endpoint[generated_audio] = float(audio_endpoint)

    condition_audio = audio_indices[:num_condition_audio_rows]
    timestep[condition_audio] = float(condition_audio_timestep)
    endpoint[condition_audio] = float(condition_audio_timestep)

    for rows, value in ((pin_video_rows, condition_video_timestep),
                        (pin_audio_rows, condition_audio_timestep)):
        if len(rows):
            index = torch.as_tensor(list(rows), dtype=torch.long)
            timestep[index] = float(value)
            endpoint[index] = float(value)

    pairs = torch.stack((timestep, endpoint), dim=-1)
    unique_pairs, indices = torch.unique(pairs, dim=0, sorted=True, return_inverse=True)
    return unique_pairs[:, 0].contiguous(), unique_pairs[:, 1].contiguous(), indices


# ---------------------------------------------------------------------------
# trap tests: python extend.py
# ---------------------------------------------------------------------------

def _self_test():
    for k in range(0, 40):
        n = FRAMES_PER_GROUP * k + FRAME_BASE
        t = frames_to_latents(n)
        assert t == LATENTS_PER_GROUP * k + LATENT_BASE, (n, t)
        assert latents_to_frames(t) == n, (n, t)

    assert frame_at_latent(0) == 0
    assert frame_at_latent(2) == 5
    assert frame_at_latent(37) == 124 == latents_to_frames(37)
    assert pixel_frames(12) == 39
    assert pixel_frames(37) == 124
    assert steps_for_frames(39) == 12
    assert steps_for_frames(124) == 37
    assert steps_for_frames(2) is None
    assert audio_total(124) == 207
    assert audio_total(39) == 65
    assert audio_total(124) - audio_total(85) == 65

    # the engine's own geometry: a 124-frame clip's tail window
    assert tail_slice_start(37, 12) == 25
    assert 25 % LATENTS_PER_GROUP == 0
    assert handover_frames(39, 12, "before", decode_edge=True) == 35
    assert handover_frames(39, 12, "before", decode_edge=False) == 39
    assert handover_frames(39, 12, "after", decode_edge=True) == 5
    assert snap_window(39) == 39 and snap_window(30) == 22 and snap_window(3) is None

    # the row plan: window rows pinned, unique pairs unchanged in count
    import torch
    text, audio_t, video_t = 300, 207, 37
    rows_per_frame = 4032
    window_rows, window_steps, window_audio = 12, 12, 65
    seq = text + audio_t * 2 + video_t * rows_per_frame
    video_indices = torch.arange(text + audio_t * 2, seq)
    audio_indices = torch.arange(text, text + audio_t * 2)
    pin_video = video_indices[:window_rows * rows_per_frame].tolist()
    pin_audio = audio_indices[:window_audio].tolist() + audio_indices[audio_t:audio_t + window_audio].tolist()
    t_p, r_p, idx = build_extend_row_pairs(
        video_indices=video_indices, audio_indices=audio_indices,
        num_condition_video_rows=0, num_condition_audio_rows=0, num_text_tokens=text,
        video_timestep=0.0, video_endpoint=0.1, audio_timestep=0.2, audio_endpoint=0.3,
        condition_video_timestep=0.999, condition_audio_timestep=1.0,
        pin_video_rows=pin_video, pin_audio_rows=pin_audio)
    assert t_p.numel() == 4, t_p.tolist()          # video, audio, cond video, cond audio
    assert idx.numel() == seq
    # every window row maps to the cond pair; the rest keep their stream pair
    pair_index = {(round(a, 6), round(b, 6)): i for i, (a, b) in enumerate(zip(t_p.tolist(), r_p.tolist()))}
    cond_v = pair_index[(0.999, 0.999)]
    cond_a = pair_index[(1.0, 1.0)]
    assert idx[pin_video[0]].item() == cond_v and idx[pin_video[-1]].item() == cond_v
    assert idx[pin_audio[0]].item() == cond_a and idx[pin_audio[-1]].item() == cond_a
    assert idx[video_indices[window_rows * rows_per_frame].item()].item() == pair_index[(0.0, 0.1)]
    assert idx[audio_indices[window_audio].item()].item() == pair_index[(0.2, 0.3)]
    return True


if __name__ == "__main__":
    _self_test()
    print("extend.py: all trap tests passed")
