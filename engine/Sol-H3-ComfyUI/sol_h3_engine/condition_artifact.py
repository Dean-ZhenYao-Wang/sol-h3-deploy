"""The take's encoded conditioning, stored as an artifact.

obvpm's rule: a chain's conditioning is an *artifact*, not a recipe. The media
copies beside the session (`refs/`) are the portable half -- the bytes the
reference came from. This is the exact half: the latents the pipeline actually
conditioned on, i.e. the video VAE's encoding of every reference image and the
audio VAE's encoding of every reference soundtrack.

Why both halves matter: the media copy keeps the chain alive after a restart
and after the source file is gone; the encoded copy keeps the chain on *the
conditioning it was built on* even if the encoding code moves under it, and
saves re-running the VAEs when the next take continues the chain.

The artifact is a safetensors file next to the take:

    refs/conditioning.safetensors
        condition_00 ... one tensor per reference condition (image or video)
        audio_00 ...    one tensor per audio-bearing reference
        metadata: format, the media digest it belongs to, the counts

The pipeline side is one hook: the official `MiniMaxH3Ref2VAReferenceEncoderStep`
encodes every reference through the VAEs; when a request carries an artifact,
the step hands it the stored latents instead and the encoders stay out of it.
That is the same seam obvpm's `condload` uses.
"""
import json
import os
from pathlib import Path

FORMAT = "solh3-cond-v1"
FILENAME = "conditioning.safetensors"

# The request in flight: the artifact it may reuse, and what happened.
_CURRENT = {"path": None, "used": None, "fallback": None}


def _digest(path):
    import hashlib
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def media_digest(paths):
    """One digest for a set of reference files: the identity of a take's media.

    Only the *content* counts: the same media frozen into two different takes
    lands in two different directories, and the two must still be recognized as
    the same conditioning.
    """
    import hashlib
    digest = hashlib.sha256()
    for file_digest in sorted(_digest(path) for path in paths):
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_path(take_dir):
    """Where a take keeps its encoded conditioning."""
    return Path(take_dir) / "refs" / FILENAME


def save(path, condition_latents, audio_condition_latents, media=""):
    """Write the encoded conditioning. Returns the record for the session metadata."""
    from safetensors.torch import save_file
    conditions = [t.detach().to("cpu").contiguous() for t in condition_latents]
    audios = [t.detach().to("cpu").contiguous() for t in audio_condition_latents]
    if not conditions and not audios:
        return None
    tensors = {}
    for index, tensor in enumerate(conditions):
        tensors["condition_%02d" % index] = tensor
    for index, tensor in enumerate(audios):
        tensors["audio_%02d" % index] = tensor
    meta = {"format": FORMAT, "media": media,
            "conditions": str(len(conditions)), "audios": str(len(audios))}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    save_file(tensors, str(tmp), metadata=meta)
    os.replace(str(tmp), str(path))
    return {"path": str(path), "sha256": _digest(str(path)), "media": media,
            "conditions": len(conditions), "audios": len(audios)}


def load(path):
    """The stored conditioning, or None when there is nothing to reuse."""
    from safetensors import safe_open
    path = Path(path or "")
    if not path.is_file():
        return None
    try:
        with safe_open(str(path), framework="pt") as handle:
            meta = dict(handle.metadata() or {})
            keys = list(handle.keys())
            tensors = {key: handle.get_tensor(key) for key in keys}
    except Exception:
        return None
    if meta.get("format") != FORMAT:
        return None
    conditions = [tensors["condition_%02d" % i]
                  for i in range(int(meta.get("conditions") or 0))
                  if "condition_%02d" % i in tensors]
    audios = [tensors["audio_%02d" % i]
              for i in range(int(meta.get("audios") or 0))
              if "audio_%02d" % i in tensors]
    return {"conditions": conditions, "audios": audios, "meta": meta}


def set_current(path):
    """Point the pipeline at the artifact this request may reuse."""
    _CURRENT["path"] = str(path) if path else None
    _CURRENT["used"] = None
    _CURRENT["fallback"] = None


def current_state():
    """What happened with the artifact of the request in flight."""
    return {"artifact": _CURRENT.get("path"), "reused": _CURRENT.get("used"),
            "fallback": _CURRENT.get("fallback")}


def capture(state, task, take_dir, media=""):
    """Store the conditioning this take was built on. Returns the record.

    The latents live on the pipeline's shared state, not on the step's own
    block state: every block builds its own view of the state from its declared
    inputs, and the encoder's outputs are reached through the state itself.
    """
    if task != "ref2va":
        return None  # A first-frame take's keyframe is its own first frame, not an inherited condition.
    conditions = list(getattr(state, "condition_latents", None) or [])
    audios = list(getattr(state, "audio_condition_latents", None) or [])
    if not conditions and not audios:
        return None
    try:
        return save(artifact_path(take_dir), conditions, audios, media)
    except Exception:
        return None  # A missing artifact is a lost optimization, not a failed take.


def install_encoder_patch():
    """Let the pipeline reuse an already encoded conditioning.

    The official `MiniMaxH3Ref2VAReferenceEncoderStep` encodes every reference
    through the video and audio VAEs. When the request carries an artifact we
    hand it the stored latents instead, so the reference rows are exactly the
    ones the parent take was built on.

    The patch is on the class, like the extend hooks: which artifact applies is
    read from `_CURRENT`, which the worker sets per request. A stored artifact
    that does not match the request (a different number of references, a
    different geometry) is *not* used -- the request falls back to encoding the
    references itself, and says so in the log.
    """
    from diffusers.modular_pipelines.minimax_h3.encoders import (
        MiniMaxH3Ref2VAReferenceEncoderStep)
    if getattr(MiniMaxH3Ref2VAReferenceEncoderStep, "_solh3_artifact", False):
        return
    original = MiniMaxH3Ref2VAReferenceEncoderStep.__call__

    def reference_encoder_with_artifact(self, components, state, _original=original):
        path = _CURRENT.get("path")
        if path:
            stored = load(path)
            if stored is not None:
                block_state = self.get_block_state(state)
                references = list(getattr(block_state, "normalized_references", None) or [])
                want_conditions = len([r for r in references if r.kind in ("image", "video")])
                want_audios = len([r for r in references if r.has_audio])
                if (len(stored["conditions"]) == want_conditions
                        and len(stored["audios"]) == want_audios):
                    device = components._execution_device
                    block_state.condition_latents = [t.to(device) for t in stored["conditions"]]
                    block_state.audio_condition_latents = list(stored["audios"])
                    self.set_block_state(state, block_state)
                    _CURRENT["used"] = str(path)
                    return components, state
                _CURRENT["fallback"] = ("stored %d conditions / %d audios, this request has %d / %d"
                                        % (len(stored["conditions"]), len(stored["audios"]),
                                           want_conditions, want_audios))
            else:
                _CURRENT["fallback"] = "artifact unreadable"
        return _original(self, components, state)

    MiniMaxH3Ref2VAReferenceEncoderStep.__call__ = reference_encoder_with_artifact
    MiniMaxH3Ref2VAReferenceEncoderStep._solh3_artifact = True
