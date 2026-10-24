"""Small ComfyUI surface: profile, prompt and seed -> audio/video + timings."""
import json
import os
from pathlib import Path
import time

from . import bridge
from .config import RESOLUTIONS, DEFAULT_STEPS, load_profile, profile_resolutions, profile_modes, profile_names

MODES = ["text", "first_frame", "reference", "extend"]
GENERATE_MODES = ["text", "first_frame", "reference"]
MODE_LABELS = {"text": "text", "first_frame": "first-frame", "reference": "reference-image", "extend": "extend"}
BACKENDS = ["local"]
# Latest update per node for GET /solh3/progress (polling fallback for reloads).
PROGRESS = {}


def publish(node, prompt_id, stage, started, **fields):
    value = {"node": node, "prompt_id": prompt_id, "stage": stage, "step": None, "steps": None,
             "elapsed_s": round(time.monotonic() - started, 2), "message": ""}
    value.update(fields)
    PROGRESS[str(node)] = value
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("solh3_progress", value)
    except Exception:
        pass  # No server (CPU tests) or no client: polling still sees PROGRESS.
    return value


def resolve_images(image):
    """Comma-separated uploaded names (ComfyUI input dir) -> absolute paths."""
    import folder_paths
    paths = []
    for name in (part.strip() for part in image.split(",")):
        if name:
            path = Path(folder_paths.get_annotated_filepath(name))
            if not path.is_file():
                raise ValueError("Uploaded image not found: %s" % name)
            paths.append(str(path.resolve()))
    return paths


def check_request(profile, mode, image, backend, resolution="768p"):
    if backend not in BACKENDS:
        raise ValueError("Unknown backend: %s" % backend)
    if mode not in MODES:
        raise ValueError("Unknown mode: %s" % mode)
    if resolution not in RESOLUTIONS:
        raise ValueError("Unknown resolution: %s" % resolution)
    if resolution not in profile_resolutions(profile):
        raise ValueError("This pipeline renders %s only" % ", ".join(profile_resolutions(profile)))
    effective = mode
    if effective != "text":
        if effective not in profile_modes(profile):
            raise ValueError("This pipeline does not support %s mode" % MODE_LABELS[effective])
        if not image.strip():
            raise ValueError("Upload an image for %s mode" % MODE_LABELS[effective])
        if effective == "first_frame" and len([n for n in image.split(",") if n.strip()]) != 1:
            raise ValueError("First-frame mode takes exactly one image")


def latest_session(output_root):
    """Newest Sol-H3 session sidecar across all output sessions (Studio auto-continue)."""
    import glob as _glob
    found = _glob.glob(str(Path(output_root) / "sol_h3" / "*" / "request-*" / "session.session.safetensors"))
    if not found:
        raise ValueError("No take to continue yet: generate a video first (any mode), then Extend")
    return max(found, key=os.path.getmtime)


def safe_stem(name):
    """A file stem that stays on one path segment (no separators, no traversal)."""
    stem = str(name or "").strip().replace("\\", "/").split("/")[-1]
    stem = stem.replace("\x00", "").strip()
    if stem in (".", ".."):
        return ""
    return stem


def resolve_output_dir(output_root, output_dir, mode):
    """Where a take is saved: a folder under the ComfyUI output root."""
    root = Path(output_root).resolve()
    raw = str(output_dir or "").strip().replace("\\", "/").strip("/")
    if not raw:
        raw = "SolH3/%s" % mode
    if ".." in raw.split("/"):
        raise ValueError("output_dir must stay inside the ComfyUI output folder")
    folder = (root / raw).resolve()
    try:
        folder.relative_to(root)
    except ValueError:
        raise ValueError("output_dir must stay inside the ComfyUI output folder") from None
    return folder


def default_stem(mode, seed):
    """A readable default: solh3_<mode>_<time>_s<seed>."""
    return "solh3_%s_%s_s%d" % (mode, time.strftime("%Y%m%d-%H%M%S"), seed)


def place_take(output_root, result, output_dir, filename, mode, seed):
    """Move the engine MP4 to a human-readable path and return that path.

    The engine writes into an opaque work folder (sol_h3/<uuid>/request-NNNNN), so
    the take is moved out to <output_dir>/<name>.mp4 and the output tree reads
    SolH3/<mode>/solh3_<mode>_<time>_s<seed>.mp4 unless the caller names it.
    """
    source = Path(result["output"])
    folder = resolve_output_dir(output_root, output_dir, mode)
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(filename) or default_stem(mode, seed)
    target = folder / (stem + ".mp4")
    index = 1
    while target.exists():
        target = folder / ("%s_%d.mp4" % (stem, index))
        index += 1
    os.replace(str(source), str(target))
    result["engine_output"] = str(source)
    result["output"] = str(target)
    return target


def write_sidecar(output_root, prompt_id, result, mode, seed, resolution="768p"):
    """Library record; the take itself lives at the readable path."""
    folder = Path(output_root) / "SolH3" / "library"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"id": prompt_id, "engine_output": result.get("engine_output") or result.get("output"),
              "output_path": result.get("output"),
              "delivered_output": result.get("delivered_output"), "session": result.get("session"),
              "prompt": result.get("prompt"), "mode": mode, "profile": result.get("profile"),
              "seed": seed, "resolution": resolution, "e2e_s": result.get("e2e_s"), "created": time.time()}
    temporary = folder / (prompt_id + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    os.replace(temporary, folder / (prompt_id + ".json"))


def current_prompt_id():
    try:
        from server import PromptServer
        return getattr(PromptServer.instance, "last_prompt_id", None)
    except Exception:
        return None


class SolH3Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "profile": (profile_names(),),
            "prompt": ("STRING", {"multiline": True, "default": "A bird glides over a mountain lake. overall_soundscape: Soft wind and distant birds."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**53 - 1}),
            "keep_loaded": ("BOOLEAN", {"default": True}),
        }, "optional": {
            # Optional with defaults so workflows saved before Studio still load.
            "mode": (GENERATE_MODES, {"default": "text"}),
            "image": ("STRING", {"default": ""}),
            "audio": ("STRING", {"default": ""}),
            "backend": (BACKENDS, {"default": "local"}),
            "resolution": (list(RESOLUTIONS), {"default": "768p"}),
            # New widgets sit last so workflows saved earlier keep their mapping.
            "output_dir": ("STRING", {"default": "", "tooltip": "Save folder, relative to the ComfyUI output folder. Empty = SolH3/<mode>."}),
            "filename": ("STRING", {"default": "", "tooltip": "File name without extension. Empty = solh3_<mode>_<time>_s<seed>."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video_with_audio", "timing_json", "session")
    FUNCTION = "generate"
    CATEGORY = "Sol Engine"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Every queued request must run: never time a ComfyUI cache hit.
        return float("nan")

    def generate(self, profile, prompt, seed, keep_loaded=True, mode="text", image="", audio="",
                 backend="local", resolution="768p", output_dir="", filename="", unique_id=None):
        if not prompt.strip():
            raise ValueError("Enter a nonempty prompt")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seed must be a nonnegative 63-bit integer")
        check_request(profile, mode, image or "", backend, resolution)
        import folder_paths
        import comfy.model_management as mm
        from comfy_api.latest import InputImpl
        images = resolve_images(image or "") if mode != "text" else []
        audios = resolve_images(audio) if (mode == "reference" and audio.strip()) else []
        if audios and mode != "reference":
            raise ValueError("Audio references need reference mode")
        if len(audios) > 3:
            raise ValueError("At most 3 audio references")
        extend = None
        node, prompt_id, started = unique_id, current_prompt_id() or "", time.monotonic()
        publish(node, prompt_id, "queued", started)
        output_root = folder_paths.get_output_directory()
        try:
            settings = load_profile(profile)
            mm.unload_all_models()
            mm.soft_empty_cache()
            cold = not bridge.is_warm(settings, output_root)
            steps = settings.get("steps", DEFAULT_STEPS.get(settings["precision"]))
            publish(node, prompt_id, "loading" if cold else "conditioning", started, steps=steps)

            def progress(update):
                stage = update.get("stage", "generating")
                if stage == "generating" and cold:
                    stage = "loading"  # Without worker updates a cold start is mostly loading.
                publish(node, prompt_id, stage, started, step=update.get("step"),
                        steps=update.get("steps", steps), message=update.get("message", ""))

            try:
                result = bridge.generate(settings, output_root, prompt, seed,
                                         cancel=mm.throw_exception_if_processing_interrupted,
                                         mode=mode, images=images, audios=audios, extend=extend,
                                         progress=progress, resolution=resolution)
            finally:
                if not keep_loaded:
                    bridge.close()
            publish(node, prompt_id, "saving", started, steps=steps)
            place_take(output_root, result, output_dir, filename, mode, seed)
            if prompt_id:
                write_sidecar(output_root, prompt_id, result, mode, seed, resolution)
        except BaseException as error:
            publish(node, prompt_id, "error", started, message=str(error) or type(error).__name__)
            raise
        publish(node, prompt_id, "done", started, steps=steps, step=steps,
                message="%.1f s" % result.get("e2e_s", 0))
        report = json.dumps(result, indent=2, ensure_ascii=False)
        # SaveVideo owns the native player. Keeping one player avoids duplicate
        # preview sizing conflicts with a legacy node's timing widget.
        # Lets Studio show the engine MP4 at once (and in graphs without SaveVideo).
        relative = Path(result["output"]).resolve().relative_to(Path(output_root).resolve())
        video = {"filename": relative.name, "subfolder": relative.parent.as_posix(), "type": "output"}
        # The session sidecar carries this take's latents for the next extend.
        session = result.get("session") or ""
        return {"ui": {"text": [report], "solh3_video": [video]},
                "result": (InputImpl.VideoFromFile(result["output"]), report, session)}


class SolH3Release:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"after": ("VIDEO",)}}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "release"
    CATEGORY = "Sol Engine"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def release(self, after):
        bridge.close()
        return (after,)


class SolH3Extend:
    """Continue a previous take: its tail window rides as pinned context and the
    engine generates the next segment after it (masked extend). Wire the previous
    take's session output in; chain the output's session again to keep going."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "profile": (profile_names(),),
            "prompt": ("STRING", {"multiline": True, "default": "The woman keeps walking, slows down and turns her head toward a cat sitting under an awning, then smiles. overall_soundscape: rain continues, a cat meows softly."}),
            "seed": ("INT", {"default": 43, "min": 0, "max": 2**53 - 1}),
            "session": ("STRING", {"default": "", "tooltip": "The previous take's session sidecar: the session output of a Generate/Extend node, or the .session.safetensors path."}),
            "window_frames": ("INT", {"default": 39, "min": 5, "max": 124, "tooltip": "Frames of the previous take's tail pinned as context. From a 124-frame take the legal lengths are 39/90 (shared AV grid, recommended), 56, 22, 5; the engine refuses the rest."}),
            "keep_loaded": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "resolution": (list(RESOLUTIONS), {"default": "768p"}),
            # New widgets sit last so workflows saved earlier keep their mapping.
            "output_dir": ("STRING", {"default": "", "tooltip": "Save folder, relative to the ComfyUI output folder. Empty = SolH3/extend."}),
            "filename": ("STRING", {"default": "", "tooltip": "File name without extension. Empty = solh3_extend_<time>_s<seed>."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video_with_audio", "timing_json", "session")
    FUNCTION = "generate"
    CATEGORY = "Sol Engine"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Every queued request must run: never time a ComfyUI cache hit.
        return float("nan")

    def generate(self, profile, prompt, seed, session, window_frames=39, keep_loaded=True,
                 resolution="768p", output_dir="", filename="", unique_id=None):
        if not prompt.strip():
            raise ValueError("Enter a nonempty prompt")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Seed must be a nonnegative 63-bit integer")
        import folder_paths
        output_root = folder_paths.get_output_directory()
        # Empty session => continue the newest take (Studio auto-continue / node convenience).
        session = (session or "").strip() or latest_session(output_root)
        if not 5 <= int(window_frames) <= 124:
            raise ValueError("window_frames must be between 5 and 124")
        check_request(profile, "text", "", "local", resolution)
        from comfy_api.latest import InputImpl
        import comfy.model_management as mm
        node, prompt_id, started = unique_id, current_prompt_id() or "", time.monotonic()
        publish(node, prompt_id, "queued", started)
        try:
            settings = load_profile(profile)
            mm.unload_all_models()
            mm.soft_empty_cache()
            cold = not bridge.is_warm(settings, output_root)
            steps = settings.get("steps", DEFAULT_STEPS.get(settings["precision"]))
            publish(node, prompt_id, "loading" if cold else "conditioning", started, steps=steps)

            def progress(update):
                stage = update.get("stage", "generating")
                if stage == "generating" and cold:
                    stage = "loading"  # Without worker updates a cold start is mostly loading.
                publish(node, prompt_id, stage, started, step=update.get("step"),
                        steps=update.get("steps", steps), message=update.get("message", ""))

            try:
                result = bridge.generate(settings, output_root, prompt, seed,
                                         cancel=mm.throw_exception_if_processing_interrupted,
                                         mode="text", progress=progress, resolution=resolution,
                                         extend={"session": session.strip(),
                                                 "window_frames": int(window_frames)})
            finally:
                if not keep_loaded:
                    bridge.close()
            publish(node, prompt_id, "saving", started, steps=steps)
            place_take(output_root, result, output_dir, filename, "extend", seed)
            if prompt_id:
                write_sidecar(output_root, prompt_id, result, "extend", seed, resolution)
        except BaseException as error:
            publish(node, prompt_id, "error", started, message=str(error) or type(error).__name__)
            raise
        publish(node, prompt_id, "done", started, steps=steps, step=steps,
                message="%.1f s" % result.get("e2e_s", 0))
        report = json.dumps(result, indent=2, ensure_ascii=False)
        relative = Path(result["output"]).resolve().relative_to(Path(output_root).resolve())
        video = {"filename": relative.name, "subfolder": relative.parent.as_posix(), "type": "output"}
        session_out = result.get("session") or ""
        return {"ui": {"text": [report], "solh3_video": [video]},
                "result": (InputImpl.VideoFromFile(result["output"]), report, session_out)}


NODE_CLASS_MAPPINGS = {"SolH3Generate": SolH3Generate, "SolH3Release": SolH3Release,
                       "SolH3Extend": SolH3Extend}
NODE_DISPLAY_NAME_MAPPINGS = {"SolH3Generate": "Sol-H3 · Generate Video",
                              "SolH3Release": "Sol-H3 · Release GPU",
                              "SolH3Extend": "Sol-H3 · Extend Video"}
