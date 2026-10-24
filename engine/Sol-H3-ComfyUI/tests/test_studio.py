"""CPU tests for the Studio backend contract: capabilities, request forwarding, progress, library."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from test_bridge import FAKE, ROOT, bridge
from sol_engine_test.sol_h3_engine import comfy_nodes, comfy_routes, config

# Records each request the fake pipeline sees, and writes a worker progress file.
RECORD = FAKE.replace('''  output=self.root/(case["case_id"]+".mp4")''', '''  (self.root.parent/"request.json").write_text(json.dumps(REQUEST))
  if case["prompt"]=="SLOW":
   Path(REQUEST["progress"]).write_text(json.dumps({"stage":"denoising","step":2,"steps":4}))
   time.sleep(1.5)
  output=self.root/(case["case_id"]+".mp4")''')


class StudioConfig(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "config.json"
        path.write_text(json.dumps({"profiles": {
            "FP8": {"precision": "fp8", "label": "Fast", "python": "x", "paths": "y"},
            "Image": {"precision": "bf16", "steps": 8, "modes": ["text", "first_frame"],
                      "resolutions": ["768p", "480p"], "latency_s": 67.2},
            "Two": {"precision": "two_stage"}}}))
        self.env = patch.dict(os.environ, {"SOL_ENGINE_CONFIG": str(path)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_capabilities(self):
        value = json.loads(json.dumps(config.capabilities()))
        self.assertEqual(value["profiles"], [
            {"name": "FP8", "label": "Fast", "steps": 4, "pipeline": "one_stage", "modes": ["text"],
             "resolutions": ["768p"], "latency_s": None, "default": True},
            {"name": "Image", "label": "Image", "steps": 8, "pipeline": "one_stage",
             "modes": ["text", "first_frame"], "resolutions": ["768p", "480p"], "latency_s": 67.2, "default": False},
            {"name": "Two", "label": "Two", "steps": 7, "pipeline": "two_stage", "modes": ["text"],
             "resolutions": ["768p"], "latency_s": None, "default": False}])
        self.assertEqual(value["device"], {"id": "rtx5090", "label": "RTX 5090"})

    def test_node_rejects_unsupported_requests(self):
        node = comfy_nodes.SolH3Generate()
        with self.assertRaisesRegex(ValueError, "does not support first-frame mode"):
            node.generate("FP8", "a scene", 1, mode="first_frame", image="a.png")
        with self.assertRaisesRegex(ValueError, "does not support reference-image mode"):
            node.generate("Image", "a scene", 1, mode="reference", image="a.png")
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            node.generate("FP8", "a scene", 1, backend="remote")
        with self.assertRaisesRegex(ValueError, "renders 768p only"):
            node.generate("FP8", "a scene", 1, resolution="480p")
        with self.assertRaisesRegex(ValueError, "Upload an image"):
            node.generate("Image", "a scene", 1, mode="first_frame", image="")
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            node.generate("Image", "a scene", 1, mode="first_frame", image="a.png,b.png")
        optional = comfy_nodes.SolH3Generate.INPUT_TYPES()["optional"]
        self.assertEqual([optional[k][1]["default"] for k in ("mode", "image", "backend", "resolution")],
                         ["text", "", "local", "768p"])


class StudioBridge(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT / "sol_h3_engine/worker.py", self.root / "worker.py")
        # The fake pipeline reads the raw request through a module global set by worker.py's loop.
        worker = (self.root / "worker.py").read_text().replace(
            '            request = json.loads(line)\n',
            '            request = json.loads(line)\n            sys.modules["runtime.pipeline"].REQUEST = request\n')
        (self.root / "worker.py").write_text(worker)
        runtime = self.root / "runtimes/h3_fp8/runtime"
        runtime.mkdir(parents=True)
        (runtime / "__init__.py").touch()
        (runtime / "pipeline.py").write_text(RECORD)
        (runtime / "config.py").write_text('def load_paths(p): return {}\ndef load_recipe(): return {"stage1":{"width":1344}}\ndef normalize_case(c): return c\n')
        self.old_root = bridge.ROOT
        bridge.ROOT = self.root
        self.profile = dict(python=sys.executable, precision="fp8", gpu="0", paths="unused", name="fake")

    def tearDown(self):
        bridge.close()
        bridge.ROOT = self.old_root
        self.temp.cleanup()

    def test_forwards_mode_and_images(self):
        image = str(self.root / "frame.png")
        bridge.generate(self.profile, self.root, "scene", 7, mode="first_frame", images=[image])
        request = json.loads(next((self.root / "sol_h3").glob("*/request.json")).read_text())
        self.assertEqual((request["mode"], request["images"], request["resolution"]), ("first_frame", [image], "768p"))
        bridge.generate(self.profile, self.root, "scene", 8, resolution="480p")
        request = json.loads(next((self.root / "sol_h3").glob("*/request.json")).read_text())
        self.assertEqual((request["mode"], request["images"], request["resolution"]), ("text", [], "480p"))

    def test_progress_callback(self):
        updates = []
        bridge.generate(self.profile, self.root, "SLOW", 7, progress=updates.append)
        self.assertTrue(updates)
        self.assertTrue(all(isinstance(u["elapsed_s"], float) for u in updates))
        # Before the worker writes its file the stage is generic ("generating"); afterwards it is the worker's.
        # The first poll can land on either side of that write, so only the stage set is checked.
        self.assertLessEqual({u["stage"] for u in updates}, {"generating", "denoising"})
        self.assertIn({"stage": "denoising", "step": 2, "steps": 4},
                      [{k: u.get(k) for k in ("stage", "step", "steps")} for u in updates])


class StudioLibrary(unittest.TestCase):
    def test_listing_merges_sidecars(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "SolH3").mkdir()
            now = time.time()
            for index, name in enumerate(("studio_00001_.mp4", "studio_00002_.mp4", "other.mp4")):
                path = root / "SolH3" / name
                path.write_bytes(b"mp4")
                os.utime(path, (now + index * 60, now + index * 60))
            (root / "unrelated.mp4").write_bytes(b"outside library")
            comfy_nodes.write_sidecar(root, "p1", {"output": "/engine/a.mp4", "prompt": "first",
                                                   "profile": "FP8", "e2e_s": 12.5}, "text", 1)
            record = root / "SolH3/library/p1.json"
            data = json.loads(record.read_text()) | {"created": now - 1}
            record.write_text(json.dumps(data))
            (root / "SolH3/library/p2.json").write_text(json.dumps(
                {"id": "p2", "prompt": "second", "mode": "text", "seed": 2, "created": now + 59}))
            items = comfy_routes.list_videos(root, limit=8)["items"]
            self.assertEqual([i["file"] for i in items],
                             ["other.mp4", "studio_00002_.mp4", "studio_00001_.mp4"])
            self.assertEqual([i["prompt"] for i in items], [None, "second", "first"])
            self.assertEqual((items[2]["subfolder"], items[2]["type"], items[2]["e2e_s"]),
                             ("SolH3", "output", 12.5))
            self.assertEqual(len(comfy_routes.list_videos(root, limit=1)["items"]), 1)
            with self.assertRaises(ValueError):
                comfy_routes.video_path(root, "../unrelated.mp4")


class WebSyntax(unittest.TestCase):
    def test_web_modules_parse(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("node is not installed")
        for path in sorted((ROOT / "web").glob("*.js")):
            with self.subTest(path=path.name):
                # --check needs module syntax: import/export would fail as a script.
                subprocess.run([node, "--input-type=module", "--check"], input=path.read_text(),
                               text=True, check=True, timeout=30, capture_output=True)


class StudioGallery(unittest.TestCase):
    def test_listing_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "b-koi.mp4").write_bytes(b"mp4")
            (root / "b-koi.json").write_text(json.dumps({"prompt": "koi", "image": "b-koi.png", "order": 2}))
            (root / "b-koi.png").write_bytes(b"png")
            (root / "a-eagle.mp4").write_bytes(b"mp4")
            (root / "a-eagle.json").write_text(json.dumps({"prompt": "eagle", "title": "Eagle", "order": 1}))
            (root / "c-orphan.mp4").write_bytes(b"no sidecar")
            (root / "d-bad.mp4").write_bytes(b"mp4")
            (root / "d-bad.json").write_text(json.dumps({"title": "no prompt"}))
            items = comfy_routes.list_gallery(root)["items"]
            self.assertEqual([(i["name"], i["mode"], i["image"], i["title"]) for i in items],
                             [("a-eagle", "text", None, "Eagle"), ("b-koi", "first_frame", "b-koi.png", "b koi")])
            self.assertEqual(comfy_routes.gallery_file(root, "b-koi.png")[1], "image/png")
            for bad in ("../x.mp4", "b-koi.json", ".hidden.mp4", "", "missing.mp4"):
                with self.assertRaises(ValueError):
                    comfy_routes.gallery_file(root, bad)
            self.assertEqual(comfy_routes.list_gallery(None), {"items": []})


class StudioReferences(unittest.TestCase):
    """Native IMAGE / AUDIO sockets: a wired reference becomes a file the engine can read."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        path = Path(self.temp.name) / "config.json"
        path.write_text(json.dumps({"profiles": {
            "FP8": {"precision": "fp8", "label": "Fast", "python": "x", "paths": "y"}}}))
        self.env = patch.dict(os.environ, {"SOL_ENGINE_CONFIG": str(path)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_image_batch_becomes_png_files_in_order(self):
        import numpy as np
        from PIL import Image
        batch = np.zeros((2, 8, 6, 3), dtype="float32")
        batch[0] = 1.0
        batch[1, 0, 0, :] = 0.5
        paths = comfy_nodes.materialize_images(batch)
        self.assertEqual(len(paths), 2)
        self.assertEqual(Image.open(paths[0]).size, (6, 8))
        self.assertEqual(Image.open(paths[0]).getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(Image.open(paths[1]).getpixel((0, 0)), (128, 128, 128))

    def test_audio_becomes_a_readable_wav(self):
        import numpy as np
        import wave
        waveform = np.zeros((1, 2, 800), dtype="float32")
        waveform[0, 0, :400] = 0.5
        paths = comfy_nodes.materialize_audio({"waveform": waveform, "sample_rate": 16000})
        self.assertEqual(len(paths), 1)
        with wave.open(paths[0]) as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnchannels(), 2)
            self.assertEqual(handle.getnframes(), 800)

    def test_reference_collection_rules(self):
        import numpy as np
        batch = np.zeros((1, 4, 4, 3), dtype="float32")
        # A wired image is collected in reference mode.
        images, audios = comfy_nodes.collect_references("reference", "", "", batch, ())
        self.assertEqual(len(images), 1)
        self.assertEqual(audios, [])
        # A text take takes no references at all.
        self.assertEqual(comfy_nodes.collect_references("text", "", "", batch, ()), ([], []))
        # Audio without an image is refused: the engine needs an image to go with it.
        slot = {"waveform": np.zeros((1, 1, 4), dtype="float32"), "sample_rate": 16000}
        with self.assertRaisesRegex(ValueError, "need a reference image"):
            comfy_nodes.collect_references("reference", "", "", None, (slot,))
        # Audio with an image is fine -- this is the ref2va extend case.
        images, audios = comfy_nodes.collect_references("reference", "", "", batch, (slot,))
        self.assertEqual((len(images), len(audios)), (1, 1))

    def test_reference_records_and_inheritance(self):
        import tempfile
        import torch
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "ref.png"
            image.write_bytes(b"pixels")
            digest = comfy_nodes._extend.file_digest(str(image))
            sidecar = Path(temp) / "session.session.safetensors"
            save_file({"video": torch.zeros((1, 24, 4, 8, 8)), "audio": torch.zeros((2, 32, 4))},
                      str(sidecar), metadata={
                          "format": "solx-v1",
                          "references": json.dumps([{"kind": "image", "path": str(image),
                                                    "sha256": digest}])})
            # The take's own references come back, verified.
            images, audios = comfy_nodes.inherited_references(str(sidecar))
            self.assertEqual(images, [str(image)])
            self.assertEqual(audios, [])
            # A file that changed under the chain is refused by name, not swapped.
            image.write_bytes(b"other pixels")
            with self.assertRaisesRegex(ValueError, "has changed since"):
                comfy_nodes.inherited_references(str(sidecar))
            # A missing file is refused too.
            image.unlink()
            with self.assertRaisesRegex(ValueError, "which is gone"):
                comfy_nodes.inherited_references(str(sidecar))

    def test_extend_takes_wired_references(self):
        import numpy as np
        # The Extend node routes a wired reference into the engine's reference mode.
        node = comfy_nodes.SolH3Extend()
        self.assertIn("image_input", comfy_nodes.SolH3Extend.INPUT_TYPES()["optional"])
        self.assertIn("audio_1", comfy_nodes.SolH3Extend.INPUT_TYPES()["optional"])
        images, _ = comfy_nodes.collect_references(
            "reference", "", "", np.zeros((1, 4, 4, 3), dtype="float32"), ())
        self.assertEqual("reference" if images else "text", "reference")

    def test_image_rules_use_the_resolved_count(self):
        node = comfy_nodes.SolH3Generate()
        with self.assertRaisesRegex(ValueError, "does not support reference-image mode"):
            node.generate("FP8", "a scene", 1, mode="reference")
        with self.assertRaisesRegex(ValueError, "Upload an image for reference-image mode"):
            comfy_nodes.check_images("reference", 0)
        with self.assertRaisesRegex(ValueError, "exactly one image"):
            comfy_nodes.check_images("first_frame", 2)
        self.assertIsNone(comfy_nodes.check_images("reference", 1))
        self.assertIsNone(comfy_nodes.check_images("text", 0))


if __name__ == "__main__":
    unittest.main()
