"""Local model planning and resume without downloading weights or using a GPU."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate_images
import local_generation


class FakeEngine:
    calls = []

    def __init__(self, *args):
        pass

    def generate(self, row):
        self.calls.append((row["id"], row["seed"], row["prompt"]))
        return local_generation.PNG_HEADER + b"test"


class LocalGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "out"
        self.addCleanup(FakeEngine.calls.clear)

    def run_model(self, model, count=2, limit=None, dry=False):
        args = ["--model", model, "--count", str(count), "--out", str(self.out),
                "--seed", "20261105"]
        if limit is not None:
            args += ["--limit", str(limit)]
        if dry:
            args += ["--dry-run"]
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_images.main(args)

    def log(self):
        return json.loads((self.out / "prompts.json").read_text(encoding="utf-8"))

    def test_sd35_uses_both_prompts_and_resumes(self):
        with patch.object(local_generation, "SD35Generator", FakeEngine):
            self.assertEqual(self.run_model("sd3.5-large", count=6, limit=2), 0)
            rows = self.log()["images"]
            self.assertEqual(FakeEngine.calls[0][0:2], ("pair_001_A", 20261105))
            self.assertEqual(FakeEngine.calls[1][0:2], ("pair_001_B", 20261105))
            self.assertNotEqual(FakeEngine.calls[0][2], FakeEngine.calls[1][2])
            self.assertTrue(all(r["prompt_applied"] for r in rows))
            self.assertEqual(self.run_model("sd3.5-large", count=6), 0)
            self.assertEqual(len(FakeEngine.calls), 6)
            self.assertEqual(self.run_model("sd3.5-large", count=8, limit=2), 0)
            self.assertEqual([c[0] for c in FakeEngine.calls[-2:]], ["pair_004_A", "pair_004_B"])

    def test_stylegan_has_distinct_seed_samples_and_no_prompt(self):
        with patch.object(local_generation, "StyleGANGenerator", FakeEngine):
            self.assertEqual(self.run_model("stylegan3", count=2), 0)
            rows = self.log()["images"]
            self.assertEqual([r["seed"] for r in rows], [20261105, 20261106])
            self.assertTrue(all(r["prompt"] is None and not r["prompt_applied"] for r in rows))
            self.assertEqual(self.run_model("stylegan3", count=4), 0)
            self.assertEqual([c[0] for c in FakeEngine.calls],
                             ["pair_001_A", "pair_001_B", "pair_002_A", "pair_002_B"])
            self.assertEqual(len(list(self.out.glob("*.png"))), 4)

    def test_dry_run_needs_no_engine_or_output(self):
        with patch.object(local_generation, "SD35Generator", side_effect=AssertionError("loaded")):
            self.assertEqual(self.run_model("sd3.5-large", dry=True), 0)
        self.assertFalse(self.out.exists())

    def test_model_switch_rejects_existing_folder(self):
        with patch.object(local_generation, "StyleGANGenerator", FakeEngine), patch.object(
                local_generation, "SD35Generator", FakeEngine):
            self.assertEqual(self.run_model("stylegan2"), 0)
            before = (self.out / "prompts.json").read_bytes()
            self.assertEqual(self.run_model("sd3.5-large"), 1)
            self.assertEqual((self.out / "prompts.json").read_bytes(), before)

    def test_interrupted_local_generation_retries_without_duplicate_file(self):
        class InterruptedEngine(FakeEngine):
            def generate(self, row):
                raise KeyboardInterrupt()

        with patch.object(local_generation, "SD35Generator", InterruptedEngine):
            self.assertEqual(self.run_model("sd3.5-large"), 130)
        self.assertEqual(self.log()["images"][0]["status"], "pending")
        with patch.object(local_generation, "SD35Generator", FakeEngine):
            self.assertEqual(self.run_model("sd3.5-large"), 0)
        self.assertEqual(len(list(self.out.glob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
