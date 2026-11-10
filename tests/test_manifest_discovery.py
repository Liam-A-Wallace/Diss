"""Compatibility manifest discovery preserves the curation directory."""
import csv
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


class ManifestDiscoveryTests(unittest.TestCase):
    def test_compatibility_command_prefers_local_images_and_falls_back(self):
        repo = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "curation_manifest_test", repo / "stimulus-curation/build_manifest.py")
        command = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(command)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            curation = root / "stimulus-curation"
            scripts = root / "dataset/scripts"
            scripts.mkdir(parents=True)
            shutil.copyfile(repo / "dataset/scripts/build_60k_manifest.py",
                            scripts / "build_60k_manifest.py")
            local = curation / "deepfake-vs-real-60k"
            fallback = root / "raw/deepfake-vs-real-60k"
            for base, name in ((local, "local"), (fallback, "fallback")):
                (base / "Real").mkdir(parents=True)
                (base / "Fake").mkdir()
                (base / "Real" / f"{name}.png").touch()
                (base / "Fake" / f"{name}.jpg").touch()
            output = curation / "DFR60K_manifest.csv"
            with patch.object(command, "__file__", str(curation / "build_manifest.py")), \
                    patch.dict("os.environ", {"RAW_DATA_ROOT": str(root / "raw")}):
                for name in ("local", "fallback"):
                    with self.subTest(source=name):
                        self.assertEqual(command.find_dataset_root(),
                                         local if name == "local" else fallback)
                        self.assertEqual(command.main([]), 0)
                        with output.open(encoding="utf-8", newline="") as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual({r["image_id"] for r in rows},
                                         {f"dfr60k_real_{name}", f"dfr60k_fake_{name}"})
                        self.assertTrue(all(name in r["image_url"] for r in rows))
                    if name == "local":
                        shutil.rmtree(local)
                shutil.rmtree(fallback)
                self.assertIsNone(command.find_dataset_root())


if __name__ == "__main__":
    unittest.main()
