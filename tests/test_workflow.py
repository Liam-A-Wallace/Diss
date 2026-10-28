"""Regressions found during the simplicity and correctness review."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image
import analyse
from analysis.face_aois import AOIMap, generate, sha256
from analysis.fixations import detect_fixations
from analysis.ingest import load_display
from analysis.process import continuous, process
from test_analysis import frame, sample


class WorkflowTests(unittest.TestCase):
    def session(self, root, name="session", test=False):
        image = root / "face.png"
        Image.new("RGB", (1000, 1000), "grey").save(image)
        data = root / "data" / "test" if test else root / "data"
        data.mkdir(parents=True, exist_ok=True)
        base = data / name
        metadata = dict(status="complete", participant={"participant": "SYNTHETIC"},
                        project_root=str(root), stimuli_sha256={"face.png": sha256(image)},
                        planned_trials=[{"stimulus": "face.png"}], practice=[{"stimulus": "practice.png"}])
        Path(str(base) + "_session.json").write_text(json.dumps(metadata))
        events = [frame(), {"kind": "offset", "trigger": 1, "timestamp_ns": 1250000000}]
        Path(str(base) + "_display.jsonl").write_text("\n".join(map(json.dumps, events)))
        with Path(str(base) + "_gaze_raw.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "status", "calibrated_gaze_position_x", "calibrated_gaze_position_y"])
            writer.writerows((1000000000 + i * 50000000, 1, 640, 360) for i in range(5))
        return base, image

    def manifest(self, root, image, reviewed=True):
        directory = root / "aois" / "main"
        directory.mkdir(parents=True)
        mask = directory / "mask.png"
        Image.new("L", (1000, 1000), 2).save(mask)
        manifest = {"labels": {"2": "nose"}, "stimuli": {
            sha256(image): dict(stimulus=str(image), mask="mask.png", reviewed=reviewed,
                                reviewed_mask_sha256=sha256(mask), notes=[])}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory, mask

    def test_short_command_runs_and_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, image = self.session(root)
            self.session(root, "newer_test", test=True)
            with patch.object(analyse, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                analyse.main(["--no-open", "--no-plots"])
                analyse.main(["--no-open", "--no-plots"])
            self.assertTrue((root / "processed/session/index.html").is_file())
            self.assertTrue((root / "processed/session_2/data_results/features.csv").is_file())
            self.assertFalse((root / "processed/newer_test").exists())
            output = root / "processed/session"
            expected = {"features.csv", "samples.csv", "fixations.csv", "aoi_dwell.csv", "transitions.csv", "scanpaths.csv"}
            self.assertEqual({p.name for p in (output / "data_results").glob("*.csv")}, expected)
            self.assertEqual(list(output.glob("*.csv")), [])
            report = (output / "index.html").read_text(encoding="utf-8")
            for name in expected:
                self.assertIn('href="data_results/' + name + '"', report)
            with (root / "processed/session/data_results/features.csv").open(newline="") as f:
                self.assertEqual(next(csv.DictReader(f))["fixation_count"], "1")

    def test_no_output_created_when_masks_are_unreviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, image = self.session(root)
            directory, _ = self.manifest(root, image, reviewed=False)
            with self.assertRaisesRegex(ValueError, "analyse.py review"):
                process(base, root / "output", directory / "manifest.json", plots=False)
            self.assertFalse((root / "output").exists())

    def test_review_saves_mask_hash_without_json_editing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, image = self.session(root)
            directory, mask = self.manifest(root, image, reviewed=False)
            with patch("builtins.input", return_value="reviewed"), contextlib.redirect_stdout(io.StringIO()):
                analyse.review_aois(directory, open_report=False)
            manifest = json.loads((directory / "manifest.json").read_text())
            record = manifest["stimuli"][sha256(image)]
            self.assertTrue(record["reviewed"])
            self.assertEqual(record["reviewed_mask_sha256"], sha256(mask))
            self.assertTrue((directory / "index.html").exists())

    def test_edited_mask_requires_review_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, image = self.session(root)
            directory, mask = self.manifest(root, image)
            Image.new("L", (1000, 1000), 0).save(mask)
            with self.assertRaisesRegex(ValueError, "changed after review"):
                AOIMap(directory / "manifest.json").load(str(image))

    def test_short_dropout_cannot_reconnect_fixations(self):
        rows = [sample(t) for t in (0, 50, 100)]
        rows += [sample(150, usable=0)]
        rows += [sample(t, x=0.6) for t in (200, 250, 300)]
        fixations = detect_fixations(rows)
        self.assertEqual(len(fixations), 2)
        self.assertLessEqual(fixations[1]["start_ms"] - fixations[0]["end_ms"], 100)
        self.assertFalse(continuous(*fixations, 100))

    def test_truncated_display_tail_preserves_complete_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "display.jsonl"
            path.write_text(json.dumps(frame()) + '\n{"kind":')
            with self.assertWarnsRegex(RuntimeWarning, "incomplete final"):
                events = load_display(path)
            self.assertEqual(len(events), 1)
            path.write_text('{invalid}\n' + json.dumps(frame()))
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_display(path)

    def test_aois_include_recorded_practice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.session(root)
            self.assertEqual(analyse.stimulus_paths(root / "data"),
                             [root / "face.png", root / "practice.png"])

    def test_running_session_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, _ = self.session(root)
            path = Path(str(base) + "_session.json")
            metadata = json.loads(path.read_text())
            metadata["status"] = "running"
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "still marked as running"):
                analyse.latest_session(root / "data")

    def test_repeated_generation_preserves_review_and_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, image = self.session(root)
            directory, mask = self.manifest(root, image)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            path = directory / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest.update(model_files={"config.json": sha256(model / "config.json")}, min_confidence=0.6)
            path.write_text(json.dumps(manifest))
            original_manifest, original_mask = path.read_bytes(), mask.read_bytes()
            fake_transformers = types.SimpleNamespace(SegformerImageProcessor=None, SegformerForSemanticSegmentation=None)
            with patch.dict("sys.modules", torch=types.SimpleNamespace(), transformers=fake_transformers), contextlib.redirect_stdout(io.StringIO()):
                generate([image], model, directory)
            self.assertEqual(path.read_bytes(), original_manifest)
            self.assertEqual(mask.read_bytes(), original_mask)


if __name__ == "__main__":
    unittest.main()
