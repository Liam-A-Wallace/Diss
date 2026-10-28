"""Regressions for export validation, worker configuration and scanpath gaps."""
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from analysis.scanpaths import scanpath
from gaze_worker import worker_entry


class RevisionTests(unittest.TestCase):
    def test_missing_export_source_preserves_existing_set(self):
        path = Path(__file__).parents[1] / "dataset/scripts/export_to_diss.py"
        spec = importlib.util.spec_from_file_location("export_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dest = root / "experiment"
            (dest / "stimuli/real").mkdir(parents=True)
            original = dest / "stimuli/real/existing.png"
            original.write_bytes(b"preserve me")
            trials = dest / "trials.csv"
            trials.write_text("existing trials")
            picks = root / "picked.csv"
            picks.write_text("label,picked_file\nREAL,missing.png\n")
            with patch.object(module, "RAW_DATA_ROOT", root), self.assertRaises(SystemExit):
                module.main(["--picked", str(picks), "--diss", str(dest)])
            self.assertEqual(original.read_bytes(), b"preserve me")
            self.assertEqual(trials.read_text(), "existing trials")

    def test_worker_initializes_patch_and_forwards_model_and_exposure(self):
        calls = []
        factory = lambda kind, component, kwargs: calls.append((component, kwargs))
        worker = types.SimpleNamespace(create_component=factory)
        def run(*args):
            worker.create_component("mgazenet", "gaze_estimator")
            worker.create_component("webcam", "camera", {"webcam_id": 1})
        worker.mp_worker_entry = run
        with patch.dict("sys.modules", {
            "gazefollower_patch": types.SimpleNamespace(apply=lambda: calls.append("patched")),
            "gazefollower": types.ModuleType("gazefollower"),
            "gazefollower.multiprocess": types.SimpleNamespace(mp_worker=worker),
            "gazefollower.camera": types.SimpleNamespace(WebCamCamera=lambda **kw: calls.append(("camera", kw)))
        }):
            worker_entry(model_path="selected.mnn", lock_exposure=True, exposure=-5.5)
        self.assertEqual(calls, ["patched", ("gaze_estimator", {"model_path": "selected.mnn"}),
                                ("camera", {"webcam_id": 1, "lock_exposure": True, "exposure": -5.5})])
        self.assertIs(worker.create_component, factory)

    def test_scanpath_does_not_connect_gaps_and_measures_reversals(self):
        def fixation(i, x, start, segment=0):
            return dict(fixation=i, image_u=x, image_v=0.5, start_ms=start,
                        end_ms=start + 100, duration_ms=100, segment=segment)
        items = [fixation(1, 0.2, 0), fixation(2, 0.5, 150), fixation(3, 0.2, 300),
                 fixation(4, 0.9, 2000), fixation(5, 0.1, 2150, segment=1)]
        summary, steps = scanpath(items)
        self.assertEqual(len(steps), 2)
        self.assertAlmostEqual(summary["scanpath_straightness"], 0)
        self.assertEqual(summary["scanpath_mean_turn_degrees"], 180)
        self.assertEqual(summary["scanpath_segment_count"], 3)
        self.assertAlmostEqual(sum(s["distance_image_units"] for s in steps), 0.6)
        self.assertEqual(scanpath([])[0]["scanpath_segment_count"], 0)


if __name__ == "__main__":
    unittest.main()