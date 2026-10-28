import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from image_view import ImageView, map_gaze
from analysis.ingest import align_samples, load_display
from analysis.fixations import detect_fixations
from analysis.process import process, sample_durations, smooth_dwell
from analysis.face_aois import sha256, AOIMap, derive_regions


def frame(ts=1000000000, **kwargs):
    return dict(kind="present", timestamp_ns=ts, trial=1, trigger=1,
                stimulus="face.png", is_practice=0, condition="real",
                window_width=1920, window_height=1080, tracker_width=1280,
                tracker_height=720, image_width=1000, image_height=1000,
                left=460, top=40, scale=1, zoom=1, viewport=[0, 40, 1920, 1000], **kwargs)


def sample(t, x=0.5, usable=1, frame_index=0):
    return dict(t_rel_ms=t, image_u=x, image_v=0.5, image_x=x * 1000,
                image_y=500, usable=usable, frame_index=frame_index)


class HeatmapTests(unittest.TestCase):
    def test_display_smoothing_preserves_dwell_even_at_image_edges(self):
        import numpy as np
        heat = np.zeros((40, 40))
        heat[0, 0], heat[39, 39], heat[20, 10] = 0.1, 0.4, 1.2
        original = heat.copy()
        blurred = smooth_dwell(heat)
        self.assertEqual(blurred.shape, heat.shape)
        self.assertAlmostEqual(float(blurred.sum()), float(heat.sum()), places=12)
        self.assertTrue(np.all(blurred >= 0))
        np.testing.assert_array_equal(heat, original)
        np.testing.assert_array_equal(smooth_dwell(np.zeros((40, 40))), np.zeros((40, 40)))


class GeometryTests(unittest.TestCase):
    def test_different_tracker_resolution_and_y_origin(self):
        self.assertEqual(map_gaze(640, 360, frame()), (500, 500, 0.5, 0.5, True))
        self.assertEqual(map_gaze(460 / 1.5, 40 / 1.5, frame())[:2], (0, 0))

    def test_offscreen_not_clamped(self):
        mapped = map_gaze(-100, 0, frame())
        self.assertLess(mapped[0], 0)
        self.assertFalse(mapped[-1])
        self.assertFalse(map_gaze(float("nan"), 360, frame())[-1])

    def test_zoom_anchor_and_bounds(self):
        view = ImageView(1000, 1000, (0, 0, 1000, 1000))
        anchor = (300, 400)
        before = ((anchor[0] - view.left) / view.scale, (anchor[1] - view.top) / view.scale)
        view.scroll(3, *anchor)
        after = ((anchor[0] - view.left) / view.scale, (anchor[1] - view.top) / view.scale)
        self.assertAlmostEqual(before[0], after[0])
        self.assertAlmostEqual(before[1], after[1])
        view.scroll(1000, *anchor)
        self.assertEqual(view.zoom, 4)
        view.pan(100000, -100000)
        self.assertEqual((view.left, view.top), (0, -3000))
        view.reset()
        self.assertEqual((view.left, view.top, view.zoom), (0, 0, 1))

    def test_wide_and_portrait_fit(self):
        for width, height in ((4000, 1000), (500, 2000)):
            v = ImageView(width, height, (0, 75, 1920, 900))
            self.assertLessEqual(width * v.scale, 1920)
            self.assertLessEqual(height * v.scale, 900)
            before = v.snapshot()
            v.scroll(10, 20, 0)
            self.assertEqual(before, v.snapshot())


class PipelineTests(unittest.TestCase):
    def raw(self, directory, timestamps):
        path = directory / "raw.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "status", "calibrated_gaze_position_x", "calibrated_gaze_position_y"])
            writer.writerows((t, 1, 640, 360) for t in timestamps)
        return path

    def test_exact_transform_boundary_and_exclusive_offset(self):
        initial = frame()
        zoomed = dict(frame(1100000000), left=-40, top=-460, scale=2, zoom=2)
        events = [initial, zoomed, dict(kind="offset", timestamp_ns=1200000000, trigger=1)]
        with tempfile.TemporaryDirectory() as d:
            rows = list(align_samples(self.raw(Path(d), [999999999, 1000000000, 1099999999,
                                                        1100000000, 1200000000]), events))
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["frame_index"] for r in rows], [0, 0, 1])
        self.assertEqual(rows[-1]["image_x"], 500)

    def test_duplicate_and_unclosed_samples(self):
        with tempfile.TemporaryDirectory() as d:
            raw = self.raw(Path(d), [1000000000, 1000000000])
            self.assertEqual(list(align_samples(raw, [frame()])), [])
            rows = list(align_samples(raw, [frame(), dict(kind="offset", timestamp_ns=1200000000, trigger=1)]))
            self.assertEqual([r["usable"] for r in rows], [1, 0])

    def test_fixation_gap_and_transform_split(self):
        rows = [sample(t) for t in (0, 50, 100)] + [sample(150, usable=0)]
        rows += [sample(t) for t in (200, 250, 300)]
        rows += [sample(t, frame_index=1) for t in (350, 400, 450)]
        rows += [sample(t, frame_index=1) for t in (1000, 1050, 1100)]
        fixations = detect_fixations(rows)
        self.assertEqual(len(fixations), 4)
        self.assertEqual([r["duration_ms"] for r in fixations], [100] * 4)
        sample_durations(rows, 100)
        self.assertEqual(sum(r["dwell_ms"] for r in rows), 400)

    def test_dispersion_rejects_movement(self):
        rows = [sample(t, x=i * 0.2) for i, t in enumerate((0, 50, 100, 150))]
        self.assertEqual(detect_fixations(rows), [])

    def test_end_to_end_with_reviewed_aoi_and_zero_samples(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            image = d / "face.png"
            Image.new("RGB", (1000, 1000), "gray").save(image)
            digest = sha256(image)
            base = d / "session"
            meta = dict(participant=dict(participant="SYNTHETIC"), stimuli_sha256={str(image): digest})
            Path(str(base) + "_session.json").write_text(json.dumps(meta))
            f = dict(frame(), stimulus=str(image))
            events = [f, dict(kind="offset", timestamp_ns=1250000000, trigger=1),
                      dict(f, trigger=2, trial=2, timestamp_ns=1300000000),
                      dict(kind="offset", timestamp_ns=1400000000, trigger=2, interrupted=True)]
            Path(str(base) + "_display.jsonl").write_text("\n".join(map(json.dumps, events)))
            self.raw(d, [1000000000 + i * 50000000 for i in range(5)]).rename(str(base) + "_gaze_raw.csv")
            Image.new("L", (1000, 1000), 2).save(d / "mask.png")
            manifest = d / "manifest.json"
            manifest.write_text(json.dumps(dict(labels={2: "nose"}, stimuli={digest: dict(mask="mask.png", reviewed=True)})))
            rows = process(base, d / "processed", manifest, plots=False)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["first_fixation_aoi"], "nose")
            self.assertEqual(rows[0]["fixation_count"], 1)
            self.assertEqual(rows[0]["observed_dwell_ms"], 200)
            self.assertEqual(rows[1]["sample_count"], 0)
            self.assertEqual(rows[1]["interrupted"], 1)
            self.assertTrue((d / "processed" / "data_results" / "samples.csv").exists())
            self.assertTrue((d / "processed" / "analysis.json").exists())
            with self.assertRaises(FileExistsError):
                process(base, d / "processed", plots=False)

    def test_unreviewed_aoi_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "face.png"
            Image.new("RGB", (10, 10)).save(p)
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps(dict(stimuli={sha256(p): dict(reviewed=False)})))
            with self.assertRaises(ValueError):
                AOIMap(manifest).load(str(p))

    def test_derived_aois_and_uncertainty(self):
        import numpy as np
        labels = np.ones((100, 100), dtype=int)
        labels[:20] = 13
        labels[50:55] = 6
        confidence = np.ones((100, 100))
        confidence[80:90] = 0.1
        mask, notes = derive_regions(labels, confidence, {1:"skin", 6:"l_brow", 13:"hair"})
        self.assertEqual(mask[30, 50], 14)
        self.assertEqual(mask[20, 50], 15)
        self.assertEqual(mask[85, 50], 255)
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
