"""Manual correction persistence, provenance and analysis integration."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from analysis.detailed_aois import build_regions, LABELS
from analysis.face_aois import AOIMap, sha256
from tools.landmark_editor import Editor


class EditorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.root_patch = patch("tools.landmark_editor.ROOT", self.root)
        self.root_patch.start()
        (self.root / "analysis").mkdir()
        (self.root / "analysis/detailed_aois.py").write_text("fixture")
        model = self.root / "models/face-parsing"
        model.mkdir(parents=True)
        names = {1: "skin", 4: "l_eye", 5: "r_eye", 6: "l_brow", 7: "r_brow"}
        (model / "config.json").write_text(json.dumps(dict(id2label=names)))
        self.source = self.root / "automatic"
        self.source.mkdir()
        self.output = self.root / "manual"
        self.points = np.tile([50.0, 50.0, 0.0], (478, 1))
        for k, q in {
            33: (25, 35),
            133: (40, 35),
            362: (60, 35),
            263: (75, 35),
            152: (50, 95),
            10: (50, 5),
            1: (50, 50),
            234: (10, 50),
            454: (90, 50),
            105: (30, 25),
            334: (70, 25),
            17: (50, 75),
            61: (35, 70),
            291: (65, 70),
        }.items():
            self.points[k, :2] = q
        raw = np.ones((100, 100), dtype="uint8")
        raw[33:38, 25:40] = 4
        raw[33:38, 60:75] = 5
        raw[24:28, 25:40] = 6
        raw[24:28, 60:75] = 7
        conf = np.ones(raw.shape, dtype="float32")
        mask, _ = build_regions(raw, conf, names, self.points)
        manifest = dict(
            schema_version="detailed-v4",
            geometry_sha256=sha256(self.root / "analysis/detailed_aois.py"),
            labels=LABELS,
            min_confidence=0.6,
            stimuli={},
        )
        self.ids = []
        for n in range(2):
            image = self.root / (str(n) + ".png")
            Image.new("RGB", (100, 100), (n, 60, 90)).save(image)
            key = sha256(image)
            self.ids.append(key)
            Image.fromarray(mask).save(self.source / (key + ".png"))
            np.savez_compressed(
                self.source / (key + "_parser.npz"), labels=raw, confidence=conf
            )
            with (self.source / (key + "_landmarks.csv")).open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "image_x", "image_y", "relative_z"])
                w.writerows((i, *p) for i, p in enumerate(self.points))
            manifest["stimuli"][key] = dict(
                stimulus=str(image),
                width=100,
                height=100,
                mask=key + ".png",
                landmarks=key + "_landmarks.csv",
                reviewed=True,
                reviewed_mask_sha256=sha256(self.source / (key + ".png")),
            )
        (self.source / "manifest.json").write_text(json.dumps(manifest))
        self.editor = Editor(self.source, self.output)

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def test_save_reload_preserves_automatic_and_includes_unedited_faces(self):
        before = (self.source / "manifest.json").read_bytes()
        key = self.ids[0]
        points = self.points.copy()
        points[105, 1] -= 5
        result = self.editor.save(key, points.tolist(), None, "Adjusted brow")
        state = self.editor.state(key)
        self.assertEqual(state["points"], points.tolist())
        self.assertEqual(state["note"], "Adjusted brow")
        self.assertEqual(result["changed"], 1)
        self.assertEqual((self.source / "manifest.json").read_bytes(), before)
        m = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(len(m["stimuli"]), 2)
        self.assertFalse(m["stimuli"][key]["reviewed"])
        self.assertNotIn("reviewed_mask_sha256", m["stimuli"][key])
        maps = AOIMap(self.output / "manifest.json")
        with self.assertRaisesRegex(ValueError, "Unreviewed"):
            maps.load(m["stimuli"][key]["stimulus"])
        self.assertIsNotNone(maps.load(m["stimuli"][self.ids[1]]["stimulus"]))
        with self.assertRaisesRegex(ValueError, "Another editor"):
            self.editor.save(key, points.tolist(), None)

    def test_invalid_edits_never_write_masks(self):
        points = self.points.copy()
        points[10, 0] = -1
        with self.assertRaisesRegex(ValueError, "within the image"):
            self.editor.save(self.ids[0], points.tolist(), None)
        points = self.points.copy()
        points[10, 2] = 1
        with self.assertRaisesRegex(ValueError, "Only image"):
            self.editor.render(self.ids[0], points)
        with self.assertRaisesRegex(ValueError, "Unknown image"):
            self.editor.state("../manifest")
        self.assertFalse(self.output.exists())

    def test_every_feature_label_can_be_edited(self):
        key = self.ids[0]
        self.assertEqual(
            set(self.editor.state(key)["original_regions"]),
            {str(k) for k in range(1, 25)},
        )
        for label in range(1, 25):
            with self.subTest(label=label):
                mask, _, _ = self.editor.render(
                    key, self.points, {str(label): [[5, 5], [15, 5], [15, 15], [5, 15]]}
                )
                self.assertEqual(int(mask[10, 10]), label)

    def test_changed_mapping_opens_without_regeneration(self):
        baseline=(self.source/"manifest.json").read_bytes()
        (self.root/"analysis/detailed_aois.py").write_text("updated mapping")
        editor=Editor(self.source,self.output)
        key=self.ids[0]
        state=editor.state(key)
        mask,flags,_=editor.render(key,state["points"])
        np.testing.assert_array_equal(mask,editor.load(key)[3])
        self.assertTrue(any("differs from generation" in f for f in flags))
        editor.save(key,state["points"],None)
        correction=json.loads((self.output/(key+"_correction.json")).read_text())
        self.assertEqual(correction["editor_geometry_sha256"],sha256(self.root/"analysis/detailed_aois.py"))
        self.assertNotEqual(correction["editor_geometry_sha256"],correction["geometry_sha256"])
        self.assertEqual((self.source/"manifest.json").read_bytes(),baseline)

    def test_landmark_changes_do_not_change_primary_masks(self):
        key = self.ids[0]
        points = self.points.copy()
        points[33, 0] += 5
        original = self.editor.load(key)[3]
        mask, _, _ = self.editor.render(key, points)
        fixed = ((original < 19) & (original != 1)) | (original == 255)
        np.testing.assert_array_equal(mask[fixed], original[fixed])

    def test_eyebrow_override_changes_mask_and_survives_reload(self):
        key = self.ids[0]
        regions = {"6": [[26, 24], [35, 24], [35, 26], [26, 26]]}
        mask, _, _ = self.editor.render(key, self.points, regions)
        self.assertEqual(int(mask[25, 39]), 255)  # removed automatic brow
        self.assertEqual(int(mask[25, 30]), 6)
        self.assertEqual(int(mask[25, 65]), 7)  # other side untouched
        result = self.editor.save(
            key, self.points.tolist(), None, "Corrected eyebrow", regions
        )
        self.assertEqual(self.editor.state(key)["regions"], regions)
        with Image.open(self.output / (key + ".png")) as saved:
            np.testing.assert_array_equal(np.asarray(saved), mask)
        restored = self.editor.save(
            key, self.points.tolist(), result["version"], "Restore automatic", {}
        )
        with Image.open(self.output / (key + ".png")) as saved:
            self.assertEqual(int(np.asarray(saved)[25, 39]), 6)

    def test_brow_override_protects_eye_pixels(self):
        key = self.ids[0]
        original = self.editor.load(key)[3]
        mask, flags, _ = self.editor.render(
            key, self.points, {"6": [[20, 20], [45, 20], [45, 39], [20, 39]]}
        )
        np.testing.assert_array_equal(mask[original == 4], original[original == 4])
        self.assertTrue(any("protect" in f for f in flags))
        self.assertEqual(int(mask[21, 21]), 6)

    def test_overlaps_resolve_by_priority_and_crossed_contours_are_rejected(self):
        polygon = [[20, 20], [40, 20], [40, 30], [20, 30]]
        overrides={"6":polygon,"7":polygon}
        mask,flags,_=self.editor.render(self.ids[0],self.points,overrides,["6","7"])
        self.assertEqual(int(mask[25,30]),7)
        self.assertTrue(any("resolved automatically" in flag for flag in flags))
        other,_,_=self.editor.render(self.ids[0],self.points,overrides,["7","6"])
        self.assertEqual(int(other[25,30]),6)
        self.editor.save(self.ids[0],self.points.tolist(),None,"Overlapping regions",overrides,["7","6"])
        self.assertEqual(self.editor.state(self.ids[0])["region_order"],["7","6"])
        with Image.open(self.output/(self.ids[0]+".png")) as saved:
            np.testing.assert_array_equal(np.asarray(saved),other)
        with self.assertRaisesRegex(ValueError,"priority"):
            self.editor.render(self.ids[0],self.points,overrides,["7","7"])
        with self.assertRaises(ValueError):
            self.editor.render(
                self.ids[0],
                self.points,
                {"6": [[20, 20], [40, 30], [20, 30], [40, 20]]},
            )
        with self.assertRaises(ValueError):
            self.editor.render(
                self.ids[0],
                self.points,
                {"6": [[20, float("nan")], [40, 20], [20, 30]]},
            )
