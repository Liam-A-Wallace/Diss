"""Tests for aggregated study summary, region-visit scanpaths, entropy, and group heatmaps."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import analyse
from analysis.face_aois import AOIMap, sha256
from analysis.process import process
from analysis.summary import (
    collapse_region_visits,
    compute_scanpath_metrics,
    group_aoi,
    levenshtein_similarity,
    summarise,
)
from unittest.mock import patch
from test_analysis import frame


def _write_session(root, name, response, correct, gaze_xy_seq, reason=""):
    image = root / "face.png"
    if not image.exists():
        Image.new("RGB", (200, 200), "grey").save(image)
    digest = sha256(image)
    base = root / name
    meta = dict(
        participant={"participant": name},
        stimuli_sha256={str(image): digest},
    )
    Path(str(base) + "_session.json").write_text(json.dumps(meta), encoding="utf-8")
    f = dict(
        frame(1000000000),
        stimulus=str(image),
        window_width=200,
        window_height=200,
        tracker_width=200,
        tracker_height=200,
        image_width=200,
        image_height=200,
        left=0,
        top=0,
        scale=1,
        viewport=[0, 0, 200, 200],
        condition="fake",
    )
    last_ts = 1000000000 + len(gaze_xy_seq) * 50_000_000
    events = [f, dict(kind="offset", timestamp_ns=last_ts, trigger=1)]
    Path(str(base) + "_display.jsonl").write_text(
        "\n".join(map(json.dumps, events)), encoding="utf-8"
    )
    with Path(str(base) + "_gaze_raw.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "status", "calibrated_gaze_position_x", "calibrated_gaze_position_y"])
        for i, (x, y) in enumerate(gaze_xy_seq):
            w.writerow([1000000000 + i * 50_000_000, 1, x, y])
    with Path(str(base) + ".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trigger", "difficulty", "response", "correct", "confidence", "decision_reason"])
        w.writerow([1, "openai_xhigh_A", response, correct, 4, reason])

    # Top half = image_left_eye (4), bottom half = chin_derived (24)
    mask_path = root / "mask.png"
    if not mask_path.exists():
        mask = Image.new("L", (200, 200), 4)
        for y in range(100, 200):
            for x in range(200):
                mask.putpixel((x, y), 24)
        mask.save(mask_path)
    manifest = root / "manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                dict(
                    labels={"4": "image_left_eye", "24": "chin_derived"},
                    stimuli={digest: dict(mask="mask.png", reviewed=True)},
                )
            ),
            encoding="utf-8",
        )
    out = root / ("processed_" + name)
    process(base, out, aois=manifest, plots=True)
    return out


class SummaryTests(unittest.TestCase):
    def test_region_visit_collapse_and_entropy_metrics(self):
        fixations = [
            dict(fixation=1, start_ms=0, end_ms=200, duration_ms=200, image_x=50, image_y=50, aoi="image_left_eye", segment=0),
            dict(fixation=2, start_ms=220, end_ms=420, duration_ms=200, image_x=60, image_y=50, aoi="image_right_eye", segment=0),
            dict(fixation=3, start_ms=450, end_ms=650, duration_ms=200, image_x=50, image_y=150, aoi="chin_derived", segment=0),
            dict(fixation=4, start_ms=700, end_ms=900, duration_ms=200, image_x=55, image_y=50, aoi="image_left_eye", segment=1),
        ]
        visits = collapse_region_visits(fixations)
        self.assertEqual([v["region"] for v in visits], ["eyes", "mouth_chin", "eyes"])
        self.assertEqual(visits[0]["duration_ms"], 400.0)
        self.assertEqual(visits[0]["fixation_count"], 2)
        self.assertTrue(visits[2]["segment_break_before"])

        metrics = compute_scanpath_metrics(fixations)
        self.assertEqual(metrics["collapsed_aoi_sequence"], "eyes > mouth_chin > eyes")
        self.assertEqual(metrics["region_visit_count"], 3)
        self.assertEqual(metrics["unique_regions_visited"], 2)
        self.assertEqual(metrics["revisit_count"], 1)
        self.assertEqual(metrics["revisited_regions"], "eyes")
        self.assertGreater(metrics["stationary_entropy_bits"], 0)
        # Only (eyes -> mouth_chin) is within the same segment; mouth_chin -> eyes crosses segment 0 -> 1.
        self.assertEqual(metrics["transition_entropy_bits"], 0.0)

    def test_unknown_aoi_not_lumped_into_periphery_and_segment_breaks_stop_merge(self):
        self.assertEqual(group_aoi("mystery_label"), "uncertain")
        fixations = [
            dict(fixation=1, start_ms=0, end_ms=200, duration_ms=200, image_x=50, image_y=50, aoi="image_left_eye", segment=0),
            dict(fixation=2, start_ms=250, end_ms=450, duration_ms=200, image_x=50, image_y=50, aoi="uncertain", segment=1),
            dict(fixation=3, start_ms=500, end_ms=700, duration_ms=200, image_x=50, image_y=50, aoi="image_left_eye", segment=1),
        ]
        visits = collapse_region_visits(fixations)
        self.assertEqual(len(visits), 2)
        self.assertTrue(visits[1]["segment_break_before"])
        metrics = compute_scanpath_metrics(fixations)
        self.assertEqual(metrics["collapsed_aoi_sequence"], "eyes")
        self.assertEqual(metrics["revisit_count"], 0)
        self.assertEqual(metrics["transition_entropy_bits"], "")

    def test_early_ms_validation_and_difficulty_column_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = _write_session(root, "P1", "fake", 1, [(50, 50)] * 5 + [(50, 150)] * 5)
            with self.assertRaisesRegex(ValueError, "early_ms"):
                summarise([p1], root / "bad", plots=False, early_ms=0)
            out = root / "ok"
            summarise([p1], out, plots=False, early_ms=2000)
            with (out / "data_results" / "behaviour_by_difficulty.csv").open(newline="", encoding="utf-8") as f:
                self.assertEqual(csv.DictReader(f).fieldnames[0], "difficulty")

    def test_duplicate_sessions_rejected_and_all_filters_by_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = [(50, 50)] * 5 + [(50, 150)] * 5
            p1 = _write_session(root, "P1", "fake", 1, seq)
            with self.assertRaisesRegex(ValueError, "Duplicate session"):
                summarise([p1, p1], root / "dup_out", plots=False)

            # Verify --all isolates data/ from data/test/
            (root / "processed" / "main_run" / "data_results").mkdir(parents=True)
            (root / "processed" / "test_run" / "data_results").mkdir(parents=True)
            (root / "processed" / "main_run" / "data_results" / "features.csv").write_text("session\nmain\n")
            (root / "processed" / "test_run" / "data_results" / "features.csv").write_text("session\ntest\n")
            (root / "processed" / "main_run" / "analysis.json").write_text(
                json.dumps({"source_session": str(root / "data" / "s1")})
            )
            (root / "processed" / "test_run" / "analysis.json").write_text(
                json.dumps({"source_session": str(root / "data" / "test" / "s2")})
            )
            with patch.object(analyse, "ROOT", root):
                main_dirs = analyse._find_processed_sessions(root / "data", [], use_all=True)
                test_dirs = analyse._find_processed_sessions(root / "data" / "test", [], use_all=True)
            self.assertEqual([p.name for p in main_dirs], ["main_run"])
            self.assertEqual([p.name for p in test_dirs], ["test_run"])

    def test_detailed_unreviewed_error_includes_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = root / "face.png"
            Image.new("RGB", (20, 20)).save(img)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    dict(
                        schema_version="detailed-v4",
                        stimuli={sha256(img): dict(reviewed=False)},
                    )
                )
            )
            with self.assertRaisesRegex(ValueError, "--detailed"):
                AOIMap(manifest).load(str(img))

    def test_levenshtein_similarity(self):
        self.assertEqual(levenshtein_similarity(["eyes", "mouth_chin"], ["eyes", "mouth_chin"]), 1.0)
        self.assertEqual(levenshtein_similarity(["eyes", "mouth_chin"], ["periphery", "forehead_hair"]), 0.0)
        self.assertAlmostEqual(levenshtein_similarity(["eyes", "mouth_chin", "eyes"], ["eyes", "mouth_chin"]), 2 / 3)

    def test_multi_session_summary_generates_tables_similarity_and_group_heatmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # P1 looks at eyes then chin then eyes (correct)
            seq1 = [(50, 50)] * 5 + [(50, 150)] * 5 + [(50, 50)] * 5
            # P2 looks at chin then eyes (incorrect)
            seq2 = [(50, 150)] * 6 + [(50, 50)] * 6
            p1 = _write_session(root, "P1", "fake", 1, seq1, reason="eyes looked odd")
            p2 = _write_session(root, "P2", "real", 0, seq2, reason="")
            self.assertTrue((p1 / "trial_1_qc.png").is_file())

            out = root / "summary"
            report = summarise([p1, p2], out, plots=True)
            self.assertTrue(report.is_file())
            tables = out / "data_results"
            expected = {
                "behaviour_by_source.csv",
                "behaviour_by_difficulty.csv",
                "aoi_dwell_by_source.csv",
                "aoi_dwell_by_outcome.csv",
                "early_vs_late_aoi.csv",
                "time_to_first_fixation.csv",
                "scanpath_metrics.csv",
                "scanpath_similarity.csv",
            }
            self.assertEqual({p.name for p in tables.glob("*.csv")}, expected)
            with (tables / "scanpath_similarity.csv").open(newline="", encoding="utf-8") as f:
                sim_rows = list(csv.DictReader(f))
            self.assertEqual(len(sim_rows), 1)
            self.assertEqual(sim_rows[0]["pair_outcome"], "mixed")
            self.assertTrue(list(out.glob("group_heatmap_*.png")))
            html = report.read_text(encoding="utf-8")
            self.assertIn("eyes looked odd", html)
            self.assertIn("openai_xhigh", html)


if __name__ == "__main__":
    unittest.main()
