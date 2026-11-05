"""Separate sample continuity, fixation linking and descriptive data quality."""
import unittest

from analysis.fixations import detect_fixations
from analysis.quality import recording_quality
from analysis.scanpaths import link_decisions, scanpath


def fixation(number, start, segment=0, frame=0, x=0.5):
    return dict(fixation=number, start_ms=start, end_ms=start + 100, duration_ms=100,
                segment=segment, frame_index=frame, image_u=x, image_v=0.5)


class ScanpathQualityTests(unittest.TestCase):
    def test_long_continuously_observed_interval_can_be_linked(self):
        rows = [dict(t_rel_ms=t, image_u=x, image_v=0.5, image_x=x * 1000,
                     image_y=500, usable=1, frame_index=0)
                for t, x in ((0, .2), (50, .2), (100, .2), (150, .4), (200, .6),
                             (250, .7), (300, .8), (350, .8), (400, .8))]
        events = detect_fixations(rows, max_gap_ms=100)
        self.assertEqual(len(events), 2)
        # The sample-gap rule is unchanged; only the independent link rule changes.
        self.assertEqual(events[0]["segment"], events[1]["segment"])
        self.assertEqual(scanpath(events, max_gap_ms=None)[0]["scanpath_step_count"], 1)
        strict = link_decisions(events, max_gap_ms=100)
        self.assertEqual(strict[0]["reason"], "link_interval_exceeded")
        self.assertEqual(strict[0]["accepted"], 0)

    def test_unlimited_link_interval_never_bridges_missing_data_or_views(self):
        events = [fixation(1, 0), fixation(2, 150, segment=1),
                  fixation(3, 300, segment=2, frame=1)]
        decisions = link_decisions(events, max_gap_ms=None)
        self.assertEqual([d["accepted"] for d in decisions], [0, 0])
        self.assertEqual([d["reason"] for d in decisions], ["segment_break", "view_change"])
        summary, steps = scanpath(events, None, decisions)
        self.assertEqual(steps, [])
        self.assertEqual(summary["scanpath_rejected_count"], 2)
        self.assertEqual(summary["scanpath_rejected_view_change"], 1)
        self.assertEqual(summary["scanpath_segment_count"], 3)

    def test_overlapping_events_are_never_linked(self):
        decisions = link_decisions([fixation(1, 0), fixation(2, 50)], None)
        self.assertEqual(decisions[0]["reason"], "overlapping_events")
        self.assertEqual(decisions[0]["accepted"], 0)

    def test_empty_and_single_fixation_have_no_candidate_pairs(self):
        for events in ([], [fixation(1, 0)]):
            summary, steps = scanpath(events, None)
            self.assertEqual(summary["scanpath_candidate_count"], 0)
            self.assertEqual(steps, [])

    def test_invalid_link_thresholds_are_rejected_even_without_fixations(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                link_decisions([], value)

    def test_quality_detects_silent_gaps_despite_valid_sample_flags(self):
        rows = [dict(t_rel_ms=t, valid=1, usable=1, duplicate=0, on_image=1)
                for t in (0, 70, 140, 500)]
        quality = recording_quality(rows, view_ms=1000)
        self.assertEqual(quality["saved_sample_hz"], 4)
        self.assertEqual(quality["valid_sample_hz"], 4)
        self.assertEqual(quality["invalid_sample_count"], 0)
        self.assertEqual(quality["median_sample_interval_ms"], 70)
        self.assertEqual(quality["p95_sample_interval_ms"], 360)
        self.assertEqual(quality["sample_gap_count"], 1)
        self.assertIn("100 ms", quality["quality_notes"])

    def test_quality_keeps_off_image_gaze_separate_from_invalid_samples(self):
        rows = [dict(t_rel_ms=0, valid=1, usable=0, duplicate=0, on_image=0),
                dict(t_rel_ms=50, valid=1, usable=1, duplicate=0, on_image=1),
                dict(t_rel_ms=50, valid=0, usable=0, duplicate=1, on_image=1)]
        quality = recording_quality(rows, 100)
        self.assertEqual(quality["saved_sample_hz"], 20)
        self.assertEqual(quality["duplicate_sample_count"], 1)
        self.assertEqual(quality["valid_off_image_count"], 1)
        self.assertEqual(quality["invalid_sample_count"], 1)

    def test_empty_trial_quality_remains_explicit(self):
        quality = recording_quality([], 1000)
        self.assertEqual(quality["saved_sample_hz"], 0)
        self.assertEqual(quality["median_sample_interval_ms"], "")
        self.assertIn("No gaze samples", quality["quality_notes"])


if __name__ == "__main__":
    unittest.main()
