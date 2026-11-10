"""Check image selection, retry logging, and rejection of stale gaze."""

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image
import calibration_check
from errors import QuitExperiment


class CalibrationCheckTests(unittest.TestCase):
    def exercise(self, keys, timestamp):
        class Stim:
            def __init__(self, *args, **kwargs):
                pass

            def draw(self):
                pass

        keyboard = iter(keys)
        fake = types.SimpleNamespace(
            event=types.SimpleNamespace(
                clearEvents=lambda: None, getKeys=lambda **kw: next(keyboard)
            ),
            visual=types.SimpleNamespace(ImageStim=Stim, Circle=Stim),
        )
        gaze = types.SimpleNamespace(
            mark_trial=lambda value: None,
            poll=lambda: (500, 500),
            recent_gaze=lambda value: True,
            _last_info_ts=timestamp,
            screen_size=(1000, 1000),
        )
        win = types.SimpleNamespace(size=(1000, 1000), flip=lambda: None)
        with tempfile.TemporaryDirectory() as tmp:
            source, log = Path(tmp) / "photo.png", Path(tmp) / "check.jsonl"
            Image.new("RGB", (100, 100)).save(source)
            with patch.dict(
                "sys.modules", psychopy=fake, ui=types.SimpleNamespace(text=Stim)
            ), patch.object(
                calibration_check.time,
                "time_ns",
                side_effect=iter(range(1000000000, 10000000000, 1200000000)),
            ):
                try:
                    result = calibration_check.check_image(win, gaze, source, log)
                except QuitExperiment:
                    result = "aborted"
            return result, json.loads(log.read_text())

    def test_retry_is_logged(self):
        result, entry = self.exercise([["r"]], 0)
        self.assertFalse(result)
        self.assertEqual(entry["outcome"], "retry")

    def test_stale_dot_cannot_be_accepted(self):
        result, entry = self.exercise([["return"], ["escape"]], 0)
        self.assertEqual(result, "aborted")
        self.assertEqual(entry["outcome"], "aborted")

    def test_fresh_dot_can_be_accepted(self):
        result, entry = self.exercise([["return"]], 2000000000)
        self.assertTrue(result)
        self.assertEqual(entry["outcome"], "accepted")
