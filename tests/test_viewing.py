"""Exercise viewing lifecycle without opening a window or starting a camera."""
import importlib.util
import itertools
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
from PIL import Image
from errors import QuitExperiment
from session_log import SessionLog, export_image_gaze
from analysis.ingest import load_display


class Window:
    size = (1920, 1080)
    nDroppedFrames = 0

    def __init__(self):
        self.callbacks = []

    def callOnFlip(self, callback, *args):
        self.callbacks.append((callback, args))

    def flip(self):
        callbacks, self.callbacks = self.callbacks, []
        for callback, args in callbacks:
            callback(*args)


class Stim:
    def __init__(self, *args, **kwargs):
        pass

    def draw(self):
        pass


class Mouse:
    def getPos(self):
        return (0, 0)

    def getPressed(self):
        return (False, False, False)

    def getWheelRel(self):
        return (0, 0)


class ViewingTests(unittest.TestCase):
    def exercise(self, interrupt):
        key_events = iter([[], ["escape" if interrupt else "return"]])
        event = types.SimpleNamespace(getKeys=lambda **kwargs: next(key_events))
        fake_psychopy = types.SimpleNamespace(event=event, visual=types.SimpleNamespace(Rect=Stim))
        fake_overlay = types.SimpleNamespace(draw=lambda: None, hide=lambda: None)
        spec = importlib.util.spec_from_file_location("viewing_test_module", Path(__file__).parents[1] / "viewing.py")
        viewing = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", psychopy=fake_psychopy, overlay=fake_overlay):
            spec.loader.exec_module(viewing)
        ticks = itertools.count(1000000000, 16666667)
        viewing.time = types.SimpleNamespace(time_ns=lambda: next(ticks))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "face.png"
            Image.new("RGB", (100, 100)).save(path)
            base = str(Path(tmp) / "session")
            session = SessionLog(base, {"participant": "SYNTHETIC"}, (1920, 1080),
                                 (1920, 1080), {}, [{"stimulus": str(path)}], [])
            windows = []
            gaze = types.SimpleNamespace(mark_trial=lambda value: None,
                                         record_window=lambda *args: windows.append(args))
            args = (Window(), Mouse(), Stim(), str(path), Stim(), Stim(), gaze,
                    session, 1, 1, False, "real")
            if interrupt:
                with self.assertRaises(QuitExperiment):
                    viewing.view_image(*args)
            else:
                stamps, dropped = viewing.view_image(*args)
                self.assertGreater(stamps["offset"], stamps["onset"])
            session.finish("aborted" if interrupt else "complete")
            events = load_display(base + "_display.jsonl")
            self.assertEqual([e["kind"] for e in events], ["present", "offset"])
            self.assertEqual(events[-1]["interrupted"], interrupt)
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0][-2:], (events[0]["timestamp_ns"], events[1]["timestamp_ns"]))

    def test_normal_completion(self):
        self.exercise(False)

    def test_escape_preserves_viewing_window(self):
        self.exercise(True)


if __name__ == "__main__":
    unittest.main()
