"""Text fitting must measure refreshed glyphs after each font-size change."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from errors import QuitExperiment


class TextFitTests(unittest.TestCase):
    def test_cached_bounds_do_not_shrink_text_to_nothing(self):
        spec = importlib.util.spec_from_file_location(
            "ui_under_test", Path(__file__).parents[1] / "ui.py")
        ui = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "psychopy": types.SimpleNamespace(visual=None),
        }):
            spec.loader.exec_module(ui)

        class CachedText:
            win = types.SimpleNamespace(size=(1920, 1080))
            height = 0.032
            boundingBox = (400, 60)

            def setText(self, value, log=False):
                # Like PsychoPy, only rebuilding the layout updates bounds.
                self.boundingBox = (400 * self.height / 0.032,
                                    60 * self.height / 0.032)

        stim = CachedText()
        ui.fit_text(stim, 1.08, 0.04)
        self.assertGreater(stim.height, 0.02)
        self.assertLessEqual(stim.boundingBox[1], 0.04 * 1080)

    def test_window_close_is_turned_into_quit(self):
        spec = importlib.util.spec_from_file_location(
            "ui_close_under_test", Path(__file__).parents[1] / "ui.py")
        ui = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "psychopy": types.SimpleNamespace(visual=None),
        }):
            spec.loader.exec_module(ui)

        class Window:
            def __init__(self):
                self.flips = 0
                self.winHandle = types.SimpleNamespace(has_exit=False)

            def flip(self):
                self.flips += 1

        win = Window()
        ui.install_close_guard(win)
        win.flip()
        self.assertEqual(win.flips, 1)

        win.winHandle.has_exit = True
        with self.assertRaises(QuitExperiment):
            win.flip()
        # Cleanup flips still run so the partial trial can be saved.
        win.flip()
        self.assertEqual(win.flips, 2)


if __name__ == "__main__":
    unittest.main()
