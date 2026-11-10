"""Text fitting must measure refreshed glyphs after each font-size change."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
