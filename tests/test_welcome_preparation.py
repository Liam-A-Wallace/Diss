"""Welcome stays responsive while background camera preparation completes."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch


class WelcomePreparationTests(unittest.TestCase):
    def run_screen(self, ready, keys):
        footers = []
        stims = []

        class Stim:
            def __init__(self, *args, **kwargs):
                self.text = args[1] if len(args) > 1 else ""
                stims.append(self)

            def draw(self):
                pass

        keyboard = iter(keys)
        readiness = iter(ready)
        fake_psychopy = types.SimpleNamespace(
            core=None,
            event=types.SimpleNamespace(clearEvents=lambda: None,
                                        getKeys=lambda **kwargs: next(keyboard)),
            visual=types.SimpleNamespace(Rect=Stim))
        path = Path(__file__).parents[1] / "screens.py"
        spec = importlib.util.spec_from_file_location("welcome_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "psychopy": fake_psychopy,
            "ui": types.SimpleNamespace(Button=None, rounded_rect=Stim, text=Stim),
            "overlay": types.SimpleNamespace(draw=lambda: None),
        }):
            spec.loader.exec_module(module)
            win = types.SimpleNamespace(flip=lambda: footers.append(stims[-1].text))
            result = module.info_screen(win, "Welcome", allow_recalibration=False,
                                        continue_when=lambda: next(readiness))
        return result, footers

    def test_continue_waits_for_preparation_without_leaving_welcome(self):
        result, footers = self.run_screen([False, True], [["return"], ["return"]])
        self.assertTrue(result)
        self.assertEqual(footers, ["Preparing eye tracker...", "Press ENTER or SPACE to continue"])

    def test_escape_works_while_preparing(self):
        result, footers = self.run_screen([False], [["escape"]])
        self.assertFalse(result)
        self.assertEqual(footers, ["Preparing eye tracker..."])


if __name__ == "__main__":
    unittest.main()
