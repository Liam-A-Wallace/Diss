"""Optional explanations preserve text, allow skipping, and clean up on exit."""
import ast
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

from errors import QuitExperiment

ROOT = Path(__file__).parents[1]


class DecisionReasonTests(unittest.TestCase):
    def screen(self, value="", escape=False):
        boxes = []

        class Stim:
            def __init__(self, *args, **kwargs):
                self.text = kwargs.get("text", "")
                self.editable = kwargs.get("editable", False)

            def draw(self, *args):
                pass

            def contains(self, pos):
                return True

        def textbox(*args, **kwargs):
            box = Stim(*args, **kwargs)
            boxes.append(box)
            return box

        presses = iter([True, True, False, True])
        mouse = types.SimpleNamespace(getPos=lambda: (0, 0),
                                      getPressed=lambda: (next(presses), False, False))
        flips = []

        def flip():
            flips.append(1)
            boxes[-1].text = value

        win = types.SimpleNamespace(callOnFlip=lambda callback: callback(), flip=flip)
        event = types.SimpleNamespace(clearEvents=Mock(),
                                      getKeys=lambda **kwargs: ["escape"] if escape else [])
        psychopy = types.SimpleNamespace(
            event=event, core=types.SimpleNamespace(Clock=lambda: types.SimpleNamespace(
                reset=lambda: None, getTime=lambda: 1.25)),
            visual=types.SimpleNamespace(TextBox2=textbox))
        spec = importlib.util.spec_from_file_location("reason_screens", ROOT / "screens.py")
        screens = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "psychopy": psychopy,
            "ui": types.SimpleNamespace(Button=Stim, rounded_rect=Stim, text=Stim,
                                        fit_text=lambda *args: None),
            "overlay": types.SimpleNamespace(draw=lambda: None),
        }):
            spec.loader.exec_module(screens)
            if escape:
                with self.assertRaises(QuitExperiment):
                    screens.explanation_screen(win, mouse, "real", Stim())
                result = None
            else:
                result = screens.explanation_screen(win, mouse, "fake", Stim())
        self.assertFalse(boxes[0].editable)
        return result, flips

    def test_preserves_multiline_punctuation_and_waits_for_fresh_click(self):
        result, flips = self.screen('  Skin looked unusual, especially "around the eyes".\nHair too.  ')
        self.assertEqual(result, ('Skin looked unusual, especially "around the eyes".\nHair too.', 1.25))
        self.assertEqual(len(flips), 3)

    def test_blank_explanation_can_be_skipped(self):
        self.assertEqual(self.screen()[0], ("", 1.25))

    def test_escape_quits_and_disables_editing(self):
        self.screen(escape=True)

    def test_trial_collects_reason_after_confidence_and_returns_csv_fields(self):
        # Load just the trial function, avoiding camera/GUI imports in a unit test.
        tree = ast.parse((ROOT / "experiment.py").read_text(encoding="utf-8"))
        trial_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_trial")
        stages = []

        def click(*args, **kwargs):
            stages.append("click")
            return ("fake", 0.5) if len(stages) == 1 else ("4", 0.6)

        def reason(*args):
            self.assertEqual(stages, ["click", "click"])
            self.assertEqual(args[2], "fake")
            return "Odd skin texture", 2.5

        noop = lambda *args, **kwargs: None
        namespace = dict(visual=types.SimpleNamespace(ImageStim=noop), text=noop,
                         TRIAL_NUMBER_POS=(0, 0), TRIAL_NUMBER_HEIGHT=0.02, MUTED="grey",
                         overlay=types.SimpleNamespace(reset=noop, show=noop),
                         run_ready=noop, event=types.SimpleNamespace(clearEvents=noop),
                         view_image=lambda *args: ({"onset": 1_000_000_000, "offset": 2_000_000_000}, 0),
                         GAZE_MIN_VALID_FRAC=0.7, wait_for_click=click,
                         explanation_screen=reason, run_iti=noop)
        exec(compile(ast.Module(body=[trial_fn], type_ignores=[]), "experiment.py", "exec"), namespace)
        result = namespace["run_trial"](
            None, None, {"stimulus": "face.png", "condition": "fake"}, None, [], [],
            dict.fromkeys(["view_hint", "decision", "conf_prompt", "conf_left", "conf_right"]),
            types.SimpleNamespace(window_quality=lambda *args: (20, 1.0)), None)
        self.assertEqual(result["decision_reason"], "Odd skin texture")
        self.assertEqual(result["rt_decision_reason"], 2.5)
        self.assertEqual(result["rt_classification"], 0.5)
        self.assertEqual(result["rt_confidence"], 0.6)


if __name__ == "__main__":
    unittest.main()
