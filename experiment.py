"""Study 1 experiment — fixation, face, real/fake response, confidence.

Run with: python experiment.py
"""
import csv
import os

from psychopy import core, data, event, gui, visual

from gaze import GazeRecorder


WIN_SIZE = (1280, 720)
FULLSCREEN = False
BACKGROUND = (-0.15, -0.15, -0.15)
TEXT_COLOR = "white"

FIXATION_DURATION = 0.7
ITI_DURATION = 0.5
STIM_HEIGHT = 0.62
FEEDBACK_DURATION = 1.0
PRACTICE_TRIALS = 2

TRIALS_CSV = "trials.csv"
DATA_DIR = "data"
CONFIDENCE_LABELS = ["1", "2", "3", "4", "5"]


class QuitExperiment(Exception):
    pass


def show_text_and_wait(win, text, keys=("space",)):
    msg = visual.TextStim(win, text=text, color=TEXT_COLOR, height=0.05,
                          wrapWidth=1.6)
    msg.draw()
    win.flip()
    pressed = event.waitKeys(keyList=list(keys) + ["escape"])
    return not (pressed and pressed[0] == "escape")


def make_button(win, text, pos, size=(0.24, 0.12)):
    rect = visual.Rect(win, size=size, pos=pos, fillColor=(0.25, 0.25, 0.25),
                       lineColor="white", lineWidth=2)
    label = visual.TextStim(win, text=text, pos=pos, color=TEXT_COLOR,
                            height=0.045)
    return rect, label


def wait_for_click(win, mouse, buttons, prompt=None, extras=()):
    prompt_stim = None
    if prompt:
        prompt_stim = visual.TextStim(win, text=prompt, color=TEXT_COLOR,
                                      height=0.05, pos=(0, 0.40), wrapWidth=1.6)

    def draw_screen():
        if prompt_stim:
            prompt_stim.draw()
        for stim in extras:
            stim.draw()
        for rect, label, _ in buttons:
            rect.draw()
            label.draw()

    # wait out a held button so the previous click doesn't carry over
    mouse.clickReset()
    for _ in range(120):
        draw_screen()
        win.flip()
        if not mouse.getPressed()[0]:
            break
    event.clearEvents()

    rt_clock = core.Clock()
    while True:
        if "escape" in event.getKeys(keyList=["escape"]):
            raise QuitExperiment()
        draw_screen()
        for rect, label, value in buttons:
            if mouse.isPressedIn(rect, buttons=[0]):
                return value, rt_clock.getTime()
        win.flip()


def run_fixation(win, fixation):
    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)


def run_iti(win):
    win.flip()
    core.wait(ITI_DURATION)


def run_trial(win, mouse, trial, fixation, classification_buttons,
              confidence_buttons, gaze, feedback=False):
    stim_path = trial["stimulus"]
    true_class = trial["condition"]
    trial_id = f"{true_class}:{os.path.basename(stim_path)}"

    gaze.start_trial(trial_id)
    run_fixation(win, fixation)

    image = visual.ImageStim(win, image=stim_path, units="height")
    try:
        ar_w, ar_h = image.aspectRatio
        image.size = (STIM_HEIGHT * ar_w / ar_h, STIM_HEIGHT)
    except Exception:
        image.size = (STIM_HEIGHT, STIM_HEIGHT)
    image.pos = (0, 0.08)

    response, rt_class = wait_for_click(
        win, mouse, classification_buttons,
        prompt="Is this face REAL or AI-GENERATED?",
        extras=(image,),
    )
    correct = 1 if response == true_class else 0

    confidence, rt_conf = wait_for_click(
        win, mouse, confidence_buttons,
        prompt="How confident are you in your decision?\n"
               "(1 = guessing, 5 = very sure)",
    )
    confidence = int(confidence)

    if feedback:
        fb = visual.TextStim(win, text="Correct!" if correct else "Incorrect",
                             color=TEXT_COLOR, height=0.06)
        fb.draw()
        win.flip()
        core.wait(FEEDBACK_DURATION)

    run_iti(win)
    gaze.stop_trial()

    return {
        "stimulus": stim_path,
        "condition": true_class,
        "difficulty": trial.get("difficulty", ""),
        "response": response,
        "correct": correct,
        "rt_classification": round(rt_class, 4),
        "confidence": confidence,
        "rt_confidence": round(rt_conf, 4),
    }


def load_trials(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    exp_info = {"participant": "001", "session": "001"}
    dlg = gui.DlgFromDict(exp_info, title="Seeing the Fake",
                          order=["participant", "session"])
    if not dlg.OK:
        core.quit()
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    base_name = os.path.join(
        DATA_DIR,
        f"{exp_info['participant']}_{exp_info['session']}_{data.getDateStr()}")

    this_exp = data.ExperimentHandler(
        name="SeeingTheFake", version="0.1", extraInfo=exp_info,
        savePickle=False, saveWideText=False, dataFileName=base_name)

    gaze = GazeRecorder(base_name + "_gaze.csv", exp_info["participant"])

    win = visual.Window(size=WIN_SIZE, fullscr=FULLSCREEN, color=BACKGROUND,
                        units="height")
    mouse = event.Mouse(win=win)
    fixation = visual.TextStim(win, text="+", color=TEXT_COLOR, height=0.08)

    real_rect, real_label = make_button(win, "REAL", pos=(-0.28, -0.35),
                                        size=(0.3, 0.12))
    fake_rect, fake_label = make_button(win, "FAKE", pos=(0.28, -0.35),
                                        size=(0.3, 0.12))
    classification_buttons = [
        (real_rect, real_label, "real"),
        (fake_rect, fake_label, "fake"),
    ]

    confidence_buttons = []
    for x, label in zip([-0.4, -0.2, 0.0, 0.2, 0.4], CONFIDENCE_LABELS):
        rect, text = make_button(win, label, pos=(x, -0.25), size=(0.15, 0.12))
        confidence_buttons.append((rect, text, label))

    trial_rows = load_trials(TRIALS_CSV)
    trials = data.TrialHandler(trial_rows, nReps=1, method="random",
                               extraInfo=exp_info, name="trials")
    this_exp.addLoop(trials)

    try:
        gaze.start_session()

        if not show_text_and_wait(
                win,
                "In this task you will see faces.\n\n"
                "Decide whether each face is REAL or AI-GENERATED, then rate "
                "your confidence.\n\n"
                "Click the buttons to respond.\n\n"
                "Press SPACE to continue.",
        ):
            raise QuitExperiment()

        if PRACTICE_TRIALS:
            if not show_text_and_wait(
                    win,
                    "Practice:\n\n"
                    "A few trials with feedback so you can get used to the "
                    "task.\n\nPress SPACE to begin.",
            ):
                raise QuitExperiment()
            for row in trial_rows[:PRACTICE_TRIALS]:
                run_trial(win, mouse, row, fixation, classification_buttons,
                          confidence_buttons, gaze, feedback=True)

        if not show_text_and_wait(
                win,
                "The main experiment is about to start.\n\n"
                "There will be no feedback from now on.\n\n"
                "Press SPACE to begin.",
        ):
            raise QuitExperiment()

        for trial in trials:
            result = run_trial(win, mouse, trial, fixation,
                               classification_buttons, confidence_buttons, gaze)
            for key, value in result.items():
                trials.addData(key, value)
            this_exp.nextEntry()

        show_text_and_wait(
            win,
            "Thank you for participating.\n\nPress SPACE to finish.",
        )

    except QuitExperiment:
        pass
    finally:
        gaze.stop_session()
        this_exp.saveAsWideText(base_name + ".csv", delim=",")
        this_exp.saveAsPickle(base_name + ".psydat")
        this_exp.close()
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
