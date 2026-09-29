"""Study 1 experiment - fixation, face, real/fake response, confidence.

Run with: python experiment.py
"""

import csv
import os
import sys

from psychopy import core, data, event, visual

from config import (BG, BUTTON_Y, CAMERA_INDEX, CONFIDENCE_LABELS, DATA_DIR,
                    FULLSCREEN, IMAGE_POS, MUTED, PRACTICE_TRIALS,
                    PROMPT_HEIGHT, PROMPT_POS, QuitExperiment, SCREEN,
                    SHOW_GAZE, STIM_HEIGHT, TRIAL_NUMBER_HEIGHT,
                    TRIAL_NUMBER_POS, TRIALS_CSV, WIN_SIZE)
from gaze import GazeRecorder
from screens import (info_screen, participant_screen, run_fixation, run_iti,
                     wait_for_click)
from ui import Button, make_fixation, text


#################### Trial ####################
def run_trial(
    win,
    mouse,
    trial,
    fixation,
    classification_buttons,
    confidence_buttons,
    gaze,
    trial_index=0,
    total_trials=0,
    practice=False,
    show_gaze=False,
):
    stim_path = trial["stimulus"]
    true_class = trial["condition"]

    # Load the image before fixation so nothing loads between fixation
    # and stimulus onset.
    image = visual.ImageStim(
        win,
        image=stim_path,
        units="height",
    )

    # scale to STIM_HEIGHT, keeping the image's own aspect ratio
    try:
        w, h = image.size
        image.size = (STIM_HEIGHT * w / h, STIM_HEIGHT)
    except Exception:
        image.size = (STIM_HEIGHT, STIM_HEIGHT)

    image.pos = IMAGE_POS

    photo_label = (
        f"Practice {trial_index}"
        if practice
        else f"Photo {trial_index}"
    )

    counter = text(
        win,
        photo_label,
        TRIAL_NUMBER_POS,
        height=TRIAL_NUMBER_HEIGHT,
        color=MUTED,
        bold=True,
        anchor_h="left",
    )

    # fixation is a visual attention anchor only - nothing is recorded yet
    run_fixation(win, fixation)

    # Stage 1: view the face
    ready_hint = text(
        win,
        "ENTER or SPACE to continue",
        (0, -0.465),
        height=0.022,
        color=MUTED,
    )

    gaze_dot = None
    if show_gaze:
        gaze_dot = visual.Circle(win, radius=0.012, fillColor="red",
                                 lineColor=None)

    # start recording as the face appears; the trigger stamps this trial's
    # number as an onset marker in the gaze stream
    win.mouseVisible = False
    gaze.start_trial(trial_index)

    while True:
        image.draw()
        counter.draw()
        ready_hint.draw()
        if gaze_dot is not None:
            sample = gaze.poll()
            if sample is not None:
                px_x, px_y = sample
                win_w, win_h = win.size  # screen pixels
                x = (px_x / win_w - 0.5) * (win_w / win_h)
                y = 0.5 - px_y / win_h
                gaze_dot.pos = (x, y)
                gaze_dot.draw()
        win.flip()

        keys = event.getKeys(keyList=["return", "num_enter", "space", "escape"])
        if "escape" in keys:
            raise QuitExperiment()
        if any(key in keys for key in ("return", "num_enter", "space")):
            break

    # any advance key (ENTER/SPACE) pauses recording here, so the decision
    # and confidence screens are not part of this stimulus's gaze trace
    gaze.stop_trial()

    # Stage 2: real / fake decision
    decision_prompt = text(
        win,
        "Is this face real or AI-generated?",
        PROMPT_POS,
        height=PROMPT_HEIGHT,
        bold=True,
        wrap=1.6,
    )

    response, rt_class = wait_for_click(
        win,
        mouse,
        classification_buttons,
        extras=[counter, decision_prompt],
    )

    correct = 1 if response == true_class else 0

    # Stage 3: confidence rating
    conf_prompt = text(
        win,
        "How confident are you in your decision?",
        (0, 0.30),
        height=PROMPT_HEIGHT,
        bold=True,
        wrap=1.6,
    )

    # captions sit under the first and last confidence buttons
    conf_left = text(
        win,
        "Guessing",
        (confidence_buttons[0].shape.pos[0], -0.135),
        height=0.03,
        color=MUTED,
    )

    conf_right = text(
        win,
        "Very sure",
        (confidence_buttons[-1].shape.pos[0], -0.135),
        height=0.03,
        color=MUTED,
    )

    confidence, rt_conf = wait_for_click(
        win,
        mouse,
        confidence_buttons,
        extras=[
            counter,
            conf_prompt,
            conf_left,
            conf_right,
        ],
    )

    confidence = int(confidence)

    run_iti(win)

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


REQUIRED_COLUMNS = {"stimulus", "condition", "difficulty"}


class StimulusError(Exception):
    pass


def load_and_verify_trials(csv_path):
    if not os.path.exists(csv_path):
        raise StimulusError(f"Trial file not found: '{csv_path}'")

    trials = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        missing_cols = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_cols:
            raise StimulusError(
                f"CSV missing required columns: {missing_cols}. "
                f"Found headers: {reader.fieldnames}"
            )

        missing_files = []
        # header is row 1, so data starts at 2
        for row_idx, row in enumerate(reader, start=2):
            norm_path = os.path.normpath(row["stimulus"].strip())
            row["stimulus"] = norm_path
            if not os.path.isfile(norm_path):
                missing_files.append((row_idx, norm_path))
            trials.append(row)

    if not trials:
        raise StimulusError(f"Trial file '{csv_path}' contains no trial data.")

    if missing_files:
        details = "\n".join(f"  Row {row}: {path}" for row, path in missing_files[:5])
        if len(missing_files) > 5:
            details += f"\n  ... and {len(missing_files) - 5} more."
        raise StimulusError(
            f"Found {len(missing_files)} missing stimulus image(s):\n{details}"
        )

    return trials


#################### Main ####################
def main():
    show_gaze = SHOW_GAZE or "-demo" in sys.argv or "--demo" in sys.argv
    if show_gaze:
        print("[demo] gaze overlay ON")

    try:
        trial_rows = load_and_verify_trials(TRIALS_CSV)
        print(f"[OK] Verified {len(trial_rows)} trials from '{TRIALS_CSV}'")
    except StimulusError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return

    # start the camera before the window exists so preview/calibration
    # never fight PsychoPy's fullscreen GL context
    gaze = GazeRecorder(camera_index=CAMERA_INDEX)
    try:
        gaze.start_session()
    except Exception as err:
        print(f"[ERROR] Gaze setup failed: {err}", file=sys.stderr)
        return

    win = visual.Window(
        size=WIN_SIZE,
        fullscr=FULLSCREEN,
        screen=SCREEN,
        color=BG,
        units="height",
        useFBO=True,
    )

    try:
        exp_info = participant_screen(win)

    except QuitExperiment:
        gaze.stop_session()
        win.close()
        core.quit()
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    base_name = os.path.join(
        DATA_DIR,
        f"{exp_info['participant']}_{exp_info['session']}_{data.getDateStr()}",
    )

    this_exp = data.ExperimentHandler(
        name="SeeingTheFake",
        version="0.2",
        extraInfo=exp_info,
        savePickle=False,
        saveWideText=False,
        dataFileName=base_name,
    )

    gaze.output_path = base_name + "_gaze.csv"

    mouse = event.Mouse(win=win)
    fixation = make_fixation(win)

    classification_buttons = [
        Button(
            win, "REAL",
            (-0.28, BUTTON_Y),
            (0.34, 0.11),
            "real",
        ),
        Button(
            win, "FAKE",
            (0.28, BUTTON_Y),
            (0.34, 0.11),
            "fake",
        ),
    ]

    confidence_buttons = [
        Button(
            win,
            label,
            (x, -0.02),
            (0.15, 0.15),
            label,
            text_height=0.055,
        )
        for x, label in zip(
            [-0.42, -0.21, 0.0, 0.21, 0.42],
            CONFIDENCE_LABELS,
        )
    ]

    trials = data.TrialHandler(
        trial_rows,
        nReps=1,
        method="random",
        extraInfo=exp_info,
        name="trials",
    )

    this_exp.addLoop(trials)

    try:
        if not info_screen(
            win,
            "Welcome",
            body=(
                "You will see a series of faces. Some are photographs "
                "of real people; others are AI-generated."
            ),
            steps=[
                (
                    "Look at the face",
                    "A face will appear on its own. Take your time to look at it.",
                ),
                (
                    "Make your decision",
                    "Press ENTER when you are ready, then click REAL or FAKE.",
                ),
                (
                    "Rate your confidence",
                    "Click a number from 1 (guessing) to 5 (very sure).",
                ),
            ],
        ):
            raise QuitExperiment()

        if PRACTICE_TRIALS:
            if not info_screen(
                win,
                "Practice",
                body=(
                    "First, you will complete one practice trial to get used "
                    "to the task.\n\nYou will not be told whether your answer "
                    "is correct."
                ),
                footer="Press SPACE to begin",
            ):
                raise QuitExperiment()

            for i, row in enumerate(
                trial_rows[:PRACTICE_TRIALS]
            ):
                run_trial(
                    win,
                    mouse,
                    row,
                    fixation,
                    classification_buttons,
                    confidence_buttons,
                    gaze,
                    trial_index=i + 1,
                    total_trials=PRACTICE_TRIALS,
                    practice=True,
                    show_gaze=show_gaze,
                )

        if not info_screen(
            win,
            "Ready?",
            body=(
                "The main task starts now.\n\nAs in the practice, you "
                "will not receive feedback. Respond at your own pace "
                "and go with your first impression."
            ),
            footer="Press SPACE to begin",
        ):
            raise QuitExperiment()

        total_trials = trials.nTotal

        for i, trial in enumerate(trials):
            result = run_trial(
                win,
                mouse,
                trial,
                fixation,
                classification_buttons,
                confidence_buttons,
                gaze,
                trial_index=i + 1,
                total_trials=total_trials,
                show_gaze=show_gaze,
            )

            for key, value in result.items():
                trials.addData(key, value)

            this_exp.nextEntry()

        info_screen(
            win,
            "Thank you",
            body=(
                "You have completed the task.\n\nYour participation "
                "is greatly appreciated."
            ),
            footer="Press SPACE to finish",
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
