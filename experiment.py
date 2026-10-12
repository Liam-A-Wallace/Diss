"""Study 1 experiment - fixation, face, real/fake response, confidence.

Run with: python experiment.py [--demo] [--placeholder] [--test]
"""

import argparse
import os
import sys
import time

from psychopy import core, data, event, visual

from config import (BG, BREAK_AT_FRACTION, BUTTON_Y, CAMERA_INDEX,
                    CHECKPOINT_EVERY, CONF_BUTTON_SPACING, CONF_BUTTON_Y,
                    CONF_CAPTION_Y, CONF_PROMPT_POS, CONFIDENCE_LABELS,
                    DATA_DIR, FULLSCREEN, GAZE_MIN_VALID_FRAC, IMAGE_POS,
                    MUTED, PLACEHOLDER_CSV, PRACTICE_CSV, PRACTICE_TRIALS,
                    PRACTICE_TRIGGER_OFFSET, PROMPT_HEIGHT, PROMPT_POS,
                    SCREEN, SHOW_GAZE, STIM_HEIGHT, TRIAL_NUMBER_HEIGHT,
                    TRIAL_NUMBER_POS, TRIALS_CSV, VIEW_HINT_POS, WIN_SIZE)
import overlay
import texts

from errors import QuitExperiment, RecalibrateRequest, StimulusError
from gaze import GazeRecorder
from screens import (info_screen, participant_screen, run_iti, run_ready,
                     wait_for_click)
from trials import load_and_verify_trials
from ui import Button, make_fixation, text


#################### Trial ####################
def build_trial_stims(win, confidence_buttons):
    """Text stimuli that never change between trials, built once."""
    return {
        "view_hint": text(win, "ENTER or SPACE to continue", VIEW_HINT_POS,
                          height=0.022, color=MUTED),
        "decision": text(win, "Is this face real or AI-generated?",
                         PROMPT_POS, height=PROMPT_HEIGHT, bold=True,
                         wrap=1.6),
        "conf_prompt": text(win, "How confident are you in your decision?",
                            CONF_PROMPT_POS, height=PROMPT_HEIGHT, bold=True,
                            wrap=1.6),
        # captions sit under the first and last confidence buttons
        "conf_left": text(win, "Guessing",
                          (confidence_buttons[0].shape.pos[0], CONF_CAPTION_Y),
                          height=0.03, color=MUTED),
        "conf_right": text(win, "Very sure",
                           (confidence_buttons[-1].shape.pos[0],
                            CONF_CAPTION_Y),
                           height=0.03, color=MUTED),
    }


def run_trial(
    win,
    mouse,
    trial,
    fixation,
    classification_buttons,
    confidence_buttons,
    stims,
    gaze,
    trial_index=0,
    practice=False,
    trigger=None,
):
    stim_path = trial["stimulus"]
    true_class = trial["condition"]

    # unique label for the raw gaze stream; practice gets a high offset so it
    # never collides with the numbered main trials
    trigger_value = trigger if trigger is not None else trial_index

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

    # the camera is already sampling; the ready cross only gates on live gaze
    # and measures drift before the face appears
    overlay.reset()
    overlay.show()
    drift = run_ready(win, fixation, gaze)
    event.clearEvents()

    # Stage 1: view the face. Onset/offset are stamped on the flips that put
    # the image up and take it down, in the same epoch-ns clock as the gaze
    # samples, so the window is exactly what the participant saw.
    stamps = {}
    win.mouseVisible = False
    gaze.mark_trial(trigger_value)
    dropped_before = win.nDroppedFrames
    first_frame = True

    while True:
        image.draw()
        counter.draw()
        stims["view_hint"].draw()
        overlay.draw()
        if first_frame:
            win.callOnFlip(lambda: stamps.setdefault("onset", time.time_ns()))
            first_frame = False
        win.flip()

        keys = event.getKeys(keyList=["return", "num_enter", "space", "escape"])
        if "escape" in keys:
            raise QuitExperiment()
        if any(key in keys for key in ("return", "num_enter", "space")):
            break

    # the image is still up until this next flip, so that flip is the offset
    overlay.hide()
    win.callOnFlip(lambda: stamps.setdefault("offset", time.time_ns()))
    win.flip()

    onset_ns, offset_ns = stamps["onset"], stamps["offset"]
    dropped = win.nDroppedFrames - dropped_before
    gaze.record_window(trial_index, trigger_value, practice, true_class,
                       stim_path, onset_ns, offset_ns)
    n_samples, valid_frac = gaze.window_quality(onset_ns, offset_ns)
    if valid_frac < GAZE_MIN_VALID_FRAC:
        print(f"[gaze] trial {trial_index}: low quality "
              f"({n_samples} samples, {valid_frac:.0%} valid)")

    # Stage 2: real / fake decision
    response, rt_class = wait_for_click(
        win,
        mouse,
        classification_buttons,
        extras=[counter, stims["decision"]],
    )

    correct = 1 if response == true_class else 0

    # Stage 3: confidence rating
    confidence, rt_conf = wait_for_click(
        win,
        mouse,
        confidence_buttons,
        extras=[counter, stims["conf_prompt"], stims["conf_left"],
                stims["conf_right"]],
    )

    confidence = int(confidence)

    run_iti(win)

    return {
        "stimulus": stim_path,
        "trigger": trigger_value,
        "is_practice": int(practice),
        "condition": true_class,
        "difficulty": trial.get("difficulty", ""),
        "response": response,
        "correct": correct,
        "rt_classification": round(rt_class, 4),
        "confidence": confidence,
        "rt_confidence": round(rt_conf, 4),
        "view_time": round((offset_ns - onset_ns) / 1e9, 4),
        "onset_ns": onset_ns,
        "offset_ns": offset_ns,
        "drift_px": "" if drift is None else round(drift, 1),
        "gaze_samples": n_samples,
        "gaze_valid_frac": round(valid_frac, 3),
        "dropped_frames": dropped,
    }


def _show_info(win, gaze, **kwargs):
    """Show an info screen; pressing R recalibrates and re-shows it."""
    while True:
        try:
            return info_screen(win, **kwargs)
        except RecalibrateRequest:
            gaze.recalibrate(win)
            event.clearEvents()


def _run_trial_with_recal(win, gaze, **kwargs):
    """Run a trial; if R is pressed on the ready cross, recalibrate and redo."""
    while True:
        try:
            return run_trial(win, gaze=gaze, **kwargs)
        except RecalibrateRequest:
            gaze.recalibrate(win)
            event.clearEvents()


def _load_trials(use_placeholder):
    """Returns (main_rows, practice_rows); practice never repeats a main image."""
    if use_placeholder:
        from make_placeholders import generate
        trial_csv = generate(PLACEHOLDER_CSV)
        print(f"[placeholder] generated trial list at '{trial_csv}'")
    else:
        trial_csv = TRIALS_CSV

    rows = load_and_verify_trials(trial_csv)
    print(f"[OK] Verified {len(rows)} trials from '{trial_csv}'")

    practice = []
    if PRACTICE_TRIALS:
        if not use_placeholder and os.path.exists(PRACTICE_CSV):
            practice = load_and_verify_trials(PRACTICE_CSV)[:PRACTICE_TRIALS]
        else:
            practice, rows = rows[:PRACTICE_TRIALS], rows[PRACTICE_TRIALS:]
            print(f"[practice] no '{PRACTICE_CSV}' found; using the first "
                  f"{len(practice)} trial(s) as practice, not repeated in "
                  "the main task")
        if not rows:
            raise StimulusError("No trials left after reserving practice.")
    return rows, practice


def _checkpoint(this_exp, gaze, base_name):
    try:
        this_exp.saveAsWideText(base_name + ".csv", delim=",")
    except Exception as err:
        print(f"[checkpoint] trial data save failed: {err}", file=sys.stderr)
    gaze.checkpoint()


def _parse_args():
    parser = argparse.ArgumentParser(description="Seeing the Fake - Study 1")
    parser.add_argument("--demo", action="store_true",
                        help="draw the live gaze point on screen")
    parser.add_argument("--placeholder", action="store_true",
                        help="generate and use placeholder stimuli")
    parser.add_argument("--test", action="store_true",
                        help="write output to data/test instead of data/")
    return parser.parse_args()


#################### Main ####################
def main():
    args = _parse_args()
    show_gaze = SHOW_GAZE or args.demo
    if show_gaze:
        print("[demo] gaze overlay ON")

    try:
        trial_rows, practice_rows = _load_trials(args.placeholder)
    except StimulusError as err:
        print(f"[ERROR] {err}", file=sys.stderr)
        return

    # build the eye tracker and calibrate it in GazeFollower's own display
    # BEFORE the PsychoPy window opens, so the two fullscreen contexts never
    # overlap
    gaze = GazeRecorder(camera_index=CAMERA_INDEX)
    try:
        gaze.init()
        gaze.calibrate()
    except Exception as err:
        print(f"[ERROR] Gaze setup failed: {err}", file=sys.stderr)
        gaze.stop_session()
        return

    win = visual.Window(
        size=WIN_SIZE,
        fullscr=FULLSCREEN,
        screen=SCREEN,
        color=BG,
        units="height",
        useFBO=True,
    )

    # needed for win.nDroppedFrames to count
    win.recordFrameIntervals = True

    print("[PsychoPy] win.size =", win.size)
    print("[GazeFollower] screen size =", gaze.tracker.config.screen_size)

    overlay.setup(win, gaze, show_gaze)

    try:
        exp_info = participant_screen(win)

    except QuitExperiment:
        gaze.stop_session()
        win.close()
        core.quit()
        return

    out_dir = os.path.join(DATA_DIR, "test") if args.test else DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.join(
        out_dir,
        f"{exp_info['participant']}_{exp_info['session']}_{data.getDateStr()}",
    )

    # stored with the data so the trial order can be reproduced
    exp_info["seed"] = int(time.time()) % 1_000_000

    this_exp = data.ExperimentHandler(
        name="SeeingTheFake",
        version="0.3",
        extraInfo=exp_info,
        savePickle=False,
        saveWideText=False,
        dataFileName=base_name,
    )

    gaze.set_output(base_name)

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

    # centred row of rating buttons, one per label
    n_conf = len(CONFIDENCE_LABELS)
    confidence_buttons = [
        Button(
            win,
            label,
            ((i - (n_conf - 1) / 2) * CONF_BUTTON_SPACING, CONF_BUTTON_Y),
            (0.15, 0.15),
            label,
            text_height=0.055,
        )
        for i, label in enumerate(CONFIDENCE_LABELS)
    ]
    stims = build_trial_stims(win, confidence_buttons)

    trial_kwargs = dict(
        mouse=mouse,
        fixation=fixation,
        classification_buttons=classification_buttons,
        confidence_buttons=confidence_buttons,
        stims=stims,
    )

    total_trials = len(trial_rows)
    break_after = (round(total_trials * BREAK_AT_FRACTION)
                   if BREAK_AT_FRACTION and total_trials >= 2 else None)

    try:
        if not _show_info(win, gaze, **texts.WELCOME):
            raise QuitExperiment()

        # the camera now samples for the whole session; only the image
        # windows are kept when the gaze file is written
        gaze.start_sampling()

        if practice_rows:
            if not _show_info(win, gaze, **texts.PRACTICE):
                raise QuitExperiment()

            for i, row in enumerate(practice_rows):
                result = _run_trial_with_recal(
                    win, gaze,
                    trial=row,
                    trial_index=i + 1,
                    practice=True,
                    trigger=PRACTICE_TRIGGER_OFFSET + i + 1,
                    **trial_kwargs,
                )
                for key, value in result.items():
                    this_exp.addData(key, value)
                this_exp.nextEntry()

        if not _show_info(win, gaze, **texts.READY):
            raise QuitExperiment()

        trials = data.TrialHandler(
            trial_rows,
            nReps=1,
            method="random",
            extraInfo=exp_info,
            name="trials",
            seed=exp_info["seed"],
        )
        this_exp.addLoop(trials)

        for i, trial in enumerate(trials):
            result = _run_trial_with_recal(
                win, gaze,
                trial=trial,
                trial_index=i + 1,
                **trial_kwargs,
            )

            for key, value in result.items():
                trials.addData(key, value)
            this_exp.nextEntry()

            if CHECKPOINT_EVERY and (i + 1) % CHECKPOINT_EVERY == 0:
                _checkpoint(this_exp, gaze, base_name)

            if break_after and i + 1 == break_after and i + 1 < total_trials:
                if not _show_info(win, gaze, **texts.BREAK):
                    raise QuitExperiment()

        _show_info(win, gaze, **texts.THANKS)

    except QuitExperiment:
        pass

    finally:
        # each step is isolated so one failure cannot skip the rest
        for step in (gaze.stop_session,
                     lambda: this_exp.saveAsWideText(base_name + ".csv",
                                                     delim=","),
                     lambda: this_exp.saveAsPickle(base_name + ".psydat"),
                     this_exp.close):
            try:
                step()
            except Exception as err:
                print(f"[shutdown] {err}", file=sys.stderr)
        win.close()
        core.quit()


if __name__ == "__main__":
    main()
