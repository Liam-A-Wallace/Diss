"""Subjective post-calibration image check, separate from experiment trials."""
import hashlib
import json
from pathlib import Path
import time


def choose_image(trials, practice):
    excluded = {hashlib.sha256(Path(r["stimulus"]).read_bytes()).hexdigest() for r in trials + practice}
    folder = Path(__file__).resolve().parent.parent / "Dataset/datasets/picked/real"
    for path in sorted(folder.glob("*.png")):
        if hashlib.sha256(path.read_bytes()).hexdigest() not in excluded:
            return str(path)
    return (practice or trials)[0]["stimulus"]


def check_image(win, gaze, stimulus, log_path=None):
    from psychopy import event, visual
    from PIL import Image
    from errors import QuitExperiment
    from ui import text
    with Image.open(stimulus) as image:
        width, height = image.size
    scale = min(win.size[0] * 0.75 / width, win.size[1] * 0.70 / height)
    photo = visual.ImageStim(win, image=stimulus, units="pix", size=(width * scale, height * scale))
    dot = visual.Circle(win, units="pix", radius=10, fillColor="#00e5ff", lineColor="black", lineWidth=2)
    title = text(win, "Check your eye tracking", (0, 0.43), height=0.038, bold=True)
    instruction = text(win, "Look at the eyes, nose and mouth. Does the blue dot follow where you look?",
                       (0, 0.38), height=0.023, wrap=1.6)
    hint = text(win, "ENTER / SPACE: satisfactory   |   R: calibrate again   |   ESC: quit",
                (0, -0.44), height=0.022, wrap=1.6)
    status = text(win, "Waiting for fresh gaze...", (0, -0.39), height=0.022)
    event.clearEvents()
    started = time.time_ns()
    gaze.mark_trial(0)
    outcome = "aborted"
    try:
        while True:
            sample = gaze.poll()
            fresh = (sample is not None and gaze.recent_gaze(0.3)
                     and gaze._last_info_ts is not None and gaze._last_info_ts >= started)
            for item in (photo, title, instruction, hint):
                item.draw()
            if fresh:
                sw, sh = gaze.screen_size
                dot.pos = (sample[0] / sw * win.size[0] - win.size[0] / 2,
                           win.size[1] / 2 - sample[1] / sh * win.size[1])
                dot.draw()
            status.text = "Live gaze dot (filtered; may lag slightly)" if fresh else "Tracking lost or waiting: face the camera"
            status.draw()
            win.flip()
            keys = event.getKeys(keyList=["return", "num_enter", "space", "r", "escape"])
            if "escape" in keys:
                raise QuitExperiment()
            if "r" in keys:
                outcome = "retry"
                return False
            if fresh and time.time_ns() - started >= 1_000_000_000 and any(
                    key in keys for key in ("return", "num_enter", "space")):
                outcome = "accepted"
                return True
    finally:
        if log_path:
            with open(log_path, "a", encoding="utf-8") as target:
                target.write(json.dumps(dict(started_ns=started, ended_ns=time.time_ns(),
                                             stimulus=str(Path(stimulus).resolve()), outcome=outcome,
                                             subjective_check=True)) + "\n")
        event.clearEvents()