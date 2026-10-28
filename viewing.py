"""PsychoPy viewing stage; geometry and exports stay independently testable."""
import time
from PIL import Image
from psychopy import event, visual
import overlay
from config import BG, IMAGE_POS, STIM_HEIGHT, MAX_ZOOM
from errors import QuitExperiment
from image_view import ImageView


def view_image(win, mouse, image, stimulus, counter, hint, gaze, session,
               trial_index, trigger, practice, condition):
    with Image.open(stimulus) as source:
        iw, ih = source.size
    ww, wh = map(int, win.size)
    vh = STIM_HEIGHT * wh
    top = wh / 2 - IMAGE_POS[1] * wh - vh / 2
    viewport = (0, top, ww, vh)
    view = ImageView(iw, ih, viewport, MAX_ZOOM)
    # Opaque bands clip the image while preserving the counter and control hint.
    bands = [visual.Rect(win, units="pix", width=ww, height=top,
                         pos=(0, (wh - top) / 2), fillColor=BG, lineColor=BG),
             visual.Rect(win, units="pix", width=ww, height=wh - top - vh,
                         pos=(0, -(top + vh) / 2), fillColor=BG, lineColor=BG)]
    image.units = "pix"
    win.mouseVisible = True
    mouse.getWheelRel()  # discard scrolling on previous screens
    previous = None
    dragging = False
    was_pressed = bool(mouse.getPressed()[0])
    last_frame = None
    stamps = {}
    interrupted = True
    dropped_before = win.nDroppedFrames
    gaze.mark_trial(trigger)

    def present(frame):
        ts = time.time_ns()
        stamps.setdefault("onset", ts)
        session.present(dict(frame, timestamp_ns=ts))

    def end():
        stamps["offset"] = time.time_ns()
        session.end_view(trigger, interrupted, stamps["offset"])

    try:
        while True:
            mx, my = mouse.getPos()  # window uses height units
            point = (ww / 2 + mx * wh, wh / 2 - my * wh)
            pressed = bool(mouse.getPressed()[0])
            wheel = float(mouse.getWheelRel()[1])
            if wheel:
                view.scroll(wheel, *point)
                dragging = False
            if pressed and not was_pressed:
                dragging = view.contains(*point)
            if dragging and pressed and previous is not None:
                view.pan(point[0] - previous[0], point[1] - previous[1])
            if not pressed:
                dragging = False
            previous, was_pressed = point, pressed
            keys = event.getKeys(keyList=["return", "num_enter", "space", "escape", "home"])
            if "escape" in keys:
                raise QuitExperiment()
            if "home" in keys:
                view.reset()
                dragging = False
            if "onset" in stamps and any(k in keys for k in ("return", "num_enter", "space")):
                interrupted = False
                break
            image.size = (iw * view.scale, ih * view.scale)
            image.pos = (view.left + iw * view.scale / 2 - ww / 2,
                         wh / 2 - view.top - ih * view.scale / 2)
            frame = dict(view.snapshot(), stimulus=stimulus, trial=trial_index,
                         trigger=trigger, is_practice=int(practice), condition=condition)
            image.draw()
            for band in bands:
                band.draw()
            counter.draw()
            hint.text = f"Scroll to zoom ({view.zoom:.1f}x) | Drag to move | HOME to reset | ENTER to continue"
            hint.draw()
            overlay.draw()
            if frame != last_frame:
                win.callOnFlip(present, frame)
                last_frame = frame
            win.flip()
    finally:
        overlay.hide()
        if "onset" in stamps:
            win.callOnFlip(end)
            win.flip()
            gaze.record_window(trial_index, trigger, practice, condition, stimulus,
                               stamps["onset"], stamps["offset"])
    return stamps, win.nDroppedFrames - dropped_before
