"""Instruction, participant-entry, and response screens."""
from psychopy import core, event, visual

import re

from config import (ACCENT, BTN_FILL, BTN_HOVER, BTN_LINE, DRIFT_WARN_PX,
                    DRIFT_WINDOW_S, FACE_LOST_AFTER, FIXATION_DURATION,
                    FONT, GAZE_READY_WINDOW, ITI_DURATION, MUTED, PANEL, PANEL_LINE,
                    TEXT)
from errors import QuitExperiment, RecalibrateRequest
from ui import Button, fit_text, rounded_rect, text

import overlay


def _safe_id(value, default):
    # IDs become part of file names, so keep only filename-safe characters
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", value.strip())
    return cleaned or default


# Instruction screens
def info_screen(win, title, body=None, steps=None,
                footer="Press ENTER or SPACE to continue", allow_recalibration=True,
                continue_when=None):
    card = rounded_rect(
        win, (1.28, 0.84), (0, 0),
        radius=0.035, fill=PANEL, line=PANEL_LINE
    )

    stims = [
        card,
        text(
            win, "SEEING THE FAKE", (0, 0.355),
            height=0.026, color=TEXT, bold=True
        ),
        text(
            win, title, (0, 0.27),
            height=0.06, bold=True, wrap=1.12, max_height=0.075
        ),
        visual.Rect(
            win, size=(0.10, 0.005), pos=(0, 0.21),
            fillColor=ACCENT, lineColor=None
        ),
    ]

    if steps:
        start_y = 0.15

        if body:
            stims.append(
                text(
                    win, body, (0, 0.14),
                    height=0.027, wrap=1.08, max_height=0.09,
                    color=MUTED, anchor_v="top"
                )
            )
            start_y = 0.025

        # Each step owns a separate region above the footer; descriptions can
        # wrap without running into the following heading.
        spacing = (start_y + 0.32) / len(steps)

        for i, (heading, desc) in enumerate(steps):
            y = start_y - i * spacing

            stims.append(
                text(
                    win, f"{i + 1}. {heading}", (0, y),
                    height=0.032, bold=True, anchor_v="top",
                    wrap=1.08, max_height=0.04
                )
            )
            stims.append(
                text(
                    win, desc, (0, y - 0.045),
                    height=0.025, color=MUTED, wrap=1.08,
                    anchor_v="top", max_height=spacing - 0.06
                )
            )

    elif body:
        stims.append(
            text(
                win, body, (0, 0.08),
                height=0.038, wrap=1.08, max_height=0.40,
                color=TEXT, anchor_v="top"
            )
        )

    stims.append(
        text(
            win, footer, (0, -0.375),
            height=0.024, color=MUTED, wrap=1.12, max_height=0.035
        )
    )

    event.clearEvents()
    while True:
        ready = continue_when is None or continue_when()
        current_footer = footer if ready else "Preparing eye tracker..."
        if stims[-1].text != current_footer:
            stims[-1].text = current_footer
        for stim in stims:
            stim.draw()
        overlay.draw()
        win.flip()

        keys = event.getKeys(keyList=["space", "return", "num_enter",
                                      "escape"] + (["r"] if allow_recalibration else []))
        if keys:
            if "r" in keys:
                raise RecalibrateRequest()
            if "escape" in keys:
                return False
            if ready:
                return True


# Participant entry
def participant_screen(win, participant="001", session="001"):
    fields = [list(participant), list(session)]
    labels = ("Participant ID", "Session")
    active = 0

    field_width, field_height = 0.86, 0.08
    field_y_positions = [0.025, -0.14]
    label_offset_y = 0.060

    # static stimuli, created once so the form doesn't rebuild them every frame
    card = rounded_rect(win, (1.24, 0.86), (0, 0), radius=0.035, fill=PANEL, line=PANEL_LINE)
    header = text(win, "SEEING THE FAKE", (0, 0.285), height=0.019, color=MUTED, bold=True)
    title = text(win, "Participant details", (0, 0.205), height=0.05, bold=True,
                 wrap=1.10, max_height=0.065)
    hint = text(
        win, "TAB switch field   •   ENTER or SPACE continue   •   ESC quit",
        (0, -0.38), height=0.020, color=MUTED, wrap=1.10, max_height=0.035
    )

    label_stims = []
    field_boxes = []
    value_stims = []
    box_shapes = []
    for y_box, label in zip(field_y_positions, labels):
        label_stims.append(
            text(win, label, (0.0, y_box + label_offset_y),
                 height=0.024, color=MUTED,
                 anchor_h="center", anchor_v="bottom")
        )
        field_boxes.append(
            rounded_rect(win, (field_width, field_height), (0.0, y_box),
                         radius=0.018, fill=BTN_FILL, line=BTN_LINE, line_width=2)
        )
        value_stims.append(
            text(win, " ", (0.0, y_box + 0.004),
                 height=0.038, color=TEXT, bold=True,
                 anchor_h="center", anchor_v="center", wrap=100)
        )
        # invisible rect, only used to hit-test which field gets clicked
        box_shapes.append(
            visual.Rect(win, width=field_width, height=field_height,
                        pos=(0.0, y_box), units="height")
        )

    btn_continue = Button(
        win, "CONTINUE", (0.0, -0.285), (0.32, 0.08), value="continue", text_height=0.035
    )

    mouse = event.Mouse(win=win)
    win.mouseVisible = True

    def draw():
        mouse_pos = mouse.getPos()
        card.draw()
        header.draw()
        title.draw()

        for i in range(len(labels)):
            value = "".join(fields[i]) or " "
            if value_stims[i].text != value:
                value_stims[i].text = value
                value_stims[i].height = 0.038
                fit_text(value_stims[i], field_width - 0.05, field_height - 0.02)
            is_active = (i == active)
            is_hovered = box_shapes[i].contains(mouse_pos)
            field_boxes[i].lineColor = ACCENT if is_active else (BTN_HOVER if is_hovered else BTN_LINE)
            field_boxes[i].fillColor = BTN_HOVER if (is_active or is_hovered) else BTN_FILL
            label_stims[i].draw()
            field_boxes[i].draw()
            value_stims[i].draw()

        hint.draw()
        btn_continue.draw(hovered=btn_continue.contains(mouse_pos))
        overlay.draw()

    while True:
        draw()
        win.flip()

        mouse_pos = mouse.getPos()

        if mouse.getPressed()[0]:
            if btn_continue.contains(mouse_pos):
                while mouse.getPressed()[0]:
                    core.wait(0.01)
                break

            for i, box in enumerate(box_shapes):
                if box.contains(mouse_pos):
                    active = i
                    break

        keys = event.getKeys()

        if "escape" in keys:
            raise QuitExperiment()

        if any(key in keys for key in ("return", "num_enter", "space")):
            break

        if "tab" in keys:
            active = 1 - active

        for key in keys:
            if key == "backspace":
                if fields[active]:
                    fields[active].pop()
            elif len(key) == 1:
                fields[active].append(key)

    return {
        "participant": _safe_id("".join(fields[0]), "001"),
        "session": _safe_id("".join(fields[1]), "001"),
    }


# Response collection
def wait_for_click(win, mouse, buttons, extras=()):
    def draw_screen(hovered=None):
        for stim in extras:
            stim.draw()

        for button in buttons:
            button.draw(hovered is button)

        overlay.draw()

    # RT is zeroed on the flip where this screen first appears.
    rt_clock = core.Clock()
    win.callOnFlip(rt_clock.reset)
    win.mouseVisible = True

    # Wait out a held button so the previous click does not carry over.
    while True:
        if "escape" in event.getKeys(keyList=["escape"]):
            raise QuitExperiment()
        draw_screen()
        win.flip()

        if not mouse.getPressed()[0]:
            break

    event.clearEvents()

    while True:
        if "escape" in event.getKeys(keyList=["escape"]):
            raise QuitExperiment()

        pos = mouse.getPos()
        hovered = next(
            (button for button in buttons if button.contains(pos)),
            None,
        )

        draw_screen(hovered)
        win.flip()

        if hovered is not None and mouse.getPressed()[0]:
            return hovered.value, rt_clock.getTime()


def explanation_screen(win, mouse, response, counter):
    """Collect an optional explanation after the decision and confidence rating."""
    label = "AI-generated" if response == "fake" else "real"
    prompt = text(win, f"What led you to decide this image was {label}?",
                  (0, 0.31), height=0.039, bold=True, wrap=1.35, max_height=0.085)
    hint = text(win, "Optional: write any reasons, or leave blank. Click CONTINUE when finished.",
                (0, 0.215), height=0.025, color=MUTED, wrap=1.3, max_height=0.055)
    box = visual.TextBox2(
        win, text="", font=FONT, pos=(0, -0.015), units="height",
        size=(1.30, 0.34), letterHeight=0.03, color=TEXT,
        fillColor=PANEL, borderColor=ACCENT, padding=0.02,
        alignment="top-left", overflow="scroll", editable=True, autoLog=False,
        placeholder="Type your reasons here (optional)...",
    )
    button = Button(win, "CONTINUE", (0, -0.32), (0.34, 0.085),
                    "continue", text_height=0.032)
    clock = core.Clock()
    event.clearEvents()
    win.callOnFlip(clock.reset)
    win.mouseVisible = True
    box.hasFocus = True
    was_pressed = bool(mouse.getPressed()[0])
    try:
        while True:
            if "escape" in event.getKeys(keyList=["escape"]):
                raise QuitExperiment()
            hovered = button.contains(mouse.getPos())
            counter.draw()
            prompt.draw()
            hint.draw()
            box.draw()
            button.draw(hovered)
            overlay.draw()
            win.flip()
            pressed = bool(mouse.getPressed()[0])
            if pressed and not was_pressed and hovered:
                return box.text.strip(), clock.getTime()
            was_pressed = pressed
    finally:
        # Remove the editable from the window so later screens don't receive text.
        box.editable = False
        event.clearEvents()


def run_ready(win, fixation, gaze):
    """Self-paced ready cross. Returns the measured drift in px (or None).

    Blocks until the tracker has fresh gaze. If gaze sits far from the cross,
    the first ENTER shows a recalibrate prompt; a second ENTER continues.
    """
    win.mouseVisible = False
    hint = text(
        win,
        "Press ENTER or SPACE when ready   •   R to recalibrate",
        (0, -0.42),
        height=0.024,
        color=MUTED,
        max_height=0.055,
    )
    status = text(win, " ", (0, -0.33), height=0.026, color=MUTED, bold=True)
    # Red: waiting for tracking; amber: off-centre; green: near the cross.
    ring = visual.Circle(
        win,
        radius=0.07,
        units="height",
        fillColor=None,
        lineColor="red",
        lineWidth=5,
    )

    gaze_dot = visual.Circle(win, radius=10, units="pix", fillColor="#00e5ff",
                             lineColor="black", lineWidth=2)
    event.clearEvents()
    floor = core.Clock()
    drift_warned = False
    drift = None
    last_status = None

    while True:
        sample = gaze.poll()
        fresh = sample is not None and gaze.recent_gaze(GAZE_READY_WINDOW)
        drift = gaze.drift_px(DRIFT_WINDOW_S)
        # Refresh the tracking status each frame.
        if gaze.seconds_since_valid() > FACE_LOST_AFTER:
            status.text = ("Eye tracker cannot see you — please face the "
                           "camera and keep your head still")
        elif drift_warned:
            status.text = ("Gaze looks off-centre. Press R to recalibrate, "
                           "or ENTER to continue anyway")
        else:
            status.text = "Blue dot = your gaze. Look at the centre cross."

        if status.text != last_status:
            status.height = 0.026
            fit_text(status, 1.0, 0.08)
            last_status = status.text

        ready = (floor.getTime() >= FIXATION_DURATION
                 and gaze.recent_gaze(GAZE_READY_WINDOW))
        aligned = drift is not None and drift <= DRIFT_WARN_PX
        ring.lineColor = "green" if ready and aligned else ("#D58B00" if ready else "red")

        ring.draw()
        if fresh:
            gaze_w, gaze_h = gaze.screen_size
            screen_w, screen_h = win.size
            gaze_dot.pos = (sample[0] / gaze_w * screen_w - screen_w / 2,
                            screen_h / 2 - sample[1] / gaze_h * screen_h)
            gaze_dot.draw()
        fixation.draw()
        hint.draw()
        status.draw()
        win.flip()

        keys = event.getKeys(keyList=["return", "num_enter", "space",
                                      "escape", "r"])
        if "escape" in keys:
            raise QuitExperiment()
        if "r" in keys:
            raise RecalibrateRequest()
        if not any(key in keys for key in ("return", "num_enter", "space")):
            continue

        # only advance while the camera is currently producing gaze, so the
        # recording is live at image onset
        if not ready:
            continue

        drift = gaze.drift_px(DRIFT_WINDOW_S)
        if drift is not None and drift > DRIFT_WARN_PX and not drift_warned:
            drift_warned = True
            continue
        return drift


def run_iti(win):
    win.mouseVisible = False
    t = core.Clock()
    while t.getTime() < ITI_DURATION:
        overlay.draw()
        win.flip()
