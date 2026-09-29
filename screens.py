"""Instruction, participant-entry, and response screens."""
from psychopy import core, event, visual

from config import (ACCENT, BTN_FILL, BTN_HOVER, BTN_LINE, FIXATION_DURATION,
                    ITI_DURATION, MUTED, PANEL, PANEL_LINE, QuitExperiment,
                    TEXT)
from ui import Button, rounded_rect, text

import overlay


#################### Instruction screens ####################
def info_screen(win, title, body=None, steps=None,
                footer="Press ENTER or SPACE to continue"):
    card = rounded_rect(
        win, (1.28, 0.84), (0, 0),
        radius=0.035, fill=PANEL, line=PANEL_LINE
    )

    stims = [
        card,
        text(
            win, "SEEING THE FAKE", (0, 0.355),
            height=0.019, color=MUTED, bold=True
        ),
        text(
            win, title, (0, 0.27),
            height=0.06, bold=True
        ),
        visual.Rect(
            win, size=(0.10, 0.005), pos=(0, 0.21),
            fillColor=ACCENT, lineColor=None
        ),
    ]

    if steps:
        start_y = 0.12
        spacing = 0.13

        if body:
            stims.append(
                text(
                    win, body, (0, 0.14),
                    height=0.027, wrap=1.0,
                    color=MUTED, anchor_v="top"
                )
            )
            start_y = 0.02
            spacing = 0.115

        for i, (heading, desc) in enumerate(steps):
            y = start_y - i * spacing

            stims.append(
                text(
                    win, f"{i + 1}. {heading}", (0, y),
                    height=0.034, bold=True
                )
            )
            stims.append(
                text(
                    win, desc, (0, y - 0.032),
                    height=0.025, color=MUTED, wrap=1.0
                )
            )

    elif body:
        stims.append(
            text(
                win, body, (0, 0.08),
                height=0.038, wrap=1.0,
                color=TEXT, anchor_v="top"
            )
        )

    stims.append(
        text(
            win, footer, (0, -0.375),
            height=0.024, color=MUTED
        )
    )

    event.clearEvents()
    while True:
        for stim in stims:
            stim.draw()
        overlay.draw()
        win.flip()

        keys = event.getKeys(keyList=["space", "return", "num_enter", "escape"])
        if keys:
            return "escape" not in keys


#################### Participant entry ####################
def participant_screen(win, participant="001", session="001"):
    fields = [list(participant), list(session)]
    labels = ("Participant ID", "Session")
    active = 0

    field_width, field_height = 0.86, 0.08
    field_y_positions = [0.08, -0.10]
    label_offset_y = 0.058

    # static stimuli, created once so the form doesn't rebuild them every frame
    card = rounded_rect(win, (1.18, 0.74), (0, 0), radius=0.035, fill=PANEL, line=PANEL_LINE)
    header = text(win, "SEEING THE FAKE", (0, 0.285), height=0.019, color=MUTED, bold=True)
    title = text(win, "Participant details", (0, 0.22), height=0.05, bold=True)
    hint = text(
        win, "TAB switch field   •   ENTER or SPACE continue   •   ESC quit",
        (0, -0.325), height=0.020, color=MUTED
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
                 anchor_h="center", anchor_v="center")
        )
        # invisible rect, only used to hit-test which field gets clicked
        box_shapes.append(
            visual.Rect(win, width=field_width, height=field_height,
                        pos=(0.0, y_box), units="height")
        )

    btn_continue = Button(
        win, "CONTINUE", (0.0, -0.25), (0.32, 0.08), value="continue", text_height=0.035
    )

    mouse = event.Mouse(win=win)
    win.mouseVisible = True

    def draw():
        mouse_pos = mouse.getPos()
        card.draw()
        header.draw()
        title.draw()

        for i in range(len(labels)):
            value_stims[i].text = "".join(fields[i]) or " "
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
        "participant": "".join(fields[0]).strip() or "001",
        "session": "".join(fields[1]).strip() or "001",
    }


#################### Response collection ####################
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
    for _ in range(120):
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


def run_ready(win, fixation):
    # self-paced "ready" gate: show the fixation cross with a hint and wait
    # for ENTER/SPACE, so the eye tracker has time to warm up before the face
    win.mouseVisible = False
    hint = text(
        win,
        "Press ENTER or SPACE when ready",
        (0, -0.42),
        height=0.024,
        color=MUTED,
    )

    event.clearEvents()
    floor = core.Clock()

    while True:
        fixation.draw()
        hint.draw()
        overlay.draw()
        win.flip()

        keys = event.getKeys(keyList=["return", "num_enter", "space", "escape"])
        if "escape" in keys:
            raise QuitExperiment()
        if floor.getTime() >= FIXATION_DURATION and any(
            key in keys for key in ("return", "num_enter", "space")
        ):
            break


def run_iti(win):
    win.mouseVisible = False
    t = core.Clock()
    while t.getTime() < ITI_DURATION:
        overlay.draw()
        win.flip()
