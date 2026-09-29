"""Instruction, participant-entry, and response screens."""
from psychopy import core, event, visual

from config import (ACCENT, BTN_FILL, BTN_HOVER, BTN_LINE, FIXATION_DURATION,
                    ITI_DURATION, MUTED, PANEL, PANEL_LINE, QuitExperiment,
                    TEXT)
from ui import Button, rounded_rect, text


#################### Instruction screens ####################
def info_screen(win, title, body=None, steps=None,
                footer="Press SPACE to continue"):
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

    for stim in stims:
        stim.draw()

    win.flip()
    pressed = event.waitKeys(keyList=["space", "escape"])
    return not (pressed and pressed[0] == "escape")


#################### Participant entry ####################
def participant_screen(win, participant="001", session="001"):
    fields = [list(participant), list(session)]
    labels = ("Participant ID", "Session")
    active = 0

    card_size = (1.18, 0.74)
    card_pos = (0, 0)

    field_width, field_height = 0.86, 0.08
    field_y_positions = [0.08, -0.10]
    label_offset_y = 0.058

    # invisible rects, only used to hit-test which field gets clicked
    box_shapes = [
        visual.Rect(win, width=field_width, height=field_height, pos=(0.0, y), units="height")
        for y in field_y_positions
    ]

    btn_continue = Button(
        win, "CONTINUE", (0.0, -0.25), (0.32, 0.08), value="continue", text_height=0.035
    )

    mouse = event.Mouse(win=win)
    win.mouseVisible = True

    def draw():
        card = rounded_rect(
            win, card_size, card_pos,
            radius=0.035, fill=PANEL, line=PANEL_LINE
        )

        stims = [
            card,
            text(
                win, "SEEING THE FAKE", (0, 0.285),
                height=0.019, color=MUTED, bold=True
            ),
            text(
                win, "Participant details", (0, 0.22),
                height=0.05, bold=True
            ),
        ]

        mouse_pos = mouse.getPos()

        for i, label in enumerate(labels):
            y_box = field_y_positions[i]
            y_label = y_box + label_offset_y
            value = "".join(fields[i])

            is_active = (i == active)
            is_hovered = box_shapes[i].contains(mouse_pos)

            stims.append(
                text(
                    win, label, (0.0, y_label),
                    height=0.024, color=MUTED,
                    anchor_h="center", anchor_v="bottom"
                )
            )

            line_color = ACCENT if is_active else (BTN_HOVER if is_hovered else BTN_LINE)
            fill_color = BTN_HOVER if (is_active or is_hovered) else BTN_FILL

            stims.append(
                rounded_rect(
                    win, (field_width, field_height), (0.0, y_box),
                    radius=0.018, fill=fill_color, line=line_color, line_width=2
                )
            )

            # the +0.004 nudges it down so it reads centred, not just measured
            stims.append(
                text(
                    win, value or " ", (0.0, y_box + 0.004),
                    height=0.038, color=TEXT, bold=True,
                    anchor_h="center", anchor_v="center"
                )
            )

        stims.append(
            text(
                win, "TAB switch field   •   ENTER continue   •   ESC quit",
                (0, -0.325),
                height=0.020, color=MUTED
            )
        )

        for stim in stims:
            stim.draw()

        btn_continue.draw(hovered=btn_continue.contains(mouse_pos))

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

        if "return" in keys or "num_enter" in keys:
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


def run_fixation(win, fixation):
    win.mouseVisible = False
    fixation.draw()
    win.flip()
    core.wait(FIXATION_DURATION)


def run_iti(win):
    win.mouseVisible = False
    win.flip()
    core.wait(ITI_DURATION)
