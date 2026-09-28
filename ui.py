"""Reusable drawing primitives for the experiment."""
import math

from psychopy import visual

from config import (ACCENT, BTN_FILL, BTN_HOVER, BTN_LINE, FONT, PANEL,
                    PANEL_LINE, TEXT)


#################### Drawing helpers ####################
def rounded_vertices(w, h, r, n=8):
    r = min(r, w / 2, h / 2)
    # each corner is a quarter-arc; the 3rd value is its start angle
    corners = [
        (w / 2 - r, h / 2 - r, 0),
        (-w / 2 + r, h / 2 - r, 90),
        (-w / 2 + r, -h / 2 + r, 180),
        (w / 2 - r, -h / 2 + r, 270),
    ]

    pts = []
    for cx, cy, start in corners:
        for k in range(n + 1):
            a = math.radians(start + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

    return pts


def rounded_rect(win, size, pos, radius=0.02, fill=PANEL, line=None,
                 line_width=2):
    # no outline passed means borderless, so reuse the fill colour
    return visual.ShapeStim(
        win,
        vertices=rounded_vertices(size[0], size[1], radius),
        pos=pos,
        fillColor=fill,
        lineColor=line if line else fill,
        lineWidth=line_width,
        closeShape=True,
    )


def text(win, txt, pos, height=0.04, color=TEXT, bold=False, wrap=1.0,
         anchor_h="center", anchor_v="center"):
    return visual.TextStim(
        win,
        text=txt,
        pos=pos,
        height=height,
        color=color,
        bold=bold,
        font=FONT,
        wrapWidth=wrap,
        anchorHoriz=anchor_h,
        anchorVert=anchor_v,
        # anchor pins the box to pos, alignText lines the text up inside it
        alignText=anchor_h,
    )


class Button:
    def __init__(self, win, label, pos, size, value, text_height=0.05):
        self.value = value
        self.shape = rounded_rect(
            win, size, pos, radius=0.022,
            fill=BTN_FILL, line=BTN_LINE
        )
        self.label = text(
            win, label, pos, height=text_height, bold=True
        )

    def contains(self, mouse_pos):
        return self.shape.contains(mouse_pos)

    def draw(self, hovered=False):
        self.shape.fillColor = BTN_HOVER if hovered else BTN_FILL
        self.shape.lineColor = ACCENT if hovered else BTN_LINE
        self.shape.draw()
        self.label.draw()


def make_fixation(win):
    a, t = 0.03, 0.004
    # a = arm length, t = half-thickness; the vertices trace a plus sign
    verts = [
        (-t, a), (t, a), (t, t), (a, t),
        (a, -t), (t, -t), (t, -a),
        (-t, -a), (-t, -t), (-a, -t),
        (-a, t), (-t, t),
    ]
    return visual.ShapeStim(
        win,
        vertices=verts,
        fillColor=TEXT,
        lineColor=None,
        closeShape=True,
    )
