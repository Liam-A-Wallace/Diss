"""Live gaze-dot overlay for demo mode.

A module-level singleton so every screen can draw the dot without threading
the gaze recorder through every function signature. Call ``setup`` once after
the window exists, then ``draw`` right before each ``win.flip()``.
"""

from psychopy import visual

_win = None
_gaze = None
_dot = None
_enabled = False

_last = None    # last valid gaze point, in screen pixels
_smooth = None  # exponentially-smoothed point, in screen pixels

# higher = smoother but laggier dot
ALPHA = 0.35


def setup(win, gaze, enabled):
    global _win, _gaze, _enabled, _dot
    _win = win
    _gaze = gaze
    _enabled = enabled
    _dot = None
    if enabled:
        _dot = visual.Circle(win, radius=0.012, fillColor="red",
                             lineColor=None)


def draw():
    global _last, _smooth
    if not _enabled or _dot is None or _gaze is None:
        return

    sample = _gaze.poll()
    if sample is not None:
        _last = sample
    if _last is None:
        return

    px_x, px_y = _last
    if _smooth is None:
        _smooth = (px_x, px_y)
    else:
        _smooth = (
            ALPHA * px_x + (1.0 - ALPHA) * _smooth[0],
            ALPHA * px_y + (1.0 - ALPHA) * _smooth[1],
        )

    win_w, win_h = _win.size
    x = (_smooth[0] / win_w - 0.5) * (win_w / win_h)
    y = 0.5 - _smooth[1] / win_h
    _dot.pos = (x, y)
    _dot.draw()
