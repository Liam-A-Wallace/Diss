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
    if _gaze is None:
        return

    # always poll so the latest gaze is tracked even when the dot is hidden
    sample = _gaze.poll()
    if sample is not None:
        _last = sample

    if not _enabled or _dot is None or _last is None:
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
    # GazeFollower reports gaze in screen pixels with (0,0) at the top-left.
    # PsychoPy's "height" units run -1..+1 vertically and -aspect..+aspect
    # horizontally, so the factor of 2 maps a full screen width/height.
    x = (_smooth[0] / win_w - 0.5) * 2.0 * (win_w / win_h)
    y = (0.5 - _smooth[1] / win_h) * 2.0
    _dot.pos = (x, y)
    _dot.draw()


def reset():
    global _last, _smooth
    _last = None
    _smooth = None


def has_gaze():
    return _last is not None
