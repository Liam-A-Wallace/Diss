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

    sample = _gaze.poll()

    if sample is not None:
        _last = sample

    if not _enabled or _dot is None or _last is None:
        return

    raw_gx, raw_gy = _last

    win_w, win_h = _win.size

    # GazeFollower uses screen pixels with (0, 0) at the
    # top-left. Convert to PsychoPy's centred pixel coordinates.
    px_x = raw_gx - win_w / 2
    px_y = win_h / 2 - raw_gy

    # Convert PsychoPy pixel coordinates to height units.
    x = px_x / (win_h / 2)
    y = px_y / (win_h / 2)

    if _smooth is None:
        _smooth = (x, y)
    else:
        _smooth = (
            ALPHA * x + (1.0 - ALPHA) * _smooth[0],
            ALPHA * y + (1.0 - ALPHA) * _smooth[1],
        )

    _dot.pos = _smooth
    _dot.draw()
def reset():
    global _last, _smooth
    _last = None
    _smooth = None


def has_gaze():
    return _last is not None
