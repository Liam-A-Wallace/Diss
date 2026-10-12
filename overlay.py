"""Live gaze-dot overlay for demo mode."""

from psychopy import visual

from config import GAZE_DOT_SMOOTHING

_win = None
_gaze = None
_dot = None
_enabled = False

_last = None
_smooth = None


def setup(win, gaze, enabled):
    global _win, _gaze, _enabled, _dot

    _win = win
    _gaze = gaze
    _enabled = enabled
    _dot = None

    if enabled:
        _dot = visual.Circle(
            win,
            radius=25,
            units="pix",
            fillColor="red",
            lineColor=None,
        )


def draw():
    global _last, _smooth

    if _gaze is None:
        return

    sample = _gaze.poll()

    if sample is not None:
        _last = sample

    if not _enabled or _dot is None or _last is None:
        return

    raw_x, raw_y = _last

    # GazeFollower reports calibrated gaze in the pixel space of ITS screen
    # size (config.screen_size), with (0, 0) = top-left and +y downward.
    # Normalise by that source resolution, then remap into the PsychoPy
    # window's own pixel space, so a monitor/resolution mismatch between the
    # tracker and the window can't skew the dot (and when they match, this is
    # identical to a plain offset/flip).
    gaze_w, gaze_h = _gaze.screen_size or (1920, 1080)
    screen_w, screen_h = _win.size

    norm_x = raw_x / gaze_w
    norm_y = raw_y / gaze_h

    # PsychoPy "pix" units: (0, 0) = centre, +x right, +y up.
    target = (
        norm_x * screen_w - screen_w / 2,
        screen_h / 2 - norm_y * screen_h,
    )

    # Exponential moving average: the dot eases toward the latest sample
    # instead of snapping, which kills the raw-sample jitter. Lower alpha =
    # smoother but laggier (tune via GAZE_DOT_SMOOTHING in config.py).
    if _smooth is None:
        _smooth = target
    else:
        alpha = GAZE_DOT_SMOOTHING
        _smooth = (
            alpha * target[0] + (1 - alpha) * _smooth[0],
            alpha * target[1] + (1 - alpha) * _smooth[1],
        )

    _dot.pos = _smooth

    _dot.draw()


def show():
    global _enabled
    _enabled = True
    reset()


def hide():
    global _enabled
    _enabled = False


def reset():
    global _last, _smooth
    _last = None
    _smooth = None


def has_gaze():
    # delegate to the recorder's fresh-sample check so the gate reflects the
    # *current* tracking state, not a single historical sample.
    if _gaze is None:
        return False
    return _gaze.recent_gaze()