"""Configuration and shared constants for the experiment."""


WIN_SIZE = (1920, 1080)   # native resolution of the experiment monitor
FULLSCREEN = True
SCREEN = 0          # which monitor PsychoPy uses (0 = primary)
SHOW_GAZE = False   # demo: draw the live gaze point on screen
CAMERA_INDEX = 0   # which webcam GazeFollower opens

# Camera exposure. Setting LOCK_CAMERA_EXPOSURE = True stops the webcam from
# adapting to bright/dark stimulus changes (a bright cross screen followed by
# a dark photo was biasing the gaze estimate downward). CAMERA_EXPOSURE is the
# fixed exposure value once auto-exposure is off; the unit is backend-specific
# (OpenCV CAP_PROP_EXPOSURE), so it may need tuning per webcam.
LOCK_CAMERA_EXPOSURE = True
CAMERA_EXPOSURE = -6
CALIBRATION_POINTS = 13  # gaze calibration pattern: 5, 9, or 13 points
# Gaze smoothing. ENABLE_FACE_FILTER smooths the face/eye landmarks before
# they reach the gaze model (reduces jitter at the source). GAZE_FILTER picks
# temporal smoothing of the final gaze point: "one_euro" (adaptive, recommended)
# or "heuristic" (GazeFollower's default spike-remover).
ENABLE_FACE_FILTER = True
GAZE_FILTER = "one_euro"
FONT = "Open Sans"

# Palette (Soft Light / Paper Theme)
BG = "#EFEFEF"          # Soft, low-glare off-white background
PANEL = "#F7F7F7"       # Very light grey panel cards
PANEL_LINE = "#D6D6D6"  # Gentle grey borders
TEXT = "#1E2022"       # Almost-black text
MUTED = "#686D76"      # Muted slate text
ACCENT = "#0066FF"     # Bright electric blue accent
BTN_FILL = "#E4E4E4"   # Light neutral button background
BTN_HOVER = "#D0D0D0"  # Hover state
BTN_LINE = "#CCCCCC"   # Button borders

# Timing (seconds)
FIXATION_DURATION = 0.7
ITI_DURATION = 0.5
PRACTICE_TRIALS = 1

# Eye-tracker readiness gate (the "ready" cross).
# GAZE_READY_WINDOW: a valid calibrated sample must have arrived within this
#   many seconds for the tracker to count as "ready" (guards against stale
#   samples when the camera briefly loses the participant).
# READY_TIMEOUT: hard ceiling after which the ready screen advances even if
#   gaze never stabilises, so the session can't hang indefinitely.
GAZE_READY_WINDOW = 0.3
READY_TIMEOUT = 15.0

# Demo gaze dot smoothing.
# GAZE_DOT_TAU: exponential-smoothing time constant in seconds. The dot eases
#   toward the latest gaze sample with this lag; higher = smoother but slower.
GAZE_DOT_TAU = 0.15

# Practice trials are stamped with this offset so their gaze data never
# collides with the numbered main trials (which use 1, 2, 3, ...).
PRACTICE_TRIGGER_OFFSET = 9000

# Warm-up samples (recorded while the camera spins up behind the ready
# cross) are stamped with this so they can be filtered out later.
WARMUP_TRIGGER = 8000

# Experiment layout.
# PsychoPy uses height units, so the screen is roughly +/- 0.89 horizontally
# and +/- 0.5 vertically at 16:9.
STIM_HEIGHT = 0.86
IMAGE_POS = (0, -0.005)

TRIAL_NUMBER_POS = (-0.80, 0.455)
TRIAL_NUMBER_HEIGHT = 0.025

PROMPT_POS = (0, 0.34)
PROMPT_HEIGHT = 0.05
BUTTON_Y = -0.38

TRIALS_CSV = "trials.csv"
PLACEHOLDER_CSV = "trials_placeholder.csv"
DATA_DIR = "data"
CONFIDENCE_LABELS = ["1", "2", "3", "4", "5"]


class QuitExperiment(Exception):
    pass


class RecalibrateRequest(Exception):
    pass
