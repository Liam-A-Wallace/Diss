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
PRACTICE_CSV = "practice.csv"  # optional; falls back to the first trials

# Eye-tracker readiness gate (the "ready" cross).
# GAZE_READY_WINDOW: a valid calibrated sample must have arrived within this
#   many seconds for the tracker to count as "ready" (guards against stale
#   samples when the camera briefly loses the participant).
# FACE_LOST_AFTER: seconds without valid gaze before the ready screen asks
#   the participant to face the camera.
GAZE_READY_WINDOW = 0.3
FACE_LOST_AFTER = 2.0

# Drift check on the ready cross: median distance (px, in the tracker's
# screen space) of the last DRIFT_WINDOW_S seconds of gaze from the cross.
# Above DRIFT_WARN_PX the first ENTER shows a recalibrate prompt instead.
DRIFT_WINDOW_S = 0.5
DRIFT_WARN_PX = 100

# Rest break after this fraction of the main trials (None disables).
BREAK_AT_FRACTION = 0.5

# Save progress every N main trials so a crash cannot lose the session.
CHECKPOINT_EVERY = 5

# A trial's gaze is flagged low quality below this fraction of valid samples.
GAZE_MIN_VALID_FRAC = 0.7

# Demo gaze dot smoothing.
# GAZE_DOT_TAU: exponential-smoothing time constant in seconds. The dot eases
#   toward the latest gaze sample with this lag; higher = smoother but slower.
GAZE_DOT_TAU = 0.15

# Practice trials are stamped with this offset so their gaze data never
# collides with the numbered main trials (which use 1, 2, 3, ...).
PRACTICE_TRIGGER_OFFSET = 9000

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
VIEW_HINT_POS = (0, -0.465)
CONF_PROMPT_POS = (0, 0.30)
CONF_CAPTION_Y = -0.135
CONF_BUTTON_Y = -0.02
CONF_BUTTON_SPACING = 0.21

TRIALS_CSV = "trials.csv"
PLACEHOLDER_CSV = "trials_placeholder.csv"
DATA_DIR = "data"
CONFIDENCE_LABELS = ["1", "2", "3", "4", "5"]
