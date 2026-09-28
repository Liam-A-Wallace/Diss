"""Configuration and shared constants for the experiment."""


WIN_SIZE = (1280, 720)
FULLSCREEN = True
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
DATA_DIR = "data"
CONFIDENCE_LABELS = ["1", "2", "3", "4", "5"]


class QuitExperiment(Exception):
    pass
