# Seeing the Fake — Study 1 (bare-bones PsychoPy experiment)

Minimal, git-friendly PsychoPy **Coder** experiment for the Study 1 core trial
loop: fixation cross → face image → real/fake classification → confidence
rating. Eye tracking is handled by GazeFollower (see `gaze.py`).

## Layout

- `experiment.py` — main experiment script (window, trial loop, data saving)
- `gaze.py` — `GazeRecorder` wrapper around the GazeFollower SDK (continuous sampling, image-window trimming)
- `gazefollower_patch.py` — camera exposure lock, applied on top of the unmodified GazeFollower checkout
- `make_placeholders.py` — generates placeholder face images + `trials_placeholder.csv`
- `config.py` / `ui.py` / `screens.py` / `texts.py` / `errors.py` / `trials.py` — constants, drawing primitives, screens, participant text, exceptions, trial-list loading
- `trials.csv` — stimulus list (`stimulus`, `condition`, `difficulty`)
- `practice.csv` — optional practice stimuli (same columns); otherwise the first trial(s) of the list are used and not repeated
- `data/` — per-participant output (gitignored); `--test` runs go to `data/test/`

## Setup

Requires Python 3.10 or 3.11 (3.10 recommended for PsychoPy wheel compatibility).

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Or with conda:

```powershell
conda create -n seeingthefake python=3.10
conda activate seeingthefake
pip install -r requirements.txt
```

Note: running the experiment needs a display (it opens a window). Windows also
requires working OpenGL drivers.

## Running

```powershell
# Run the experiment on the real stimulus list (trials.csv)
python experiment.py

# Or try it with generated placeholder stimuli
python experiment.py --placeholder
```

### Command-line flags

`python experiment.py` accepts the following flags:

| Flag | Effect |
|------|--------|
| `--demo` | Draw the live gaze point on screen (debug overlay). |
| `--placeholder` | Generate placeholder stimuli on the fly and run on `trials_placeholder.csv` instead of `trials.csv` — useful while the real stimulus set is still being curated. |
| `--test` | Write output to `data/test/` so test runs don't mix with participant data. |

Flags can be combined, e.g.:

```powershell
python experiment.py --demo --placeholder
```

### Keyboard controls

- **ENTER / SPACE** — continue / advance through screens.
- **R** — recalibrate the eye tracker (on instruction, break and ready screens).
- **ESC** — quit early (data collected so far is still saved).

Flow: the eye tracker is previewed and calibrated first, then:

1. Enter a participant ID and read the instructions.
2. Complete the practice trial(s) (no feedback).
3. For each trial: ready cross (checks the tracker sees the participant and
   that gaze is near the cross; offers R if it has drifted), face image
   (ENTER/SPACE to continue), click REAL or FAKE, then click a 1–5
   confidence button.
4. A rest break (with the option to recalibrate) halfway through.

Progress is checkpointed every few trials, so a crash does not lose the session.

## Output

For each session, `data/` receives:

- `<participant>_<session>_<date>.csv` — one row per trial (practice rows have
  `is_practice=1`) with `stimulus`, `condition`, `difficulty`, `response`,
  `correct`, `rt_classification`, `confidence`, `rt_confidence`, `view_time`,
  `onset_ns`/`offset_ns`, `drift_px`, `gaze_samples`, `gaze_valid_frac`,
  `dropped_frames`, plus participant/session/seed metadata.
- `<participant>_<session>_<date>_gaze.csv` — gaze samples recorded only while
  an image was on screen, with `trial`, `is_practice`, `condition` and
  `t_rel_ms` (time since image onset).
- `<participant>_<session>_<date>_gaze_raw.csv` — every sample of the session.
- `<participant>_<session>_<date>_windows.csv` — image onset/offset per trial;
  with the raw file it can rebuild the trimmed file after a crash
  (`gaze.trim_gaze_file`).
- `<participant>_<session>_<date>.psydat` — PsychoPy pickle of the full run.

## Eye tracking

Gaze tracking uses [GazeFollower](https://github.com/Gancheng-Zhu/GazeFollower)
(Zhu et al., 2025, "GazeFollower: An open-source system for deep learning-based
gaze tracking with web cameras", ACM CGIT), licensed under
CC BY-NC-SA 4.0. Use here is strictly non-commercial academic research.

If you request the 32M base model from the authors, use a university email and
state that it is for academic research only.

## Notes / next steps

- The stimulus set is a placeholder until a real dataset is chosen (see the
  dissertation plan). `make_placeholders.py` writes `trials_placeholder.csv`
  only; edit `trials.csv` by hand for the real set.
- `stimuli/` and `data/` are gitignored.
- Camera exposure is locked (`LOCK_CAMERA_EXPOSURE` in `config.py`) so bright
  and dark stimuli don't bias the gaze estimate.
- Eye tracking is handled by GazeFollower via `gaze.py`; no C++ wrapper needed.
