# Seeing the Fake — Study 1

Minimal, git-friendly PsychoPy **Coder** experiment for the Study 1 core trial
loop: fixation cross → face image → real/fake classification → confidence
rating. Eye tracking is handled by GazeFollower (see `gaze.py`).

## Layout

- `experiment.py` — main experiment script (window, trial loop, data saving)
- `gaze.py` — `GazeRecorder` wrapper around the GazeFollower SDK (continuous sampling, image-window trimming)
- `gazefollower_patch.py` — camera exposure lock, applied on top of the unmodified GazeFollower checkout
- `tools/make_placeholders.py` — generates placeholder face images + `trials_placeholder.csv`
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

### View gaze analysis

Install once, then analyse your latest participant session:

~~~powershell
python -m pip install -r requirements.txt
python analyse.py
~~~

A local results page opens with heatmaps, fixations and CSV tables. For a test
session, use python analyse.py --test. To add facial regions, install
requirements-aois.txt, then run python analyse.py aois and python analyse.py review
once for your stimulus set. The model download is automatic.

See [the analysis guide](../analysis/README.md) for AOI review and data definitions.
Scroll zoom (1–4x), drag-to-pan and HOME reset are available during image viewing.

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

The gaze model is selected by `GAZE_MODEL` in `config.py` (default
`base_32M.mnn`, the newer 32M model). It is passed to GazeFollower at runtime
via `gaze.py`, so the upstream GazeFollower checkout is left untouched — no
model files are replaced in the repo.

## Notes / next steps

- The stimulus set is a placeholder until a real dataset is chosen (see the
  dissertation plan). `tools/make_placeholders.py` writes `trials_placeholder.csv`
  only; edit `trials.csv` by hand for the real set.
- `stimuli/` and `data/` are gitignored.
- Camera exposure is currently fixed at -5 for this webcam. Check the preview
  and tune `CAMERA_EXPOSURE` in `config.py` before calibration; automatic mode
  is available through `LOCK_CAMERA_EXPOSURE = False`.
- Eye tracking is handled by GazeFollower via `gaze.py`; no C++ wrapper needed.

### Five-photo pilot

See [five-photo guide](../docs/DEMO.md) for a five-photo run using actual curated portraits, facial-landmark previews and measured scanpath analysis.

