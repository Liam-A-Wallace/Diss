# Developer notes

The experiment runs in PsychoPy. GazeFollower estimates webcam gaze in a
separate process. Its checkout is in `../GazeFollower`; raw image datasets are
in `../Dataset`. SDK overrides belong in `gaze_worker.py` and
`gazefollower_patch.py`.

## Main files

| File | Purpose |
| --- | --- |
| `experiment.py` | Setup, trial loop, responses and saving |
| `demo.py` | Five curated images, with output under `data/test` |
| `config.py` | Display, camera, exposure, timing and calibration settings |
| `gaze.py` | Tracker setup, sampling, calibration, drift checks and exports |
| `gaze_worker.py` | Passes the chosen model and exposure settings to the worker |
| `gaze_preview.py` | Prepares the first camera preview in the background during Welcome |
| `gazefollower_patch.py` | Adds exposure control to the SDK camera |
| `calibration_check.py` | Portrait and gaze dot shown after calibration |
| `screens.py`, `ui.py`, `texts.py` | Screens, drawing helpers and instructions |
| `image_view.py` | Zoom/pan geometry and mapping gaze to image pixels |
| `viewing.py` | Image viewing, input handling and display timestamps |
| `overlay.py` | Optional gaze dot during image trials |
| `session_log.py` | Session metadata, display history and coordinate export |
| `trials.py` | Reads trial lists and checks image paths |
| `analyse.py` | Analysis command and HTML reports |
| `analysis/` | Sample mapping, fixations, scanpaths, AOIs and plots |

## Collection flow

Participant details and instructions come before calibration. Preview,
calibration and the portrait check use the task window. The check prefers an
image outside the planned trials, with a practice/task image as a fallback.
Outcomes are logged separately in `_calibration_checks.jsonl`.

A trial loads its image before the ready cross, then runs image viewing,
classification, confidence rating and the interval between trials. The cross
shows a live gaze dot. The first attempt to continue with large drift gives a
warning; a second attempt can override it. R recalibrates and repeats the check.

`QuitExperiment` and `RecalibrateRequest` return control to the trial or screen
loop. Shutdown saves each output separately so one failed save does not prevent
the other steps. Keep that cleanup in `finally` blocks.

## Timing and coordinates

Gaze is sampled continuously during collection and paused for calibration.
Image onset, display changes and offset are stamped in flip callbacks using
epoch nanoseconds. The raw stream is trimmed to image intervals on shutdown.
Webcam timestamps follow the camera read; they do not measure sensor exposure
or physical display timing.

GazeFollower coordinates start at the screen's top-left. PsychoPy pixel
coordinates start at the centre, with positive y upwards. Scale by tracker and
window resolutions before converting between them. Image coordinates use the
original image's top-left, with positive y downwards. Preserve the logged
zoom/pan transforms when changing the viewing code.

Raw gaze and `_windows.csv` can rebuild the trimmed file with
`gaze.trim_gaze_file`. Current analysis also needs `_session.json` and
`_display.jsonl`. It checks image hashes, so replacing a recorded stimulus can
prevent reconstruction.

## Data and tools

`data/` holds recordings, `aois/` holds masks and reviews, and `processed/` holds
reports. Each report has a `data_results/` folder for its CSVs. These folders
and image/model assets are ignored by Git.

The current selection tool is `dataset/scripts/review.py`. `export_to_diss.py`
checks sources and stages copies before replacing the experiment set.
`curate.py` is the older Easy/Medium/Hard workflow. `stimulus-curation/`
contains optional image enrichment tools.

Run from `Diss`:

```powershell
python demo.py
python analyse.py --test
python -m unittest discover -s tests -v
```

Tests mock the display and tracker. Check live webcam accuracy, camera driver
settings and OpenGL behaviour on the collection setup.

See the [experiment guide](EXPERIMENT.md), [analysis notes](../analysis/README.md)
and [dataset guide](../dataset/README.md) for controls and file definitions.
