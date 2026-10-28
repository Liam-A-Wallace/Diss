# Running gaze analysis

Run these commands from Diss, using your Python environment.

## After an experiment

Install the small analysis dependency set once:

~~~powershell
python -m pip install -r requirements.txt
~~~

Then use:

~~~powershell
python analyse.py
~~~

This selects the latest participant session, saves the tables and plots, and opens a local results page in your browser. Results go in processed/SESSION_NAME; repeating the command creates a numbered folder and preserves earlier results. The terminal prints the selected session and output location.

For a run collected with experiment.py --test:

~~~powershell
python analyse.py --test
~~~

Participant and test sessions have separate AOI folders. A currently running session is rejected. Older sessions without the new display/session sidecars cannot be reconstructed by this pipeline.

## Add facial AOIs

Install the optional model dependencies once:

~~~powershell
python -m pip install -r requirements-aois.txt
~~~

Generate and inspect your AOIs:

~~~powershell
python analyse.py aois
python analyse.py review
~~~

The aois command downloads the model on its first run and processes images locally. It uses the exact main and practice stimulus lists from your latest session. Before any session exists, it uses trials.csv and the optional practice.csv. Repeating it generates missing masks and preserves existing masks and reviews.

The review command opens a gallery of coloured overlays. Inspect every image, then return to the terminal and type reviewed to approve the set. Pressing Enter leaves it unapproved. No JSON editing is required. You can correct the single-channel label PNGs in aois/main while retaining their dimensions and integer IDs; run review again afterwards. Editing a reviewed mask invalidates its saved review hash.

Now run python analyse.py again to include AOI dwell and transitions. Add --test to all three commands for test data. If AOIs have not been generated, ordinary gaze analysis still works and labels regions unmapped. If AOIs exist but are missing or unreviewed, the command explains which short command to run next.

The [face-parsing model](https://huggingface.co/jonathandinu/face-parsing) supplies eyes, eyebrows, nose, mouth/lips, ears, hair, glasses and other skin/background classes. Left/right features are combined into broad regions. Forehead and hairline are explicitly derived approximations based on skin, brows and hair. Tilt, occlusion, bangs and baldness can invalidate them. Low-confidence pixels are uncertain; the confidence score is not a calibrated probability. The model specifies non-commercial research/educational use and needs validation on your generated faces.

## What to open

The web report stays at `processed/SESSION_NAME/index.html`. All analysis CSVs are grouped in `processed/SESSION_NAME/data_results/`, with download links in the report. Plot images and `analysis.json` remain beside the report. Rerunning analysis creates a new report folder and preserves previous exports.

The results page links to these files:

| File | Contents |
| --- | --- |
| trial_*_qc.png | Original image with a dwell heatmap and numbered fixations |
| data_results/features.csv | One row per trial: response, confidence, quality, fixation statistics and AOI dwell columns |
| data_results/samples.csv | Individual gaze samples, image coordinates, AOIs and quality flags |
| data_results/fixations.csv | Exploratory fixations with location, duration, segment and AOI |
| data_results/aoi_dwell.csv | Per-region observed dwell, fractions and first fixation latency |
| data_results/scanpaths.csv | Linked fixation steps: endpoints, distances, directions and intervals |
| data_results/transitions.csv | AOI changes within a continuous observation segment |
| analysis.json | Parameters and hashes of the source files and AOI masks |

Practice and interrupted trials are explicitly flagged. Trials with no gaze samples are retained. Missing responses stay blank. Tables identify both participant and session; repeated sessions should not be combined using participant ID alone. All manifest AOI labels have dwell columns, including zero-dwell regions.

## Recording files and coordinate conventions

The normal experiment automatically saves _gaze_image.csv, _display.jsonl and _session.json alongside the existing raw gaze, trial-window and behavioural files. Display changes are stamped in flip callbacks. Original image pixels have a top-left origin, +x right and +y down; image_u/image_v divide by original dimensions. usable=1 means a valid, finite, nonduplicate sample on the visible image. Off-image gaze is never clamped to its edges.

Default analysis uses calibrated screen coordinates. Filtered coordinates remain available for sensitivity analysis because filtering introduces temporal lag. Webcam timestamps are recorded after camera read, so camera exposure and physical display timing remain unmeasured. Collection currently requires fullscreen on the primary calibration monitor.

ESC preserves the partial viewing interval. An incomplete final display-log line after a hard crash is ignored with a warning; trials without a known offset are omitted and listed in analysis.json. Existing raw-gaze checkpointing still applies; this is not power-loss-proof storage.

## Interpretation and pilot validation

Observed dwell only counts time between usable samples in the same view with gaps of at most 100 ms. It does not fill missing gaze or extend the final sample. Fractions of total viewing time can therefore sum to less than 1. Heatmaps use this time in a 40 by 40 grid, with Gaussian display smoothing (sigma 1.2 grid cells) and transparent low-density edges. Colour scales are set separately for each trial, so colours are not absolute comparisons across trials. Smoothing affects the picture only; CSV dwell values, fixation detection and AOI calculations use the original samples. The grid and smoothing parameters are recorded in analysis.json.

Exploratory I-DT defaults are a summed x/y range of 0.08 in normalised image units, a 100 ms minimum duration and a 100 ms maximum gap. Invalid samples, long gaps and transform changes split segments. Scanpath length and transition plots do not reconnect across these boundaries. Scanpath length uses normalised image distance, not visual angle; no automatic saccade classification is claimed.

Before participant collection, check known locations at 1x, 2x and 4x after zoom, pan and reset; inspect calibration, dropouts and coverage; review every AOI; and compare fixation thresholds against pilot accuracy. Zoom mapping does not correct calibration error.

## Optional controls

~~~powershell
python analyse.py data/OLDER_SESSION_BASE
python analyse.py --no-open
python analyse.py --no-plots
~~~

If matplotlib is unavailable, the short command reports this and still saves CSVs. The lower-level python -m analysis.process --help exposes coordinate-source and fixation-threshold settings for methodological comparisons. Routine use does not need those options.

Run regression tests with python -m unittest discover -s tests -v. The sandbox tests use Python 3.12, synthetic data and a mocked display. Live PsychoPy/webcam behaviour, model inference and matplotlib rendering remain unvalidated here because the intended Python 3.10 environment and those dependencies are unavailable.

## Scanpath analysis

Each analysis now exports `data_results/scanpaths.csv`: one row for each linked
pair of fixations, including endpoints, AOIs, image distance, direction and time
between fixations. Links stop at invalid tracking segments, display-transform
changes and gaps beyond the configured maximum. No links are inferred across
these boundaries.

`features.csv` additionally includes linked step count, segment count (including
isolated fixations), mean/max step distance, total fixation duration, summed
segment endpoint displacement, straightness and mean absolute turning angle.
Straightness is summed endpoint displacement divided by summed path length;
zero-length paths and measures with no supporting steps are blank. A straight
path scores 1; a path returning to its starting point scores 0. These measures
are descriptive and depend on the fixation thresholds and calibration quality.

Distances use normalised original-image coordinates, not degrees of visual
angle. Directions use image coordinates: right = 0 degrees, down = +90, up = -90.
The interval between fixations is not a measured saccade duration. These are
exploratory scanpath steps, not automatic saccade detections.

Rerun `python analyse.py --test` for a new report with these exports. Earlier
recordings made before the worker fix may have used the SDK default gaze model
and camera exposure rather than the settings listed in session metadata. The
worker fix affects new collections; it cannot correct previous recordings.
