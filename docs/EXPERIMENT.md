# Experiment setup and controls

Collection needs a display, webcam and working OpenGL drivers. Install Python
3.10 or 3.11 and the dependencies in a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

GazeFollower is imported from `../GazeFollower`. Install its runtime dependencies
in the same environment; see [the SDK README](../../GazeFollower/README.md).

Run from `Diss`:

| Command | Run |
| --- | --- |
| `python experiment.py` | Full experiment using `trials.csv` |
| `python demo.py` | Five curated images; output in `data/test` |
| `python experiment.py --test` | Full trial list; output in `data/test` |
| `python experiment.py --demo` | Full trial list with a gaze dot during image viewing |
| `python experiment.py --placeholder` | Generate and use cartoon placeholders |

Flags can be combined. Cartoon test images are isolated under
`stimuli/placeholders/{real,fake}`; normal runs use the photos in `trials.csv`.

`trials.csv` needs `stimulus`, `condition` and `difficulty` columns. Image paths
are relative to `Diss`. Optional `practice.csv` uses the same format. Otherwise,
the configured number of initial trials is reserved for practice.

## Calibration and image check

Participant details and task instructions come first. Camera preview,
calibration and trials use the same fullscreen window. Collection requires the
primary calibration monitor (`FULLSCREEN=True`, `SCREEN=0`).

A portrait appears after each calibration. Look at its eyes, nose and mouth and
check whether the blue gaze dot follows. Enter/Space accepts, R repeats
calibration, and Esc quits at the check. Acceptance requires fresh tracking and
at least one second of viewing. The dot hides when tracking is lost; filtered
gaze can lag slightly behind eye movement.

The check prefers an unused curated portrait, with a practice/task image as a
fallback. Session metadata records the image and its hash. Attempts are logged
in `_calibration_checks.jsonl` and excluded from trial analysis. This is a
subjective check, not a measured accuracy validation.

## Trial controls

The full experiment currently has 30 main trials, no practice trials
(`PRACTICE_TRIALS=0`) and a halfway break. Adjust these defaults in `config.py`.
The five-image demo skips normal practice and the break. Each trial has a ready
cross, image viewing, real/fake classification, confidence rating from 1 to 5,
and an optional text box asking what led to the participant's decision. Click
CONTINUE to submit the explanation or leave it blank. Enter adds a new line;
Space types a space. The image is not shown again during explanation entry.
This step also appears during practice and in the five-image demo.

The trial CSV saves the explanation as `decision_reason` and the time spent on
that screen as `rt_decision_reason` (seconds). A skipped explanation is an empty
string. Classification and confidence timings are recorded before this step.

| Control | Action |
| --- | --- |
| Enter/Space | Advance or accept the calibration check |
| Mouse wheel over the image | Zoom between 1x and 4x |
| Left-button drag | Pan the image |
| Home | Reset the view |
| R on a ready/instruction screen | Recalibrate and repeat the portrait check |
| Esc on a task/check screen | Quit and save the partial run |

The ready cross shows a blue gaze dot. Its ring is red while waiting for
tracking, amber for off-centre gaze and green when recent gaze is near the
cross. Look at the fixed cross rather than following the dot. A drift warning
can be overridden on the second attempt to continue.

## Camera settings

`CAMERA_INDEX`, `LOCK_CAMERA_EXPOSURE` and `CAMERA_EXPOSURE` are in `config.py`.
Check the preview in the room and lighting used for collection. Supported
exposure values depend on the webcam driver. Restart after changing them.

`GAZE_MODEL` selects a model by filename in the SDK model folder, or by path.
`gaze_worker.py` passes that choice and the exposure settings to the worker.
Recordings made before this fix may have used SDK defaults despite different
settings in their session metadata.

## Recording files

Each session in `data/` (or `data/test/`) produces:

| Suffix | Contents |
| --- | --- |
| `.csv` | Responses, confidence, timing and quality fields |
| `_gaze_raw.csv` | Session gaze stream |
| `_gaze.csv` | Samples trimmed to image-viewing intervals |
| `_windows.csv` | Image onset and offset timestamps |
| `_gaze_image.csv` | Optional gaze mapped to original image coordinates; enable with `--export-image-gaze` |
| `_display.jsonl` | Image positions, scales and display changes |
| `_session.json` | Session details, settings and image hashes |
| `_calibration_checks.jsonl` | Portrait-check outcomes |
| `_error.txt` | Traceback saved if the session fails |
| `.psydat` | PsychoPy session pickle |

Keep source recordings, timing sidecars and stimulus images together.
Checkpoints reduce losses from a crash but cannot guarantee recovery after
power loss. See the [analysis guide](../analysis/README.md) for processing.

## GazeFollower

This project uses [GazeFollower](https://github.com/Gancheng-Zhu/GazeFollower)
(Zhu et al., 2025, ACM CGIT), licensed under CC BY-NC-SA 4.0, for non-commercial
academic research. Models are selected at runtime rather than replaced in the
SDK checkout. Requests for the 32M model should use a university email and
state the academic research purpose.
