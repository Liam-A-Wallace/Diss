# Five-photo processing demo

The demo uses three genuine portraits and two generated faces from the curated
`Dataset/datasets/picked` collection. These are actual image stimuli, not the
cartoon files currently referenced by the ordinary `trials.csv`. It copies just
five files to `stimuli/demo` and writes `data/demo_trials.csv`. It does not
replace the main stimulus list. The order within collection is randomised by
the normal experiment and recorded in the session logs.

From `Diss`, using a working Python environment with `requirements.txt` installed:

```powershell
python -m tools.demo_landmarks
python demo.py
python analyse.py --test
```

1. `tools/demo_landmarks.py` opens a preview of facial landmarks on the five stimulus
   images and saves original-pixel and normalised coordinates to
   `processed/demo_landmarks/landmarks.csv`. It uses the existing local
   `GazeFollower/gazefollower/res/model_weights/face_landmarker.task` and
   MediaPipe Tasks API. Zero or multiple detected faces are flagged for review.
   The landmark preview folder is refreshed on rerun.
2. `demo.py` runs calibration, five self-paced image trials, real/fake responses
   and confidence ratings. There is no practice trial or halfway break. All
   recordings go to `data/test`, with the ordinary timing and zoom/pan mapping.
   Try looking at the eyes, nose and mouth, and zoom/pan on one image.
   ENTER/SPACE advances, R recalibrates on ready screens, ESC saves a partial run.
3. `analyse.py --test` selects the latest completed test session and opens its
   heatmaps, numbered fixation paths and tables. Check the printed session name
   if you have other test recordings. The plots are based on measured gaze;
   neither demo script invents eye movements.

`python demo.py --prepare` prepares just the five images and list.
`python demo.py --gaze-dot` also shows a live gaze dot; leave it off for a less
influenced viewing check.

Landmark and gaze coordinates share original-image pixel space: top-left origin,
x right, y down. Join `landmarks.csv` to analysis `samples.csv` or `fixations.csv`
by stimulus, then compare `image_x` and `image_y`. Landmark IDs identify points;
they do not by themselves define facial area masks. These landmarks describe
the displayed faces, not the participant's webcam face.

For per-region dwell and transitions, after recording the demo:

```powershell
python -m pip install -r requirements-aois.txt
python analyse.py aois --test
python analyse.py review --test
python analyse.py --test
```

Inspect the generated AOI masks before approving them in the terminal. The
ordinary scanpath analysis also runs without masks, with regions marked unmapped.
Webcam accuracy and fixation thresholds still need checking against the locations
you deliberately looked at; this pilot is a processing check.

