# Five-image demo

Run from `Diss` with the experiment environment active:

```powershell
python demo.py
```

The demo uses three real portraits and two generated faces from
`../Dataset/datasets/picked`. It copies them to `stimuli/demo`, writes
`data/demo_trials.csv`, and randomises their order during collection.

Enter the participant details and read the instructions, then complete the
camera preview and calibration in the same window. The portrait check shows
a blue gaze dot: Enter/Space accepts, R recalibrates, and Esc quits. This check
is separate from the five trials. The demo skips normal practice and the break.

The ready cross also shows a blue gaze dot. During image viewing, scroll to
zoom, drag to pan and press Home to reset. Enter/Space advances to the real/fake
decision and confidence rating. Esc on a task/check screen saves a partial run.

## Results

```powershell
python analyse.py --test
```

The terminal prints the session and output folder. The browser report shows
heatmaps and fixation paths, with links to the CSVs in `data_results/`. Check the
session name if you have several test recordings.

For facial-region dwell and transitions, install the AOI dependencies once,
then generate and inspect the masks:

```powershell
python -m pip install -r requirements-aois.txt
python analyse.py aois --test
python analyse.py review --test
python analyse.py --test
```

Approve the masks after checking the overlays. Gaze plots also work without
masks; region labels will be `unmapped`.

## Other commands

| Command | Purpose |
| --- | --- |
| `python demo.py --prepare` | Prepare the images and list without opening the camera |
| `python demo.py --gaze-dot` | Also show a gaze dot during image viewing |
| `python -m tools.demo_landmarks` | Preview landmarks on the five displayed faces |

Landmarks are saved to `processed/demo_landmarks/landmarks.csv`. That preview
folder is refreshed on rerun. The detector uses the local MediaPipe model in
`../GazeFollower/gazefollower/res/model_weights/face_landmarker.task`. Images
with zero or multiple detected faces are flagged for review.

Landmark and gaze exports use original image pixels: x right, y down, origin at
the top-left. Join by `stimulus` to compare `image_x` and `image_y` in the
landmark, sample and fixation tables. Landmark points are separate from the
masks used for AOI dwell.

Check calibration, dropouts, zoom mapping and fixation settings before
collecting participant data. The portrait check is subjective; it does not
measure accuracy. See the [calibration controls](EXPERIMENT.md#calibration-and-image-check).
