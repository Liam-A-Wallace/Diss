# Seeing the Fake - Study 1

A PsychoPy experiment that records webcam gaze while participants classify face
images as real or generated, rate their confidence, and optionally explain their decision.

## Running the experiment

Run these from `Diss`, with your Python environment active:

| Task | Command |
| --- | --- |
| Five-image demo | `python demo.py` |
| Full experiment using `trials.csv` | `python experiment.py` |
| Analyse the latest five-photo/test recording | `python analyse.py --test` |
| Generate facial AOI masks for that recording | `python analyse.py aois --test` |
| Inspect and approve those masks | `python analyse.py review --test` |
| Analyse the latest participant recording | `python analyse.py` |
| Preview landmarks on the five demo images | `python -m tools.demo_landmarks` |

After approving masks, rerun the analysis to include facial-region dwell and
transitions. The five-photo demo saves to `data/test`. `--demo` on the full
experiment means a live gaze dot; use `demo.py` for the shortened experiment.

`trials.csv` selects the active real photos and generated images from the shared-style,
high and extra-high folders. The demo selects three real images and two shared-style
fakes from that list, using their original paths without copying them.
The full list contains 15 real and 15 generated images, with one image per pair.
Real images are local copies in `stimuli/real/`; the picked source set remains
in the sibling `Dataset` folder.

## Project folders

| Location | Contents |
| --- | --- |
| `experiment.py`, `demo.py`, `analyse.py` | Main commands |
| `config.py` | Webcam, exposure, display, timing and experiment settings |
| Other root Python modules | Experiment UI, tracking, display mapping and recording |
| `analysis/` | Gaze processing, fixations and facial AOIs |
| `tools/` | Landmark previews and placeholder generation |
| `docs/` | Demo, experiment and developer guides |
| `dataset/` | Curation tools, source manifests, metadata and selection outputs |
| `tests/` | Regression checks |
| `stimuli/` | Images shown by the experiment |
| `data/` | Source recordings and session metadata; retain for analysis |
| `aois/` | Generated masks and their review records |
| `processed/` | Web reports and plots; each report has a `data_results/` folder containing the analysis CSVs |
| `models/` | Downloaded face-parsing model |
| `.venv/` | Installed Python environment |

Raw image corpora and GazeFollower live in the sibling `../Dataset/` and
`../GazeFollower/` folders.

`experiment.py` accepts `--trials PATH`, `--practice-trials N` and `--no-break`;
the demo passes these options explicitly. Add `--export-image-gaze` to produce
the convenience `_gaze_image.csv` at shutdown. Normal analysis maps the preserved
raw gaze and display logs without needing that extra export.

## Guides

- [Five-image demo and landmark preview](docs/DEMO.md)
- [Gaze analysis and AOI definitions](analysis/README.md)
- [Full experiment setup, controls and recording files](docs/EXPERIMENT.md)
- [Developer guide](docs/DEV_GUIDE.md)
- [Dataset selection and export](dataset/README.md)

Install base dependencies with `python -m pip install -r requirements.txt`.
For more detailed facial AOIs, use `python analyse.py aois --detailed --test`,
then `python analyse.py review --detailed --test`. After reviewing the masks,
run `python analyse.py --detailed --test` for analysis. This version preserves
separate eyes, eyebrows, ears and lip components and adds approximate under-eye,
cheek, forehead and chin regions. See [detailed AOI definitions](docs/DETAILED_AOIS.md).

For manual landmark and feature-mask correction, run
`python analyse.py edit --detailed --test`. Choose any feature in the editor's
Edit selector to drag its contour handles, then save and run
`python analyse.py review --manual --test`. Use `python analyse.py --manual --test`
for gaze analysis with those corrections. See the [editor controls](docs/DETAILED_AOIS.md#manually-correct-individual-landmarks-and-features).

Facial AOI generation also needs
`python -m pip install -r requirements-aois.txt`. Follow the setup guides for
the local GazeFollower dependencies and webcam configuration.
