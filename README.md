# Seeing the Fake - Study 1

A PsychoPy experiment that records webcam gaze while participants classify face
images as real or generated and rate their confidence.

## Everyday commands

Run these from `Diss`, with your Python environment active:

| Task | Command |
| --- | --- |
| Five-photo pilot with actual curated images | `python demo.py` |
| Full experiment using `trials.csv` | `python experiment.py` |
| Analyse the latest five-photo/test recording | `python analyse.py --test` |
| Generate facial AOI masks for that recording | `python analyse.py aois --test` |
| Inspect and approve those masks | `python analyse.py review --test` |
| Analyse the latest participant recording | `python analyse.py` |
| Preview landmarks on the five demo images | `python -m tools.demo_landmarks` |

After approving masks, rerun the analysis to include facial-region dwell and
transitions. The five-photo demo saves to `data/test`. `--demo` on the full
experiment means a live gaze dot; use `demo.py` for the shortened experiment.

The normal `trials.csv` still points to the original stimulus paths. Check its
images before using the full experiment: the five-photo demo prepares its own
curated photos and does not replace that list.

## Where things live

| Location | Contents |
| --- | --- |
| `experiment.py`, `demo.py`, `analyse.py` | Main commands |
| `config.py` | Webcam, exposure, display, timing and experiment settings |
| Other root Python modules | Experiment UI, tracking, display mapping and recording |
| `analysis/` | Gaze processing, fixations and facial AOIs |
| `tools/` | Occasional landmark preview and placeholder generation |
| `docs/` | Demo, experiment and developer guides; cleanup review |
| `dataset/` | Curation tools, source manifests, metadata and selection outputs |
| `stimulus-curation/` | Optional image enrichment and an older manifest builder |
| `tests/` | Regression checks |
| `stimuli/` | Images shown by the experiment |
| `data/` | Source recordings and session metadata; retain for analysis |
| `aois/` | Generated masks and their review records |
| `processed/` | Web reports and plots; each report has a `data_results/` folder containing the analysis CSVs |
| `models/` | Downloaded face-parsing model |
| `.venv/` | Installed Python environment |

Raw image corpora and GazeFollower live in the sibling `../Dataset/` and
`../GazeFollower/` folders.

## Guides

- [Five-photo demo and landmark preview](docs/DEMO.md)
- [Gaze analysis and AOI definitions](analysis/README.md)
- [Full experiment setup, controls and recording files](docs/EXPERIMENT.md)
- [Developer guide](docs/DEV_GUIDE.md)
- [Dataset selection and export](dataset/README.md)
- [Files to review for possible removal](docs/CLEANUP_REVIEW.md)

Install base dependencies with `python -m pip install -r requirements.txt`.
Automatic facial AOIs additionally need
`python -m pip install -r requirements-aois.txt`. Follow the setup guides for
the local GazeFollower dependencies and webcam configuration.
