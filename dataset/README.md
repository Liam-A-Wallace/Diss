# Dataset curation for "Seeing the Fake" — Study 1

Tools to go from source CSVs + image folders to a balanced, curated stimulus
set that plugs straight into the PsychoPy experiment at the repo root (`../`).

## Folder structure

```
dataset/                 # this folder, inside the Diss project
  scripts/               # all curation/review/export tools (common.py lives here too)
  manifests/             # source data + generated manifests (FINAL_DATASET.csv, *_manifest.csv, *_male/female.csv)
  metadata/              # enriched per-image metadata (image_metadata.csv, *_metadata.csv)
  outputs/               # curation results (picked.csv, review_log.csv, curated.csv)

../Dataset/datasets/     # the raw image folders, kept OUTSIDE the project
                         # (Deepfake-vs-Real-60K, generated_faces, luca_stimuli, ...) + picked/ copies
```

The raw image corpora are **not** committed — they live in a sibling `Dataset/`
folder next to `Diss/` (override with the `RAW_DATA_ROOT` env var if you keep
them elsewhere). Image paths inside the manifests are **repo-root-relative**
(e.g. `Dataset/datasets/generated_faces/images/face (1).jpg`, relative to
`Diss/`'s parent), so the tree stays portable across machines.

## Scripts (`dataset/scripts/`)

| Script | Purpose |
| --- | --- |
| `common.py` | shared CSV loading, image download/cache, path resolution |
| `review.py` | the current **keep/axe** pick tool (per-label target, `--append`, `--labels`, `--skip-avatars`, gender hints) |
| `export_to_diss.py` | copies `outputs/picked.csv` images into `Diss/stimuli/` and writes `Diss/trials.csv` |
| `build_60k_manifest.py` | builds `manifests/DFR60K_manifest.csv` from the 60K Real/Fake folders |
| `extract_final_fakes.py` | extracts gender-labelled fakes from `manifests/FINAL_DATASET.csv` |
| `import_generated_faces.py` | builds a FAKE manifest from a folder of generated faces |
| `curate.py` | the original 6-cell (Easy/Medium/Hard) review tool |
| `audit.py` | prints a dataset audit (distributions, confounds) |
| `smoke_test.py` | headless end-to-end check of the curation pipeline |

## Quick start (current workflow)

Use the Diss venv:

```powershell
cd c:\Users\LiamW\Documents\uni\y4\Diss
.venv\Scripts\activate

# 1. review + pick images (defaults to manifests/DFR60K_manifest.csv)
python dataset\scripts\review.py --per-label 25

# 2. add more picks to the existing set (e.g. only fakes, appending)
python dataset\scripts\review.py --csv dataset\manifests\FINAL_DATASET.csv --labels fake --per-label 19 --append dataset\outputs\picked.csv --skip-avatars

# 3. export the picked set into the experiment
python dataset\scripts\export_to_diss.py
```

`review.py` writes `outputs/picked.csv` (with a `picked_file` column) and copies
the images into `datasets/picked/{real,fake}/` (under the external `Dataset/`
folder). `export_to_diss.py` then moves those into `stimuli/` and regenerates
`trials.csv`.

## `curate.py` (legacy 6-cell review)

The original design assigned Easy/Medium/Hard per cell (5 per cell x 6 cells =
30 images). See the review window's keep/axe/undo keys and the difficulty
re-derivation note below. Outputs go to `outputs/curated.csv`,
`outputs/review_log.csv`, and `stimuli/`.

## Face pre-filter (optional, in `curate.py`)

Pass `--face-filter` to auto-axe candidates that aren't a single, prominent
face (no face, multiple faces, or face too small). It uses OpenCV's Haar
cascade (bundled with most wheels; downloaded once on first use otherwise).

## The difficulty re-derivation (important, `curate.py` only)

`detection_difficulty` is confounded with label in the source data
(REAL = all "Easy", FAKE = only "Medium"/"Hard"), so it cannot populate the
6-cell design. There is no predicted-class column, so we **assume**
`confidence_score` is the detector's confidence in the *correct* label and
treat it as **model-derived** difficulty:

- REAL: high confidence -> Easy, low -> Hard (looks least "real" = hardest).
- FAKE: high confidence -> Easy, low -> Hard (confidently fake = easiest).

This is model-derived difficulty, **not** human difficulty.

Images are cached under `dataset/.image_cache/` so re-runs don't re-download.
