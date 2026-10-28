# Stimulus Curation (enrichment)

Offline image enrichment for the "Seeing the Fake" study. `enrich.py` reads
one or more source CSVs, downloads/caches each image, computes per-image
metadata, and writes `image_metadata.csv`:

```
image_id, is_photo, photo_score, num_faces, max_face_fraction,
face_center_off, face_ok, ethnicity, gender, age_group, status, error
```

## Files

| File | Purpose |
| --- | --- |
| `enrich.py` | the enrichment CLI |
| `build_manifest.py` | builds `DFR60K_manifest.csv` from the sibling `Dataset/` 60K `{Real,Fake}` folders |
| `common.py` | shared CSV loading + image download/cache helpers |
| `curated.csv` | the 30-image curated stimulus sample (output of the review tool) |

## Prerequisites (Linux)

- Python 3.10 or 3.11 (recommended)
- `pip` and `venv`
- ~4 GB free RAM to run every task at once; run one task at a time on smaller
  machines

## Setup

```bash
cd Diss/stimulus-curation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Base imports (always required):
pip install pillow requests

# Per-task backends (install only what you need):
pip install torch open_clip_torch          # photo  task (CLIP)
pip install mediapipe                      # face   task (MediaPipe)
pip install transformers huggingface_hub    # race / gender / age tasks
```

To run everything, install all of the above in one go:

```bash
pip install pillow requests torch open_clip_torch mediapipe transformers huggingface_hub
```

## Running

### From the local dataset folders (recommended)

The raw image folders (`Deepfake-vs-Real-60K`, etc.) live **outside** this
project, in the sibling `Dataset/datasets/` folder (resolved automatically via
the `RAW_DATA_ROOT` env var — default `../../Dataset`). `enrich.py` and
`build_manifest.py` find them automatically:

```bash
python build_manifest.py            # all images in the folders
python build_manifest.py --sample 300

python enrich.py --sample 50                # every task (needs ~4 GB RAM)
python enrich.py --tasks face --sample 50   # one task at a time
```

The manifest's `image_url` column is repo-relative, so both `build_manifest.py`
and `enrich.py` work on any machine/checkout location (see the note on path
resolution at the bottom).

### From the shared manifests

`enrich.py` reads its source CSVs from the dataset toolchain's
`../dataset/manifests/` by default (`FINAL_DATASET.csv` and
`DFR60K_manifest.csv`), and writes `image_metadata.csv` next to itself. Point
the flags elsewhere if you keep them somewhere different:

```bash
# quick smoke test on 10 images
python enrich.py \
  --csv ../dataset/manifests/FINAL_DATASET.csv \
  --extra-csv ../dataset/manifests/DFR60K_manifest.csv \
  --sample 10

# low-memory: one task per invocation (recommended < 16 GB RAM)
python enrich.py --tasks face   --sample 50
python enrich.py --tasks photo  --skip-existing
python enrich.py --tasks gender --skip-existing
python enrich.py --tasks age    --skip-existing
python enrich.py --tasks race   --skip-existing

# everything at once (needs ~4 GB free RAM)
python enrich.py --sample 50
```

Useful flags:

- `--csv PATH` — primary source CSV (default `FINAL_DATASET.csv` next to the script)
- `--extra-csv PATH` — second source CSV (default `DFR60K_manifest.csv`)
- `--out PATH` — output CSV (default `image_metadata.csv`)
- `--tasks face,photo,race,gender,age` — comma-separated subset
- `--sample N` / `--limit N` — process at most N images
- `--start N` — skip the first N rows (resume helper)
- `--skip-existing` — only process rows whose requested columns are empty
- `--force` — recompute even if already present

## Gated FairFace model (race task)

The `race` task uses `HuggingFaceM4/FairFace`, which is gated on Hugging Face.
Before running it:

1. Open <https://huggingface.co/HuggingFaceM4/FairFace> and accept the terms.
2. Log in once:

   ```bash
   pip install huggingface_hub
   huggingface-cli login      # or: export HF_TOKEN=hf_...
   ```

The other tasks (`photo`, `face`, `gender`, `age`) need no token.

## Notes

- Images are cached under `.image_cache/` so re-runs do not re-download.
- `image_metadata.csv`, `.image_cache/`, and `deepfake-vs-real-60k/` are
  git-ignored.
- The raw `Deepfake-vs-Real-60K` corpus (~23 GB) lives **outside** this project,
  in the sibling `Dataset/datasets/` folder, and is intentionally kept out of
  version control.
- `curated.csv` is a 30-row manifest, not an input to `enrich.py`. Its
  `image_url` column stores **repo-root-relative** paths (e.g.
  `Dataset/Deepfake-vs-Real-60K/...`), and `common.py` resolves local image
  paths relative to the script, the `RAW_DATA_ROOT` folder, or — as a last
  resort — by filename under the local 60K folders. So the same CSV works on
  any machine and any checkout location, including on Linux.
