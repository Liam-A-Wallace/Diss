"""Build a FINAL_DATASET-compatible manifest from the local 60K image folders.

Deepfake-vs-Real-60K ships as two folders of images (no CSV):

    Deepfake-vs-Real-60K/deepfake-vs-real-60k/Real/*.png  -> label REAL
    Deepfake-vs-Real-60K/deepfake-vs-real-60k/Fake/*.jpg  -> label FAKE

This script globs those folders and writes ``Dataset/DFR60K_manifest.csv`` so
the images can flow through the same curation pipeline as ``FINAL_DATASET.csv``.

Usage (from anywhere, using the Diss venv):

    python Dataset/build_60k_manifest.py --sample 300
    python Dataset/build_60k_manifest.py --sample 300 --seed 1
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE.parent / "manifests" / "DFR60K_manifest.csv"
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT") or HERE.parent.parent.parent / "Dataset")
REPO = RAW_DATA_ROOT.parent                        # root image_url paths are relative to

SOURCE = "Deepfake-vs-Real-60K"
SOURCE_DATASET = "dfr60k"
VERSION = "v1.0"

# Same column order as FINAL_DATASET.csv, plus ``source_dataset`` at the end.
FIELDS = [
    "image_id", "image_url", "label", "label_numeric", "category", "gender",
    "age_group", "source", "fake_method", "image_quality", "resolution",
    "confidence_score", "detection_difficulty", "dataset_split",
    "date_collected", "version", "year", "source_dataset",
]


def _has_real_fake(d: Path) -> bool:
    """True if ``d`` directly contains ``Real/`` and ``Fake/`` (case-sensitive).

    Name strings are compared exactly so lowercase ``stimuli/real|fake`` style
    folders are not mistaken for the dataset on case-insensitive filesystems.
    """
    try:
        names = {p.name for p in d.iterdir() if p.is_dir()}
    except OSError:
        return False
    return "Real" in names and "Fake" in names


def _dirs_under(base: Path, depth: int) -> list[Path]:
    """All directories up to ``depth`` levels below ``base`` (base included)."""
    out = [base]
    if depth <= 0:
        return out
    try:
        children = sorted(p for p in base.iterdir() if p.is_dir())
    except OSError:
        return out
    for child in children:
        out.extend(_dirs_under(child, depth - 1))
    return out


def find_dataset_root(preferred_roots=()) -> Path | None:
    """Find the folder that directly contains ``Real/`` and ``Fake/``.

    Search caller-provided locations first, then the dataset directory and
    RAW_DATA_ROOT. Each search is bounded to three directory levels.
    """
    for base in (*preferred_roots, HERE.parent, RAW_DATA_ROOT):
        for cand in _dirs_under(base, depth=3):
            if _has_real_fake(cand):
                return cand
    return None


def _portable_image_url(path: Path) -> str:
    """Image path repo-root-relative (portable across machines)."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path.resolve())


def make_row(path: Path, label: str) -> dict:
    cond = "real" if label == "REAL" else "fake"
    stem = path.stem.strip().replace(" ", "_")
    image_id = f"dfr60k_{cond}_{stem}"
    today = datetime.date.today()
    return {
        "image_id": image_id,
        "image_url": _portable_image_url(path),
        "label": label,
        "label_numeric": "1" if label == "REAL" else "0",
        "category": "",
        "gender": "",
        "age_group": "",
        "source": SOURCE,
        "fake_method": "",
        "image_quality": "",
        "resolution": "",
        "confidence_score": "",
        "detection_difficulty": "",
        "dataset_split": "",
        "date_collected": today.isoformat(),
        "version": VERSION,
        "year": str(today.year),
        "source_dataset": SOURCE_DATASET,
    }


def main(argv: list[str] | None = None, *, default_out=OUT_CSV, default_sample=300,
         preferred_roots=()) -> int:
    root = find_dataset_root(preferred_roots)
    real_default = str(root / "Real") if root else ""
    fake_default = str(root / "Fake") if root else ""

    parser = argparse.ArgumentParser(
        description="Build DFR60K_manifest.csv from the local 60K folders.")
    parser.add_argument("--sample", type=int, default=default_sample,
                        help="max rows per label; 0 = all (default %(default)s)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--real-dir", default=real_default)
    parser.add_argument("--fake-dir", default=fake_default)
    parser.add_argument("--out", default=str(default_out))
    args = parser.parse_args(argv)
    if args.sample < 0:
        parser.error("--sample must be nonnegative")
    if not args.real_dir or not args.fake_dir:
        parser.error("Dataset not found; pass --real-dir and --fake-dir")

    real_dir = Path(args.real_dir)
    fake_dir = Path(args.fake_dir)
    if not real_dir.is_dir():
        raise SystemExit(f"[error] real dir not found: {real_dir}")
    if not fake_dir.is_dir():
        raise SystemExit(f"[error] fake dir not found: {fake_dir}")

    real_paths = sorted(real_dir.glob("*.png"))
    fake_paths = sorted(fake_dir.glob("*.jpg"))

    rng = random.Random(args.seed)
    rng.shuffle(real_paths)
    rng.shuffle(fake_paths)

    if args.sample:
        real_paths, fake_paths = real_paths[:args.sample], fake_paths[:args.sample]
    rows = ([make_row(p, "REAL") for p in real_paths]
            + [make_row(p, "FAKE") for p in fake_paths])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_real, n_fake = len(real_paths), len(fake_paths)
    print(f"[saved] {out}")
    print(f"        real={n_real}  fake={n_fake}  total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
