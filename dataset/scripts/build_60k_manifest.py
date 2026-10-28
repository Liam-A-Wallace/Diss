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


def find_dataset_root() -> Path | None:
    """Find the folder that directly contains ``Real/`` and ``Fake/``.

    Checks the usual ``Deepfake-vs-Real-60K/deepfake-vs-real-60k`` layout first,
    then falls back to a bounded search under the Dataset folder and repo root,
    so the script works wherever the image folders end up.
    """
    for base in (HERE.parent, RAW_DATA_ROOT):
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


def main(argv: list[str] | None = None) -> int:
    root = find_dataset_root()
    if root is None:
        raise SystemExit(
            "[error] could not find a folder containing Real/ and Fake/ under "
            f"{HERE.parent} or {RAW_DATA_ROOT}; pass --real-dir and --fake-dir explicitly."
        )
    real_default = str(root / "Real")
    fake_default = str(root / "Fake")

    parser = argparse.ArgumentParser(
        description="Build DFR60K_manifest.csv from the local 60K folders.")
    parser.add_argument("--sample", type=int, default=300,
                        help="max rows to sample per label (default 300)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--real-dir", default=real_default)
    parser.add_argument("--fake-dir", default=fake_default)
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args(argv)

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

    rows = ([make_row(p, "REAL") for p in real_paths[: args.sample]]
            + [make_row(p, "FAKE") for p in fake_paths[: args.sample]])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n_real = min(args.sample, len(real_paths))
    n_fake = min(args.sample, len(fake_paths))
    print(f"[saved] {out}")
    print(f"        real={n_real}  fake={n_fake}  total={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
