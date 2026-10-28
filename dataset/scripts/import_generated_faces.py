"""Build a FAKE-only manifest from a folder of AI-generated face images.

Useful for importing a downloaded generated-face dataset (e.g. the
"This Person Does Not Exist" StyleGAN2 set) into the review pipeline.
Image paths are written repo-root-relative so they resolve from both
``Dataset/`` and ``stimulus-curation/``.

Usage (from anywhere, using the Diss venv):

    python Dataset/import_generated_faces.py \
        --dir Dataset/this-person-does-not-exist/images \
        --out Dataset/generated_faces_manifest.csv

Output: a manifest with one FAKE row per image (image_id, image_url, label, ...).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_CSV = HERE.parent / "manifests" / "generated_faces_manifest.csv"
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT") or HERE.parent.parent.parent / "Dataset")
REPO = RAW_DATA_ROOT.parent                        # root image_url paths are relative to

FIELDS = [
    "image_id", "image_url", "label", "label_numeric", "category", "gender",
    "age_group", "source", "fake_method", "image_quality", "resolution",
    "confidence_score", "detection_difficulty", "dataset_split",
    "date_collected", "version", "year", "source_dataset",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a FAKE manifest from a folder of generated faces.")
    parser.add_argument("--dir", required=True,
                        help="folder containing the generated images")
    parser.add_argument("--out", default=str(OUT_CSV))
    parser.add_argument("--source", default="ThisPersonDoesNotExist",
                        help="source label for the manifest")
    parser.add_argument("--prefix", default="tpdn",
                        help="image_id prefix")
    args = parser.parse_args(argv)

    img_dir = Path(args.dir)
    if not img_dir.is_dir():
        raise SystemExit(f"[error] image dir not found: {img_dir}")

    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise SystemExit(f"[error] no images found in {img_dir}")

    today = datetime.date.today()
    rows = []
    for p in paths:
        stem = p.stem.strip().replace(" ", "_")
        try:
            url = p.relative_to(REPO).as_posix()
        except ValueError:
            url = p.as_posix()
        rows.append({
            "image_id": f"{args.prefix}_{stem}",
            "image_url": url,
            "label": "FAKE",
            "label_numeric": "0",
            "category": "AI Generated",
            "gender": "",
            "age_group": "",
            "source": args.source,
            "fake_method": "",
            "image_quality": "",
            "resolution": "",
            "confidence_score": "",
            "detection_difficulty": "",
            "dataset_split": "",
            "date_collected": today.isoformat(),
            "version": "v1.0",
            "year": str(today.year),
            "source_dataset": args.prefix,
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
