"""Extract gender-diverse FAKE images from FINAL_DATASET.csv.

The 60K dataset's fakes are all women, but FINAL_DATASET.csv's FAKE rows have a
small gender-labelled subset (randomuser.me ``men|women`` portraits). This
script pulls those out, downloads them locally, and writes a manifest with
portable local paths so they can be reviewed (or merged into ``picked.csv``).

Usage (from anywhere, using the Diss venv):

    python Dataset/extract_final_fakes.py                     # male + female
    python Dataset/extract_final_fakes.py --gender male       # male only
    python Dataset/extract_final_fakes.py --gender female
    python Dataset/extract_final_fakes.py --all               # include Unknown-gender avatars

Output:
    Dataset/final_fake_gendered.csv
    Dataset/final_fake_gendered/male/male_001.jpg ...
    Dataset/final_fake_gendered/female/female_001.jpg ...
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE.parent / "manifests" / "FINAL_DATASET.csv"
OUT_CSV = HERE.parent / "manifests" / "final_fake_gendered.csv"
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT") or HERE.parent.parent.parent / "Dataset")
REPO = RAW_DATA_ROOT.parent                        # root image_url paths are relative to
OUT_DIR = RAW_DATA_ROOT / "datasets" / "final_fake_gendered"

_HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset curation script)"}

# Keep the same column order as FINAL_DATASET.csv plus ``source_dataset``.
FIELDS = [
    "image_id", "image_url", "label", "label_numeric", "category", "gender",
    "age_group", "source", "fake_method", "image_quality", "resolution",
    "confidence_score", "detection_difficulty", "dataset_split",
    "date_collected", "version", "year", "source_dataset",
]


def url_gender(url: str) -> str | None:
    if "/men/" in url:
        return "Male"
    if "/women/" in url:
        return "Female"
    return None


def _download(url: str, dest: Path, timeout: float = 20.0) -> bool:
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=timeout, headers=_HEADERS)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return dest.stat().st_size > 0
        except Exception:
            time.sleep(0.3)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract gender-labelled FAKE images from FINAL_DATASET.csv.")
    parser.add_argument("--csv", default=str(SOURCE_CSV))
    parser.add_argument("--gender", choices=["male", "female"],
                        help="only this gender")
    parser.add_argument("--all", action="store_true",
                        help="include Unknown-gender avatars too")
    parser.add_argument("--out", default=str(OUT_CSV))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--no-download", action="store_true",
                        help="write the CSV but skip downloading images")
    args = parser.parse_args(argv)

    rows = [r for r in csv.DictReader(open(args.csv, encoding="utf-8-sig"))
            if r["label"] == "FAKE"]

    out: list[dict] = []
    for r in rows:
        g = url_gender(r["image_url"]) or r.get("gender", "")
        if g not in ("Male", "Female"):
            if not args.all:
                continue
            g = r.get("gender", "") or "Unknown"
        if args.gender and g.lower() != args.gender:
            continue
        row = {k: r.get(k, "") for k in FIELDS}
        row["gender"] = g
        row["source_dataset"] = "final"
        out.append(row)

    n_male = sum(1 for r in out if r["gender"] == "Male")
    n_female = sum(1 for r in out if r["gender"] == "Female")
    n_unknown = sum(1 for r in out if r["gender"] not in ("Male", "Female"))
    print(f"[found] male={n_male}  female={n_female}  unknown={n_unknown}")

    out_dir = Path(args.out_dir)
    if not args.no_download and out:
        counters: dict[str, int] = {}
        for r in out:
            g = "male" if r["gender"] == "Male" else "female"
            counters[g] = counters.get(g, 0) + 1
            dest_dir = out_dir / g
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{g}_{counters[g]:03d}.jpg"
            if _download(r["image_url"], dest):
                r["image_url"] = dest.relative_to(REPO).as_posix()
            else:
                print(f"  [warn] download failed: {r['image_id']} {r['image_url']}")

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)

    n_local = sum(1 for r in out if not r["image_url"].startswith(("http://", "https://")))
    print(f"[saved] {len(out)} rows -> {out_path} ({n_local} local images)")
    if not args.no_download:
        print(f"[downloaded] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
