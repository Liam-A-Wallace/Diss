"""Dataset audit for "Seeing the Fake" — Study 1 stimulus curation.

Answers the pre-selection questions:

  1. What does ``confidence_score`` represent?
  2. Is it confidence in the predicted class or the true class?
  3. What are the confidence distributions for REAL and FAKE?
  4. How many images are available in each class / cell?
  5. What happens to the existing ``detection_difficulty`` labels?

Usage (from anywhere):

    python Dataset/audit.py                # use Dataset/FINAL_DATASET.csv
    python Dataset/audit.py path/to.csv    # or point at another CSV
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def _fmt_row(cols: list[str]) -> str:
    return " | ".join(cols)


def _hist(values: list[float], step: float = 0.05) -> str:
    buckets: dict[float, int] = {}
    for v in values:
        b = round(v / step) * step
        buckets[b] = buckets.get(b, 0) + 1
    return ", ".join(f"{b:.2f}:{n}" for b, n in sorted(buckets.items()))


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else common.DATASET_CSV
    rows = common.load_rows(csv_path)
    print(f"===== DATASET AUDIT: {Path(csv_path).name} =====")
    print(f"rows: {len(rows)}")
    print(f"columns: {list(rows[0].keys())}")
    print()

    # ---- 4. counts by class and by existing cell --------------------------
    print("---- counts by label ----")
    for k, v in sorted(Counter(r["label"] for r in rows).items()):
        print(f"  {k:6s} {v}")
    print()
    print("---- existing detection_difficulty x label ----")
    cells = Counter((r["label"], r["detection_difficulty"]) for r in rows)
    for lab in common.LABELS:
        line = "  " + lab.ljust(6)
        for d in common.DIFFICULTIES:
            line += f"  {d}:{cells.get((lab, d), 0):<7}"
        print(line)
    print()

    # ---- 3. confidence distributions -------------------------------------
    print("---- confidence_score by label ----")
    for lab in common.LABELS:
        vals = sorted(float(r["confidence_score"]) for r in rows if r["label"] == lab)
        mean = sum(vals) / len(vals)
        print(f"  {lab:6s} n={len(vals)}  min={min(vals):.2f}  mean={mean:.3f}  max={max(vals):.2f}")
        print(f"         buckets: {_hist(vals)}")
    print()

    # ---- 5. does confidence separate the existing difficulty labels? ------
    print("---- confidence vs existing difficulty (FAKE) ----")
    for d in ("Medium", "Hard"):
        vals = [float(r["confidence_score"]) for r in rows
                if r["label"] == "FAKE" and r["detection_difficulty"] == d]
        if vals:
            mean = sum(vals) / len(vals)
            print(f"  FAKE {d:7s} n={len(vals):<5} min={min(vals):.2f} mean={mean:.3f} max={max(vals):.2f}")
    print("  => existing Medium/Hard FAKE labels are NOT separable by confidence_score.")
    print()

    # ---- 1./2. what confidence_score means --------------------------------
    print("---- what does confidence_score represent? ----")
    has_predicted = any("predict" in c.lower() or "prob" in c.lower() for c in rows[0])
    print("  * No predicted-class column present in the CSV "
          f"({'yes' if has_predicted else 'no'} 'predicted'/'prob' column found).")
    print("  * We cannot distinguish 'confidence in the true label' from")
    print("    'confidence in the predicted label' from the data alone.")
    print("  * ASSUMPTION for curation: confidence_score = detector confidence")
    print("    in the CORRECT REAL/FAKE classification (high = easy, low = hard).")
    print("    This is a model-derived difficulty, not human difficulty.")
    print()

    # ---- confounds --------------------------------------------------------
    print("---- other confounds with label ----")
    print("  source  :", dict(Counter((r["label"], r["source"]) for r in rows)))
    print("  resolution:", dict(Counter((r["label"], r["resolution"]) for r in rows)))
    print("  image_quality:", dict(Counter((r["label"], r["image_quality"]) for r in rows)))
    print()

    # ---- re-derived difficulty preview ------------------------------------
    print("---- re-derived difficulty (confidence tertiles within label) ----")
    difficulty_of = common.derive_difficulty(rows, high_conf_is_easy=True)
    derived = Counter(common.cell_key(r, difficulty_of) for r in rows)
    for lab in common.LABELS:
        line = "  " + lab.ljust(6)
        for d in common.DIFFICULTIES:
            line += f"  {d}:{derived.get((lab, d), 0):<7}"
        print(line)
    print()

    # confidence range of each derived tertile (so you can sanity-check it)
    print("---- confidence ranges of each derived tertile ----")
    for lab in common.LABELS:
        for d in common.DIFFICULTIES:
            vals = sorted(float(r["confidence_score"]) for r in rows
                          if common.cell_key(r, difficulty_of) == (lab, d))
            if vals:
                print(f"  {lab} {d:7s} n={len(vals):<5} conf {min(vals):.2f}..{max(vals):.2f}")
    print()

    # ---- duplicates -------------------------------------------------------
    print("---- duplicates ----")
    ids = Counter(r["image_id"] for r in rows)
    urls = Counter(r["image_url"] for r in rows)
    print(f"  duplicate image_ids: {sum(1 for v in ids.values() if v > 1)}")
    print(f"  duplicate image_urls: {sum(1 for v in urls.values() if v > 1)}")
    print()

    # ---- metadata coverage ------------------------------------------------
    print("---- metadata coverage ----")
    print("  gender   :", dict(Counter(r["gender"] for r in rows)))
    print("  age_group:", dict(Counter(r["age_group"] for r in rows)))
    print("  dataset_split:", dict(Counter(r["dataset_split"] for r in rows)))
    print()


if __name__ == "__main__":
    main()
