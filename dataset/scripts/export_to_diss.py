"""Copy the picked stimulus set into the PsychoPy experiment (Diss/) and write
its ``trials.csv``.

Reads ``Dataset/picked.csv``, copies each picked image into
``Diss/stimuli/{real,fake}/``, and writes ``Diss/trials.csv`` in the schema the
experiment expects (``stimulus, condition, difficulty``).

Usage (from anywhere, using the Diss venv):

    python Dataset/export_to_diss.py
    python Dataset/export_to_diss.py --picked Dataset/picked.csv

Note: ``difficulty`` is written empty because the pick workflow doesn't assign
an Easy/Medium/Hard tier (that was the old 6-cell design). Populate it here or
in trials.csv afterwards if your design still needs it.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISS_DIR = HERE.parent.parent                        # Diss/
PICKED_CSV = HERE.parent / "outputs" / "picked.csv"
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT") or HERE.parent.parent.parent / "Dataset")
STIM_DIR = DISS_DIR / "stimuli"
TRIALS_CSV = DISS_DIR / "trials.csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export picked stimuli to the Diss experiment.")
    parser.add_argument("--picked", default=str(PICKED_CSV))
    parser.add_argument("--diss", default=str(DISS_DIR))
    args = parser.parse_args(argv)

    picked_csv = Path(args.picked)
    diss_dir = Path(args.diss)
    if not picked_csv.exists():
        raise SystemExit(f"[error] picked CSV not found: {picked_csv}")

    with picked_csv.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise SystemExit("[error] picked CSV is empty")

    # Validate the entire selection before touching an existing experiment set.
    plan, destinations, errors = [], set(), []
    for number, row in enumerate(rows, 2):
        label = row.get("label", "").strip().upper()
        fname = Path(row.get("picked_file", "")).name
        if label not in ("REAL", "FAKE") or not fname:
            errors.append(f"Row {number}: invalid label or missing picked_file")
            continue
        cond = label.lower()
        src = RAW_DATA_ROOT / "datasets" / "picked" / cond / fname
        key = (cond, fname.casefold())
        if key in destinations:
            errors.append(f"Row {number}: duplicate destination {cond}/{fname}")
        if not src.is_file():
            errors.append(f"Row {number}: missing source {src}")
        destinations.add(key)
        plan.append((src, cond, fname))
    if errors:
        raise SystemExit("[error] Export cancelled; existing files preserved:\n" + "\n".join(errors))
    for cond in ("real", "fake"):
        if (diss_dir / "stimuli" / cond).is_symlink():
            raise SystemExit("[error] Refusing to replace a symlinked stimulus directory")

    trial_rows = [dict(stimulus=f"stimuli/{cond}/{fname}", condition=cond, difficulty="")
                  for _, cond, fname in plan]
    diss_dir.mkdir(parents=True, exist_ok=True)
    # Stage every copy first; unreadable sources/copy errors preserve the old set.
    with tempfile.TemporaryDirectory(prefix=".stimulus-export-", dir=diss_dir) as temporary:
        stage = Path(temporary)
        for src, cond, fname in plan:
            (stage / cond).mkdir(exist_ok=True)
            shutil.copy2(src, stage / cond / fname)
        staged_csv = stage / "trials.csv"
        with staged_csv.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=["stimulus", "condition", "difficulty"])
            writer.writeheader()
            writer.writerows(trial_rows)
        for cond in ("real", "fake"):
            dest = diss_dir / "stimuli" / cond
            dest.mkdir(parents=True, exist_ok=True)
            for existing in dest.iterdir():
                if existing.is_file():
                    existing.unlink()
        for _, cond, fname in plan:
            shutil.copy2(stage / cond / fname, diss_dir / "stimuli" / cond / fname)
        trials_path = diss_dir / "trials.csv"
        staged_csv.replace(trials_path)
    copied = len(plan)

    n_real = sum(1 for t in trial_rows if t["condition"] == "real")
    n_fake = sum(1 for t in trial_rows if t["condition"] == "fake")
    print(f"[copied] {copied} images -> {diss_dir / 'stimuli'}")
    print(f"[saved]  {trials_path} ({n_real} real + {n_fake} fake)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
