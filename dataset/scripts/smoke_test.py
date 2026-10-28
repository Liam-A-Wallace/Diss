"""Headless end-to-end smoke test for the curation pipeline.

Simulates a manual keep session (5 per cell = 30) WITHOUT opening the GUI,
writes curated.csv + stimuli + trials.csv into a temp folder, and verifies
they load (including via the experiment's own trial verifier).

Usage (from anywhere, using the Diss venv):

    python Dataset/smoke_test.py              # 60K rows only (fast, offline)
    python Dataset/smoke_test.py --full       # also mix in FINAL_DATASET (downloads)
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))          # Diss/

import common  # noqa: E402
import curate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the curation pipeline headlessly.")
    parser.add_argument("--full", action="store_true",
                        help="include FINAL_DATASET rows (requires network)")
    args = parser.parse_args()

    rows = common.load_rows(common.DATASET_CSV) if args.full else []
    manifest = HERE.parent / "manifests" / "DFR60K_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"[error] manifest missing: {manifest} — run build_60k_manifest.py first")
    rows = rows + common.load_rows(manifest)

    hint_of, label_pools = curate.build_pools(rows, candidates_per_label=150, mix_ratio=0.5)
    curate.print_pool_summary(label_pools)

    # Simulate manual keep: 5 per (label, difficulty) cell, Easy -> Medium -> Hard.
    kept = {c: [] for c in curate.CELL_ORDER}
    pool = {l: list(label_pools[l]) for l in common.LABELS}
    for label in common.LABELS:
        for difficulty in ("Easy", "Medium", "Hard"):
            for _ in range(curate.CELL_TARGETS[(label, difficulty)]):
                if not pool[label]:
                    raise SystemExit(f"[error] ran out of {label} candidates during smoke keep")
                row = pool[label].pop(0)
                row["derived_difficulty"] = difficulty
                kept[(label, difficulty)].append(row)

    final_rows = []
    for cell in curate.CELL_ORDER:
        for row in kept[cell][: curate.CELL_TARGETS[cell]]:
            out = dict(row)
            out["derived_difficulty"] = cell[1]
            final_rows.append(out)

    # Write outputs to a temp folder (don't touch the real curated.csv/stimuli).
    tmp = Path(tempfile.mkdtemp(prefix="curate_smoke_"))
    curate.OUT_CSV = tmp / "curated.csv"
    curate.STIM_REAL_DIR = tmp / "stimuli" / "real"
    curate.STIM_FAKE_DIR = tmp / "stimuli" / "fake"
    curate.TRIALS_CSV = tmp / "trials.csv"
    curate.REVIEW_LOG = tmp / "review_log.csv"

    curate.save_outputs(final_rows)
    curate.write_trials_csv(final_rows)

    assert len(final_rows) == 30, f"expected 30, got {len(final_rows)}"
    real = [r for r in final_rows if r["label"] == "REAL"]
    fake = [r for r in final_rows if r["label"] == "FAKE"]
    n_final = sum(1 for r in final_rows if r.get("source_dataset") != "dfr60k")
    n_dfr = sum(1 for r in final_rows if r.get("source_dataset") == "dfr60k")
    print(f"[smoke] final set: real={len(real)} fake={len(fake)} "
          f"final_src={n_final} dfr60k={n_dfr}")
    assert len(real) == 15 and len(fake) == 15, "expected 15 real + 15 fake"

    n_stim = len(list((tmp / "stimuli").rglob("*.png")))
    print(f"[smoke] stimuli files written: {n_stim}")
    assert n_stim == 30, f"expected 30 stimuli, got {n_stim}"

    # Verify trials.csv through the experiment's own loader (needs cwd = tmp
    # because trials.csv stores stimulus paths relative to the project root).
    import experiment  # noqa: E402
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp)
        trials = experiment.load_and_verify_trials("trials.csv")
        print(f"[smoke] experiment.load_and_verify_trials OK: {len(trials)} trials")
    finally:
        os.chdir(old_cwd)

    print(f"[smoke] PASS — outputs written to {tmp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
