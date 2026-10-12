"""Trial-list loading and validation."""
import csv
import os

from errors import StimulusError

REQUIRED_COLUMNS = {"stimulus", "condition", "difficulty"}


def load_and_verify_trials(csv_path):
    if not os.path.exists(csv_path):
        raise StimulusError(f"Trial file not found: '{csv_path}'")

    trials = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        missing_cols = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_cols:
            raise StimulusError(
                f"CSV missing required columns: {missing_cols}. "
                f"Found headers: {reader.fieldnames}"
            )

        missing_files = []
        # header is row 1, so data starts at 2
        for row_idx, row in enumerate(reader, start=2):
            norm_path = os.path.normpath(row["stimulus"].strip())
            row["stimulus"] = norm_path
            if not os.path.isfile(norm_path):
                missing_files.append((row_idx, norm_path))
            trials.append(row)

    if not trials:
        raise StimulusError(f"Trial file '{csv_path}' contains no trial data.")

    if missing_files:
        details = "\n".join(
            f"  Row {row}: {path}" for row, path in missing_files[:5])
        if len(missing_files) > 5:
            details += f"\n  ... and {len(missing_files) - 5} more."
        raise StimulusError(
            f"Found {len(missing_files)} missing stimulus image(s):\n{details}"
        )

    return trials
