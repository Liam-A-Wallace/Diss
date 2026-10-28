"""Align camera timestamps to the display state then undo image transforms."""
import bisect
import csv
import json
import math
import warnings
from pathlib import Path
from image_view import map_gaze


def resolve_stimulus(stimulus, metadata):
    """Resolve recorded relative paths independently of the analysis working folder."""
    path = Path(stimulus)
    if path.is_absolute():
        return path
    recorded_root = metadata.get("project_root")
    if recorded_root:
        return Path(recorded_root) / path
    # Compatibility with early sessions recorded without project_root.
    project = Path(__file__).resolve().parents[1]
    return project / path


def load_display(path):
    events = []
    lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith("\n"):
                warnings.warn("Ignoring an incomplete final display record after a crash", RuntimeWarning)
                continue
            raise ValueError(f"Invalid display record on line {index + 1}") from None
    if any(a["timestamp_ns"] >= b["timestamp_ns"] for a, b in zip(events, events[1:])):
        raise ValueError("Display timestamps must strictly increase; check clock discontinuities")
    return events


def align_samples(raw_path, events, coordinate_source="calibrated"):
    if coordinate_source not in ("calibrated", "filtered"):
        raise ValueError("Use calibrated or filtered gaze")
    times = [e["timestamp_ns"] for e in events]
    onsets = {}
    closed = {}
    for e in events:
        if e["kind"] == "present":
            onsets.setdefault(e["trigger"], e["timestamp_ns"])
        elif e["kind"] == "offset":
            closed[e["trigger"]] = e
    prefix = coordinate_source + "_gaze_position_"
    with open(raw_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "status", prefix + "x", prefix + "y"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Raw gaze columns missing: " + str(required - set(reader.fieldnames or [])))
        previous_ts = None
        for raw_index, row in enumerate(reader):
            ts = int(row["timestamp"])
            if previous_ts is not None and ts < previous_ts:
                raise ValueError("Gaze timestamps went backwards; check recording clock")
            duplicate = ts == previous_ts
            previous_ts = ts
            index = bisect.bisect_right(times, ts) - 1
            if index < 0 or events[index]["kind"] != "present":
                continue
            frame = events[index]
            trigger = frame["trigger"]
            if trigger not in closed:  # unknown offset after a hard crash: do not invent duration
                continue
            try:
                x, y = float(row[prefix + "x"]), float(row[prefix + "y"])
                ix, iy, u, v, inside = map_gaze(x, y, frame)
            except (ValueError, TypeError):
                ix = iy = u = v = float("nan")
                inside = False
            finite = math.isfinite(u) and math.isfinite(v)
            valid = str(row.get("status", "")).strip().lower() in ("1", "true") and finite and not duplicate
            yield dict(row, raw_index=raw_index, raw_trigger=row.get("trigger", ""), trigger=trigger, trial=frame["trial"],
                       stimulus=frame["stimulus"], is_practice=frame["is_practice"],
                       condition=frame["condition"], frame_index=index,
                       interrupted=int(closed[trigger].get("interrupted", False)),
                       t_rel_ms=(ts - onsets[trigger]) / 1e6,
                       image_x=ix if finite else "", image_y=iy if finite else "",
                       image_u=u if finite else "", image_v=v if finite else "",
                       zoom=frame["zoom"], coordinate_source=coordinate_source,
                       duplicate=int(duplicate), valid=int(valid), on_image=int(inside),
                       usable=int(valid and inside))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
