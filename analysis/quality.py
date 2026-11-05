"""Descriptive recording quality; no automatic trial exclusion or interpolation."""
import math
import statistics


def recording_quality(rows, view_ms, sample_max_gap_ms=100):
    intervals = [b["t_rel_ms"] - a["t_rel_ms"] for a, b in zip(rows, rows[1:])
                 if b["t_rel_ms"] > a["t_rel_ms"]]
    valid = sum(bool(r["valid"]) for r in rows)
    unique = sum(not r["duplicate"] for r in rows)
    usable = sum(bool(r["usable"]) for r in rows)
    gaps = [dt for dt in intervals if dt > sample_max_gap_ms]
    ordered = sorted(intervals)
    notes = []
    if not rows:
        notes.append("No gaze samples recorded")
    elif not usable:
        notes.append("No usable on-image samples")
    if gaps:
        notes.append(f"{len(gaps)} sample intervals exceed {sample_max_gap_ms:g} ms")
    return dict(saved_sample_hz=unique * 1000 / view_ms if view_ms > 0 else 0,
                valid_sample_hz=valid * 1000 / view_ms if view_ms > 0 else 0,
                median_sample_interval_ms=statistics.median(intervals) if intervals else "",
                p95_sample_interval_ms=ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)] if ordered else "",
                max_sample_interval_ms=max(intervals) if intervals else "",
                sample_gap_count=len(gaps), sample_gap_total_ms=sum(gaps),
                duplicate_sample_count=len(rows) - unique,
                invalid_sample_count=len(rows) - valid,
                valid_off_image_count=sum(bool(r["valid"]) and not r["on_image"] for r in rows),
                quality_notes="; ".join(notes))
