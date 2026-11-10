"""Exploratory I-DT fixations in normalised original-image coordinates.

Thresholds must be checked against pilot accuracy. Transform changes, invalid
samples and long gaps split segments; no interpolation hides missing tracking.
"""

import statistics


def detect_fixations(samples, dispersion=0.08, min_duration_ms=100, max_gap_ms=100):
    if dispersion <= 0 or min_duration_ms <= 0 or max_gap_ms <= 0:
        raise ValueError("Fixation parameters must be positive")
    segments, current = [], []
    for sample in samples:
        broken = not sample["usable"] or (
            current
            and (
                sample["frame_index"] != current[-1]["frame_index"]
                or sample["t_rel_ms"] - current[-1]["t_rel_ms"] > max_gap_ms
            )
        )
        if broken and current:
            segments.append(current)
            current = []
        if sample["usable"]:
            current.append(sample)
    if current:
        segments.append(current)
    result = []
    for segment_id, segment in enumerate(segments):
        start = 0
        while start < len(segment):
            end = start + 1
            while (
                end < len(segment)
                and segment[end]["t_rel_ms"] - segment[start]["t_rel_ms"]
                < min_duration_ms
            ):
                end += 1
            if end == len(segment):
                break
            window = segment[start : end + 1]
            min_u = min(r["image_u"] for r in window)
            max_u = max(r["image_u"] for r in window)
            min_v = min(r["image_v"] for r in window)
            max_v = max(r["image_v"] for r in window)
            if max_u - min_u + max_v - min_v > dispersion:
                start += 1
                continue
            while end + 1 < len(segment):
                candidate = segment[end + 1]
                low_u, high_u = min(min_u, candidate["image_u"]), max(
                    max_u, candidate["image_u"]
                )
                low_v, high_v = min(min_v, candidate["image_v"]), max(
                    max_v, candidate["image_v"]
                )
                if high_u - low_u + high_v - low_v > dispersion:
                    break
                min_u, max_u, min_v, max_v = low_u, high_u, low_v, high_v
                end += 1
            group = segment[start : end + 1]
            result.append(
                dict(
                    fixation=len(result) + 1,
                    segment=segment_id,
                    start_ms=group[0]["t_rel_ms"],
                    end_ms=group[-1]["t_rel_ms"],
                    duration_ms=group[-1]["t_rel_ms"] - group[0]["t_rel_ms"],
                    image_x=statistics.mean(r["image_x"] for r in group),
                    image_y=statistics.mean(r["image_y"] for r in group),
                    image_u=statistics.mean(r["image_u"] for r in group),
                    image_v=statistics.mean(r["image_v"] for r in group),
                    sample_count=len(group),
                    frame_index=group[0]["frame_index"],
                )
            )
            start = end + 1
    return result
