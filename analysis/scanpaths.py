"""Exploratory fixation-to-fixation paths in normalised image coordinates.

Steps are not automatically classified saccades. Timing between fixations is
not a measured saccade duration. Image distance is not visual angle.
"""

import math
import statistics


def link_decisions(fixations, max_gap_ms=100):
    """Classify every neighbouring pair. None allows all within-segment intervals.

    A link denotes observed fixation order, never a measured saccade trajectory.
    Missing samples and display changes remain hard boundaries in either mode.
    """
    if max_gap_ms is not None and (not math.isfinite(max_gap_ms) or max_gap_ms <= 0):
        raise ValueError("Link interval must be positive and finite, or None")
    decisions = []
    for a, b in zip(fixations, fixations[1:]):
        interval = b["start_ms"] - a["end_ms"]
        if a.get("frame_index") != b.get("frame_index"):
            reason = "view_change"
        elif a["segment"] != b["segment"]:
            reason = "segment_break"
        elif interval < 0:
            reason = "overlapping_events"
        elif max_gap_ms is not None and interval > max_gap_ms:
            reason = "link_interval_exceeded"
        else:
            reason = "within_segment"
        decisions.append(
            dict(
                from_fixation=a["fixation"],
                to_fixation=b["fixation"],
                inter_fixation_ms=interval,
                accepted=int(reason == "within_segment"),
                reason=reason,
                from_segment=a["segment"],
                to_segment=b["segment"],
            )
        )
    return decisions


def scanpath(fixations, max_gap_ms=100, decisions=None):
    decisions = (
        link_decisions(fixations, max_gap_ms) if decisions is None else decisions
    )
    steps = []
    segment = 0
    for a, b, decision in zip(fixations, fixations[1:], decisions):
        if not decision["accepted"]:
            segment += 1
            continue
        dx, dy = b["image_u"] - a["image_u"], b["image_v"] - a["image_v"]
        steps.append(
            dict(
                path_segment=segment,
                from_fixation=a["fixation"],
                to_fixation=b["fixation"],
                link_kind="fixation_order",
                from_aoi=a.get("aoi", "unmapped"),
                to_aoi=b.get("aoi", "unmapped"),
                start_ms=a["end_ms"],
                end_ms=b["start_ms"],
                inter_fixation_ms=b["start_ms"] - a["end_ms"],
                from_u=a["image_u"],
                from_v=a["image_v"],
                to_u=b["image_u"],
                to_v=b["image_v"],
                distance_image_units=math.hypot(dx, dy),
                direction_degrees=math.degrees(math.atan2(dy, dx)) if dx or dy else "",
            )
        )
    distances = [s["distance_image_units"] for s in steps]
    displacement = 0.0
    for group in sorted({s["path_segment"] for s in steps}):
        members = [s for s in steps if s["path_segment"] == group]
        displacement += math.hypot(
            members[-1]["to_u"] - members[0]["from_u"],
            members[-1]["to_v"] - members[0]["from_v"],
        )
    length = sum(distances)
    turns = []
    for a, b in zip(steps, steps[1:]):
        if (
            a["path_segment"] == b["path_segment"]
            and a["to_fixation"] == b["from_fixation"]
            and a["direction_degrees"] != ""
            and b["direction_degrees"] != ""
        ):
            turns.append(
                abs((b["direction_degrees"] - a["direction_degrees"] + 180) % 360 - 180)
            )
    summary = dict(
        scanpath_step_count=len(steps),
        scanpath_candidate_count=len(decisions),
        scanpath_rejected_count=len(decisions) - len(steps),
        scanpath_segment_count=len(fixations) - len(steps),
        scanpath_mean_step_image_units=statistics.mean(distances) if distances else "",
        scanpath_max_step_image_units=max(distances) if distances else "",
        scanpath_net_displacement_image_units=displacement,
        scanpath_straightness=displacement / length if length else "",
        scanpath_mean_turn_degrees=statistics.mean(turns) if turns else "",
        total_fixation_duration_ms=sum(f["duration_ms"] for f in fixations),
    )
    for reason in (
        "view_change",
        "segment_break",
        "overlapping_events",
        "link_interval_exceeded",
    ):
        summary["scanpath_rejected_" + reason] = sum(
            d["reason"] == reason for d in decisions
        )
    return summary, steps
