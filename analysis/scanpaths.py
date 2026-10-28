"""Exploratory fixation-to-fixation paths in normalised image coordinates.

Steps are not automatically classified saccades. Timing between fixations is
not a measured saccade duration. Image distance is not visual angle.
"""
import math
import statistics


def scanpath(fixations, max_gap_ms=100):
    steps = []
    segment = 0
    for a, b in zip(fixations, fixations[1:]):
        if a["segment"] != b["segment"] or not 0 <= b["start_ms"] - a["end_ms"] <= max_gap_ms:
            segment += 1
            continue
        dx, dy = b["image_u"] - a["image_u"], b["image_v"] - a["image_v"]
        steps.append(dict(path_segment=segment, from_fixation=a["fixation"], to_fixation=b["fixation"],
                          from_aoi=a.get("aoi", "unmapped"), to_aoi=b.get("aoi", "unmapped"),
                          start_ms=a["end_ms"], end_ms=b["start_ms"],
                          inter_fixation_ms=b["start_ms"] - a["end_ms"],
                          from_u=a["image_u"], from_v=a["image_v"],
                          to_u=b["image_u"], to_v=b["image_v"],
                          distance_image_units=math.hypot(dx, dy),
                          direction_degrees=math.degrees(math.atan2(dy, dx)) if dx or dy else ""))
    distances = [s["distance_image_units"] for s in steps]
    displacement = 0.0
    for group in sorted({s["path_segment"] for s in steps}):
        members = [s for s in steps if s["path_segment"] == group]
        displacement += math.hypot(members[-1]["to_u"] - members[0]["from_u"],
                                   members[-1]["to_v"] - members[0]["from_v"])
    length = sum(distances)
    turns = []
    for a, b in zip(steps, steps[1:]):
        if (a["path_segment"] == b["path_segment"] and a["to_fixation"] == b["from_fixation"]
                and a["direction_degrees"] != "" and b["direction_degrees"] != ""):
            turns.append(abs((b["direction_degrees"] - a["direction_degrees"] + 180) % 360 - 180))
    summary = dict(scanpath_step_count=len(steps),
                   scanpath_segment_count=len(fixations) - len(steps),
                   scanpath_mean_step_image_units=statistics.mean(distances) if distances else "",
                   scanpath_max_step_image_units=max(distances) if distances else "",
                   scanpath_net_displacement_image_units=displacement,
                   scanpath_straightness=displacement / length if length else "",
                   scanpath_mean_turn_degrees=statistics.mean(turns) if turns else "",
                   total_fixation_duration_ms=sum(f["duration_ms"] for f in fixations))
    return summary, steps