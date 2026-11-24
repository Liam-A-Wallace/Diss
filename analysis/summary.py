"""Aggregated study analysis: AOI dwell, early/late gaze, region scanpaths, entropy & group heatmaps."""
from collections import Counter, defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics

from .ingest import write_csv
from .process import HEATMAP_BINS, smooth_dwell

GROUP_ORDER = ("eyes", "nose_cheeks", "mouth_chin", "forehead_hair", "periphery")
GROUP_LABELS = {
    "eyes": "Eye region",
    "nose_cheeks": "Nose & cheeks",
    "mouth_chin": "Mouth & chin",
    "forehead_hair": "Forehead & hair",
    "periphery": "Ears, neck & periphery",
}
GROUP_COLOURS = {
    "eyes": "#1f77b4",
    "nose_cheeks": "#2ca02c",
    "mouth_chin": "#d62728",
    "forehead_hair": "#9467bd",
    "periphery": "#ff7f0e",
}

_AOI_TO_GROUP = {
    "image_left_eye": "eyes",
    "image_right_eye": "eyes",
    "image_left_eyebrow": "eyes",
    "image_right_eyebrow": "eyes",
    "image_left_under_eye_derived": "eyes",
    "image_right_under_eye_derived": "eyes",
    "eyes": "eyes",
    "eyebrows": "eyes",
    "glasses": "eyes",
    "nose": "nose_cheeks",
    "image_left_cheek_derived": "nose_cheeks",
    "image_right_cheek_derived": "nose_cheeks",
    "other_skin": "nose_cheeks",
    "mouth_interior": "mouth_chin",
    "upper_lip": "mouth_chin",
    "lower_lip": "mouth_chin",
    "chin_derived": "mouth_chin",
    "mouth": "mouth_chin",
    "forehead_derived": "forehead_hair",
    "hairline_derived": "forehead_hair",
    "hair": "forehead_hair",
    "hat": "forehead_hair",
    "image_left_ear": "periphery",
    "image_right_ear": "periphery",
    "ears": "periphery",
    "earring": "periphery",
    "necklace": "periphery",
    "neck": "periphery",
    "clothing": "periphery",
    "background": "periphery",
    "uncertain": "uncertain",
    "invalid_or_outside": "uncertain",
    "outside_image": "uncertain",
    "unmapped": "unmapped",
    "": "unmapped",
}


def group_aoi(label):
    """Map a fine or broad AOI label to a robust functional face zone."""
    return _AOI_TO_GROUP.get((label or "").strip(), "uncertain")


def source_group(difficulty, condition):
    """Group prompt A/B variants into their generator/quality source."""
    diff = (difficulty or "").strip()
    if diff.endswith("_A") or diff.endswith("_B"):
        return diff[:-2]
    return diff or (condition or "unknown").strip()


def collapse_region_visits(fixations):
    """Collapse consecutive within-segment fixations in the same functional region."""
    visits = []
    last_seen_segment = None
    for f in fixations:
        seg = f.get("segment")
        region = group_aoi(f.get("aoi", ""))
        if region in ("unmapped", "uncertain"):
            if seg is not None:
                last_seen_segment = seg
            continue
        dur = max(float(f.get("duration_ms", 0.0)), 1e-6)
        fx, fy = float(f["image_x"]), float(f["image_y"])
        start_ms = float(f["start_ms"])
        end_ms = float(f.get("end_ms", start_ms + dur))
        same_segment = (
            bool(visits)
            and seg is not None
            and visits[-1]["last_segment"] == seg
            and (last_seen_segment is None or last_seen_segment == seg)
        )
        segment_break = (
            bool(visits)
            and seg is not None
            and (
                visits[-1]["last_segment"] != seg
                or (last_seen_segment is not None and last_seen_segment != seg)
            )
        )
        if visits and visits[-1]["region"] == region and (same_segment or seg is None):
            v = visits[-1]
            total = v["duration_ms"] + dur
            v["image_x"] = (v["image_x"] * v["duration_ms"] + fx * dur) / total
            v["image_y"] = (v["image_y"] * v["duration_ms"] + fy * dur) / total
            v["duration_ms"] = total
            v["end_ms"] = max(v["end_ms"], end_ms)
            v["fixation_count"] += 1
            v["last_segment"] = seg
        else:
            visits.append(
                dict(
                    visit=len(visits) + 1,
                    region=region,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    duration_ms=dur,
                    fixation_count=1,
                    image_x=fx,
                    image_y=fy,
                    last_segment=seg,
                    segment_break_before=segment_break,
                )
            )
        if seg is not None:
            last_seen_segment = seg
    return visits


def compute_scanpath_metrics(fixations):
    """Compute region-visit sequence, revisits, stationary & transition entropy."""
    visits = collapse_region_visits(fixations)
    # Collapse consecutive identical regions across segment breaks so a brief
    # frame drop while gazing continuously at one region is not counted as a revisit.
    seq = []
    for v in visits:
        if not seq or seq[-1] != v["region"]:
            seq.append(v["region"])
    unique = set(seq)
    revisits = max(0, len(seq) - len(unique))
    counts = Counter(seq)
    revisited_regions = sorted(r for r, c in counts.items() if c >= 2)

    dwell_by_region = defaultdict(float)
    for v in visits:
        dwell_by_region[v["region"]] += v["duration_ms"]
    total_dur = sum(dwell_by_region.values())
    if total_dur > 0:
        stat_h = -sum(
            (d / total_dur) * math.log2(d / total_dur)
            for d in dwell_by_region.values()
            if d > 0
        )
        stat_h_norm = stat_h / math.log2(len(GROUP_ORDER))
    else:
        stat_h = ""
        stat_h_norm = ""

    transitions = [
        (a["region"], b["region"])
        for a, b in zip(visits, visits[1:])
        if not b["segment_break_before"] and a["region"] != b["region"]
    ]
    if transitions:
        from_counts = Counter(a for a, _ in transitions)
        pair_counts = Counter(transitions)
        n_trans = len(transitions)
        trans_h = 0.0
        for a, count_a in from_counts.items():
            p_a = count_a / n_trans
            for (src, _), count_ab in pair_counts.items():
                if src == a:
                    p_b_given_a = count_ab / count_a
                    trans_h -= p_a * p_b_given_a * math.log2(p_b_given_a)
        trans_h = round(trans_h, 4)
    else:
        trans_h = ""

    return dict(
        collapsed_aoi_sequence=" > ".join(seq),
        region_visit_count=len(seq),
        unique_regions_visited=len(unique),
        revisit_count=revisits,
        revisit_rate=round(revisits / len(seq), 4) if seq else "",
        revisited_regions=", ".join(revisited_regions),
        stationary_entropy_bits=round(stat_h, 4) if stat_h != "" else "",
        stationary_entropy_norm=round(stat_h_norm, 4) if stat_h_norm != "" else "",
        transition_entropy_bits=trans_h,
    )


def levenshtein_similarity(seq_a, seq_b):
    """Normalised edit similarity in [0, 1] between two region sequences."""
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0
    m, n = len(seq_a), len(seq_b)
    prev = list(range(n + 1))
    for i, item_a in enumerate(seq_a, 1):
        curr = [i] + [0] * n
        for j, item_b in enumerate(seq_b, 1):
            cost = 0 if item_a == item_b else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[n] / max(m, n)


def _read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _to_float(value, default=None):
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _mean(values):
    clean = [v for v in values if v is not None]
    return statistics.mean(clean) if clean else None


def _median(values):
    clean = [v for v in values if v is not None]
    return statistics.median(clean) if clean else None


def _fmt(value, digits=3):
    return "" if value is None else round(value, digits)


def load_processed_session(processed_dir):
    """Load data_results CSVs from one processed session folder."""
    processed_dir = Path(processed_dir)
    tables = processed_dir / "data_results"
    if not (tables / "features.csv").is_file():
        raise ValueError(f"Not a processed session folder (missing data_results/features.csv): {processed_dir}")
    features = _read_csv(tables / "features.csv")
    fixations = _read_csv(tables / "fixations.csv")
    aoi_dwell = _read_csv(tables / "aoi_dwell.csv")
    samples = _read_csv(tables / "samples.csv")
    return dict(
        path=processed_dir,
        features=features,
        fixations=fixations,
        aoi_dwell=aoi_dwell,
        samples=samples,
    )


def _plot_behaviour(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return None
    labels = [r["source"] for r in rows]
    acc = [(r["accuracy"] or 0.0) * 100 for r in rows]
    conf = [r["mean_confidence"] or 0.0 for r in rows]
    view = [r["mean_view_s"] or 0.0 for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor="white")
    try:
        colours = ["#2b6cb0" if l == "real" else "#c05621" for l in labels]
        axes[0].bar(x, acc, color=colours, alpha=0.88)
        axes[0].set_ylim(0, 112)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Classification accuracy")
        for xi, v, r in zip(x, acc, rows):
            acc_label = f"{v:.0f}%" if r["accuracy"] != "" else "N/A"
            axes[0].text(xi, v + 2.5, f"{acc_label}\n(n={r['n_trials']})", ha="center", va="bottom", fontsize=8)

        axes[1].bar(x, conf, color=colours, alpha=0.88)
        axes[1].set_ylim(0, 5.5)
        axes[1].set_ylabel("Mean confidence (1-5)")
        axes[1].set_title("Confidence rating")
        for xi, v, r in zip(x, conf, rows):
            conf_label = f"{v:.2f}" if r["mean_confidence"] != "" else "N/A"
            axes[1].text(xi, v + 0.12, conf_label, ha="center", va="bottom", fontsize=8)

        axes[2].bar(x, view, color=colours, alpha=0.88)
        top_view = max(view, default=0.0)
        if top_view > 0:
            axes[2].set_ylim(0, top_view * 1.15)
        axes[2].set_ylabel("Mean viewing time (s)")
        axes[2].set_title("Viewing duration")
        for xi, v in zip(x, view):
            axes[2].text(xi, v + max(top_view, 1.0) * 0.02, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)

        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    return out_path


def _plot_grouped_aoi(rows, key_field, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = []
    by_key = defaultdict(dict)
    for r in rows:
        k = r[key_field]
        if k not in keys:
            keys.append(k)
        by_key[k][r["region"]] = (r["mean_dwell_fraction"] or 0.0) * 100
    if not keys:
        return None
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 1.35), 4.8), facecolor="white")
    try:
        x = np.arange(len(keys))
        width = 0.72 / len(GROUP_ORDER)
        for i, region in enumerate(GROUP_ORDER):
            vals = [by_key[k].get(region, 0.0) for k in keys]
            offset = (i - (len(GROUP_ORDER) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width, label=GROUP_LABELS[region],
                   color=GROUP_COLOURS[region], alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=22, ha="right", fontsize=9)
        ax.set_ylabel("Mean observed dwell (%)")
        ax.set_title(title)
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    return out_path


def _plot_early_late(rows, out_path, early_ms=3000.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return None
    conditions = sorted({r["condition"] for r in rows})
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.2 * len(conditions), 4.4),
                             facecolor="white", squeeze=False)
    try:
        by_key = {(r["condition"], r["phase"], r["region"]): (r["mean_fixation_fraction"] or 0.0) * 100
                  for r in rows}
        x = np.arange(len(GROUP_ORDER))
        for ax, cond in zip(axes[0], conditions):
            early = [by_key.get((cond, "early", g), 0.0) for g in GROUP_ORDER]
            late = [by_key.get((cond, "late", g), 0.0) for g in GROUP_ORDER]
            ax.bar(x - 0.18, early, width=0.35, label=f"Early (first {early_ms / 1000:g} s)", color="#2b6cb0", alpha=0.88)
            ax.bar(x + 0.18, late, width=0.35, label="Late (remainder)", color="#dd6b20", alpha=0.88)
            ax.set_xticks(x)
            ax.set_xticklabels([GROUP_LABELS[g] for g in GROUP_ORDER], rotation=22, ha="right", fontsize=8)
            ax.set_ylabel("Share of fixation time (%)")
            ax.set_title(f"Early vs late gaze: {cond}")
            ax.legend(frameon=False, fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    return out_path


def _plot_scanpath_summary(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not rows:
        return None
    labels = [r["source"] for r in rows]
    revisits = [r["mean_revisit_count"] or 0.0 for r in rows]
    breadth = [r["mean_unique_regions"] or 0.0 for r in rows]
    entropy = [r["mean_stationary_entropy_bits"] or 0.0 for r in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor="white")
    try:
        colours = ["#2b6cb0" if l == "real" else "#c05621" for l in labels]
        for ax, vals, ylabel, title in (
            (axes[0], revisits, "Mean revisits / trial", "Region revisits (verification)"),
            (axes[1], breadth, "Unique regions (0-5)", "Exploration breadth"),
            (axes[2], entropy, "Shannon entropy (bits)", "Gaze distribution entropy"),
        ):
            ax.bar(x, vals, color=colours, alpha=0.88)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    return out_path


def _plot_group_heatmaps(stimulus_trials, samples_by_trial, output_dir):
    """Generate Right vs Wrong group heatmaps whenever >=2 sessions viewed the same image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    import numpy as np
    from PIL import Image

    created = []
    for digest, entries in sorted(stimulus_trials.items()):
        sessions = {e["session"] for e in entries}
        if len(sessions) < 2:
            continue
        stimulus_path = Path(entries[0]["stimulus"])
        if not stimulus_path.is_file():
            continue
        with Image.open(stimulus_path) as im:
            image = im.convert("RGB")
        w, h = image.size

        def avg_heat(subset):
            grids = []
            for e in subset:
                srows = samples_by_trial.get((e["session"], str(e["trigger"])), [])
                usable = [
                    r for r in srows
                    if _to_float(r.get("usable"), 0) == 1 and (_to_float(r.get("dwell_ms"), 0) or 0) > 0
                ]
                if not usable:
                    continue
                total = sum(float(r["dwell_ms"]) for r in usable)
                heat, _, _ = np.histogram2d(
                    [float(r["image_y"]) for r in usable],
                    [float(r["image_x"]) for r in usable],
                    bins=HEATMAP_BINS,
                    range=((0, h), (0, w)),
                    weights=[float(r["dwell_ms"]) / total for r in usable],
                )
                grids.append(smooth_dwell(heat))
            return np.mean(grids, axis=0) if grids else None

        correct_entries = [e for e in entries if e["correct"] == 1]
        wrong_entries = [e for e in entries if e["correct"] == 0]
        h_right = avg_heat(correct_entries)
        h_wrong = avg_heat(wrong_entries)
        if h_right is None and h_wrong is None:
            continue

        out_path = output_dir / f"group_heatmap_{stimulus_path.stem}_{digest[:8]}.png"
        has_both = h_right is not None and h_wrong is not None
        ncols = 3 if has_both else 1
        fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.4), facecolor="white", squeeze=False)
        try:
            panels = (
                [
                    (f"Correct (n={len(correct_entries)})", h_right, "pos"),
                    (f"Incorrect (n={len(wrong_entries)})", h_wrong, "pos"),
                    ("Difference (Correct - Incorrect)", h_right - h_wrong, "diff"),
                ]
                if has_both
                else [(
                    f"{'Correct' if h_right is not None else 'Incorrect'} "
                    f"(n={len(correct_entries) if h_right is not None else len(wrong_entries)})",
                    h_right if h_right is not None else h_wrong,
                    "pos",
                )]
            )
            vmax_pos = max(
                float(h_right.max()) if h_right is not None else 0.0,
                float(h_wrong.max()) if h_wrong is not None else 0.0,
                1e-6,
            )
            for ax, (title, grid, mode) in zip(axes[0], panels):
                ax.imshow(image)
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.axis("off")
                ax.set_title(title, fontsize=11)
                if mode == "pos":
                    norm = Normalize(vmin=0, vmax=vmax_pos)
                    rgba = plt.get_cmap("turbo")(norm(grid))
                    rgba[..., 3] = 0.78 * np.power(norm(grid), 0.65)
                else:
                    lim = max(float(np.abs(grid).max()), 1e-6)
                    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
                    rgba = plt.get_cmap("coolwarm")(norm(grid))
                    rgba[..., 3] = 0.78 * np.power(np.abs(grid) / lim, 0.65)
                ax.imshow(rgba, extent=(0, w, h, 0), origin="upper", interpolation="bilinear")
            fig.suptitle(
                f"{stimulus_path.name} ({entries[0]['condition']} / {entries[0]['source']})",
                fontsize=12,
                fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(out_path, dpi=160, facecolor="white")
        finally:
            plt.close(fig)
        created.append((
            f"Group heatmap: {stimulus_path.name} ({entries[0]['source']})",
            out_path,
            f"{len(entries)} recordings ({len(correct_entries)} correct, {len(wrong_entries)} incorrect). "
            + ("Red = relatively more dwell in correct trials; blue = more in incorrect trials." if has_both else ""),
        ))
    return created


def summarise(processed_dirs, output_dir, *, plots=True, early_ms=3000.0):
    """Aggregate one or more processed sessions into dissertation-ready tables and figures."""
    if not processed_dirs:
        raise ValueError("Provide at least one processed session directory.")
    if not math.isfinite(early_ms) or early_ms <= 0:
        raise ValueError("early_ms must be positive and finite.")
    sessions = [load_processed_session(p) for p in processed_dirs]
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Choose an empty summary output directory.")

    fixations_by_trial = defaultdict(list)
    dwell_by_trial = defaultdict(list)
    samples_by_trial = defaultdict(list)
    trials = []

    seen_sessions = {}
    for sess in sessions:
        if not sess["features"]:
            continue
        sess_id = sess["features"][0]["session"]
        if sess_id in seen_sessions:
            raise ValueError(
                f"Duplicate session '{sess_id}' found in both {seen_sessions[sess_id]} and {sess['path']}."
            )
        seen_sessions[sess_id] = sess["path"]
        for row in sess["fixations"]:
            fixations_by_trial[(row.get("session", sess_id), str(row["trigger"]))].append(row)
        for row in sess["aoi_dwell"]:
            dwell_by_trial[(row.get("session", sess_id), str(row["trigger"]))].append(row)
        for row in sess["samples"]:
            samples_by_trial[(row.get("session", sess_id), str(row["trigger"]))].append(row)
        for row in sess["features"]:
            if str(row.get("is_practice", "0")) == "1":
                continue
            key = (row["session"], str(row["trigger"]))
            fix = sorted(fixations_by_trial[key], key=lambda r: int(float(r["fixation"])))
            sp = compute_scanpath_metrics(fix)
            correct_v = _to_float(row.get("correct"))
            correct_i = int(correct_v) if correct_v is not None else None
            cond = (row.get("condition") or "").strip().lower()
            diff = (row.get("difficulty") or "").strip()
            src = source_group(diff, cond)
            outcome = (
                f"{cond}_{'correct' if correct_i == 1 else 'incorrect'}"
                if correct_i is not None and cond
                else "unscored"
            )
            view_ms = _to_float(row.get("view_ms"), 0.0) or 0.0
            obs_ms = _to_float(row.get("observed_dwell_ms"), 0.0) or 0.0

            # Group observed AOI dwell fractions + earliest TTFF per group
            group_dwell_ms = {g: 0.0 for g in GROUP_ORDER}
            group_ttff = {g: [] for g in GROUP_ORDER}
            mapped_ms = 0.0
            for drow in dwell_by_trial[key]:
                g = group_aoi(drow.get("aoi", ""))
                ms = _to_float(drow.get("dwell_ms"), 0.0) or 0.0
                if g in group_dwell_ms:
                    group_dwell_ms[g] += ms
                    mapped_ms += ms
                    ttff = _to_float(drow.get("time_to_first_fixation_ms"))
                    if ttff is not None:
                        group_ttff[g].append(ttff)

            # Early vs late fixation share per grouped region
            cutoff = min(early_ms, view_ms / 2.0) if view_ms > 0 else early_ms
            early_ms_by_g = {g: 0.0 for g in GROUP_ORDER}
            late_ms_by_g = {g: 0.0 for g in GROUP_ORDER}
            for f in fix:
                g = group_aoi(f.get("aoi", ""))
                if g not in GROUP_ORDER:
                    continue
                dur = _to_float(f.get("duration_ms"), 0.0) or 0.0
                if (_to_float(f.get("start_ms"), 0.0) or 0.0) < cutoff:
                    early_ms_by_g[g] += dur
                else:
                    late_ms_by_g[g] += dur

            trials.append(
                dict(
                    participant=row.get("participant", ""),
                    session=row["session"],
                    trial=row.get("trial", ""),
                    trigger=row["trigger"],
                    stimulus=row.get("stimulus", ""),
                    stimulus_name=Path(row.get("stimulus", "")).name,
                    stimulus_sha256=row.get("stimulus_sha256", ""),
                    condition=cond,
                    difficulty=diff,
                    source=src,
                    response=row.get("response", ""),
                    correct=correct_i,
                    outcome=outcome,
                    confidence=_to_float(row.get("confidence")),
                    view_s=view_ms / 1000.0,
                    observed_ms=obs_ms,
                    mapped_ms=mapped_ms,
                    max_zoom=_to_float(row.get("max_zoom"), 1.0) or 1.0,
                    fixation_count=_to_float(row.get("fixation_count"), 0.0) or 0.0,
                    decision_reason=(row.get("decision_reason") or "").strip(),
                    group_dwell_frac={
                        g: (group_dwell_ms[g] / obs_ms if obs_ms > 0 else None)
                        for g in GROUP_ORDER
                    },
                    group_ttff={
                        g: (min(group_ttff[g]) if group_ttff[g] else None)
                        for g in GROUP_ORDER
                    },
                    early_frac={
                        g: (
                            early_ms_by_g[g] / sum(early_ms_by_g.values())
                            if sum(early_ms_by_g.values()) > 0
                            else None
                        )
                        for g in GROUP_ORDER
                    },
                    late_frac={
                        g: (
                            late_ms_by_g[g] / sum(late_ms_by_g.values())
                            if sum(late_ms_by_g.values()) > 0
                            else None
                        )
                        for g in GROUP_ORDER
                    },
                    **sp,
                )
            )

    if not trials:
        raise ValueError("No non-practice trials found in the supplied processed session(s).")

    has_aois = any(t["mapped_ms"] > 0 for t in trials)

    # 1. Behaviour + scanpath summary by source & by difficulty
    def build_group_summary(field):
        grouped = defaultdict(list)
        for t in trials:
            grouped[t[field] or t["condition"]].append(t)
        order = sorted(grouped, key=lambda k: (k != "real", k))
        out = []
        for k in order:
            ts = grouped[k]
            mapped_ts = [t for t in ts if t["mapped_ms"] > 0]
            out.append(
                {
                    field: k,
                    "n_trials": len(ts),
                    "accuracy": _fmt(_mean([t["correct"] for t in ts]), 4),
                    "mean_confidence": _fmt(_mean([t["confidence"] for t in ts]), 3),
                    "mean_view_s": _fmt(_mean([t["view_s"] for t in ts]), 3),
                    "median_view_s": _fmt(_median([t["view_s"] for t in ts]), 3),
                    "zoomed_fraction": _fmt(_mean([1.0 if t["max_zoom"] > 1.05 else 0.0 for t in ts]), 3),
                    "mean_fixations": _fmt(_mean([t["fixation_count"] for t in ts]), 2),
                    "mean_region_visits": _fmt(_mean([t["region_visit_count"] for t in mapped_ts]), 2) if has_aois else "",
                    "mean_unique_regions": _fmt(_mean([t["unique_regions_visited"] for t in mapped_ts]), 2) if has_aois else "",
                    "mean_revisit_count": _fmt(_mean([t["revisit_count"] for t in mapped_ts]), 2) if has_aois else "",
                    "mean_revisit_rate": _fmt(_mean([_to_float(t["revisit_rate"]) for t in mapped_ts]), 4) if has_aois else "",
                    "mean_stationary_entropy_bits": (
                        _fmt(_mean([_to_float(t["stationary_entropy_bits"]) for t in mapped_ts]), 4)
                        if has_aois
                        else ""
                    ),
                    "mean_transition_entropy_bits": (
                        _fmt(_mean([_to_float(t["transition_entropy_bits"]) for t in mapped_ts]), 4)
                        if has_aois
                        else ""
                    ),
                }
            )
        return out

    behaviour_by_source = build_group_summary("source")
    behaviour_by_difficulty = build_group_summary("difficulty")

    # 1b. Grouped AOI dwell by source and by outcome (real/fake x correct/incorrect)
    def build_aoi_summary(field):
        grouped = defaultdict(list)
        for t in trials:
            if t["mapped_ms"] > 0:
                grouped[t[field]].append(t)
        order = sorted(grouped, key=lambda k: (not k.startswith("real"), k))
        rows = []
        for k in order:
            ts = grouped[k]
            for g in GROUP_ORDER:
                vals = [t["group_dwell_frac"][g] for t in ts]
                rows.append(
                    {
                        field: k,
                        "region": g,
                        "region_label": GROUP_LABELS[g],
                        "n_trials": len(ts),
                        "mean_dwell_fraction": _fmt(_mean(vals), 4),
                        "median_dwell_fraction": _fmt(_median(vals), 4),
                    }
                )
        return rows

    aoi_by_source = build_aoi_summary("source")
    aoi_by_outcome = build_aoi_summary("outcome")

    # 2. Early vs Late gaze + Time to First Fixation
    early_late_rows = []
    ttff_rows = []
    for cond in sorted({t["condition"] for t in trials if t["condition"]}):
        ts = [t for t in trials if t["condition"] == cond and t["mapped_ms"] > 0]
        if not ts:
            continue
        for phase, attr in (("early", "early_frac"), ("late", "late_frac")):
            for g in GROUP_ORDER:
                vals = [t[attr][g] for t in ts if t[attr][g] is not None]
                early_late_rows.append(
                    dict(
                        condition=cond,
                        phase=phase,
                        region=g,
                        region_label=GROUP_LABELS[g],
                        n_trials=len(vals),
                        mean_fixation_fraction=_fmt(_mean(vals), 4),
                        median_fixation_fraction=_fmt(_median(vals), 4),
                    )
                )
        for g in GROUP_ORDER:
            vals = [t["group_ttff"][g] for t in ts if t["group_ttff"][g] is not None]
            ttff_rows.append(
                dict(
                    condition=cond,
                    region=g,
                    region_label=GROUP_LABELS[g],
                    n_reached=len(vals),
                    n_trials=len(ts),
                    reached_fraction=_fmt(len(vals) / len(ts), 3),
                    median_ttff_ms=_fmt(_median(vals), 1),
                    mean_ttff_ms=_fmt(_mean(vals), 1),
                )
            )

    # 3. Per-trial scanpath metrics + between-session scanpath similarity
    scanpath_rows = [
        dict(
            participant=t["participant"],
            session=t["session"],
            trial=t["trial"],
            stimulus=t["stimulus_name"],
            condition=t["condition"],
            source=t["source"],
            difficulty=t["difficulty"],
            response=t["response"],
            correct="" if t["correct"] is None else t["correct"],
            confidence="" if t["confidence"] is None else t["confidence"],
            view_s=_fmt(t["view_s"], 3),
            region_visit_count=t["region_visit_count"],
            unique_regions_visited=t["unique_regions_visited"],
            revisit_count=t["revisit_count"],
            revisit_rate=t["revisit_rate"],
            revisited_regions=t["revisited_regions"],
            stationary_entropy_bits=t["stationary_entropy_bits"],
            stationary_entropy_norm=t["stationary_entropy_norm"],
            transition_entropy_bits=t["transition_entropy_bits"],
            collapsed_aoi_sequence=t["collapsed_aoi_sequence"],
            decision_reason=t["decision_reason"],
        )
        for t in trials
    ]

    by_stim = defaultdict(list)
    for t in trials:
        if t["stimulus_sha256"]:
            by_stim[t["stimulus_sha256"]].append(t)

    similarity_rows = []
    for digest, entries in sorted(by_stim.items()):
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                if a["session"] == b["session"]:
                    continue
                sa = [x for x in a["collapsed_aoi_sequence"].split(" > ") if x]
                sb = [x for x in b["collapsed_aoi_sequence"].split(" > ") if x]
                if not sa or not sb:
                    continue
                if a["correct"] is None or b["correct"] is None:
                    pair_type = "unscored"
                elif a["correct"] == 1 and b["correct"] == 1:
                    pair_type = "both_correct"
                elif a["correct"] == 0 and b["correct"] == 0:
                    pair_type = "both_incorrect"
                else:
                    pair_type = "mixed"
                similarity_rows.append(
                    dict(
                        stimulus=a["stimulus_name"],
                        condition=a["condition"],
                        source=a["source"],
                        session_a=a["session"],
                        session_b=b["session"],
                        pair_outcome=pair_type,
                        levenshtein_similarity=round(levenshtein_similarity(sa, sb), 4),
                    )
                )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "data_results"
    tables_dir.mkdir(exist_ok=True)

    write_csv(tables_dir / "behaviour_by_source.csv", behaviour_by_source, list(behaviour_by_source[0]))
    write_csv(tables_dir / "behaviour_by_difficulty.csv", behaviour_by_difficulty, list(behaviour_by_difficulty[0]))
    write_csv(
        tables_dir / "aoi_dwell_by_source.csv",
        aoi_by_source,
        ["source", "region", "region_label", "n_trials", "mean_dwell_fraction", "median_dwell_fraction"],
    )
    write_csv(
        tables_dir / "aoi_dwell_by_outcome.csv",
        aoi_by_outcome,
        ["outcome", "region", "region_label", "n_trials", "mean_dwell_fraction", "median_dwell_fraction"],
    )
    write_csv(
        tables_dir / "early_vs_late_aoi.csv",
        early_late_rows,
        ["condition", "phase", "region", "region_label", "n_trials", "mean_fixation_fraction", "median_fixation_fraction"],
    )
    write_csv(
        tables_dir / "time_to_first_fixation.csv",
        ttff_rows,
        ["condition", "region", "region_label", "n_reached", "n_trials", "reached_fraction", "median_ttff_ms", "mean_ttff_ms"],
    )
    write_csv(tables_dir / "scanpath_metrics.csv", scanpath_rows, list(scanpath_rows[0]))
    write_csv(
        tables_dir / "scanpath_similarity.csv",
        similarity_rows,
        ["stimulus", "condition", "source", "session_a", "session_b", "pair_outcome", "levenshtein_similarity"],
    )

    figures = []
    if plots:
        for title, fn, note in (
            (
                "Behaviour by stimulus source",
                lambda: _plot_behaviour(behaviour_by_source, output_dir / "behaviour_by_source.png"),
                "Accuracy, confidence, and viewing duration grouped by real vs generator/quality source.",
            ),
            (
                "Grouped AOI dwell by source",
                lambda: _plot_grouped_aoi(
                    aoi_by_source, "source", "Where participants looked by stimulus source",
                    output_dir / "aoi_dwell_by_source.png",
                ),
                "Mean fraction of observed dwell across five robust facial zones.",
            ),
            (
                "Grouped AOI dwell by response outcome",
                lambda: _plot_grouped_aoi(
                    aoi_by_outcome, "outcome", "Where participants looked: Correct vs Incorrect",
                    output_dir / "aoi_dwell_by_outcome.png",
                ),
                "Compares dwell distribution on correctly identified vs missed real and fake faces.",
            ),
            (
                "Early vs late gaze distribution",
                lambda: _plot_early_late(early_late_rows, output_dir / "early_vs_late_aoi.png", early_ms=early_ms),
                f"Early = first {early_ms / 1000:g} s (or first half if viewing < {2 * early_ms / 1000:g} s); Late = remainder.",
            ),
            (
                "Scanpath metrics by source",
                lambda: _plot_scanpath_summary(behaviour_by_source, output_dir / "scanpath_metrics_by_source.png") if has_aois else None,
                "Region revisits (verification checks), exploration breadth, and Shannon gaze entropy.",
            ),
        ):
            path = fn()
            if path is not None:
                figures.append((title, path, note))
        figures.extend(_plot_group_heatmaps(by_stim, samples_by_trial, output_dir))

    # Build HTML report
    reasons_html = ""
    noted = [t for t in trials if t["decision_reason"]]
    if noted:
        reason_rows = []
        for t in noted:
            top_g = sorted(
                ((t["group_dwell_frac"][g] or 0.0, GROUP_LABELS[g]) for g in GROUP_ORDER),
                reverse=True,
            )
            top_txt = ", ".join(f"{lbl} ({v:.0%})" for v, lbl in top_g[:2] if v > 0) or "unmapped"
            verdict = "ok" if t["correct"] == 1 else ("miss" if t["correct"] == 0 else "unscored")
            reason_rows.append(
                "<tr>"
                + "".join(
                    f"<td>{escape(str(val))}</td>"
                    for val in (
                        f"{t['session']} / T{t['trial']}",
                        t["source"],
                        f"{t['response']} ({verdict})" if t["response"] else verdict,
                        top_txt,
                        t["collapsed_aoi_sequence"] or "-",
                        t["decision_reason"],
                    )
                )
                + "</tr>"
            )
        reasons_html = (
            '<section class="analysis-section"><h2>Supporting decision reasons</h2>'
            "<p>Participant notes paired with their top facial zones and collapsed region scanpath.</p>"
            '<div class="table-scroll"><table><thead><tr>'
            "<th>Trial</th><th>Source</th><th>Decision</th><th>Top zones</th><th>Region scanpath</th><th>Note</th>"
            "</tr></thead><tbody>"
            + "".join(reason_rows)
            + "</tbody></table></div></section>"
        )

    sim_note = ""
    if similarity_rows:
        by_pair = defaultdict(list)
        for r in similarity_rows:
            by_pair[r["pair_outcome"]].append(r["levenshtein_similarity"])
        parts = [
            f"{k}: {statistics.mean(v):.3f} (n={len(v)} pairs)"
            for k, v in sorted(by_pair.items())
        ]
        sim_note = "<p><strong>Between-participant scanpath similarity (0-1):</strong> " + " | ".join(parts) + "</p>"
    else:
        sim_note = (
            "<p><strong>Multi-participant analyses (group heatmaps &amp; scanpath similarity):</strong> "
            "Pass 2 or more processed sessions to enable Right-vs-Wrong heatmaps and between-participant scanpath similarity.</p>"
        )

    beh_rows = []
    for r in behaviour_by_source:
        acc_s = f"{r['accuracy'] * 100:.1f}%" if r["accuracy"] != "" else "-"
        beh_rows.append(
            "<tr>"
            + "".join(
                f"<td>{escape('-' if v == '' else str(v))}</td>"
                for v in (
                    r["source"],
                    r["n_trials"],
                    acc_s,
                    r["mean_confidence"],
                    r["mean_view_s"],
                    r["mean_region_visits"],
                    r["mean_revisit_count"],
                    r["mean_stationary_entropy_bits"],
                    r["mean_transition_entropy_bits"],
                )
            )
            + "</tr>"
        )
    summary_html = (
        '<section class="analysis-section"><h2>Performance &amp; scanpath overview by source</h2>'
        '<div class="table-scroll"><table><thead><tr>'
        "<th>Source</th><th>Trials</th><th>Accuracy</th><th>Confidence</th><th>View (s)</th>"
        "<th>Region visits</th><th>Revisits</th><th>Stationary H (bits)</th><th>Transition H (bits)</th>"
        "</tr></thead><tbody>"
        + "".join(beh_rows)
        + "</tbody></table></div>"
        + sim_note
        + "</section>"
        + reasons_html
    )

    from analyse import gallery

    descriptions = {
        "behaviour_by_source.csv": "Accuracy, confidence, viewing time and scanpath metrics by generator source",
        "behaviour_by_difficulty.csv": "Same summary split by exact difficulty/prompt tag (e.g. A vs B)",
        "aoi_dwell_by_source.csv": "Grouped facial-zone dwell % by stimulus source",
        "aoi_dwell_by_outcome.csv": "Grouped facial-zone dwell % by real/fake x correct/incorrect",
        "early_vs_late_aoi.csv": "Early vs late fixation share across facial zones",
        "time_to_first_fixation.csv": "Median & mean time to first fixation per facial zone",
        "scanpath_metrics.csv": "Per-trial collapsed region scanpath, revisits, breadth and entropy",
        "scanpath_similarity.csv": "Between-participant Levenshtein scanpath similarity on shared stimuli",
    }
    items = [
        (p.name, p, False, descriptions.get(p.name, "Summary table"))
        for p in sorted(tables_dir.glob("*.csv"))
    ]
    items += [(title, path, True, note) for title, path, note in figures]
    report = gallery(
        output_dir / "index.html",
        f"Study summary: {len(sessions)} session(s), {len(trials)} trials",
        (
            "Aggregated AOI dwell, early-vs-late gaze, collapsed region-visit scanpaths, "
            "revisit/entropy metrics, and supporting notes. "
            + (
                "AOI groupings merge adjacent fine labels into 5 robust functional zones."
                if has_aois
                else "No AOIs were mapped in the input session(s); rerun analyse.py --detailed first for AOI breakdowns."
            )
        ),
        items,
        summary_html=summary_html,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            dict(
                sessions=[str(s["path"]) for s in sessions],
                n_trials=len(trials),
                early_ms=early_ms,
                aoi_groups={
                    g: sorted(k for k, v in _AOI_TO_GROUP.items() if v == g)
                    for g in GROUP_ORDER
                },
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
