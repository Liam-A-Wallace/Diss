"""Run with python -m analysis.process data/SESSION_BASE --output processed/RUN."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from .ingest import load_display, align_samples, write_csv, resolve_stimulus
from .fixations import detect_fixations
from .scanpaths import scanpath
from .face_aois import AOIMap, sha256


def sample_durations(rows, max_gap_ms):
    """Only attribute intervals bracketed by usable samples in the same view."""
    for row in rows:
        row["dwell_ms"] = 0.0
    for a, b in zip(rows, rows[1:]):
        dt = b["t_rel_ms"] - a["t_rel_ms"]
        if (a["usable"] and b["usable"] and a["frame_index"] == b["frame_index"]
                and 0 < dt <= max_gap_ms):
            a["dwell_ms"] = dt


def continuous(a, b, max_gap_ms):
    return (a["segment"] == b["segment"] and
            b["start_ms"] - a["end_ms"] <= max_gap_ms)


HEATMAP_BINS = 40
HEATMAP_SIGMA = 1.2  # grid cells; display smoothing only


def smooth_dwell(heat, sigma=HEATMAP_SIGMA):
    """Gaussian display smoothing; symmetric edges preserve total dwell."""
    import numpy as np
    radius = int(math.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    result = np.asarray(heat, dtype=float)
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (radius, radius)
        result = np.apply_along_axis(
            lambda values: np.convolve(values, kernel, mode="valid"),
            axis, np.pad(result, padding, mode="symmetric"))
    return result


def plot_trial(stimulus, rows, fixations, out, name, max_gap_ms=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import numpy as np
    from PIL import Image
    with Image.open(stimulus) as im:
        image = im.convert("RGB")
    w, h = image.size
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": "#18263b", "axes.titleweight": "bold"}):
        fig, axes = plt.subplots(1, 2, figsize=(13, 7), facecolor="#f3f6fa")
        fig.subplots_adjust(left=0.035, right=0.965, top=0.83, bottom=0.18, wspace=0.12)
        try:
            for ax in axes:
                ax.imshow(image)
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                ax.axis("off")
            usable = [r for r in rows if r["usable"] and r["dwell_ms"] > 0]
            observed_seconds = sum(r["dwell_ms"] for r in usable) / 1000
            if usable:
                heat, _, _ = np.histogram2d(
                    [r["image_y"] for r in usable], [r["image_x"] for r in usable],
                    bins=HEATMAP_BINS, range=((0, h), (0, w)),
                    weights=[r["dwell_ms"] / 1000 for r in usable])
                heat = smooth_dwell(heat)
                norm = Normalize(vmin=0, vmax=float(heat.max()))
                colours = plt.get_cmap("turbo")(norm(heat))
                colours[..., 3] = 0.78 * np.power(norm(heat), 0.65)
                axes[0].imshow(colours, extent=(0, w, h, 0), origin="upper", interpolation="bilinear")
                colourbar = fig.colorbar(ScalarMappable(norm=norm, cmap="turbo"), ax=axes[0],
                                         orientation="horizontal", fraction=0.045, pad=0.055,
                                         shrink=0.8, aspect=35)
                colourbar.set_label("Smoothed observed dwell (seconds / grid cell)", fontsize=9)
                colourbar.outline.set_visible(False)
                colourbar.ax.tick_params(labelsize=8, length=0)
                right, left = axes[1].get_position(), axes[0].get_position()
                axes[1].set_position([right.x0, left.y0, left.width, left.height])
            for a, b in zip(fixations, fixations[1:]):
                if continuous(a, b, max_gap_ms):
                    axes[1].annotate("", xy=(b["image_x"], b["image_y"]),
                                     xytext=(a["image_x"], a["image_y"]),
                                     arrowprops=dict(arrowstyle="-|>", color="#21c7e7",
                                                     lw=1.7, alpha=0.85, shrinkA=8, shrinkB=8))
            if fixations:
                axes[1].scatter([f["image_x"] for f in fixations], [f["image_y"] for f in fixations],
                                s=[max(140, f["duration_ms"] * 0.6) for f in fixations],
                                c="#123650", edgecolors="white", linewidths=1.4, alpha=0.88, zorder=3)
                for f in fixations:
                    axes[1].annotate(str(f["fixation"]), (f["image_x"], f["image_y"]),
                                     color="white", ha="center", va="center", fontsize=8,
                                     fontweight="bold", zorder=4)
            for ax, empty, message in ((axes[0], not usable, "No usable dwell recorded"),
                                       (axes[1], not fixations, "No fixations detected")):
                if empty:
                    ax.text(0.5, 0.04, message, transform=ax.transAxes, ha="center",
                            bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="none", alpha=0.9))
            axes[0].set_title("Where you looked", loc="left", pad=14, fontsize=14)
            axes[1].set_title("Order of fixations", loc="left", pad=14, fontsize=14)
            fig.text(0.04, 0.94, name, fontsize=18, fontweight="bold")
            fig.text(0.04, 0.895,
                     f"{observed_seconds:.2f} s observed dwell   /   {len(fixations)} exploratory fixations",
                     fontsize=10, color="#53667e")
            fig.text(0.04, 0.07,
                     "Heatmap: Gaussian display smoothing, 1.2 cells on a 40 x 40 grid; colour scale is per trial.\n"
                     "Fixations: number = order; circle size increases with duration. Arrows stop at tracking gaps or view changes.",
                     fontsize=9, color="#53667e", linespacing=1.6)
            fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
        finally:
            plt.close(fig)


def process(base, output, aois=None, allow_unreviewed=False, coordinate_source="calibrated",
            dispersion=0.08, min_duration_ms=100, max_gap_ms=100, plots=True):
    base = str(base)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory to preserve previous analysis")
    metadata = json.loads(Path(base + "_session.json").read_text(encoding="utf-8"))
    events = load_display(base + "_display.jsonl")
    samples = list(align_samples(base + "_gaze_raw.csv", events, coordinate_source))
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["trigger"]].append(row)
    views, offsets = {}, {}
    for e in events:
        if e["kind"] == "present":
            views.setdefault(e["trigger"], e)
        else:
            offsets[e["trigger"]] = e
    responses = {}
    response_path = Path(base + ".csv")
    if response_path.exists():
        with response_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("trigger"):
                    responses[int(float(row["trigger"]))] = row
    aoi_map = AOIMap(aois, allow_unreviewed)
    # Check every stimulus and mask before creating any output files.
    stimuli = {}
    for view in views.values():
        stimulus = resolve_stimulus(view["stimulus"], metadata)
        digest = sha256(stimulus)
        if digest != metadata["stimuli_sha256"].get(str(Path(view["stimulus"]))):
            raise ValueError("Stimulus changed since recording: " + str(stimulus))
        aoi_map.load(str(stimulus))
        stimuli[view["stimulus"]] = (str(stimulus), digest)
    if plots:
        import matplotlib  # fail early if the optional plotting dependency is missing
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "data_results"
    tables.mkdir()
    aoi_names = set(aoi_map.manifest["labels"].values()) if aois else {"unmapped"}
    features, all_fixations, dwell_rows, transitions = [], [], [], []
    all_steps = []
    for trigger, view in views.items():
        if trigger not in offsets:
            continue
        stimulus, digest = stimuli[view["stimulus"]]
        rows = grouped[trigger]
        identity = dict(participant=metadata["participant"]["participant"], session=Path(base).name,
                        trigger=trigger, trial=view["trial"], stimulus=stimulus,
                        stimulus_sha256=digest, is_practice=view["is_practice"])
        sample_durations(rows, max_gap_ms)
        dwell = defaultdict(float, {name: 0.0 for name in sorted(aoi_names)})
        for r in rows:
            r["aoi"] = aoi_map.at(stimulus, r["image_x"], r["image_y"]) if r["usable"] else "invalid_or_outside"
            dwell[r["aoi"]] += r["dwell_ms"]
        fixations = detect_fixations(rows, dispersion, min_duration_ms, max_gap_ms)
        for f in fixations:
            f["aoi"] = aoi_map.at(stimulus, f["image_x"], f["image_y"])
            all_fixations.append(dict(identity, **f))
        path_summary, steps = scanpath(fixations, max_gap_ms)
        all_steps.extend(dict(identity, **step) for step in steps)
        # Transitions only within continuous observation and an unchanged display transform.
        for a, b in zip(fixations, fixations[1:]):
            if continuous(a, b, max_gap_ms) and a["aoi"] != b["aoi"]:
                transitions.append(dict(identity, from_aoi=a["aoi"], to_aoi=b["aoi"],
                                        at_ms=b["start_ms"], from_fixation=a["fixation"], to_fixation=b["fixation"]))
        view_ms = (offsets[trigger]["timestamp_ns"] - view["timestamp_ns"]) / 1e6
        observed = sum(dwell.values())
        for aoi, duration in dwell.items():
            matching = [f for f in fixations if f["aoi"] == aoi]
            dwell_rows.append(dict(identity, aoi=aoi, dwell_ms=duration,
                                   fraction_observed=duration / observed if observed else "",
                                   fraction_view=duration / view_ms if view_ms else "",
                                   time_to_first_fixation_ms=matching[0]["start_ms"] if matching else ""))
        response = responses.get(trigger, {})
        path_length = sum(math.hypot(b["image_u"] - a["image_u"], b["image_v"] - a["image_v"])
                          for a, b in zip(fixations, fixations[1:])
                          if continuous(a, b, max_gap_ms))
        aoi_names.update(dwell)
        features.append(dict(identity, condition=view["condition"],
                             **{k: response.get(k, "") for k in
                                ("difficulty", "response", "correct", "confidence", "rt_classification", "drift_px")},
                             interrupted=int(offsets[trigger].get("interrupted", False)),
                             view_ms=view_ms, sample_count=len(rows),
                             valid_fraction=sum(r["valid"] for r in rows) / len(rows) if rows else 0,
                             on_image_fraction=sum(r["usable"] for r in rows) / len(rows) if rows else 0,
                             observed_dwell_ms=observed,
                             observed_fraction=observed / view_ms if view_ms else 0,
                             fixation_count=len(fixations),
                             scanpath_length_image_units=path_length,
                             mean_fixation_ms=statistics.mean(f["duration_ms"] for f in fixations) if fixations else "",
                             first_fixation_aoi=fixations[0]["aoi"] if fixations else "",
                             aoi_sequence=" > ".join(f["aoi"] for f in fixations),
                             max_zoom=max(e["zoom"] for e in events if e["kind"] == "present" and e["trigger"] == trigger)))
        features[-1].update(path_summary)
        for aoi, duration in dwell.items():
            features[-1][aoi + "_dwell_ms"] = duration
            features[-1][aoi + "_dwell_fraction_observed"] = duration / observed if observed else ""
        if plots:
            plot_trial(stimulus, rows, fixations, output / f"trial_{trigger}_qc.png", f"Trial {trigger}", max_gap_ms)
    write_csv(tables / "samples.csv", samples, list(samples[0]) if samples else ["timestamp", "trigger", "usable", "image_x", "image_y", "aoi"])
    for feature in features:
        for aoi in sorted(aoi_names):
            feature.setdefault(aoi + "_dwell_ms", 0)
            feature.setdefault(aoi + "_dwell_fraction_observed", 0 if feature["observed_dwell_ms"] else "")
    write_csv(tables / "features.csv", features, None if features else ["participant", "trigger", "sample_count"])
    write_csv(tables / "fixations.csv", all_fixations, list(all_fixations[0]) if all_fixations else ["participant", "trigger", "fixation", "start_ms", "duration_ms", "aoi"])
    write_csv(tables / "aoi_dwell.csv", dwell_rows, list(dwell_rows[0]) if dwell_rows else ["participant", "trigger", "aoi", "dwell_ms"])
    write_csv(tables / "transitions.csv", transitions, list(transitions[0]) if transitions else ["participant", "trigger", "from_aoi", "to_aoi", "at_ms"])
    write_csv(tables / "scanpaths.csv", all_steps, list(all_steps[0]) if all_steps else
              ["participant", "session", "trigger", "path_segment", "from_fixation", "to_fixation", "distance_image_units"])
    provenance = dict(schema_version=1, source_session=base, coordinate_source=coordinate_source,
                      dispersion=dispersion, min_duration_ms=min_duration_ms, max_gap_ms=max_gap_ms,
                      aois=str(aois) if aois else None, allow_unreviewed=allow_unreviewed,
                      tables_directory="data_results",
                      heatmap_display=dict(bins=HEATMAP_BINS, gaussian_sigma_cells=HEATMAP_SIGMA, scale="per_trial"),
                      skipped_unclosed_trials=sorted(set(views) - set(offsets)),
                      input_sha256={suffix: sha256(base + suffix) for suffix in
                                    ("_gaze_raw.csv", "_display.jsonl", "_session.json")})
    if response_path.exists():
        provenance["input_sha256"][".csv"] = sha256(response_path)
    if aois:
        provenance["aoi_manifest_sha256"] = sha256(aois)
        provenance["aoi_masks_sha256"] = {name: sha256(Path(aois).parent / r["mask"])
                                            for name, r in aoi_map.manifest["stimuli"].items()}
    (output / "analysis.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return features


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("session_base")
    p.add_argument("--output", required=True)
    p.add_argument("--aois")
    p.add_argument("--allow-unreviewed", action="store_true")
    p.add_argument("--coordinate-source", choices=("calibrated", "filtered"), default="calibrated")
    p.add_argument("--dispersion", type=float, default=0.08)
    p.add_argument("--min-duration-ms", type=float, default=100)
    p.add_argument("--max-gap-ms", type=float, default=100)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()
    if min(args.dispersion, args.min_duration_ms, args.max_gap_ms) <= 0:
        p.error("fixation parameters must be positive")
    process(args.session_base, args.output, args.aois, args.allow_unreviewed,
            args.coordinate_source, args.dispersion, args.min_duration_ms,
            args.max_gap_ms, not args.no_plots)


if __name__ == "__main__":
    main()
