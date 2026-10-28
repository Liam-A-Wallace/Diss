"""Short entry point: python analyse.py [aois | review | SESSION] [--test]."""
import argparse
import csv
from datetime import datetime, timezone
from html import escape
import importlib.util
import json
from pathlib import Path
import webbrowser

ROOT = Path(__file__).resolve().parent

REPORT_STYLE = """
* { box-sizing: border-box; }
body { max-width: 1080px; margin: 48px auto; padding: 0 32px;
       background: #fff; color: #242424; font: 15px/1.6 Arial, sans-serif; }
h1, h2 { color: #171717; font-family: Georgia, serif; font-weight: normal; }
h1 { margin: 0; font-size: 36px; line-height: 1.2; }
h2 { margin: 0; font-size: 24px; }
a { color: #234e70; text-underline-offset: 3px; }
a:hover { color: #111; }
a:focus-visible { outline: 2px solid #234e70; outline-offset: 4px; }
.report-heading { padding-bottom: 20px; border-bottom: 2px solid #242424; }
.session { margin: 12px 0 0; font: 12px/1.6 Consolas, monospace;
           color: #666; overflow-wrap: anywhere; }
.intro { max-width: 780px; margin: 20px 0 36px; }
.downloads { margin: 0 0 48px; }
.downloads p { margin: 6px 0 18px; color: #666; font-size: 13px; }
table { width: 100%; border-collapse: collapse; text-align: left; }
th { font-size: 12px; font-weight: normal; color: #666; border-bottom: 1px solid #888; }
th, td { padding: 10px 16px 10px 0; vertical-align: top; }
td { border-bottom: 1px solid #e3e3e3; }
td:first-child { width: 28%; white-space: nowrap; font-family: Consolas, monospace; font-size: 13px; }
td:last-child { color: #555; }
article { margin: 44px 0 56px; break-inside: avoid; }
article header { display: flex; justify-content: space-between; align-items: baseline;
                 gap: 16px; border-bottom: 1px solid #bbb; padding-bottom: 10px; margin-bottom: 16px; }
article header a { font-size: 12px; white-space: nowrap; }
img { display: block; width: 100%; height: auto; }
article p { margin: 12px 0 0; color: #666; font-size: 13px; max-width: 800px; }
@media (max-width: 600px) {
  body { margin: 24px auto; padding: 0 18px; font-size: 14px; }
  h1 { font-size: 30px; } h2 { font-size: 22px; }
  td:first-child { width: auto; white-space: normal; overflow-wrap: anywhere; }
  th, td { padding-right: 12px; }
}
@media print {
  body { max-width: none; margin: 0; padding: 0; color: #000; }
  article header a { display: none; }
  a { color: #000; }
}
"""


def latest_session(data_dir):
    paths = list(Path(data_dir).glob("*_session.json"))
    if not paths:
        raise ValueError("No zoom-aware sessions found. Run the experiment first.")
    path = max(paths, key=lambda p: p.stat().st_mtime_ns)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("status") == "running":
        raise ValueError("The latest session is still marked as running. Close the experiment first, or specify an older session.")
    base = Path(str(path)[:-len("_session.json")])
    for suffix in ("_display.jsonl", "_gaze_raw.csv"):
        if not Path(str(base) + suffix).is_file():
            raise ValueError("Session file missing: " + str(base) + suffix)
    return base


def unused_output(base):
    output = ROOT / "processed" / base.name
    number = 2
    while output.exists():
        output = ROOT / "processed" / (base.name + "_" + str(number))
        number += 1
    return output


def gallery(path, title, introduction, items):
    """A local research report with CSV links and full-width figures."""
    path = Path(path)
    links, cards = [], []
    for label, target, is_image, note in items:
        target = Path(target).resolve()
        try:
            from urllib.parse import quote
            url = quote(target.relative_to(path.parent.resolve()).as_posix())
        except ValueError:
            url = target.as_uri()
        url = escape(url, quote=True)
        if is_image and target.stem.startswith("trial_") and target.stem.endswith("_qc"):
            label = "Trial " + target.stem[len("trial_"):-len("_qc")]
        label, note = escape(label), escape(note)
        if is_image:
            cards.append('<article><header><h2>' + label + '</h2><a href="' + url +
                         '">Open image</a></header><a href="' + url + '"><img src="' +
                         url + '" alt="' + label + '" loading="lazy"></a><p>' + note + '</p></article>')
        else:
            links.append('<tr><td><a href="' + url + '" download>' + label +
                         '</a></td><td>' + note + '</td></tr>')
    downloads = ('<section class="downloads"><h2>Data tables</h2>'
                 '<p>CSV files are saved in the data_results folder alongside this report.</p>'
                 '<table><thead><tr><th scope="col">File</th><th scope="col">Contents</th></tr></thead>'
                 '<tbody>' + ''.join(links) + '</tbody></table></section>') if links else ''
    heading, separator, session = title.partition(": ")
    report_heading = ('<header class="report-heading"><h1>' + escape(heading) + '</h1>' +
                      ('<p class="session">' + escape(session) + '</p>' if separator else '') + '</header>')
    document = ('<!doctype html><html lang="en-GB"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>' + escape(title) + '</title><style>'
                + REPORT_STYLE + '</style></head><body><main>' + report_heading +
                '<p class="intro">' + escape(introduction) + '</p>' + downloads +
                ''.join(cards) + '</main></body></html>')
    path.write_text(document, encoding="utf-8")
    return path


def stimulus_paths(data_dir):
    """Use the session's exact main/practice set, or the configured lists before collection."""
    from analysis.ingest import resolve_stimulus
    if list(Path(data_dir).glob("*_session.json")):
        base = latest_session(data_dir)
        metadata = json.loads(Path(str(base) + "_session.json").read_text(encoding="utf-8"))
        return [resolve_stimulus(r["stimulus"], metadata)
                for r in metadata["planned_trials"] + metadata["practice"]]
    from config import TRIALS_CSV, PRACTICE_CSV
    paths = []
    for name in (TRIALS_CSV, PRACTICE_CSV):
        source = ROOT / name
        if not source.exists() and name == PRACTICE_CSV:
            continue
        with source.open(newline="", encoding="utf-8-sig") as f:
            paths.extend(ROOT / row["stimulus"] for row in csv.DictReader(f))
    return paths


def make_aois(data_dir, aoi_dir):
    from analysis.face_aois import MODEL, generate
    for module in ("torch", "transformers", "huggingface_hub"):
        if importlib.util.find_spec(module) is None:
            raise ValueError("AOI generation needs a one-off installation: python -m pip install -r requirements-aois.txt")
    paths = stimulus_paths(data_dir)
    for path in paths:
        if not path.is_file():
            raise ValueError("Stimulus missing: " + str(path))
    model_dir = ROOT / "models" / "face-parsing"
    if not (model_dir / "config.json").is_file() or not list(model_dir.glob("*.safetensors")):
        from huggingface_hub import snapshot_download
        print("Downloading the face-parsing model once. Stimulus images stay on this computer.")
        snapshot_download(MODEL, local_dir=model_dir, allow_patterns=["*.json", "*.safetensors"])
    generate(paths, model_dir, aoi_dir)
    print("AOIs are ready to inspect. Next: python analyse.py review" + (" --test" if data_dir.name == "test" else ""))


def review_aois(aoi_dir, open_report=True):
    from analysis.face_aois import preview, save_manifest, sha256
    import numpy as np
    from PIL import Image
    path = aoi_dir / "manifest.json"
    if not path.is_file():
        raise ValueError("No AOIs yet. Run python analyse.py aois first (add --test for test sessions).")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    items, reviewed_hashes = [], {}
    for digest, record in manifest["stimuli"].items():
        source = ROOT / record["stimulus"]
        if sha256(source) != digest:
            raise ValueError("Stimulus changed since AOI generation: " + str(source))
        mask_path = aoi_dir / record["mask"]
        with Image.open(source) as im, Image.open(mask_path) as mask:
            if mask.mode not in ("L", "P") or mask.size != im.size:
                raise ValueError("Invalid label mask for " + source.name)
            pixels = np.asarray(mask)
            if set(np.unique(pixels)) - set(map(int, manifest["labels"])):
                raise ValueError("Unknown label IDs in " + str(mask_path))
            preview_path = aoi_dir / (digest + "_review.png")
            preview(im.convert("RGB"), pixels, preview_path)
        reviewed_hashes[digest] = sha256(mask_path)
        items.append((source.name, preview_path, True, " ".join(record.get("notes", []))))
    report = gallery(aoi_dir / "index.html", "Review facial AOIs",
                     "Inspect every image. Forehead and hairline are approximations. Return to the terminal after checking the masks.", items)
    print("AOI gallery:", report)
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    answer = input("After checking every overlay, type reviewed to approve this set (Enter leaves it unchanged): ").strip().lower()
    if answer != "reviewed":
        print("Review status unchanged.")
        return
    for digest, record in manifest["stimuli"].items():
        if sha256(aoi_dir / record["mask"]) != reviewed_hashes[digest]:
            raise ValueError("A mask changed while reviewing. Run the review command again.")
        record.update(reviewed=True, reviewed_mask_sha256=reviewed_hashes[digest],
                      reviewed_at=datetime.now(timezone.utc).isoformat())
    save_manifest(path, manifest)
    print("AOI review saved. Run python analyse.py" + (" --test" if aoi_dir.name == "test" else ""))


def analyse(base, aoi_dir, plots=True, open_report=True):
    from analysis.process import process
    output = unused_output(base)
    manifest = aoi_dir / "manifest.json"
    aois = manifest if manifest.is_file() else None
    print("Session:", base.name)
    print("AOIs:", manifest if aois else "not generated; gaze plots and CSVs will still be produced")
    if plots and importlib.util.find_spec("matplotlib") is None:
        print("Plots unavailable. Saving CSVs; install plots with: python -m pip install -r requirements.txt")
        plots = False
    features = process(base, output, aois=aois, plots=plots)
    descriptions = {"features.csv": "Responses, quality and trial summaries",
                    "samples.csv": "Individual gaze samples and coordinates",
                    "fixations.csv": "Fixation locations, order and duration",
                    "aoi_dwell.csv": "Time spent in each facial region",
                    "scanpaths.csv": "Fixation-to-fixation distances, directions and timing",
                    "transitions.csv": "Changes between facial regions"}
    items = [(p.name, p, False, descriptions.get(p.name, "Analysis table"))
             for p in sorted((output / "data_results").glob("*.csv"))]
    items += [(p.name, p, True, "Smoothed dwell overlay and numbered exploratory fixations; arrows stop at gaps or view changes.") for p in sorted(output.glob("*_qc.png"))]
    report = gallery(output / "index.html", "Gaze analysis: " + base.name,
                     str(len(features)) + " trials. Practice and interrupted trials are flagged in the tables. " +
                     ("Reviewed AOI masks used." if aois else "No AOIs used; regions are labelled unmapped."), items)
    print("Results:", report)
    print("CSV folder:", output / "data_results")
    if open_report:
        webbrowser.open(report.resolve().as_uri())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyse the latest experiment, generate AOIs, or review them.")
    parser.add_argument("target", nargs="?", default="latest", help="latest (default), aois, review, or a session base path")
    parser.add_argument("--test", action="store_true", help="use data/test and aois/test")
    parser.add_argument("--no-open", action="store_true", help="save the report without opening a browser")
    parser.add_argument("--no-plots", action="store_true", help="CSV analysis only")
    args = parser.parse_args(argv)
    data_dir = ROOT / "data" / "test" if args.test else ROOT / "data"
    aoi_dir = ROOT / "aois" / ("test" if args.test else "main")
    try:
        if args.target == "aois":
            make_aois(data_dir, aoi_dir)
        elif args.target == "review":
            review_aois(aoi_dir, not args.no_open)
        else:
            base = latest_session(data_dir) if args.target == "latest" else ROOT / args.target
            analyse(base, aoi_dir, not args.no_plots, not args.no_open)
    except ModuleNotFoundError as error:
        parser.exit(1, "Missing dependency: " + str(error.name) + ". Install: python -m pip install -r requirements.txt\n")
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
