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
    """A local report with compact table links and image cards."""
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
        label, note = escape(label), escape(note)
        if is_image:
            cards.append('<article><header><h2>' + label + '</h2><a href="' + url +
                         '">Open image</a></header><a href="' + url + '"><img src="' +
                         url + '" alt="' + label + '" loading="lazy"></a><p>' + note + '</p></article>')
        else:
            links.append('<a class="table" href="' + url + '" download><strong>' + label +
                         '</strong><span>' + note + '</span><small>Download CSV &#8599;</small></a>')
    downloads = ('<section class="downloads"><h2>Data tables</h2>'
                 '<p>CSV files are saved in the data_results folder alongside this report.</p>'
                 '<div class="table-grid">' + ''.join(links) + '</div></section>') if links else ''
    document = ('<!doctype html><html lang="en-GB"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<title>' + escape(title) + '</title><style>'
                '*{box-sizing:border-box}body{font:16px/1.55 system-ui,sans-serif;max-width:1200px;'
                'margin:36px auto;padding:0 24px;background:#f3f6fa;color:#18263b}'
                'h1{font-size:clamp(23px,3vw,32px);overflow-wrap:anywhere;line-height:1.25}'
                'h2{font-size:19px;margin:0}p{color:#53667e}a{color:#12617e}'
                '.intro{max-width:850px;margin-bottom:28px}.downloads,article{background:white;'
                'border:1px solid #dce4ee;border-radius:14px;margin:24px 0;overflow:hidden}'
                '.downloads{padding:22px}.downloads p{margin-top:6px;font-size:14px}'
                '.table-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px}'
                '.table{display:flex;flex-direction:column;gap:7px;padding:15px;background:#f7f9fc;'
                'border:1px solid #e3e9f0;border-radius:9px;text-decoration:none}'
                '.table:hover{border-color:#16809d;background:#edf8fb}.table span{font-size:13px;color:#53667e}'
                '.table small{font-size:12px;margin-top:auto}.table strong{font-size:15px}'
                'article header{display:flex;justify-content:space-between;align-items:center;gap:12px;'
                'padding:18px 22px}article header a{font-size:13px;white-space:nowrap}'
                'img{display:block;width:100%;height:auto}article p{padding:0 22px;font-size:14px}'
                '</style><h1>' + escape(title) + '</h1><p class="intro">' + escape(introduction) +
                '</p>' + downloads + ''.join(cards) + '</html>')
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
