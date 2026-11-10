"""Compare contour mapping against cached raw model outputs, without rerunning models."""
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from analysis.detailed_aois import build_regions, LABELS
from analysis.face_aois import preview, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    from analyse import gallery
    baseline = ROOT/"processed/aoi_diagnostics_20261110_183148"
    output = ROOT/"processed"/("detailed_v4_validation_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir()
    config = json.loads((ROOT/"models/face-parsing/config.json").read_text())
    old_root = ROOT/"aois/test/detailed-v3"
    old_manifest = json.loads((old_root/"manifest.json").read_text())
    items, metrics = [], []
    with (baseline/"selection.csv").open(encoding="utf-8-sig") as f:
        selection = list(csv.DictReader(f))
    for folder, row in zip(sorted(p for p in baseline.iterdir() if p.is_dir()), selection):
        source = ROOT/row["stimulus"]
        if not folder.name.endswith(source.stem):
            raise ValueError("Diagnostic selection/folder mismatch")
        points = np.genfromtxt(folder/"landmarks.csv", delimiter=",", names=True)
        points = np.array([points["image_x"],points["image_y"],points["relative_z"]]).T
        with np.load(folder/"prediction_512.npz") as saved:
            raw, confidence = saved["labels"], saved["confidence"]
        new, flags = build_regions(raw, confidence, config["id2label"], points)
        Image.fromarray(new).save(output/(source.stem+"_labels.png"))
        old = np.asarray(Image.open(old_root/old_manifest["stimuli"][sha256(source)]["mask"]))
        confident_primary = (old < 19) & (old != 1)
        # All native feature pixels must be preserved exactly, including sided IDs.
        primary_changes = int(np.sum(new[confident_primary] != old[confident_primary]))
        invalid_derived = int(np.sum(((new>=19)&(new<=24)) & (old!=1) & ~((old>=19)&(old<=24))))
        if primary_changes or invalid_derived or np.any(new[old==255]!=255):
            raise AssertionError("Primary/uncertainty preservation failed: "+source.name)
        image = Image.open(source).convert("RGB")
        before, after = output/(source.stem+"_before.png"), output/(source.stem+"_after.png")
        preview(image, old, before, LABELS)
        preview(image, new, after, LABELS)
        solid = output/(source.stem+"_solid.png")
        preview(image, new, solid, LABELS, solid=True)
        items.extend([(source.name+" v3",before,True,"Historical ellipse AOIs"),
            (source.name+" v4",after,True,"; ".join(flags)),
            (source.name+" opaque",solid,True,"Grey is uncertain; colours match legend exactly")])
        record = dict(stimulus=source.name, primary_changed_pixels=primary_changes,
            invalid_derived_pixels=invalid_derived, uncertain_pixels=int((new==255).sum()), flags=flags)
        record["primary_components"] = {LABELS[k]: int(cv2.connectedComponents((new==k).astype("uint8"), connectivity=8)[0]-1)
            for k in (2,4,5,6,7,8,9,10,11,12,13)}
        record["raw_class_retention"] = {name: dict(raw_pixels=int((raw==int(k)).sum()),
            confident_pixels=int(((raw==int(k))&(confidence>=.6)).sum()))
            for k,name in config["id2label"].items()}
        for k in range(19,25):
            record[LABELS[k]] = dict(before=int((old==k).sum()),after=int((new==k).sum()))
        metrics.append(record)
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2))
    (output/"provenance.json").write_text(json.dumps(dict(
        validation_script_sha256=sha256(Path(__file__)),
        geometry_sha256=sha256(ROOT/"analysis/detailed_aois.py"),
        cached_model_provenance_sha256=sha256(baseline/"provenance.json"),
        historical_manifest_sha256=sha256(old_root/"manifest.json"),
        inference="same cached full-frame 512 predictions; confidence cutoff 0.6",
        anatomical_reference_masks_available=False),indent=2))
    generated_path = ROOT/"aois/test/detailed-v4/manifest.json"
    if generated_path.exists():
        generated = json.loads(generated_path.read_text())
        checks = []
        for digest, record in generated["stimuli"].items():
            old = np.asarray(Image.open(old_root/old_manifest["stimuli"][digest]["mask"]))
            new = np.asarray(Image.open(generated_path.parent/record["mask"]))
            primary = (old<19)&(old!=1)
            changed = int(np.sum(new[primary]!=old[primary]))
            uncertainty_changed = int(np.sum(new[old==255]!=255))
            bad_derived = int(np.sum(((new>=19)&(new<=24)) & ~((old==1)|((old>=19)&(old<=24)))))
            if changed or uncertainty_changed or bad_derived or set(np.unique(new))-set(LABELS):
                raise AssertionError("Generated-set integrity failure: "+record["stimulus"])
            checks.append(dict(stimulus=record["stimulus"],primary_changed_pixels=changed,
                uncertainty_changed_pixels=uncertainty_changed,invalid_derived_pixels=bad_derived))
        (output/"generated_set_checks.json").write_text(json.dumps(checks,indent=2))
    gallery(output/"index.html","Detailed AOI contour validation",
        "Eight faces, same cached raw parser predictions and mesh. Exact preservation checks are integrity checks, not anatomical accuracy. Before/after/opaque maps are shown per image.",items)
    print(output, flush=True)


if __name__ == "__main__":
    main()
