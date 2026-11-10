"""Read-only model comparison; never generates or approves production AOIs.

python -m tools.diagnose_aois
Compares the same images at 512/1024 and thresholds .4/.5/.6. Saved v2 masks
are historical comparison artifacts, not outputs of the rolled-back pipeline.
"""

import argparse
import csv
from datetime import datetime
from html import escape
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FEATURES = {
    "right_eye": [
        33,
        7,
        163,
        144,
        145,
        153,
        154,
        155,
        133,
        173,
        157,
        158,
        159,
        160,
        161,
        246,
    ],
    "left_eye": [
        263,
        249,
        390,
        373,
        374,
        380,
        381,
        382,
        362,
        398,
        384,
        385,
        386,
        387,
        388,
        466,
    ],
    "right_brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "left_brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "nose": [
        168,
        193,
        122,
        196,
        3,
        51,
        45,
        220,
        115,
        48,
        64,
        98,
        97,
        2,
        326,
        327,
        294,
        278,
        344,
        440,
        275,
        281,
        248,
        419,
        351,
        417,
    ],
    "mouth": [
        61,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        409,
        270,
        269,
        267,
        0,
        37,
        39,
        40,
        185,
    ],
}


def summary(labels, confidence, id2label, threshold):
    import cv2

    rows = []
    for k, name in id2label.items():
        raw = labels == int(k)
        retained = raw & (confidence >= threshold)
        count = int(raw.sum())
        components = (
            cv2.connectedComponents(retained.astype("uint8"), connectivity=8)[0] - 1
        )
        rows.append(
            dict(
                feature=name,
                raw_pixels=count,
                retained_pixels=int(retained.sum()),
                retained_fraction=float(retained.sum() / count) if count else "",
                mean_confidence=float(confidence[raw].mean()) if count else "",
                components=int(components),
            )
        )
    return rows


def overlay(image, labels):
    rgb = np.asarray(image).copy()
    colour = np.zeros_like(rgb)
    for k in np.unique(labels):
        colour[labels == k] = (
            (int(k) * 83 + 50) % 256,
            (int(k) * 137 + 80) % 256,
            (int(k) * 47 + 130) % 256,
        )
    colour[labels == 255] = (255, 220, 0)
    return Image.fromarray((rgb * 0.58 + colour * 0.42).astype("uint8"))


def face_crop_prediction(image, points, processor, model, torch):
    """Run the same parser on a landmark-bounded square crop; map back to source pixels."""
    if points is None:
        raise ValueError("Face crop inference requires exactly one detected face")
    width, height = image.size
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    centre = (low + high) / 2
    side = max(high - low) * 1.25
    side = int(side)
    left = round(centre[0] - side / 2)
    top = round(centre[1] - side / 2)
    crop = image.crop((left, top, left + side, top + side))
    inputs = processor(
        images=crop, return_tensors="pt", size={"height": 512, "width": 512}
    )
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = torch.nn.functional.interpolate(
            logits, size=(side, side), mode="bilinear", align_corners=False
        )
        confidence, labels = logits.softmax(dim=1).max(dim=1)
    full_labels = np.zeros((height, width), dtype="uint8")
    full_confidence = np.zeros((height, width), dtype="float32")
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + side), min(height, top + side)
    cx0, cy0 = x0 - left, y0 - top
    full_labels[y0:y1, x0:x1] = (
        labels[0, cy0 : cy0 + y1 - y0, cx0 : cx0 + x1 - x0].numpy().astype("uint8")
    )
    full_confidence[y0:y1, x0:x1] = confidence[
        0, cy0 : cy0 + y1 - y0, cx0 : cx0 + x1 - x0
    ].numpy()
    return full_labels, full_confidence, (left, top, side)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(output):
    import torch
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from analysis.face_aois import sha256

    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=False)
    # Fixed paired selection covers previously reported failures, not random
    # population sampling. It cannot estimate population-level model accuracy.
    wanted = {
        "real_001.png": "earlier brow/core-feature complaint",
        "real_003.png": "real comparator",
        "real_007.png": "small confident eye region in earlier diagnostic",
        "real_009.png": "small confident eye region in earlier diagnostic",
        "pair_001_A.png": "generated comparator",
        "pair_002_B.png": "glasses and missing confident eye",
        "pair_004_B.png": "higher uncertainty in earlier diagnostic",
        "pair_008_B.png": "unequal eye coverage in earlier diagnostic",
    }
    with (ROOT / "trials.csv").open(newline="", encoding="utf-8-sig") as f:
        selection = [
            dict(r, reason=wanted[Path(r["stimulus"]).name])
            for r in csv.DictReader(f)
            if Path(r["stimulus"]).name in wanted
        ]
    if len(selection) != 8:
        raise ValueError("Expected eight diagnostic images in trials.csv")
    write_csv(output / "selection.csv", selection)
    model_dir = ROOT / "models/face-parsing"
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_dir, local_files_only=True
    ).eval()
    processor = SegformerImageProcessor.from_pretrained(
        model_dir, local_files_only=True
    )
    landmark_path = (
        ROOT.parent / "GazeFollower/gazefollower/res/model_weights/face_landmarker.task"
    )
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(landmark_path)),
        num_faces=2,
        running_mode=vision.RunningMode.IMAGE,
    )
    old_path = ROOT / "aois/test/detailed-v2/manifest.json"
    old = (
        json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else None
    )
    metrics, comparisons, manual, sections, records = [], [], [], [], []
    palette = "; ".join(f"{k}: {name}" for k, name in model.config.id2label.items())
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for index, row in enumerate(selection, 1):
            source = ROOT / row["stimulus"]
            digest = sha256(source)
            folder = output / f"{index:02d}_{source.stem}"
            folder.mkdir()
            image = Image.open(source).convert("RGB")
            image.save(folder / "original.png")
            result = detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
            )
            points = None
            mesh = image.copy()
            draw = ImageDraw.Draw(mesh)
            if len(result.face_landmarks) == 1:
                points = np.array(
                    [
                        (p.x * image.width, p.y * image.height, p.z)
                        for p in result.face_landmarks[0]
                    ]
                )
                write_csv(
                    folder / "landmarks.csv",
                    [
                        dict(
                            id=i,
                            image_x=float(p[0]),
                            image_y=float(p[1]),
                            relative_z=float(p[2]),
                        )
                        for i, p in enumerate(points)
                    ],
                )
                for p in points:
                    x, y = p[:2]
                    draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill="cyan")
                for name, ids in FEATURES.items():
                    draw.line(
                        [tuple(p[:2]) for p in points[ids + [ids[0]]]],
                        fill="yellow",
                        width=2,
                    )
            mesh.save(folder / "landmarks.png")
            crop_prediction = None
            if points is not None:
                crop_start = time.monotonic()
                crop_labels, crop_confidence, crop_box = face_crop_prediction(
                    image, points, processor, model, torch
                )
                crop_prediction = crop_labels
                np.savez_compressed(
                    folder / "prediction_facecrop_512.npz",
                    labels=crop_labels,
                    confidence=crop_confidence,
                )
                overlay(image, crop_labels).save(folder / "raw_facecrop_512.png")
                Image.fromarray((crop_confidence * 255).astype("uint8")).save(
                    folder / "confidence_facecrop_512.png"
                )
                for threshold in (0.4, 0.5, 0.6):
                    masked = crop_labels.copy()
                    masked[crop_confidence < threshold] = 255
                    overlay(image, masked).save(
                        folder / f"threshold_facecrop_512_{threshold}.png"
                    )
                    for stat in summary(
                        crop_labels, crop_confidence, model.config.id2label, threshold
                    ):
                        metrics.append(
                            dict(
                                stimulus=row["stimulus"],
                                resolution="facecrop512",
                                threshold=threshold,
                                **stat,
                            )
                        )
                print(
                    f"{index}/8 {source.name} face crop 512: {time.monotonic()-crop_start:.1f}s",
                    flush=True,
                )
            predictions = {}
            for resolution in (512, 1024):
                start = time.monotonic()
                inputs = processor(
                    images=image,
                    return_tensors="pt",
                    size={"height": resolution, "width": resolution},
                )
                with torch.inference_mode():
                    logits = model(**inputs).logits
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=(image.height, image.width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    confidence, labels = logits.softmax(1).max(1)
                labels, confidence = (
                    labels[0].numpy().astype("uint8"),
                    confidence[0].numpy(),
                )
                predictions[resolution] = labels
                np.savez_compressed(
                    folder / f"prediction_{resolution}.npz",
                    labels=labels,
                    confidence=confidence,
                )
                overlay(image, labels).save(folder / f"raw_{resolution}.png")
                # Fixed 0..1 scale across all faces and both resolutions.
                Image.fromarray((confidence * 255).astype("uint8")).save(
                    folder / f"confidence_{resolution}.png"
                )
                for threshold in (0.4, 0.5, 0.6):
                    masked = labels.copy()
                    masked[confidence < threshold] = 255
                    overlay(image, masked).save(
                        folder / f"threshold_{resolution}_{threshold}.png"
                    )
                    for stat in summary(
                        labels, confidence, model.config.id2label, threshold
                    ):
                        metrics.append(
                            dict(
                                stimulus=row["stimulus"],
                                resolution=resolution,
                                threshold=threshold,
                                **stat,
                            )
                        )
                print(
                    f"{index}/8 {source.name} {resolution}: {time.monotonic()-start:.1f}s",
                    flush=True,
                )
            for k, name in model.config.id2label.items():
                a, b = predictions[512] == int(k), predictions[1024] == int(k)
                denom = int(a.sum() + b.sum())
                comparisons.append(
                    dict(
                        stimulus=row["stimulus"],
                        feature=name,
                        prediction_dice=(
                            float(2 * (a & b).sum() / denom) if denom else ""
                        ),
                        note="agreement between predictions, NOT accuracy",
                    )
                )
                if crop_prediction is not None:
                    a, b = predictions[512] == int(k), crop_prediction == int(k)
                    denom = int(a.sum() + b.sum())
                    comparisons.append(
                        dict(
                            stimulus=row["stimulus"],
                            feature=name,
                            prediction_dice=(
                                float(2 * (a & b).sum() / denom) if denom else ""
                            ),
                            note="full-frame 512 vs face-crop 512 agreement, NOT accuracy",
                        )
                    )
            old_record = old["stimuli"].get(digest) if old else None
            old_note = "Saved detailed-v2 unavailable"
            if old_record:
                with Image.open(old_path.parent / old_record["mask"]) as im:
                    old_mask = np.asarray(im)
                if old_mask.shape != (image.height, image.width):
                    raise ValueError("Saved AOI dimensions differ")
                overlay(image, old_mask).save(folder / "saved_detailed_v2.png")
                old_note = "Historical detailed-v2 (unchanged); label IDs differ from raw model."
            for feature in FEATURES:
                manual.append(
                    dict(
                        stimulus=row["stimulus"],
                        feature=feature,
                        landmark_alignment="unrated",
                        segmentation_512="unrated",
                        segmentation_1024="unrated",
                        reference_mask="",
                        reviewer="",
                        notes="",
                    )
                )
            records.append(
                dict(
                    stimulus=row["stimulus"],
                    sha256=digest,
                    face_count=len(result.face_landmarks),
                    resolution_disagreement=float(
                        np.mean(predictions[512] != predictions[1024])
                    ),
                )
            )
            rel = folder.name
            names = [
                ("Original", "original.png"),
                ("MediaPipe dots + feature contours", "landmarks.png"),
                ("Raw SegFormer face crop 512", "raw_facecrop_512.png"),
                ("Raw SegFormer 512 (no confidence cutoff)", "raw_512.png"),
                ("Raw SegFormer 1024 (no confidence cutoff)", "raw_1024.png"),
                ("Confidence 512: black=0, white=1", "confidence_512.png"),
                ("Confidence 1024: black=0, white=1", "confidence_1024.png"),
            ]
            names += [
                (f"{r}px threshold {t}; yellow=uncertain", f"threshold_{r}_{t}.png")
                for r in (512, 1024)
                for t in (0.4, 0.5, 0.6)
            ]
            if old_record:
                names.append((old_note, "saved_detailed_v2.png"))
            figures = "".join(
                f'<figure><figcaption>{escape(title)}</figcaption><a href="{rel}/{file}"><img loading="lazy" src="{rel}/{file}"></a></figure>'
                for title, file in names
            )
            sections.append(
                f'<section><h2>{escape(source.name)}</h2><p>{escape(row["reason"])}. Detected faces: {len(result.face_landmarks)}.</p><div class="grid">{figures}</div></section>'
            )
    write_csv(output / "feature_metrics.csv", metrics)
    write_csv(output / "resolution_agreement.csv", comparisons)
    write_csv(output / "manual_review.csv", manual)
    provenance = dict(
        selection=records,
        thresholds=[0.4, 0.5, 0.6],
        resolutions=[512, 1024],
        additional_experiment="512 square SegFormer inference on a square crop bounded by MediaPipe landmarks, with 25% margin, predictions projected back to source pixels",
        model_files={
            p.name: sha256(p)
            for p in model_dir.iterdir()
            if p.suffix in (".json", ".safetensors")
        },
        landmarker_sha256=sha256(landmark_path),
        historical_manifest_sha256=sha256(old_path) if old else None,
        torch_version=torch.__version__,
        mediapipe_version=mp.__version__,
        processor=processor.to_dict(),
        script_sha256=sha256(Path(__file__)),
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    intro = f'<h1>Face model diagnostics</h1><p>Eight fixed cases: four real, four OpenAI faces. No production masks or approvals changed. This is a failure-case comparison, not an accuracy benchmark.</p><p>Raw model label legend: {escape(palette)}</p><p>Confidence is the maximum softmax score, not a calibrated probability. Higher resolution is an experiment, not presumed improvement. <a href="manual_review.csv">Manual review template</a> | <a href="feature_metrics.csv">Feature metrics</a> | <a href="resolution_agreement.csv">Resolution agreement</a> | <a href="provenance.json">Provenance</a>.</p>'
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Face model diagnostics</title><style>body{font:16px Arial;margin:30px;color:#222}section{border-top:2px solid #bbb;margin-top:35px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}figure{margin:0}img{width:100%}figcaption{min-height:40px}p{max-width:1100px;line-height:1.5}</style>'
        + intro
        + "".join(sections),
        encoding="utf-8",
    )
    print("Report:", output / "index.html", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "processed"
        / ("aoi_diagnostics_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
    )
    run(p.parse_args().output)
