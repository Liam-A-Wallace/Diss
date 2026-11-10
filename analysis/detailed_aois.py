"""Detailed-v4: native parser features and contour-bounded skin regions.

Side names always refer to the displayed image. Derived regions are positional
approximations, not detections of eye bags, wrinkles or other appearance traits.
"""
import csv
import json
from pathlib import Path

import numpy as np

from analysis.face_aois import MODEL, preview, save_manifest, sha256

LABELS = dict(enumerate([
    "background", "other_skin", "nose", "glasses", "image_left_eye",
    "image_right_eye", "image_left_eyebrow", "image_right_eyebrow",
    "image_left_ear", "image_right_ear", "mouth_interior", "upper_lip",
    "lower_lip", "hair", "hat", "earring", "necklace", "neck", "clothing",
    "forehead_derived", "image_left_under_eye_derived", "image_right_under_eye_derived",
    "image_left_cheek_derived", "image_right_cheek_derived", "chin_derived"]))
LABELS[255] = "uncertain"
RAW_NAMES = ["background", "skin", "nose", "eye_g", "l_eye", "r_eye",
             "l_brow", "r_brow", "l_ear", "r_ear", "mouth", "u_lip", "l_lip",
             "hair", "hat", "ear_r", "neck_l", "neck", "cloth"]
PARENTS = {name: ("eyes" if k in (4, 5) else "eyebrows" if k in (6, 7)
    else "ears" if k in (8, 9) else "mouth" if k in (10, 11, 12)
    else "other_skin" if 19 <= k <= 24 else name) for k, name in LABELS.items()}
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,
        400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]


def build_regions(raw, confidence, id2label, points=None, threshold=0.6):
    """Return a non-overlapping label image and review flags. Never fill uncertainty."""
    from PIL import Image, ImageDraw
    import cv2
    raw = np.asarray(raw)
    confidence = np.asarray(confidence)
    mask = np.full(raw.shape, 255, dtype=np.uint8)
    for k, name in id2label.items():
        if name not in RAW_NAMES:
            raise ValueError("Unknown parser class: " + name)
        mask[(raw == int(k)) & (confidence >= threshold)] = RAW_NAMES.index(name)
    flags = []
    for name, ids in (("eyes", (4,5)), ("eyebrows", (6,7)), ("nose", (2,)), ("mouth", (10,11,12))):
        if not np.isin(mask, ids).any():
            flags.append("missing_confident_" + name)
    yy, xx = np.indices(raw.shape)
    # Without reliable geometry retain model laterality, explicitly flagging that
    # it is not the promised image-relative convention. Withhold those labels.
    if points is None or len(points) < 468 or not np.isfinite(points).all():
        mask[np.isin(mask, (4,5,6,7,8,9))] = 255
        return mask, flags + ["landmarks_unavailable: sided features and derived regions withheld"]
    p = np.asarray(points)[:, :2]
    centres = np.array([(p[33]+p[133])/2, (p[362]+p[263])/2])
    centres = centres[np.argsort(centres[:, 0])]
    distance = np.linalg.norm(centres[1]-centres[0])
    if distance < 8:
        mask[np.isin(mask, (4,5,6,7,8,9))] = 255
        return mask, flags + ["degenerate_eye_geometry: sided features and derived regions withheld"]
    u = (centres[1]-centres[0])/distance
    v = np.array([-u[1], u[0]])
    mid = centres.mean(axis=0)
    if np.dot(p[152]-mid, v) <= distance * .4:
        mask[np.isin(mask, (4,5,6,7,8,9))] = 255
        return mask, flags + ["unsupported_pose: sided features and derived regions withheld"]
    x = (xx-mid[0])*u[0] + (yy-mid[1])*u[1]
    y = (xx-mid[0])*v[0] + (yy-mid[1])*v[1]
    # Merge paired raw semantic classes, then assign the displayed-image side.
    # This repairs side-ID mixing without inventing additional feature pixels.
    for a, b in ((4,5), (6,7), (8,9)):
        pair = np.isin(mask, (a,b))
        mask[pair & (x < 0)] = a
        mask[pair & (x >= 0)] = b
    for k in (4,5,6,7,8,9):
        selected = (mask == k).astype("uint8")
        count, _, _, _ = cv2.connectedComponentsWithStats(selected, connectivity=8)
        if count > 2:
            flags.append(LABELS[k] + ": fragmented_prediction")
    widths = np.array([np.linalg.norm(p[33]-p[133]), np.linalg.norm(p[362]-p[263])])
    nose_x = np.dot(p[1]-mid, u)
    if widths.min()/max(widths.max(), 1) < .55 or abs(nose_x) > distance * .4:
        return mask, flags + ["pose_asymmetry: derived regions withheld"]
    def polygon(vertices):
        region = Image.new("L", (raw.shape[1], raw.shape[0]))
        ImageDraw.Draw(region).polygon([tuple(q) for q in vertices], fill=1)
        return np.asarray(region) > 0

    face = polygon(p[OVAL])
    skin = mask == 1
    # Include forehead skin beyond the mesh's upper oval only when connected
    # to facial skin. The parser's hair/hat/background boundary is the upper
    # limit; mesh point 10 is not a detected hairline.
    count, components = cv2.connectedComponents(skin.astype("uint8"), connectivity=8)
    supported = np.unique(components[skin & face])
    supported = supported[supported != 0]
    available = skin & np.isin(components, supported)

    def assign(label, candidate):
        accepted = candidate & available & (mask == 1)
        if accepted.sum() < max(4, distance**2 * .002):
            flags.append(LABELS[label] + ": insufficient_confident_skin")
        else:
            mask[accepted] = label

    # A piecewise brow contour separates forehead from the lower face. At the
    # sides it follows temple anchors; all coordinates are in the eye frame.
    brow_ids = [127,70,63,105,66,107,9,336,296,334,293,300,356]
    brow_points = p[brow_ids]
    bx = (brow_points-mid) @ u
    by = (brow_points-mid) @ v
    order = np.argsort(bx)
    brow_line = np.interp(x.ravel(), bx[order], by[order]).reshape(x.shape)
    face_x = (p[OVAL]-mid) @ u
    assign(19, (y < brow_line) & (x >= face_x.min()) & (x <= face_x.max()))
    if not (np.any(mask == 6) and np.any(mask == 7)):
        flags.append("forehead_derived: mesh_brow_boundary_requires_review")

    # Follow each actual lower eyelid, then form a curved ribbon towards the
    # upper cheek. Its inferior edge is an operational AOI boundary, not a
    # detected eye-bag contour. No parser feature or uncertain pixel is replaced.
    lids = [[33,7,163,144,145,153,154,155,133],
            [263,249,390,373,374,380,381,382,362]]
    lids.sort(key=lambda ids: p[ids].mean(axis=0)[0])
    for side, (centre, ids) in enumerate(zip(centres, lids)):
        local = ((xx-centre[0])**2 + (yy-centre[1])**2) < (distance*.3)**2
        if not np.any(local & (mask == 4+side)):
            flags.append(LABELS[20+side] + ": eye_parser_mesh_disagreement; withheld")
            continue
        lid = p[ids]
        eye_width = np.linalg.norm(lid[0]-lid[-1])
        depth = eye_width * (.18 + .22*np.sin(np.linspace(0,np.pi,len(lid))))
        bottom = lid + depth[:,None]*v
        assign(20+side, polygon(np.concatenate([lid,bottom[::-1]])) & face)

    # The lower lip is the superior chin boundary; the jaw contour is inferior.
    # Side joins run from mouth corners to the corresponding lower jaw.
    chin_ids = [61,146,91,181,84,17,314,405,321,375,291,
                367,397,365,379,378,400,377,152,148,176,149,150,136,172,138]
    assign(24, polygon(p[chin_ids]) & face)

    # Cheeks span from the lower eyes to jaw, bounded medially by the nose wings
    # and mouth corners. Skin around the lips remains other_skin. Curves adapt
    # to each side's observed face outline rather than equal-sized patches.
    cheek_loops = [[127,234,93,132,58,172,138,61,203,129,98,133,33],
                   [356,454,323,361,288,397,367,291,423,358,327,362,263]]
    cheek_loops.sort(key=lambda ids: p[ids].mean(axis=0)[0])
    for side, ids in enumerate(cheek_loops):
        assign(22+side, polygon(p[ids]) & face & (y >= brow_line))
    return mask, flags


def generate(stimuli, model_dir, output, min_confidence=0.6):
    import torch
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    from PIL import Image
    torch.set_num_threads(4)
    if not 0 <= min_confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    root = Path(__file__).resolve().parents[1]
    model_dir, output = Path(model_dir), Path(output)
    weights = root.parent / "GazeFollower/gazefollower/res/model_weights/face_landmarker.task"
    if not weights.is_file():
        raise ValueError("Detailed AOIs require local face_landmarker.task at " + str(weights))
    files = {p.name: sha256(p) for p in model_dir.iterdir() if p.suffix in (".json", ".bin", ".safetensors")}
    identity = dict(model_files=files, landmarker_sha256=sha256(weights),
                    geometry_sha256=sha256(Path(__file__)), min_confidence=min_confidence,
                    schema_version="detailed-v4")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if any(manifest.get(k) != value for k, value in identity.items()):
            raise ValueError("Detailed AOI implementation/settings changed; use a new versioned output folder.")
    else:
        manifest = dict(**identity, model=MODEL, labels=LABELS, parents=PARENTS,
            side_convention="displayed image left/right, assigned using mesh eye axis",
            label_sources={name: ("parser + landmark side assignment" if 4 <= k <= 9 else
                "approximate mesh region clipped to confident parser skin" if 19 <= k <= 24 else
                "confidence rejection" if k == 255 else "parser") for k, name in LABELS.items()},
            stimuli={})
    processor = SegformerImageProcessor.from_pretrained(model_dir, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True).eval()
    options = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(weights)),
        running_mode=vision.RunningMode.IMAGE, num_faces=2)
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for source in dict.fromkeys(map(Path, stimuli)):
            digest = sha256(source)
            record = manifest["stimuli"].get(digest)
            if record and (output/record["mask"]).is_file():
                continue
            with Image.open(source) as im:
                image = im.convert("RGB")
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image)))
            points = None
            if len(result.face_landmarks) == 1:
                points = np.array([(q.x*image.width, q.y*image.height, q.z) for q in result.face_landmarks[0]])
                with (output/(digest+"_landmarks.csv")).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f); writer.writerow(["id","image_x","image_y","relative_z"])
                    writer.writerows((i,*q) for i,q in enumerate(points))
            with torch.inference_mode():
                logits = model(**processor(images=image, return_tensors="pt")).logits
                logits = torch.nn.functional.interpolate(logits, size=(image.height,image.width), mode="bilinear", align_corners=False)
                confidence, raw = logits.softmax(1).max(1)
            mask, flags = build_regions(raw[0].numpy(), confidence[0].numpy(), model.config.id2label, points, min_confidence)
            np.savez_compressed(output/(digest+"_parser.npz"), labels=raw[0].numpy().astype("uint8"), confidence=confidence[0].numpy())
            from PIL import ImageDraw
            mesh = image.copy()
            if points is not None:
                draw = ImageDraw.Draw(mesh)
                for q in points:
                    draw.ellipse((q[0]-1,q[1]-1,q[0]+1,q[1]+1), fill="cyan")
            mesh.save(output/(digest+"_landmarks.png"))
            preview(image, raw[0].numpy(), output/(digest+"_raw_review.png"), dict(enumerate(RAW_NAMES)))
            Image.fromarray(mask).save(output/(digest+".png"))
            preview(image, mask, output/(digest+"_review.png"), LABELS)
            preview(image, mask, output/(digest+"_solid.png"), LABELS, solid=True)
            Image.fromarray((confidence[0].numpy()*255).astype("uint8")).save(output/(digest+"_confidence.png"))
            manifest["stimuli"][digest] = dict(stimulus=str(source.resolve()), width=image.width, height=image.height,
                mask=digest+".png", reviewed=False, notes=flags,
                face_count=len(result.face_landmarks), landmarks=digest+"_landmarks.csv" if points is not None else None,
                uncertain_fraction=float(np.mean(mask==255)))
            save_manifest(path, manifest)
            print("Detailed AOIs:", source.name, flush=True)
    return path
