"""Offline image enrichment for stimulus curation.

Computes per-image metadata and writes ``Dataset/image_metadata.csv``:

    image_id, is_photo, photo_score, num_faces, max_face_fraction,
    face_center_off, face_ok, ethnicity, gender, age_group, status, error

Backends (each optional; missing ones are skipped with a warning):

  * CLIP  (``open_clip`` + ``torch``)          -> photo vs drawing
  * MediaPipe (``mediapipe``)                  -> face count + bounding boxes
  * transformers pipelines                     -> ethnicity / gender / age
        - ethnicity: HuggingFaceM4/FairFace    (GATED on HF: accept terms
                                                + set HF_TOKEN)
        - gender:    rizvandwiki/gender-classification
        - age:       nateraw/vit-age-classifier

Install (Diss venv):

    .\Diss\.venv\Scripts\python.exe -m pip install torch open_clip_torch mediapipe transformers huggingface_hub

For the gated FairFace race model, first open
    https://huggingface.co/HuggingFaceM4/FairFace
and accept the terms, then log in once:
    .\Diss\.venv\Scripts\hf.exe login     (or set the HF_TOKEN environment variable)

Usage (from anywhere, using the Diss venv):

    # low-memory: one task per invocation (recommended on <16 GB RAM)
    python Dataset/enrich.py --tasks face   --sample 50
    python Dataset/enrich.py --tasks photo  --skip-existing
    python Dataset/enrich.py --tasks gender --skip-existing
    python Dataset/enrich.py --tasks age    --skip-existing
    python Dataset/enrich.py --tasks race   --skip-existing

    # everything at once (needs ~4 GB free RAM)
    python Dataset/enrich.py --sample 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Best-effort: quiet MediaPipe/TFLite's noisy per-image warning lines.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

METADATA_CSV = HERE / "image_metadata.csv"

FIELDS = [
    "image_id", "is_photo", "photo_score", "num_faces",
    "max_face_fraction", "face_center_off", "face_ok",
    "ethnicity", "gender", "age_group",
    "status", "error",
]

RACE_MODEL = "HuggingFaceM4/FairFace"
GENDER_MODEL = "rizvandwiki/gender-classification"
AGE_MODEL = "nateraw/vit-age-classifier"

MIN_FACE_FRACTION = 0.1    # face box must cover >= 10% of the image
FACE_CENTER_TOLERANCE = 0.2  # face centre within 20% of the image centre

# Which columns each task fills (tasks can be run one at a time to limit RAM).
TASK_COLS = {
    "face":   ["num_faces", "max_face_fraction", "face_center_off", "face_ok"],
    "photo":  ["is_photo", "photo_score"],
    "race":   ["ethnicity"],
    "gender": ["gender"],
    "age":    ["age_group"],
}
_PIPE_MODELS = {"race": RACE_MODEL, "gender": GENDER_MODEL, "age": AGE_MODEL}

BATCH = 32  # images per forward pass (large CPU speedup over one-at-a-time)

# MediaPipe 1.x removed the legacy ``solutions`` API; the FaceDetector task
# needs this TFLite model, downloaded on first use.
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


# --------------------------------------------------------------------------
# Lazy backends (each guarded by a "failed" flag so we warn once, not per image)
# --------------------------------------------------------------------------
_clip = None
_clip_failed = False
_face_detector = None
_face_failed = False
_pipes: dict = {}


def _load_clip():
    global _clip
    import torch  # noqa: F401
    import open_clip  # noqa: F401
    torch.set_num_threads(os.cpu_count() or 4)
    # ViT-B-32 (LAION) has a much stronger zero-shot text tower than RN50, so it
    # separates "photo" from "drawing" reliably. Batching keeps it fast enough.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = tokenizer([
        "a photograph of a person's face",
        "a drawing, cartoon, illustration or painting of a face",
    ])
    _clip = (model, preprocess, tokenizer, text)
    return _clip


def _face_model_path() -> Path:
    local = common.CACHE_DIR / "models" / "blaze_face_short_range.tflite"
    if local.exists() and local.stat().st_size > 0:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    import requests
    resp = requests.get(FACE_MODEL_URL, timeout=30, headers=common._HEADERS)
    resp.raise_for_status()
    local.write_bytes(resp.content)
    return local


def _load_face_detector():
    global _face_detector
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(_face_model_path())),
        min_detection_confidence=0.5)
    _face_detector = vision.FaceDetector.create_from_options(options)
    return _face_detector


def _load_pipe(name: str):
    """Load a single transformers pipeline (cached). Only what's requested."""
    if name in _pipes:
        return _pipes[name]
    try:
        from transformers import pipeline  # noqa: F401
        import torch  # noqa: F401
        torch.set_num_threads(os.cpu_count() or 4)
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[enrich] WARN: transformers not available ({exc})")
        _pipes[name] = None
        return None
    model_id = _PIPE_MODELS[name]
    try:
        pipe = pipeline("image-classification", model=model_id, device=-1,
                        batch_size=BATCH)
        print(f"[enrich] loaded {name} model: {model_id}")
    except Exception as exc:  # pragma: no cover - env dependent
        pipe = None
        print(f"[enrich] WARN: could not load {name} model {model_id} ({exc})")
    _pipes[name] = pipe
    return pipe


def backend_status() -> dict:
    """Report module availability WITHOUT importing them (importing torch is
    heavy — we only want to know if a task *can* run, not load it now)."""
    import importlib.util
    status = {}
    for key, mod in (("clip", "open_clip"), ("mediapipe", "mediapipe"),
                     ("transformers", "transformers")):
        status[key] = importlib.util.find_spec(mod) is not None
    return status


# --------------------------------------------------------------------------
# Per-image analysis
# --------------------------------------------------------------------------
_text_feat = None


def _text_features(model, tokenizer, text):
    """Encode the CLIP text prompts once and cache (they never change)."""
    global _text_feat
    if _text_feat is None:
        import torch
        with torch.no_grad():
            feat = model.encode_text(text)
            _text_feat = feat / feat.norm(dim=-1, keepdim=True)
    return _text_feat


def photo_classify_batch(imgs) -> list:
    """Return [(photo_score, is_photo), ...] for a batch of PIL images.

    Returns (None, None) entries if CLIP is unavailable.
    """
    global _clip_failed
    if _clip_failed or not imgs:
        return [(None, None)] * len(imgs)
    try:
        import torch  # noqa: F401
        model, preprocess, tokenizer, text = _load_clip()
        model.eval()
        batch = torch.stack([preprocess(im) for im in imgs])
        with torch.no_grad():
            feat = model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            probs = (feat @ _text_features(model, tokenizer, text).T).softmax(dim=-1)
        scores = probs[:, 0].tolist()
        return [(s, 1 if s >= 0.5 else 0) for s in scores]
    except Exception as exc:  # pragma: no cover - env dependent
        _clip_failed = True
        print(f"[enrich] WARN: CLIP failed ({exc}); photo check skipped")
        return [(None, None)] * len(imgs)


def detect_faces(img) -> list | None:
    """Return [(x, y, w, h, score), ...] or None if MediaPipe unavailable."""
    global _face_failed
    if _face_failed:
        return None
    try:
        import numpy as np
        import mediapipe as mp
        det = _load_face_detector()
        arr = np.asarray(img.convert("RGB"))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
        res = det.detect(mp_img)
        if not res.detections:
            return []
        out = []
        for d in res.detections:
            bb = d.bounding_box
            score = float(d.categories[0].score) if d.categories else 0.0
            out.append((bb.origin_x, bb.origin_y, bb.width, bb.height, score))
        return out
    except Exception as exc:  # pragma: no cover - env dependent
        _face_failed = True
        print(f"[enrich] WARN: MediaPipe failed ({exc}); face check skipped")
        return None


def crop_face(img, faces: list):
    if not faces:
        return img
    x, y, bw, bh, _ = max(faces, key=lambda f: f[2] * f[3])
    pad = 0.3
    left = max(0, int(x - bw * pad))
    top = max(0, int(y - bh * pad))
    right = min(img.width, int(x + bw * (1 + pad)))
    bottom = min(img.height, int(y + bh * (1 + pad)))
    return img.crop((left, top, right, bottom))


def _run_pipe_batch(name: str, imgs) -> list:
    """Run one transformer pipeline over a batch of PIL images.

    Returns a list of label strings ('' on failure / empty result).
    """
    pipe = _load_pipe(name)
    if not pipe:
        return [""] * len(imgs)
    try:
        results = pipe(imgs)  # pipeline batches internally (batch_size=BATCH)
        out = []
        for r in results:
            if not r:
                out.append("")
                continue
            label = str(r[0]["label"])
            if label.isdigit():
                label = str(pipe.model.config.id2label[int(label)])
            out.append(label)
        return out
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[enrich] WARN: {name} model failed ({exc})")
        return [""] * len(imgs)


def compute_photo_batch(imgs) -> list:
    results = photo_classify_batch(imgs)
    out = []
    for score, is_photo in results:
        d = {"is_photo": "", "photo_score": ""}
        if score is not None:
            d["photo_score"] = f"{score:.3f}"
            d["is_photo"] = str(is_photo)
        out.append(d)
    return out


def compute_faces(img) -> dict:
    faces = detect_faces(img)
    if faces is None:
        return {}
    out = {"num_faces": str(len(faces))}
    if faces:
        x, y, w, h, _ = max(faces, key=lambda f: f[2] * f[3])
        frac = w * h / (img.width * img.height)
        off_x = abs((x + w / 2) - img.width / 2) / img.width
        off_y = abs((y + h / 2) - img.height / 2) / img.height
        off = max(off_x, off_y)
    else:
        frac, off = 0.0, 1.0
    out["max_face_fraction"] = f"{frac:.3f}"
    out["face_center_off"] = f"{off:.3f}"
    out["face_ok"] = "1" if (len(faces) == 1
                             and frac >= MIN_FACE_FRACTION
                             and off <= FACE_CENTER_TOLERANCE) else "0"
    return out


def compute_attribute_batch(imgs, name: str) -> list:
    """Run one attribute task (race/gender/age) over a batch of images.

    Returns a list of single-key dicts ({col: label}), '' for no-face images.
    """
    col = TASK_COLS[name][0]
    crops = []
    idx_map = []
    for i, im in enumerate(imgs):
        faces = detect_faces(im)
        if faces:
            crops.append(crop_face(im, faces))
            idx_map.append(i)
    labels = _run_pipe_batch(name, crops) if crops else []
    out = [{col: ""} for _ in imgs]
    for pos, i in enumerate(idx_map):
        out[i] = {col: labels[pos]}
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _write_csv(path: Path, table: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rid in sorted(table):
            writer.writerow(table[rid])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enrich images with metadata.")
    parser.add_argument("--csv", default=str(common.DATASET_CSV),
                        help="primary source CSV")
    parser.add_argument("--extra-csv", default=str(common.MANIFESTS_DIR / "DFR60K_manifest.csv"),
                        help="second source CSV to include")
    parser.add_argument("--out", default=str(METADATA_CSV),
                        help="output metadata CSV")
    parser.add_argument("--tasks", default="face,photo,race,gender,age",
                        help="comma-separated tasks: face,photo,race,gender,age "
                             "(run one at a time on low-RAM machines)")
    parser.add_argument("--sample", type=int, default=0,
                        help="max images to process (0 = all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="alias for --sample")
    parser.add_argument("--start", type=int, default=0,
                        help="skip the first N rows (resume helper)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip images whose requested task columns are already filled")
    parser.add_argument("--force", action="store_true",
                        help="recompute requested tasks even if already present")
    args = parser.parse_args(argv)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASK_COLS]
    if unknown:
        parser.error(f"unknown tasks: {unknown} (choose from {sorted(TASK_COLS)})")

    rows: list[dict] = []
    csv_path = Path(args.csv)
    extra_path = Path(args.extra_csv)
    if csv_path.exists():
        rows += common.load_rows(csv_path)
        print(f"[enrich] loaded {len(rows)} rows from {csv_path}")
    if extra_path.exists() and extra_path != csv_path:
        n = len(rows)
        rows += common.load_rows(extra_path)
        print(f"[enrich] loaded {len(rows) - n} rows from {extra_path}")

    if not rows:
        # No source CSV found. If the dataset folders live in this project,
        # build a manifest from them so enrich can run purely off the folder.
        import build_manifest  # same-directory module (this repo)
        if build_manifest.find_dataset_root() is not None:
            print("[enrich] no source CSV found; building DFR60K_manifest.csv "
                  "from the local dataset folders")
            build_manifest.main(["--out", str(HERE / "DFR60K_manifest.csv")])
            manifest = HERE / "DFR60K_manifest.csv"
            if manifest.exists():
                rows += common.load_rows(manifest)
                print(f"[enrich] loaded {len(rows)} rows from {manifest}")

    if not rows:
        parser.error(
            "no source CSV found. Copy the dataset folders "
            "(deepfake-vs-real-60k/Real + deepfake-vs-real-60k/Fake) into this "
            "project (or run build_manifest.py), or pass --csv/--extra-csv "
            "pointing at an existing CSV."
        )

    out_path = Path(args.out)
    existing: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8-sig", newline="") as f:
            existing = {r.get("image_id", ""): r for r in csv.DictReader(f)}

    print("[enrich] backends available: "
          + ", ".join(f"{k}={v}" for k, v in backend_status().items()))
    print(f"[enrich] tasks: {', '.join(tasks)}")

    if args.skip_existing and not args.force:
        work = [r for r in rows
                if any(not existing.get(r["image_id"], {}).get(c)
                       for t in tasks for c in TASK_COLS[t])]
    else:
        work = rows
    if args.start:
        work = work[args.start:]
    n = args.sample or args.limit
    if n:
        work = work[:n]
    print(f"[enrich] {len(rows)} total rows -> {len(work)} to process")

    if not work:
        print("[enrich] nothing to do")
        return 0

    processed = 0
    next_write = 100
    t0 = time.time()
    for batch_start in range(0, len(work), BATCH):
        batch_rows = work[batch_start:batch_start + BATCH]
        recs = []
        imgs = []
        for row in batch_rows:
            rec = existing.setdefault(row["image_id"], {k: "" for k in FIELDS})
            rec["image_id"] = row["image_id"]
            recs.append(rec)
            try:
                img = common.fetch_image(row, thumbnail=True)
                if img is None:
                    imgs.append(None)
                    rec["status"] = "error"
                    rec["error"] = "image_load_failed"
                else:
                    imgs.append(img)
                    rec["status"] = "ok"
            except Exception as exc:  # never kill a long run on one image
                imgs.append(None)
                rec["status"] = "error"
                rec["error"] = str(exc)[:200]

        for task in tasks:
            cols = TASK_COLS[task]
            need = [i for i in range(len(batch_rows))
                    if imgs[i] is not None
                    and (args.force or any(not recs[i].get(c) for c in cols))]
            if not need:
                continue
            need_imgs = [imgs[i] for i in need]
            if task == "photo":
                res = compute_photo_batch(need_imgs)
            elif task == "face":
                res = [compute_faces(im) for im in need_imgs]
            else:
                res = compute_attribute_batch(need_imgs, task)
            for pos, i in enumerate(need):
                recs[i].update(res[pos])

        processed += len(batch_rows)
        if processed >= next_write:
            _write_csv(out_path, existing)
            next_write = processed + 100
        dt = time.time() - t0
        print(f"[enrich] {processed}/{len(work)} done ({dt / max(processed, 1):.2f}s/img)")

    _write_csv(out_path, existing)
    print(f"[enrich] saved {out_path} ({len(existing)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
