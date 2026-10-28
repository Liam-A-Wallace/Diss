"""Offline face parsing, reviewable label masks, and explicit derived AOIs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

LABELS = {0: "background", 1: "other_skin", 2: "nose", 3: "glasses", 4: "eyes",
          5: "eyebrows", 6: "ears", 7: "mouth", 8: "hair", 9: "hat",
          10: "earring", 11: "necklace", 12: "neck", 13: "clothing",
          14: "forehead_derived", 15: "hairline_derived", 255: "uncertain"}
MODEL = "jonathandinu/face-parsing"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_regions(labels, confidence, id2label, min_confidence=0.6):
    import numpy as np
    from PIL import Image, ImageFilter
    mapping = dict(background=0, skin=1, nose=2, eye_g=3, l_eye=4, r_eye=4,
                   l_brow=5, r_brow=5, l_ear=6, r_ear=6, mouth=7, u_lip=7,
                   l_lip=7, hair=8, hat=9, ear_r=10, neck_l=11, neck=12, cloth=13)
    mask = np.full(labels.shape, 255, dtype=np.uint8)
    for index, name in id2label.items():
        if name not in mapping:
            raise ValueError("Unknown model label: " + name)
        mask[labels == int(index)] = mapping[name]
    mask[confidence < min_confidence] = 255
    brows_y, _ = np.where(mask == 5)
    notes = []
    if brows_y.size:
        # Explicit approximation, NOT a learned forehead/hairline class.
        # Broad horizontal brow cutoff is unsuitable for strongly tilted faces.
        cutoff = int(np.percentile(brows_y, 10))
        upper = np.indices(mask.shape)[0] < cutoff
        forehead = (mask == 1) & upper
        radius = max(1, round(min(mask.shape) * 0.012))
        hair = Image.fromarray(((mask == 8) * 255).astype(np.uint8))
        near_hair = np.asarray(hair.filter(ImageFilter.MaxFilter(2 * radius + 1))) > 0
        mask[forehead] = 14
        mask[forehead & near_hair] = 15
        notes.append("Forehead: skin above 10th percentile brow y; hairline: forehead within 1.2% image-short-side of hair. Review tilt, bangs, occlusion and baldness.")
    else:
        notes.append("No reliable brows: forehead and hairline were not inferred.")
    if not np.any(mask == 2) or not np.any(mask == 4):
        notes.append("Missing nose or eyes: detection requires particular review.")
    return mask, notes


def preview(image, mask, out):
    import numpy as np
    from PIL import Image, ImageDraw
    # Stable colours independent of labels present in a particular stimulus.
    palette = {k: ((k * 83 + 50) % 256, (k * 137 + 80) % 256,
                   (k * 47 + 130) % 256) for k in LABELS}
    colour = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, rgb in palette.items():
        colour[mask == k] = rgb
    overlay = Image.blend(image.convert("RGB"), Image.fromarray(colour), 0.38)
    canvas = Image.new("RGB", (image.width + 240, max(image.height, 430)), "white")
    canvas.paste(overlay, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for row, (k, name) in enumerate(LABELS.items()):
        y = 12 + row * 24
        draw.rectangle((image.width + 10, y, image.width + 28, y + 16), fill=palette[k])
        draw.text((image.width + 34, y), name, fill="black")
    canvas.save(out)


def save_manifest(path, manifest):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def generate(stimuli, model_dir, output, min_confidence=0.6):
    """Generate missing masks; preserve existing masks and human reviews."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    model_dir, output = Path(model_dir), Path(output)
    if not 0 <= min_confidence <= 1:
        raise ValueError("Confidence must be between 0 and 1")
    model_files = {p.name: sha256(p) for p in model_dir.iterdir()
                   if p.suffix in (".safetensors", ".bin", ".json")}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model_files"] != model_files or manifest["min_confidence"] != min_confidence:
            raise ValueError("Model or threshold changed. Choose a new AOI output folder.")
    else:
        manifest = dict(schema_version=1, model=MODEL, model_files=model_files,
                        min_confidence=min_confidence, labels=LABELS, stimuli={})
    processor = model = None
    for path in dict.fromkeys(map(Path, stimuli)):
        digest = sha256(path)
        existing = manifest["stimuli"].get(digest)
        if existing and (output / existing["mask"]).is_file():
            print("Already generated:", path.name)
            continue
        if model is None:
            processor = SegformerImageProcessor.from_pretrained(model_dir, local_files_only=True)
            model = SegformerForSemanticSegmentation.from_pretrained(model_dir, local_files_only=True)
            model.eval()
        with Image.open(path) as source:
            image = source.convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width),
                                                     mode="bilinear", align_corners=False)
            confidence, labels = logits.softmax(dim=1).max(dim=1)
        mask, notes = derive_regions(labels[0].numpy(), confidence[0].numpy(),
                                     model.config.id2label, min_confidence)
        Image.fromarray(mask).save(output / (digest + ".png"))
        preview(image, mask, output / (digest + "_review.png"))
        manifest["stimuli"][digest] = dict(stimulus=str(path.resolve()), width=image.width, height=image.height,
                                         mask=digest + ".png", reviewed=False, notes=notes,
                                         uncertain_fraction=float(np.mean(mask == 255)))
        save_manifest(manifest_path, manifest)
        print("Generated:", path.name)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("aois/main"))
    parser.add_argument("--min-confidence", type=float, default=0.6)
    args = parser.parse_args()
    with args.trials.open(newline="", encoding="utf-8-sig") as source:
        stimuli = [args.trials.parent / r["stimulus"] for r in csv.DictReader(source)]
    generate(stimuli, args.model_dir, args.output, args.min_confidence)


class AOIMap:
    def __init__(self, manifest_path=None, allow_unreviewed=False):
        self.path = Path(manifest_path) if manifest_path else None
        self.manifest = json.loads(self.path.read_text(encoding="utf-8")) if self.path else None
        self.allow_unreviewed = allow_unreviewed
        self.cache = {}

    def load(self, stimulus):
        if self.manifest is None:
            return None
        from PIL import Image
        if stimulus not in self.cache:
            digest = sha256(stimulus)
            record = self.manifest["stimuli"].get(digest)
            if record is None:
                raise ValueError("AOI mask missing for " + stimulus + ". Run python analyse.py aois (add --test for test data).")
            if not record.get("reviewed") and not self.allow_unreviewed:
                raise ValueError("Unreviewed AOIs for " + stimulus + ". Run python analyse.py review (add --test for test data).")
            mask_path = self.path.parent / record["mask"]
            if record.get("reviewed_mask_sha256") and sha256(mask_path) != record["reviewed_mask_sha256"]:
                raise ValueError("AOI mask changed after review. Run python analyse.py review again.")
            with Image.open(mask_path) as im:
                mask = im.copy()
            with Image.open(stimulus) as im:
                if mask.size != im.size:
                    raise ValueError("AOI mask dimensions do not match stimulus")
            if mask.mode not in ("L", "P"):
                raise ValueError("AOI mask must be a single-channel label image")
            if {label for count, label in mask.getcolors(256)} - set(map(int, self.manifest["labels"])):
                raise ValueError("AOI mask contains unknown label IDs")
            self.cache[stimulus] = mask
        return self.cache[stimulus]

    def at(self, stimulus, x, y):
        mask = self.load(stimulus)
        if mask is None:
            return "unmapped"
        if not (0 <= x < mask.width and 0 <= y < mask.height):
            return "outside_image"
        return self.manifest["labels"][str(mask.getpixel((int(x), int(y))))]


if __name__ == "__main__":
    main()

