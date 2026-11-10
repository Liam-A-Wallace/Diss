"""Render face crops and descriptive comparison tables from a completed run."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def summarise(root):
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        required=["original.png","landmarks.png","raw_512.png","raw_facecrop_512.png","raw_1024.png","threshold_512_0.6.png","threshold_facecrop_512_0.6.png","threshold_1024_0.6.png"]
        if not all((folder/name).exists() for name in required) or not (folder/"landmarks.csv").exists():
            continue
        with (folder/"landmarks.csv").open() as f:
            points=np.array([(float(r["image_x"]),float(r["image_y"])) for r in csv.DictReader(f)])
        image=Image.open(folder/"original.png")
        raw_points=image.convert("RGB")
        point_draw=ImageDraw.Draw(raw_points)
        for x,y in points:
            point_draw.ellipse((x-1,y-1,x+1,y+1),fill="cyan")
        raw_points.save(folder/"raw_landmarks.png")
        for resolution in (512,1024):
            with np.load(folder/f"prediction_{resolution}.npz") as saved:
                Image.fromarray(saved["labels"]).save(folder/f"raw_{resolution}_labels.png")
        low,high=points.min(0),points.max(0)
        margin=(high-low)*.08
        box=(max(0,int(low[0]-margin[0])),max(0,int(low[1]-margin[1])),min(image.width,int(high[0]+margin[0])),min(image.height,int(high[1]+margin[1])))
        canvas=Image.new("RGB",(1200,1290),"white")
        draw=ImageDraw.Draw(canvas)
        for i,name in enumerate(required):
            x,y=(i%3)*400,(i//3)*430
            draw.text((x+5,y+5),name,fill="black")
            with Image.open(folder/name) as im:
                crop=im.crop(box); crop.thumbnail((395,400))
                canvas.paste(crop,(x,y+25))
        canvas.save(folder/"face_comparison.jpg",quality=94)
    if not (root/"feature_metrics.csv").exists():
        return
    with (root/"feature_metrics.csv").open() as f:
        metrics=list(csv.DictReader(f))
    focus={"eyes":["l_eye","r_eye"],"eyebrows":["l_brow","r_brow"],"nose":["nose"],"mouth":["mouth","u_lip","l_lip"]}
    aggregated=[]
    for resolution in (512,"facecrop512",1024):
        for threshold in (.4,.5,.6):
            for group,names in focus.items():
                rows=[r for r in metrics if r["resolution"]==str(resolution) and float(r["threshold"])==threshold and r["feature"] in names]
                raw=sum(int(r["raw_pixels"]) for r in rows)
                keep=sum(int(r["retained_pixels"]) for r in rows)
                aggregated.append(dict(resolution=resolution,threshold=threshold,feature=group,
                    raw_pixels=raw,retained_pixels=keep,retained_fraction=keep/raw if raw else None))
    (root/"aggregate_metrics.json").write_text(json.dumps(aggregated,indent=2))
    lines=["# Descriptive model comparison", "", "Eight failure cases; pooled pixel counts weight larger source images more heavily.",
           "These fractions measure retention, not correctness. No threshold/model has been selected.","",
           "| Input | Threshold | Feature | Raw pixels retained |", "|---|---|---|---|"]
    for r in aggregated:
        value=f'{r["retained_fraction"]:.1%}' if r["retained_fraction"] is not None else "absent"
        lines.append(f'| {r["resolution"]} | {r["threshold"]} | {r["feature"]} | {value} |')
    lines.extend(["", "Per-image crops are saved as face_comparison.jpg inside each case folder.",
        "The facecrop512 condition uses MediaPipe bounds plus 25% padding. Retention and agreement remain descriptive, not ground-truth accuracy.",
        "Manual/reference ratings remain unfilled; resolution agreement is not ground-truth accuracy."])
    (root/"metrics_summary.md").write_text("\n".join(lines),encoding="utf-8")
    report=root/"index.html"
    if report.exists():
        text=report.read_text(encoding="utf-8")
        for folder in (p for p in root.iterdir() if p.is_dir()):
            text=text.replace(f'{folder.name}/landmarks.png',f'{folder.name}/raw_landmarks.png')
        text=text.replace("MediaPipe dots + feature contours","Raw MediaPipe points (no custom contours)")
        if 'metrics_summary.md' not in text:
            text=text.replace('<h1>Face model diagnostics</h1>','<h1>Face model diagnostics</h1><p><a href="metrics_summary.md">Descriptive summary</a>. Face crop sheets and contour guides are available in each image folder.</p>')
        report.write_text(text,encoding="utf-8")
    (root/"postprocessing.json").write_text(json.dumps(dict(
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        changes=["raw landmark-only visualisations from saved CSV", "integer label PNGs from saved arrays", "crop comparisons", "pooled descriptive metrics including face-crop condition"],
        production_masks_changed=False),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    summarise(p.parse_args().folder)
