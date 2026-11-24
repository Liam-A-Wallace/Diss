"""Local browser landmark correction editor. No model inference or network service."""

import base64
import copy
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
from urllib.parse import urlsplit
import webbrowser

import numpy as np
from PIL import Image, ImageDraw

from analysis.detailed_aois import build_regions, LABELS
from analysis.face_aois import preview, save_manifest, sha256

ROOT = Path(__file__).resolve().parents[1]


class Editor:
    def __init__(self, source, output):
        self.source, self.output = Path(source).resolve(), Path(output).resolve()
        if not self.output.is_relative_to(ROOT) or self.output == self.source:
            raise ValueError("Corrections need a separate output directory inside Diss")
        self.base = json.loads(
            (self.source / "manifest.json").read_text(encoding="utf-8")
        )
        if self.base.get("schema_version") != "detailed-v4":
            raise ValueError("Generate detailed-v4 AOIs before editing")
        if {int(k): v for k, v in self.base["labels"].items()} != LABELS:
            raise ValueError("The saved label schema is incompatible with this editor")
        self.geometry_sha256 = sha256(ROOT / "analysis/detailed_aois.py")
        self.geometry_changed = self.base["geometry_sha256"] != self.geometry_sha256
        self.config = json.loads((ROOT / "models/face-parsing/config.json").read_text())
        self.cache = {}

    def record(self, key):
        if key not in self.base["stimuli"]:
            raise ValueError("Unknown image")
        return self.base["stimuli"][key]

    def correction_path(self, key):
        self.record(key)
        return self.output / (key + "_correction.json")

    def load(self, key):
        record = self.record(key)
        source = Path(record["stimulus"])
        if sha256(source) != key:
            raise ValueError("Stimulus changed since generation")
        if key not in self.cache:
            if not record.get("landmarks"):
                raise ValueError("This face has no editable landmark mesh")
            with (self.source / record["landmarks"]).open(newline="") as f:
                rows = list(csv.DictReader(f))
            points = np.array(
                [
                    [float(r["image_x"]), float(r["image_y"]), float(r["relative_z"])]
                    for r in rows
                ]
            )
            with np.load(self.source / (key + "_parser.npz")) as saved:
                raw, confidence = saved["labels"].copy(), saved["confidence"].copy()
            with Image.open(self.source / record["mask"]) as im:
                original_mask = np.array(im)
            self.cache.clear()  # one high-resolution image at a time
            self.cache[key] = (points, raw, confidence, original_mask)
        return self.cache[key]

    def state(self, key):
        original, _, _, _ = self.load(key)
        path = self.correction_path(key)
        correction = json.loads(path.read_text()) if path.exists() else None
        if correction and correction["automatic_mask_sha256"] != sha256(
            self.source / self.record(key)["mask"]
        ):
            raise ValueError(
                "Automatic mask changed since this correction; resolve the conflict before editing"
            )
        regions = self.region_contours(key)
        overrides=(correction.get("regions",{str(6+int(k)):v for k,v in correction.get("brows",{}).items()}) if correction else {})
        return dict(
            points=correction["points"] if correction else original.tolist(),
            original_points=original.tolist(),
            version=sha256(path) if correction else None,
            original_regions=regions,
            region_order=correction.get("region_order",list(overrides)) if correction else [],
            regions=(
                correction.get(
                    "regions",
                    {
                        str(6 + int(k)): v
                        for k, v in correction.get("brows", {}).items()
                    },
                )
                if correction
                else {}
            ),
            note=correction.get("note", "") if correction else "",
            labels=LABELS,
        )

    @staticmethod
    def shapes(value):
        if not isinstance(value,list):
            raise ValueError("Feature contours must be a list")
        if value and not isinstance(value[0],dict):
            value=[dict(outer=value,holes=[])]
        for shape in value:
            if not isinstance(shape,dict) or "outer" not in shape or not isinstance(shape.get("holes",[]),list):
                raise ValueError("Invalid contour shape")
        return value

    def region_contours(self, key):
        import cv2
        _,_,_,mask=self.load(key)
        result={}
        for label in range(1,25):
            contours,hierarchy=cv2.findContours((mask==label).astype("uint8"),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
            def simplify(contour):
                epsilon=max(.3,cv2.arcLength(contour,True)*.002)
                p=cv2.approxPolyDP(contour,epsilon,True)[:,0,:]
                while len(p)>128:
                    epsilon*=1.5;p=cv2.approxPolyDP(contour,epsilon,True)[:,0,:]
                if len(p)<3 or cv2.contourArea(p.astype("float32"))<1:
                    x,y,w,h=cv2.boundingRect(contour)
                    r=self.record(key)
                    x0=max(0,x-.5);y0=max(0,y-.5);x1=min(r["width"]-1,x+w+.5);y1=min(r["height"]-1,y+h+.5)
                    p=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
                return p.tolist()
            shapes=[]
            if hierarchy is not None:
                for i,contour in enumerate(contours):
                    if hierarchy[0,i,3]!=-1:continue
                    holes=[];child=int(hierarchy[0,i,2])
                    while child!=-1:
                        holes.append(simplify(contours[child]));child=int(hierarchy[0,child,0])
                    shapes.append(dict(outer=simplify(contour),holes=holes))
            result[str(label)]=shapes
        return result

    def apply_regions(self, key, mask, regions, region_order=None):
        import cv2

        if not isinstance(regions, dict) or set(regions) - {
            str(k) for k in range(1, 25)
        }:
            raise ValueError("Unknown region override")
        polygons = {}
        order=list(regions) if region_order is None else region_order
        if not isinstance(order,list) or any(not isinstance(k,str) for k in order) or len(order)!=len(set(order)) or set(order)!=set(regions):
            raise ValueError("Region priority must list each edited region once")
        r = self.record(key)
        def rasterize(vertices):
            p = np.asarray(vertices, dtype=float)
            if (
                p.ndim != 2
                or p.shape[1] != 2
                or not 3 <= len(p) <= 128
                or not np.isfinite(p).all()
            ):
                raise ValueError("Regions need 3 to 128 finite contour points")
            if (
                np.any(p < 0)
                or np.any(p[:, 0] >= r["width"])
                or np.any(p[:, 1] >= r["height"])
            ):
                raise ValueError("Region contours must stay within the image")
            if cv2.contourArea(p.astype("float32")) < 1:
                raise ValueError("Region contour has no usable area")

            def cross(a, b, c):
                d = b - a
                e = c - a
                return d[0] * e[1] - d[1] * e[0]

            for i in range(len(p)):
                a, b = p[i], p[(i + 1) % len(p)]
                for j in range(i + 2, len(p)):
                    if i == 0 and j == len(p) - 1:
                        continue
                    c, d = p[j], p[(j + 1) % len(p)]
                    if (
                        cross(a, b, c) * cross(a, b, d) < 0
                        and cross(c, d, a) * cross(c, d, b) < 0
                    ):
                        raise ValueError(
                            "Region contour crosses itself; undo the last move"
                        )
            image=Image.new("L",(r["width"],r["height"]))
            ImageDraw.Draw(image).polygon([tuple(q) for q in p],fill=1)
            return np.asarray(image)>0
        for side,value in regions.items():
            union=np.zeros(mask.shape,dtype=bool)
            for shape in self.shapes(value):
                region=rasterize(shape["outer"])
                for hole in shape.get("holes",[]):
                    region &= ~rasterize(hole)
                union |= region
            polygons[side]=union
        combined = np.zeros(mask.shape, dtype=bool)
        overlaps = False
        for region in polygons.values():
            if np.any(combined & region):
                overlaps = True
            combined |= region
        flags = []
        if overlaps:
            flags.append("Overlapping contours resolved automatically: most recently edited feature takes priority")
        for label in polygons:
            mask[mask == int(label)] = 255
        allowed = np.isin(mask, [0, 1, 19, 20, 21, 22, 23, 24, 255])
        for label in order:
            region=polygons[label]
            if np.any(region & ~allowed):
                flags.append("Region contour clipped to protect other primary features")
            mask[region & allowed] = int(label)
        return flags

    def render(self, key, points, regions=None, region_order=None):
        original, raw, confidence, automatic = self.load(key)
        points = np.asarray(points, dtype=float)
        record = self.record(key)
        if points.shape != original.shape or not np.isfinite(points).all():
            raise ValueError("Expected a finite original-sized landmark array")
        moved = np.any(points[:, :2] != original[:, :2], axis=1)
        outside = (
            (points[:, 0] < 0)
            | (points[:, 0] >= record["width"])
            | (points[:, 1] < 0)
            | (points[:, 1] >= record["height"])
        )
        if np.any(outside & moved):
            raise ValueError("Landmarks must stay within the image")
        if not np.array_equal(points[:, 2], original[:, 2]):
            raise ValueError("Only image x/y coordinates are editable")
        if not np.any(moved):
            mask, flags = automatic.copy(), list(record.get("notes", []))
        else:
            mask, flags = build_regions(raw, confidence, self.config["id2label"], points, self.base["min_confidence"])
        if self.geometry_changed:
            flags.append("Mapping code differs from generation. Unmoved landmarks retain the saved mask; moving landmarks recomputes derived regions with current code.")
        # Landmark changes affect derived skin regions only. Preserve parser
        # detections and their original sided assignment, including uncertainty.
        fixed = ((automatic < 19) & (automatic != 1)) | (automatic == 255)
        mask[fixed] = automatic[fixed]
        if regions is not None:
            flags += self.apply_regions(key, mask, regions, region_order)
        changed = int(np.any(points[:, :2] != original[:, :2], axis=1).sum())
        return mask, flags, changed

    @staticmethod
    def png(mask):
        colours = np.array(
            [
                ((k * 83 + 50) % 256, (k * 137 + 80) % 256, (k * 47 + 130) % 256, 255)
                for k in range(256)
            ],
            dtype="uint8",
        )
        colours[255] = (96, 96, 96, 255)
        buffer = BytesIO()
        Image.fromarray(colours[mask]).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def save(self, key, points, version, note="", regions=None, region_order=None):
        if not isinstance(note, str) or len(note) > 4000:
            raise ValueError("Correction note must be at most 4000 characters")
        self.state(key)  # check original input and parent consistency
        correction_path = self.correction_path(key)
        current = sha256(correction_path) if correction_path.exists() else None
        if version != current:
            raise ValueError("Another editor saved this image. Reload before saving")
        mask, flags, changed = self.render(key, points, regions, region_order)
        record = copy.deepcopy(self.record(key))
        self.output.mkdir(parents=True, exist_ok=True)
        output_manifest = self.output / "manifest.json"
        if output_manifest.exists():
            manifest = json.loads(output_manifest.read_text())
            if (
                manifest.get("automatic_directory") != str(self.source)
                or manifest.get("geometry_sha256") != self.base["geometry_sha256"]
            ):
                raise ValueError(
                    "Output folder belongs to a different automatic AOI set"
                )
        else:
            manifest = copy.deepcopy(self.base)
            manifest.update(
                schema_version="detailed-v4-manual",
                automatic_directory=str(self.source),
            )
            for fallback in manifest["stimuli"].values():
                for field in ("mask", "landmarks"):
                    if fallback.get(field):
                        fallback[field] = os.path.relpath(
                            self.source / fallback[field], self.output
                        )
                fallback["automatic_fallback"] = True
        correction = dict(
            points=points,
            note=note,
            regions=regions or {},
            region_order=list(regions or {}) if region_order is None else region_order,
            overlap_policy="last_edited_wins",
            modified_landmarks=changed,
            saved_at=datetime.now(timezone.utc).isoformat(),
            stimulus_sha256=key,
            automatic_mask_sha256=sha256(self.source / record["mask"]),
            automatic_landmarks_sha256=sha256(self.source / record["landmarks"]),
            parser_sha256=sha256(self.source / (key + "_parser.npz")),
            geometry_sha256=self.base["geometry_sha256"],
            editor_geometry_sha256=self.geometry_sha256,
        )
        # Stage every artifact, then publish the manifest last. A crash cannot
        # leave a newly written mask carrying an old approval.
        record.pop("reviewed_at", None)
        record.pop("reviewed_mask_sha256", None)
        record.update(
            reviewed=False,
            notes=flags + ["Manual landmark/region corrections; requires review"],
            correction=correction_path.name,
            landmarks=key + "_landmarks.csv",
            manual_modified_landmarks=changed,
            editor_geometry_sha256=self.geometry_sha256,
        )
        record["manual_region_labels"] = list((regions or {}).keys())
        record["manual_label_sources"] = {
            LABELS[
                int(side)
            ]: "manually edited polygon, clipped to protect other primary features"
            for side in (regions or {})
        }
        old_record = manifest["stimuli"].get(key)
        if old_record:
            old_record.pop("reviewed_at", None)
            old_record.pop("reviewed_mask_sha256", None)
            old_record["reviewed"] = False
            save_manifest(output_manifest, manifest)
        temp = self.output / (key + ".png.tmp")
        Image.fromarray(mask).save(temp, format="PNG")
        import time as _time
        for attempt in range(6):
            try:
                temp.replace(self.output / record["mask"])
                break
            except PermissionError:
                if attempt == 5:
                    raise
                _time.sleep(0.1 * (attempt + 1))
        with (self.output / record["landmarks"]).open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["id", "image_x", "image_y", "relative_z"])
            writer.writerows((i, *q) for i, q in enumerate(points))
        with Image.open(record["stimulus"]) as im:
            image = im.convert("RGB")
        preview(image, mask, self.output / (key + "_review.png"), LABELS)
        preview(image, mask, self.output / (key + "_solid.png"), LABELS, solid=True)
        draw = ImageDraw.Draw(image)
        for q in points:
            draw.ellipse((q[0] - 1, q[1] - 1, q[0] + 1, q[1] + 1), fill="cyan")
        image.save(self.output / (key + "_landmarks.png"))
        save_manifest(correction_path, correction)
        record.update(
            correction_sha256=sha256(correction_path),
            uncertain_fraction=float(np.mean(mask == 255)),
        )
        manifest["stimuli"][key] = record
        save_manifest(output_manifest, manifest)
        return dict(version=sha256(correction_path), flags=flags, changed=changed)


def serve(source, output, open_browser=True):
    editor = Editor(source, output)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value, mime="application/json"):
            data = json.dumps(value).encode() if mime == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            expected = "127.0.0.1:" + str(self.server.server_port)
            if self.headers.get("Host") != expected:
                return False
            if self.headers.get("Origin", "http://" + expected) != "http://" + expected:
                return False
            return secrets.compare_digest(self.headers.get("X-Editor-Token", ""), token)

        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                return self.reply(
                    200,
                    Path(__file__).with_name("landmark_editor.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            if not self.allowed():
                return self.reply(403, {"error": "Editor authorization failed"})
            try:
                if route == "/api/images":
                    return self.reply(
                        200,
                        [
                            dict(
                                id=k,
                                name=Path(r["stimulus"]).name,
                                width=r["width"],
                                height=r["height"],
                                editable=bool(r.get("landmarks")),
                            )
                            for k, r in editor.base["stimuli"].items()
                        ],
                    )
                key = route.rsplit("/", 1)[-1]
                if route.startswith("/api/state/"):
                    return self.reply(200, editor.state(key))
                if route.startswith("/api/image/"):
                    editor.load(key)
                    with Image.open(editor.record(key)["stimulus"]) as im:
                        buffer = BytesIO()
                        im.convert("RGB").save(buffer, format="PNG")
                    return self.reply(200, buffer.getvalue(), "image/png")
                self.reply(404, {"error": "Unknown route"})
            except (ValueError, OSError, KeyError) as error:
                self.reply(400, {"error": str(error)})

        def do_POST(self):
            if not self.allowed():
                return self.reply(403, {"error": "Editor authorization failed"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1000000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                key = data["id"]
                if self.path == "/api/preview":
                    mask, flags, changed = editor.render(
                        key, data["points"], data.get("regions"), data.get("region_order")
                    )
                    return self.reply(
                        200, dict(mask=editor.png(mask), flags=flags, changed=changed)
                    )
                if self.path == "/api/save":
                    return self.reply(
                        200,
                        editor.save(
                            key,
                            data["points"],
                            data.get("version"),
                            data.get("note", ""),
                            data.get("regions"),
                            data.get("region_order"),
                        ),
                    )
                self.reply(404, {"error": "Unknown route"})
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    url = "http://127.0.0.1:" + str(server.server_port) + "/#" + token
    print("Landmark editor:", url, flush=True)
    print(
        "Corrections:",
        output,
        "\nKeep this terminal open. Ctrl+C stops the editor.",
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
