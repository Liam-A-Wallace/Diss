"""Shared helpers for the Dataset audit + curation tools.

Run from anywhere: the scripts add their own directory to ``sys.path`` and
then ``import common``.

Notes on the data model
-----------------------
``FINAL_DATASET.csv`` has one row per image. The only continuous,
difficulty-like signal available for BOTH labels is ``confidence_score``
(``detection_difficulty`` is confounded with label: REAL is all "Easy",
FAKE is only "Medium"/"Hard"). There is no ``predicted_class`` column, so we
treat ``confidence_score`` as the detector's confidence in the *correct*
REAL/FAKE classification (high = easy, low = hard) and re-derive
Easy/Medium/Hard within each label using confidence tertiles.

This is a **model-derived** difficulty variable, not human difficulty.
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
MANIFESTS_DIR = HERE.parent / "dataset" / "manifests"   # shared manifests from the dataset/ toolchain
DATASET_CSV = MANIFESTS_DIR / "FINAL_DATASET.csv"
CACHE_DIR = HERE / ".image_cache"
RAW_DATA_ROOT = Path(os.environ.get("RAW_DATA_ROOT") or HERE.parent.parent / "Dataset")
REPO = RAW_DATA_ROOT.parent                        # root image_url paths are relative to

LABELS = ("REAL", "FAKE")
DIFFICULTIES = ("Easy", "Medium", "Hard")

_HEADERS = {"User-Agent": "Mozilla/5.0 (research dataset curation script)"}


def load_rows(csv_path: Path | str | None = None) -> list[dict]:
    """Read the dataset CSV into a list of dicts (strings, not parsed)."""
    csv_path = Path(csv_path or DATASET_CSV)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _is_numeric(value) -> bool:
    """True if ``value`` can be parsed as a float (None/blank -> False)."""
    if value is None:
        return False
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def derive_difficulty(rows: list[dict], high_conf_is_easy: bool = True) -> dict[str, str]:
    """Assign Easy/Medium/Hard by confidence tertiles *within* each label.

    Returns a mapping ``image_id -> difficulty``. Rows without a numeric
    ``confidence_score`` (e.g. the 60K manifest) are skipped — no hint is
    produced for them.

    ``high_conf_is_easy=True``  -> highest confidence third  = Easy, lowest = Hard.
    ``high_conf_is_easy=False`` -> highest confidence third  = Hard, lowest = Easy
                                   (i.e. "looks real" interpretation).
    """
    mapping: dict[str, str] = {}
    for label in LABELS:
        class_rows = [r for r in rows
                      if r.get("label") == label
                      and _is_numeric(r.get("confidence_score"))]
        # stable sort: confidence ascending, image_id as tie-break
        class_rows.sort(key=lambda r: (float(r["confidence_score"]), r["image_id"]))
        n = len(class_rows)

        base, rem = divmod(n, 3)
        sizes = [base + (1 if i < rem else 0) for i in range(3)]  # largest first
        starts = [0]
        for s in sizes[:-1]:
            starts.append(starts[-1] + s)

        # chunk 0 = lowest confidence, chunk 2 = highest confidence
        if high_conf_is_easy:
            chunk_to_difficulty = {0: "Hard", 1: "Medium", 2: "Easy"}
        else:
            chunk_to_difficulty = {0: "Easy", 1: "Medium", 2: "Hard"}

        for chunk, difficulty in chunk_to_difficulty.items():
            lo = starts[chunk]
            hi = starts[chunk] + sizes[chunk]
            for r in class_rows[lo:hi]:
                mapping[r["image_id"]] = difficulty
    return mapping


def cell_key(row: dict, difficulty_of: dict[str, str]) -> tuple[str, str | None]:
    """Return the (label, difficulty) cell for a row (difficulty may be None)."""
    return (row["label"], difficulty_of.get(row["image_id"]))


def _download_pil(row: dict, timeout: float = 8.0) -> Image.Image | None:
    url = row.get("image_url", "")
    for _ in range(2):
        try:
            resp = requests.get(url, timeout=timeout, headers=_HEADERS)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            time.sleep(0.3)
    return None


def _is_local_path(url: str) -> bool:
    """True if ``image_url`` is an existing file on disk (not a remote URL)."""
    if not url:
        return False
    if url.startswith(("http://", "https://", "ftp://")):
        return False
    return resolve_image_path(url) is not None


# --------------------------------------------------------------------------
# Local image path resolution
# --------------------------------------------------------------------------
# ``image_url`` values in the CSVs may be absolute (machine-specific), relative
# to this project's folder, or relative to the repo root. Resolve any of those
# to an actual file so the pipeline works no matter where the project (and the
# deepfake-vs-real-60k folder) lives.
_SKIP_DIR_NAMES = {".venv", ".git", "__pycache__", ".image_cache", ".cache"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

_FILENAME_INDEX: dict[str, Path] | None = None


def _filename_index() -> dict[str, Path]:
    """Lazily index image filenames under the project and repo root (cached).

    Only used as a last resort when a path can't be resolved any other way
    (e.g. an old CSV with an absolute path from another machine).
    """
    global _FILENAME_INDEX
    if _FILENAME_INDEX is not None:
        return _FILENAME_INDEX
    index: dict[str, Path] = {}
    for base in (HERE, RAW_DATA_ROOT):
        try:
            roots = [p for p in base.iterdir() if p.is_dir()]
        except OSError:
            continue
        for root in roots:
            if root.name in _SKIP_DIR_NAMES:
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
                for fname in filenames:
                    if Path(fname).suffix.lower() in _IMAGE_EXTS:
                        index.setdefault(fname.lower(), Path(dirpath) / fname)
    _FILENAME_INDEX = index
    return index


def resolve_image_path(url: str) -> Path | None:
    """Resolve an ``image_url`` to an existing local file, or ``None``.

    Resolution order: as written -> project-relative -> repo-root-relative ->
    filename search under the local image folders. Remote URLs return ``None``.
    """
    if not url:
        return None
    if url.startswith(("http://", "https://", "ftp://")):
        return None
    p = Path(url)
    if p.is_file():
        return p
    for base in (HERE, RAW_DATA_ROOT, REPO):
        cand = base / url
        if cand.is_file():
            return cand
    return _filename_index().get(p.name.lower())


def portable_image_url(url: str) -> str:
    """Return a directory-agnostic ``image_url`` for CSV output.

    Local paths are rewritten relative to the repo root (``HERE.parent``) with
    forward slashes; remote URLs and unresolvable paths are returned unchanged.
    """
    local = resolve_image_path(url)
    if local is None:
        return url
    try:
        return local.relative_to(REPO).as_posix()
    except ValueError:
        return local.as_posix()


def _load_pil(row: dict, timeout: float = 20.0) -> Image.Image | None:
    """Open the image for a row from a local path or, failing that, the web."""
    url = row.get("image_url", "")
    local = resolve_image_path(url)
    if local is not None:
        try:
            return Image.open(local).convert("RGB")
        except Exception:
            return None
    if not url.startswith(("http://", "https://", "ftp://")):
        # a local path that doesn't exist on this machine -> nothing to download
        return None
    return _download_pil(row, timeout)


def fetch_image(row: dict, thumbnail: bool = True) -> Image.Image | None:
    """Load (with on-disk cache) an image for a row.

    Works for both remote URLs and local file paths in ``image_url``.

    ``thumbnail=True``  -> cached, capped at 900px (fast review display).
    ``thumbnail=False`` -> cached at original resolution (final stimuli).
    """
    sub = "thumb" if thumbnail else "full"
    dest = CACHE_DIR / sub / f"{row['image_id']}.jpg"
    if dest.exists() and dest.stat().st_size > 0:
        try:
            return Image.open(dest).convert("RGB")
        except Exception:
            pass

    img = _load_pil(row)
    if img is None:
        return None
    if thumbnail:
        img.thumbnail((900, 900), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=92)
    return img


# --------------------------------------------------------------------------
# Face check (used by the optional "face only" pre-filter in curate.py)
# --------------------------------------------------------------------------
_HAAR_CASCADE = None

_HAAR_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)


def _haar_path():
    """Return a local path to the Haar cascade, downloading it if needed.

    Newer opencv-python wheels no longer bundle the cascade XMLs, so we fall
    back to OpenCV's official copy on GitHub.
    """
    import cv2

    bundled = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if bundled.exists():
        return bundled

    local = CACHE_DIR / "models" / "haarcascade_frontalface_default.xml"
    if local.exists() and local.stat().st_size > 0:
        return local

    local.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(_HAAR_URL, timeout=30, headers=_HEADERS)
    resp.raise_for_status()
    local.write_bytes(resp.content)
    return local


def _haar():
    """Lazily load OpenCV's frontal-face cascade."""
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        import cv2
        _HAAR_CASCADE = cv2.CascadeClassifier(str(_haar_path()))
        if _HAAR_CASCADE.empty():
            raise RuntimeError("Failed to load the face-detection cascade")
    return _HAAR_CASCADE


def face_check(img, min_face_fraction=0.1, max_faces=1):
    """Decide whether a PIL image is a "single, prominent face" photo.

    Returns a dict::

        {"ok": bool, "faces": int, "max_fraction": float, "reason": str}

    ``reason`` is one of "ok", "no_face", "multiple_faces",
    "face_too_small", or "unavailable" (OpenCV missing -> passes).
    """
    try:
        import cv2
        import numpy as np

        arr = np.asarray(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        faces = _haar().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
    except Exception:
        return {"ok": True, "faces": 0, "max_fraction": 0.0,
                "reason": "unavailable"}
    n = len(faces)
    total = gray.shape[0] * gray.shape[1]
    max_fraction = max((float(w * h) / total for (x, y, w, h) in faces),
                       default=0.0)

    if n == 0:
        return {"ok": False, "faces": 0, "max_fraction": 0.0, "reason": "no_face"}
    if n > max_faces:
        return {"ok": False, "faces": n, "max_fraction": max_fraction,
                "reason": "multiple_faces"}
    if max_fraction < min_face_fraction:
        return {"ok": False, "faces": n, "max_fraction": max_fraction,
                "reason": "face_too_small"}
    return {"ok": True, "faces": n, "max_fraction": max_fraction, "reason": "ok"}
