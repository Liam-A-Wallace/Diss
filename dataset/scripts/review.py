"""Quick manual review: cut a manifest down to a small set of "good" images.

Loads a manifest CSV (default ``DFR60K_manifest.csv``), shows each image one at
a time, and lets you keep or axe it until you have a target number per label.

    K / Space  -> keep the image
    X / A      -> axe (skip) the image
    U          -> undo the last decision
    Esc        -> quit (saves the picks so far)

Usage (from anywhere, using the Diss venv):

    python Dataset/review.py                          # keep 25 REAL + 25 FAKE
    python Dataset/review.py --per-label 30           # keep 30 per label
    python Dataset/review.py --count 50               # 50 total (split evenly)
    python Dataset/review.py --csv other.csv          # review a different CSV
    python Dataset/review.py --labels fake            # review FAKE rows only
    python Dataset/review.py --seed 7                 # different candidate order
    python Dataset/review.py --no-copy                # don't copy the picked images

    # review only the fakes from FINAL_DATASET and add them to the existing picks:
    python Dataset/review.py --csv Dataset/FINAL_DATASET.csv --labels fake \
        --per-label 19 --append Dataset/picked.csv --skip-avatars

Output:
    Dataset/picked.csv                  - the kept rows (original columns + picked_file)
    Dataset/picked/real/real_001.png    - copied images (fake_*.jpg for FAKE)
    Dataset/picked/fake/fake_001.jpg
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

try:
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
    _HAS_TK = True
except Exception:  # pragma: no cover - headless environments
    _HAS_TK = False

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parent / "manifests" / "DFR60K_manifest.csv"
METADATA_CSV = HERE.parent / "metadata" / "image_metadata.csv"
OUT_CSV = HERE.parent / "outputs" / "picked.csv"
PICKED_DIR = common.RAW_DATA_ROOT / "datasets" / "picked"

PER_LABEL = 25        # default: 25 REAL + 25 FAKE = 50
SEED = 2026

# Cartoon/illustration avatar hosts in FINAL_DATASET.csv's fake set. These are
# not photographic faces, so --skip-avatars drops them from the review.
AVATAR_HOSTS = ("dicebear.com", "multiavatar.com")

FONT = "Segoe UI"

COLORS = {
    "bg": "#F4F5F7",
    "card": "#FFFFFF",
    "border": "#E1E4E8",
    "text": "#1E2022",
    "muted": "#686D76",
    "accent": "#0066FF",
    "real": "#1E8E3E", "real_bg": "#E6F4EA",
    "fake": "#D93025", "fake_bg": "#FCE8E6",
    "keep": "#1E8E3E", "keep_active": "#176B30",
    "axe": "#D93025", "axe_active": "#A50E0E",
    "undo": "#5F6368", "undo_active": "#3C4043",
}


def load_metadata(csv_path: Path | str = METADATA_CSV) -> dict[str, dict]:
    """Read image_metadata.csv into {image_id: row} (empty dict if missing)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return {r.get("image_id", ""): r for r in csv.DictReader(f)}


def build_pools(rows: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    """Return {label: shuffled candidate list} for REAL and FAKE."""
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {}
    for label in common.LABELS:
        pool = [r for r in rows if r.get("label") == label]
        rng.shuffle(pool)
        pools[label] = pool
    return pools


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def _copy_image(row: dict, dest: Path) -> bool:
    """Copy a picked image to ``dest``.

    Local paths are copied as-is; remote URLs are downloaded at full size
    (and cached by ``common``) and saved as JPEG.
    """
    url = row.get("image_url", "")
    src = common.resolve_image_path(url)
    if src is not None:
        shutil.copy2(src, dest)
        return True
    if str(url).startswith(("http://", "https://")):
        try:
            img = common.fetch_image(row, thumbnail=False)
            if img is not None:
                img.convert("RGB").save(dest, "JPEG", quality=95)
                return True
        except Exception:
            pass
    return False


def _picked_ext(row: dict) -> str:
    """Extension for a picked copy: source suffix for local, .jpg for remote."""
    url = row.get("image_url", "")
    if common.resolve_image_path(url) is not None:
        return Path(url).suffix or ".png"
    return ".jpg"


def write_outputs(kept: dict[str, list[dict]], out_csv: Path, copy_dir: Path,
                  copy_images: bool = True) -> None:
    """Write picked.csv and copy the kept images into copy_dir/{real,fake}/."""
    order = []
    for label in common.LABELS:
        order.extend(kept[label])
    if not order:
        print("[done] no images kept — nothing to save.")
        return

    if copy_images:
        (copy_dir / "real").mkdir(parents=True, exist_ok=True)
        (copy_dir / "fake").mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    out_rows: list[dict] = []
    for row in order:
        cond = "real" if row["label"] == "REAL" else "fake"
        counters[cond] = counters.get(cond, 0) + 1
        fname = f"{cond}_{counters[cond]:03d}{_picked_ext(row)}"
        picked_file = f"datasets/picked/{cond}/{fname}"

        if copy_images and not _copy_image(row, copy_dir / cond / fname):
            print(f"  [warn] could not save {row['image_id']}, skipping copy")

        out = dict(row)
        out["picked_file"] = picked_file
        out_rows.append(out)

    fields = list(out_rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[saved] {len(out_rows)} rows -> {out_csv}")
    print(f"        real: {counters.get('real', 0)}  fake: {counters.get('fake', 0)}")
    if copy_images:
        print(f"[copied] images -> {copy_dir}")


def merge_outputs(kept_new: dict[str, list[dict]], existing_rows: list[dict],
                  out_csv: Path, copy_dir: Path, copy_images: bool = True) -> None:
    """Append newly kept rows to an existing picked CSV.

    Existing rows keep their ``picked_file`` values untouched; new rows are
    numbered after them and their images copied into ``copy_dir``.
    """
    existing = [dict(r) for r in existing_rows]
    counters: dict[str, int] = {}
    for r in existing:
        cond = "real" if r["label"] == "REAL" else "fake"
        counters[cond] = counters.get(cond, 0) + 1

    if copy_images:
        (copy_dir / "real").mkdir(parents=True, exist_ok=True)
        (copy_dir / "fake").mkdir(parents=True, exist_ok=True)

    new_rows: list[dict] = []
    for label in common.LABELS:
        for row in kept_new[label]:
            cond = "real" if row["label"] == "REAL" else "fake"
            counters[cond] = counters.get(cond, 0) + 1
            fname = f"{cond}_{counters[cond]:03d}{_picked_ext(row)}"
            picked_file = f"datasets/picked/{cond}/{fname}"

            if copy_images and not _copy_image(row, copy_dir / cond / fname):
                print(f"  [warn] could not save {row['image_id']}, skipping copy")

            out = dict(row)
            out["picked_file"] = picked_file
            new_rows.append(out)

    combined = existing + new_rows
    if not combined:
        print("[done] nothing to save.")
        return

    fields = list(combined[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)

    print(f"[saved] {len(combined)} rows -> {out_csv} "
          f"({len(existing)} existing + {len(new_rows)} new)")
    print(f"        real: {counters.get('real', 0)}  fake: {counters.get('fake', 0)}")
    if copy_images and new_rows:
        print(f"[copied] {len(new_rows)} new images -> {copy_dir}")


# --------------------------------------------------------------------------
# Review GUI
# --------------------------------------------------------------------------
class ReviewApp:
    DISPLAY = (900, 900)   # fallback box when the label hasn't been laid out yet

    def __init__(self, root: tk.Tk, pools: dict[str, list[dict]],
                 targets: dict[str, int], meta: dict[str, dict],
                 copy_images: bool = True,
                 out_csv: Path = OUT_CSV, copy_dir: Path = PICKED_DIR,
                 existing_rows: list[dict] | None = None):
        self.root = root
        self.meta = meta
        self.copy_images = copy_images
        self.out_csv = out_csv
        self.copy_dir = copy_dir
        self.existing_rows = existing_rows
        self.target = dict(targets)
        self.pool: dict[str, list[dict]] = {l: list(pools[l]) for l in common.LABELS}
        self.kept: dict[str, list[dict]] = {l: [] for l in common.LABELS}
        self.axed: dict[str, list[dict]] = {l: [] for l in common.LABELS}
        self.history: list[tuple] = []   # ("keep"|"axe", label, row)
        self.current_label: str | None = None
        self.current_row: dict | None = None
        self.photo = None
        self.done = False

        self._build_ui()
        self._show_next()

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        root = self.root
        root.title("Seeing the Fake — Pick good images")
        root.configure(bg=COLORS["bg"])

        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w = min(1080, screen_w - 40)
        win_h = min(760, screen_h - 90)
        x = (screen_w - win_w) // 2
        y = max(0, (screen_h - win_h) // 2 - 10)
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        root.minsize(820, 540)

        # header
        header = tk.Frame(root, bg=COLORS["bg"])
        header.pack(fill="x", padx=24, pady=(18, 4))
        tk.Label(header, text="Pick good images (keep / axe)",
                 font=(FONT, 17, "bold"), bg=COLORS["bg"],
                 fg=COLORS["text"]).pack(anchor="w")
        self.status_var = tk.StringVar()
        tk.Label(header, textvariable=self.status_var, font=(FONT, 11),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w")

        # body
        body = tk.Frame(root, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=12)

        image_card = tk.Frame(body, bg=COLORS["card"],
                              highlightbackground=COLORS["border"],
                              highlightthickness=1)
        image_card.pack(side="left", fill="both", expand=True, padx=(0, 16))
        self.image_label = tk.Label(image_card, bg=COLORS["card"],
                                    fg=COLORS["muted"], text="Loading…",
                                    font=(FONT, 13))
        self.image_label.pack(expand=True)

        info_card = tk.Frame(body, bg=COLORS["card"],
                             highlightbackground=COLORS["border"],
                             highlightthickness=1)
        info_card.pack(side="left", fill="y")
        inner = tk.Frame(info_card, bg=COLORS["card"])
        inner.pack(anchor="nw", padx=20, pady=18)

        self.badge_label = tk.Label(inner, text="", font=(FONT, 17, "bold"),
                                    padx=18, pady=7)
        self.badge_label.pack(anchor="w", pady=(0, 14))

        self.info_var = tk.StringVar()
        tk.Label(inner, textvariable=self.info_var, font=(FONT, 10),
                 bg=COLORS["card"], fg=COLORS["muted"], justify="left",
                 anchor="w").pack(anchor="w", pady=(0, 18))

        # controls
        controls = tk.Frame(inner, bg=COLORS["card"])
        controls.pack(anchor="w", pady=(8, 0))
        self._button(controls, "Keep  (K)", self.keep, COLORS["keep"],
                     COLORS["keep_active"]).pack(side="left", padx=(0, 8))
        self._button(controls, "Axe  (X)", self.axe, COLORS["axe"],
                     COLORS["axe_active"]).pack(side="left", padx=(0, 8))
        self._button(controls, "Undo  (U)", self.undo, COLORS["undo"],
                     COLORS["undo_active"]).pack(side="left")

        tk.Label(inner, text="Esc = finish & save", font=(FONT, 9),
                 bg=COLORS["card"], fg=COLORS["muted"]).pack(anchor="w", pady=(14, 0))

        root.bind("<KeyPress-k>", lambda e: self.keep())
        root.bind("<KeyPress-K>", lambda e: self.keep())
        root.bind("<space>", lambda e: self.keep())
        root.bind("<KeyPress-x>", lambda e: self.axe())
        root.bind("<KeyPress-X>", lambda e: self.axe())
        root.bind("<KeyPress-a>", lambda e: self.axe())
        root.bind("<KeyPress-A>", lambda e: self.axe())
        root.bind("<KeyPress-u>", lambda e: self.undo())
        root.bind("<KeyPress-U>", lambda e: self.undo())
        root.bind("<Escape>", lambda e: self.finish())

    def _button(self, parent, text, cmd, bg, active):
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg="white",
                         activebackground=active, activeforeground="white",
                         relief="flat", bd=0, padx=18, pady=8,
                         font=(FONT, 11, "bold"), cursor="hand2",
                         highlightthickness=0)

    # ---- actions ----------------------------------------------------------
    def keep(self) -> None:
        if self.done or self.current_row is None:
            return
        label = self.current_label
        row = self.current_row
        self.kept[label].append(row)
        self.history.append(("keep", label, row))
        self.current_row = None
        self._show_next()

    def axe(self) -> None:
        if self.done or self.current_row is None:
            return
        label = self.current_label
        row = self.current_row
        self.axed[label].append(row)
        self.history.append(("axe", label, row))
        self.current_row = None
        self._show_next()

    def undo(self) -> None:
        if self.done or not self.history:
            return
        action, label, row = self.history.pop()
        if action == "keep":
            self._remove(self.kept[label], row)
        else:
            self._remove(self.axed[label], row)
        self.pool[label].insert(0, row)
        self.current_label = label
        self.current_row = row
        self._display(row)
        self._refresh_status()

    @staticmethod
    def _remove(lst: list[dict], row: dict) -> None:
        for i, r in enumerate(lst):
            if r is row:
                lst.pop(i)
                return

    def _labels_done(self) -> bool:
        return all(len(self.kept[l]) >= self.target[l] for l in common.LABELS)

    def _show_next(self) -> None:
        while True:
            if self._labels_done():
                self.finish()
                return
            label = self._next_label()
            if label is None:
                # ran out of candidates entirely
                self.finish()
                return
            if not self.pool[label]:
                print(f"[warn] {label} ran out of candidates "
                      f"({len(self.kept[label])}/{self.target[label]} kept)")
                # treat this label as done at its current count
                self.target[label] = len(self.kept[label])
                continue
            self.current_label = label
            self.current_row = self.pool[label].pop(0)
            self._display(self.current_row)
            return

    def _next_label(self) -> str | None:
        """Pick the label still needing candidates (alternating)."""
        for label in common.LABELS:
            if len(self.kept[label]) < self.target[label] and self.pool[label]:
                return label
        for label in common.LABELS:
            if len(self.kept[label]) < self.target[label]:
                return label
        return None

    def _display(self, row: dict) -> None:
        self._refresh_status()
        self.image_label.configure(image="", text="Loading image…")
        self._update_info(row)
        self.root.update_idletasks()
        self.root.update()

        img = common.fetch_image(row, thumbnail=True)
        if img is None:
            self._auto_axe()
            return

        # Fit the image to the available pane, upscaling small images too so
        # low-res portraits don't sit tiny in the corner.
        bw = self.image_label.winfo_width()
        bh = self.image_label.winfo_height()
        box = (bw, bh) if (bw > 60 and bh > 60) else self.DISPLAY
        img = self._fit(img, box)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo, text="")

    @staticmethod
    def _fit(img: Image.Image, box: tuple[int, int]) -> Image.Image:
        """Scale ``img`` to fit inside ``box``, allowing upscaling."""
        w, h = img.size
        bw, bh = box
        if bw <= 0 or bh <= 0:
            return img
        scale = min(bw / w, bh / h)
        if scale >= 1.0:
            return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
        out = img.copy()
        out.thumbnail(box, Image.LANCZOS)
        return out

    def _auto_axe(self) -> None:
        label = self.current_label
        row = self.current_row
        self.axed[label].append(row)
        self.history.append(("axe", label, row))
        self.current_row = None
        self._show_next()

    def _update_info(self, row: dict) -> None:
        label = row["label"]
        self.badge_label.configure(
            text=label, fg="white",
            bg=COLORS["real"] if label == "REAL" else COLORS["fake"])
        m = self.meta.get(row.get("image_id", ""), {})
        # gender/age/ethnicity may live in the row itself (e.g. FINAL_DATASET
        # or final_fake_gendered.csv) or in the enriched metadata CSV.
        gender = row.get("gender") or m.get("gender", "")
        age = row.get("age_group") or m.get("age_group", "")
        ethnicity = row.get("ethnicity") or m.get("ethnicity", "")
        hints = []
        if gender:
            hints.append(f"gender: {gender}")
        if age:
            hints.append(f"age: {age}")
        if ethnicity:
            hints.append(f"ethnicity: {ethnicity}")
        if m.get("face_ok") == "1":
            hints.append("face: single, prominent")
        elif m.get("face_ok") == "0":
            hints.append(f"face: FAIL ({m.get('num_faces', '?')} faces)")
        if m.get("is_photo") == "0":
            hints.append("photo: DRAWING")
        hint_str = "  |  ".join(hints) if hints else "(no metadata yet)"
        self.info_var.set(
            f"image_id: {row['image_id']}\n"
            f"source_dataset: {row.get('source_dataset', '')}\n"
            f"\n{hint_str}")

    def _refresh_status(self) -> None:
        parts = []
        for l in common.LABELS:
            parts.append(f"{l}: {len(self.kept[l])}/{self.target[l]} kept")
        self.status_var.set("  ".join(parts))

    # ---- finish -----------------------------------------------------------
    def finish(self) -> None:
        if self.done:
            return
        incomplete = [l for l in common.LABELS
                      if len(self.kept[l]) < self.target[l]]
        if incomplete and self.pool_has_remaining():
            msg = ("Not all targets met:\n"
                   + "\n".join(f"  {l}: {len(self.kept[l])}/{self.target[l]}" for l in incomplete)
                   + "\n\nFinish anyway and save what you have?")
            if not messagebox.askyesno("Finish early?", msg):
                return
        self._finalize()
        self.root.quit()

    def pool_has_remaining(self) -> bool:
        return any(self.pool[l] for l in common.LABELS)

    def _finalize(self) -> None:
        kept: dict[str, list[dict]] = {
            l: self.kept[l][: self.target[l]] for l in common.LABELS}
        if self.existing_rows is not None:
            merge_outputs(kept, self.existing_rows, self.out_csv, self.copy_dir,
                          copy_images=self.copy_images)
        else:
            write_outputs(kept, self.out_csv, self.copy_dir,
                          copy_images=self.copy_images)
        self.done = True

    def _quit(self) -> None:
        self._finalize()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pick good images from a manifest.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV),
                        help="source manifest CSV (default: Dataset/DFR60K_manifest.csv)")
    parser.add_argument("--per-label", type=int, default=PER_LABEL,
                        help="images to keep per label (default 25 -> 50 total)")
    parser.add_argument("--count", type=int, default=0,
                        help="total to keep, split evenly across labels "
                             "(overrides --per-label)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(OUT_CSV), help="output CSV")
    parser.add_argument("--copy-dir", default=str(PICKED_DIR), help="image copy dir")
    parser.add_argument("--no-copy", action="store_true",
                        help="don't copy the picked images")
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing output CSV")
    parser.add_argument("--labels", default="",
                        help="comma-separated labels to review (e.g. 'fake'); "
                             "default: all labels in the CSV")
    parser.add_argument("--skip-avatars", action="store_true",
                        help="exclude cartoon avatar hosts (dicebear, multiavatar)")
    parser.add_argument("--append", default="",
                        help="append new picks to an existing picked CSV "
                             "(e.g. Dataset/picked.csv) instead of overwriting")
    parser.add_argument("--meta", default=str(METADATA_CSV),
                        help="metadata CSV for on-screen hints")
    args = parser.parse_args(argv)

    if args.count:
        args.per_label = args.count // 2

    csv_path = Path(args.csv)
    if not csv_path.exists():
        parser.error(f"manifest not found: {csv_path}")

    # In append mode the output file already exists by design, so the
    # overwrite guard below is skipped.
    append_path = Path(args.append) if args.append else None
    out_path = append_path if append_path is not None else Path(args.out)
    if append_path is None and out_path.exists() and not args.overwrite:
        parser.error(
            f"output already exists: {out_path}\n"
            "Pass --out for a new file (e.g. --out picked_extra.csv), or "
            "--overwrite to replace it.")

    rows = common.load_rows(csv_path)
    print(f"[loaded] {len(rows)} rows from {csv_path}")

    if args.labels:
        wanted = {l.strip().upper() for l in args.labels.split(",") if l.strip()}
        rows = [r for r in rows if r.get("label") in wanted]
        print(f"[filter] labels {sorted(wanted)} -> {len(rows)} rows")

    if args.skip_avatars:
        before = len(rows)
        rows = [r for r in rows
                if not any(h in (r.get("image_url") or "") for h in AVATAR_HOSTS)]
        print(f"[skip-avatars] removed {before - len(rows)} rows "
              f"({len(rows)} remain)")

    existing_rows: list[dict] | None = None
    if append_path is not None:
        if append_path.exists():
            existing_rows = common.load_rows(append_path)
            existing_ids = {r.get("image_id", "") for r in existing_rows}
            rows = [r for r in rows if r.get("image_id") not in existing_ids]
            print(f"[append] {len(existing_rows)} existing rows in {append_path}; "
                  f"{len(rows)} new candidates remain")
        else:
            existing_rows = []
            print(f"[append] {append_path} does not exist yet; starting fresh")

    pools = build_pools(rows, seed=args.seed)
    targets = {l: args.per_label for l in common.LABELS}
    for label in common.LABELS:
        if not pools[label]:
            targets[label] = 0
            print(f"[note] no {label} candidates in this CSV -> target 0")
        else:
            print(f"[pool] {label}: {len(pools[label])} candidates")
    print(f"[target] {targets['REAL']} REAL + {targets['FAKE']} FAKE")

    meta = load_metadata(args.meta)
    if meta:
        print(f"[metadata] {len(meta)} rows loaded (shown as hints)")

    if not _HAS_TK:
        print("[error] tkinter is not available in this environment; "
              "cannot open the review GUI.")
        return 1

    root = tk.Tk()
    app = ReviewApp(root, pools, targets, meta,
                    copy_images=not args.no_copy,
                    out_csv=out_path, copy_dir=Path(args.copy_dir),
                    existing_rows=existing_rows)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
