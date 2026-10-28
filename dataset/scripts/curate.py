"""Dataset curation tool for "Seeing the Fake" — Study 1.

Pipeline
--------
    FINAL_DATASET.csv  +  DFR60K_manifest.csv (Deepfake-vs-Real-60K, local files)
        -> split REAL / FAKE
        -> interleave the two sources (~50/50) into label-level candidate pools
        -> manual review: assign Easy/Medium/Hard yourself, or axe
        -> 5 kept per (label x difficulty) cell = 30 final stimuli
        -> writes curated CSV + copies images into Diss/stimuli/real|fake

Usage (from anywhere, using the Diss venv):

    python Dataset/curate.py                     # launch the review GUI
    python Dataset/curate.py --dry-run           # build the pool, print it, no GUI
    python Dataset/curate.py --trials            # also (re)write Diss/trials.csv
    python Dataset/curate.py --extra-csv other.csv  # second source (default: DFR60K_manifest.csv)

Keyboard in the review window:
    1 / 2 / 3  -> keep as Easy / Medium / Hard
    K / Space  -> keep with the suggested difficulty (default Medium)
    X / A      -> axe (pull the next candidate from the same label)
    U          -> undo last decision
    Esc        -> quit (saves progress log so far)

Candidates are drawn from a label-level pool that interleaves FINAL_DATASET
with the 60K manifest at MIX_RATIO. Difficulty is assigned by you at keep time
(keys 1/2/3), so the six (label x difficulty) cells fill up as you review.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from PIL import Image, ImageTk
    _HAS_TK = True
except Exception:  # pragma: no cover - headless environments
    _HAS_TK = False

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DISS_DIR = HERE.parent.parent                        # Diss/

OUT_CSV = HERE.parent / "outputs" / "curated.csv"          # final 30 (or fewer if finished early)
REVIEW_LOG = HERE.parent / "outputs" / "review_log.csv"    # every keep/axe decision
STIM_REAL_DIR = DISS_DIR / "stimuli" / "real"
STIM_FAKE_DIR = DISS_DIR / "stimuli" / "fake"
TRIALS_CSV = DISS_DIR / "trials.csv"    # only written with --trials
METADATA_CSV = HERE.parent / "metadata" / "image_metadata.csv"   # optional enriched labels (from enrich.py)

# Optional hard cap on candidates staged per label. None = uncapped: use every
# available row (FINAL + 60K interleaved) until the 6 cells are full.
CANDIDATES_PER_LABEL = None
# Fraction of the candidate pool drawn from FINAL_DATASET (the rest from the
# 60K manifest). 0.5 = half from each.
MIX_RATIO = 0.5

# Final target per cell (5 * 6 = 30).
CELL_TARGETS = {
    ("REAL", "Easy"): 5,
    ("REAL", "Medium"): 5,
    ("REAL", "Hard"): 5,
    ("FAKE", "Easy"): 5,
    ("FAKE", "Medium"): 5,
    ("FAKE", "Hard"): 5,
}

HIGH_CONF_IS_EASY = True   # True: high confidence = Easy. See common.py.
SEED = 2026                # reproducible candidate order

# Face pre-filter (used when --face-filter is passed).
MIN_FACE_FRACTION = 0.1    # face box must cover >= 10% of the image
MAX_FACES = 1              # a "face only" photo should have exactly one face

DIFF_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}
CELL_ORDER = [("REAL", "Easy"), ("REAL", "Medium"), ("REAL", "Hard"),
              ("FAKE", "Easy"), ("FAKE", "Medium"), ("FAKE", "Hard")]


# --------------------------------------------------------------------------
# GUI theme
# --------------------------------------------------------------------------
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
    "easy": "#1E8E3E", "easy_bg": "#E6F4EA",
    "medium": "#9A6700", "medium_bg": "#FFF3D6",
    "hard": "#C5221F", "hard_bg": "#FCE8E6",
    "keep": "#1E8E3E", "keep_active": "#176B30",
    "axe": "#D93025", "axe_active": "#A50E0E",
    "undo": "#5F6368", "undo_active": "#3C4043",
    "finish": "#0066FF", "finish_active": "#0052CC",
}

# (difficulty label, colour key) used for the difficulty chip.
DIFF_STYLE = {
    "Easy": ("Easy", "easy"),
    "Medium": ("Medium", "medium"),
    "Hard": ("Hard", "hard"),
}


def _styled_button(parent, text, command, bg, active):
    """A flat, solid-colour button (more attractive than the default)."""
    return tk.Button(parent, text=text, command=command, bg=bg, fg="white",
                     activebackground=active, activeforeground="white",
                     relief="flat", bd=0, padx=22, pady=8,
                     font=(FONT, 11, "bold"), cursor="hand2", highlightthickness=0)


def load_metadata(csv_path: Path | str = METADATA_CSV) -> dict[str, dict]:
    """Read image_metadata.csv into {image_id: row} (empty dict if missing)."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        return {r.get("image_id", ""): r for r in csv.DictReader(f)}


# --------------------------------------------------------------------------
# Candidate-pool construction
# --------------------------------------------------------------------------
def _interleave(a: list[dict], b: list[dict], ratio: float) -> list[dict]:
    """Interleave two lists so ~``ratio`` of the output comes from ``a``."""
    out: list[dict] = []
    i = j = 0
    while i < len(a) or j < len(b):
        if j >= len(b) or (i < len(a) and (i + 1) / (i + j + 1) <= ratio):
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out


def build_pools(rows: list[dict],
                seed: int = SEED,
                candidates_per_label: int | None = CANDIDATES_PER_LABEL,
                mix_ratio: float = MIX_RATIO) -> tuple[dict, dict]:
    """Return (hint_of, label_pools).

    ``hint_of`` maps image_id -> suggested difficulty (confidence tertiles),
    only for rows that have a numeric confidence_score (60K rows get no hint).

    ``label_pools`` maps "REAL"/"FAKE" -> interleaved candidate list
    (FINAL_DATASET + 60K at ``mix_ratio``), optionally capped at
    ``candidates_per_label`` (None = use every available row).
    """
    for r in rows:
        r.setdefault("source_dataset", "final")

    hint_of = common.derive_difficulty(rows, high_conf_is_easy=HIGH_CONF_IS_EASY)
    rng = random.Random(seed)

    label_pools: dict[str, list[dict]] = {}
    for label in common.LABELS:
        final_rows = [r for r in rows
                      if r.get("label") == label
                      and r.get("source_dataset") != "dfr60k"]
        dfr_rows = [r for r in rows
                    if r.get("label") == label
                    and r.get("source_dataset") == "dfr60k"]
        rng.shuffle(final_rows)
        rng.shuffle(dfr_rows)
        interleaved = _interleave(final_rows, dfr_rows, mix_ratio)
        if candidates_per_label is not None:
            interleaved = interleaved[:candidates_per_label]
        label_pools[label] = interleaved

    return hint_of, label_pools


def print_pool_summary(label_pools: dict) -> None:
    print("---- candidate pools (label level) ----")
    for label in common.LABELS:
        pool = label_pools[label]
        n_final = sum(1 for r in pool if r.get("source_dataset") != "dfr60k")
        n_dfr = len(pool) - n_final
        print(f"  {label:5s}  total={len(pool)}  final={n_final}  dfr60k={n_dfr}")
    targets = ", ".join(f"{c[0]}/{c[1]}={CELL_TARGETS[c]}" for c in CELL_ORDER)
    print(f"  targets: {targets}")
    print()


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def save_outputs(final_rows: list[dict]) -> None:
    """Copy the final stimuli into Diss/stimuli and write curated.csv."""
    final_rows.sort(key=lambda r: (r["label"], DIFF_ORDER[r["derived_difficulty"]]))

    STIM_REAL_DIR.mkdir(parents=True, exist_ok=True)
    STIM_FAKE_DIR.mkdir(parents=True, exist_ok=True)

    counters: dict[str, int] = {}
    out_rows: list[dict] = []
    for i, row in enumerate(final_rows, start=1):
        cond = "real" if row["label"] == "REAL" else "fake"
        counters[cond] = counters.get(cond, 0) + 1
        fname = f"{cond}_{counters[cond]:03d}.png"
        dest = (STIM_REAL_DIR if cond == "real" else STIM_FAKE_DIR) / fname

        img = common.fetch_image(row, thumbnail=False)
        if img is not None:
            img.save(dest, "PNG")
        else:
            print(f"  [warn] could not download image {row['image_id']}, skipping copy")

        out = dict(row)
        out["image_url"] = common.portable_image_url(row.get("image_url", ""))
        out["derived_difficulty"] = row["derived_difficulty"]
        out["cell"] = f"{row['label']}|{row['derived_difficulty']}"
        out["stimulus"] = str(Path("stimuli") / cond / fname)
        out_rows.append(out)

    fields = list(out_rows[0].keys()) if out_rows else []
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[saved] {len(out_rows)} images -> {STIM_REAL_DIR} / {STIM_FAKE_DIR}")
    print(f"[saved] {OUT_CSV}")
    print(f"        real: {counters.get('real', 0)}  fake: {counters.get('fake', 0)}")


def write_trials_csv(final_rows: list[dict]) -> None:
    """Write Diss/trials.csv (stimulus, condition, difficulty).

    Filenames are numbered the same way as ``save_outputs`` (real_001..,
    fake_001.. in label order, then Easy < Medium < Hard).
    """
    final_rows.sort(key=lambda r: (r["label"], DIFF_ORDER[r["derived_difficulty"]]))
    counters: dict[str, int] = {}
    with open(TRIALS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stimulus", "condition", "difficulty"])
        for row in final_rows:
            cond = "real" if row["label"] == "REAL" else "fake"
            counters[cond] = counters.get(cond, 0) + 1
            fname = f"{cond}_{counters[cond]:03d}.png"
            writer.writerow([
                f"stimuli/{cond}/{fname}",
                cond,
                row["derived_difficulty"].lower(),
            ])
    print(f"[saved] {TRIALS_CSV}")


def write_review_log(log_entries: list[dict]) -> None:
    if not log_entries:
        return
    fields = list(log_entries[0].keys())
    with open(REVIEW_LOG, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(log_entries)
    print(f"[saved] {REVIEW_LOG} ({len(log_entries)} decisions)")


# --------------------------------------------------------------------------
# Review GUI
# --------------------------------------------------------------------------
class ReviewApp:
    DISPLAY = (560, 560)

    def __init__(self, root: tk.Tk, rows: list[dict],
                 hint_of: dict[str, str],
                 label_pools: dict[str, list[dict]],
                 face_filter: bool = False,
                 auto_filter: bool = False,
                 min_face_fraction: float = MIN_FACE_FRACTION,
                 kept_seed: dict | None = None,
                 prior_log: list[dict] | None = None):
        self.root = root
        self.rows = rows
        self.hint_of = hint_of
        self.face_filter = face_filter
        self.auto_filter = auto_filter
        self.min_face_fraction = min_face_fraction
        self.target = dict(CELL_TARGETS)
        self.pool: dict[str, list[dict]] = {l: list(label_pools[l]) for l in common.LABELS}
        self.kept: dict[tuple[str, str], list[dict]] = {
            c: list(kept_seed[c]) if (kept_seed and c in kept_seed) else []
            for c in CELL_ORDER}
        self.axed: dict[str, list[dict]] = {l: [] for l in common.LABELS}
        self.history: list[tuple] = []   # ("keep"|"axe", label, row, difficulty_or_None)
        self.log_entries: list[dict] = list(prior_log) if prior_log else []
        self.current_label: str | None = None
        self.current_row: dict | None = None
        self.photo = None
        self.done = False

        self._build_ui()
        self._show_next()

    # ---- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        root = self.root
        root.title("Seeing the Fake — Stimulus Curation")
        root.configure(bg=COLORS["bg"])

        # Size the window to fit the usable screen (leaving room for the
        # taskbar/title bar) and centre it, so nothing is hidden.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w = min(1120, screen_w - 40)
        win_h = min(820, screen_h - 90)
        x = (screen_w - win_w) // 2
        y = max(0, (screen_h - win_h) // 2 - 10)
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        root.minsize(860, 560)

        style = ttk.Style(root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Cell.Horizontal.TProgressbar",
                        troughcolor=COLORS["border"], background=COLORS["accent"],
                        bordercolor=COLORS["border"], lightcolor=COLORS["accent"],
                        darkcolor=COLORS["accent"], thickness=10)

        # ---- header ----
        header = tk.Frame(root, bg=COLORS["bg"])
        header.pack(fill="x", padx=24, pady=(18, 4))
        tk.Label(header, text="Seeing the Fake — Stimulus Curation",
                 font=(FONT, 17, "bold"), bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor="w")
        self.overall_var = tk.StringVar()
        tk.Label(header, textvariable=self.overall_var, font=(FONT, 11),
                 bg=COLORS["bg"], fg=COLORS["muted"]).pack(anchor="w")

        # ---- body: image card + info card ----
        body = tk.Frame(root, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=24, pady=12)

        image_card = tk.Frame(body, bg=COLORS["card"],
                              highlightbackground=COLORS["border"], highlightthickness=1)
        image_card.pack(side="left", fill="both", expand=True, padx=(0, 16))
        self.image_label = tk.Label(image_card, bg=COLORS["card"], fg=COLORS["muted"],
                                    text="Loading…", font=(FONT, 13))
        self.image_label.pack(expand=True)

        info_card = tk.Frame(body, bg=COLORS["card"],
                             highlightbackground=COLORS["border"], highlightthickness=1)
        info_card.pack(side="left", fill="y")
        inner = tk.Frame(info_card, bg=COLORS["card"])
        inner.pack(anchor="nw", padx=20, pady=18)

        # badges
        self.badge_label = tk.Label(inner, text="", font=(FONT, 17, "bold"), padx=18, pady=7)
        self.badge_label.pack(anchor="w", pady=(0, 8))
        self.diff_label = tk.Label(inner, text="", font=(FONT, 11, "bold"), padx=12, pady=4)
        self.diff_label.pack(anchor="w", pady=(0, 14))

        # metadata grid
        self.meta_fields = [
            ("Source dataset", "source_dataset"),
            ("Confidence", "confidence_score"),
            ("Source", "source"),
            ("Fake method", "fake_method"),
            ("Ethnicity", "ethnicity"),
            ("Gender", "gender"),
            ("Age group", "age_group"),
            ("Image quality", "image_quality"),
            ("Resolution", "resolution"),
            ("Dataset split", "dataset_split"),
            ("Image ID", "image_id"),
        ]
        self.meta_vars: dict[str, tk.StringVar] = {}
        meta = tk.Frame(inner, bg=COLORS["card"])
        meta.pack(anchor="w")
        for i, (label, key) in enumerate(self.meta_fields):
            tk.Label(meta, text=label, font=(FONT, 10), bg=COLORS["card"],
                     fg=COLORS["muted"], anchor="w").grid(row=i, column=0, sticky="w",
                                                          padx=(0, 16), pady=2)
            var = tk.StringVar(value="")
            self.meta_vars[key] = var
            tk.Label(meta, textvariable=var, font=(FONT, 10, "bold"), bg=COLORS["card"],
                     fg=COLORS["text"], anchor="w").grid(row=i, column=1, sticky="w", pady=2)

        tk.Label(inner, text="\nReview for:\n• face visibility  • obvious artefacts\n• poor crop  • duplicates / near-duplicates\n• anything unsuitable for eye tracking",
                 font=(FONT, 9), bg=COLORS["card"], fg=COLORS["muted"],
                 justify="left", anchor="w").pack(anchor="w", pady=(14, 0))

        # ---- progress ----
        progress = tk.Frame(root, bg=COLORS["bg"])
        progress.pack(fill="x", padx=24, pady=(4, 0))
        self.cell_title_vars: dict = {}
        self.cell_bars: dict = {}
        for idx, cell in enumerate(CELL_ORDER):
            r, c = divmod(idx, 3)
            box = tk.Frame(progress, bg=COLORS["bg"])
            box.grid(row=r, column=c, sticky="ew", padx=8, pady=3)
            progress.columnconfigure(c, weight=1)
            var = tk.StringVar()
            self.cell_title_vars[cell] = var
            tk.Label(box, textvariable=var, font=(FONT, 9), bg=COLORS["bg"],
                     fg=COLORS["text"], anchor="w").pack(fill="x")
            bar = ttk.Progressbar(box, maximum=5, value=0,
                                  style="Cell.Horizontal.TProgressbar")
            bar.pack(fill="x")
            self.cell_bars[cell] = bar

        # ---- controls ----
        controls = tk.Frame(root, bg=COLORS["bg"])
        controls.pack(pady=(14, 6))
        _styled_button(controls, "Easy  (1)", lambda: self.keep_as("Easy"),
                       COLORS["easy"], COLORS["keep_active"]).pack(side="left", padx=6)
        _styled_button(controls, "Med  (2)", lambda: self.keep_as("Medium"),
                       COLORS["medium"], COLORS["medium"]).pack(side="left", padx=6)
        _styled_button(controls, "Hard  (3)", lambda: self.keep_as("Hard"),
                       COLORS["hard"], COLORS["hard"]).pack(side="left", padx=6)
        _styled_button(controls, "Axe  (X)", self.axe, COLORS["axe"],
                       COLORS["axe_active"]).pack(side="left", padx=6)
        _styled_button(controls, "Undo  (U)", self.undo, COLORS["undo"],
                       COLORS["undo_active"]).pack(side="left", padx=6)
        _styled_button(controls, "Finish", self.finish, COLORS["finish"],
                       COLORS["finish_active"]).pack(side="left", padx=6)
        tk.Label(root, text="1 / 2 / 3 = keep Easy / Med / Hard   ·   K / Space = keep suggested   ·   X / A = axe   ·   U = undo   ·   Esc = save & quit",
                 font=(FONT, 9), bg=COLORS["bg"], fg=COLORS["muted"]).pack(pady=(0, 10))

        for key, fn in (("1", lambda: self.keep_as("Easy")),
                        ("2", lambda: self.keep_as("Medium")),
                        ("3", lambda: self.keep_as("Hard")),
                        ("k", self.keep_hint), ("K", self.keep_hint),
                        (" ", self.keep_hint),
                        ("x", self.axe), ("X", self.axe), ("a", self.axe),
                        ("A", self.axe), ("u", self.undo), ("U", self.undo)):
            self.root.bind(f"<KeyPress-{key}>", lambda e, f=fn: f())
        self.root.bind("<Escape>", lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    # ---- helpers ----------------------------------------------------------
    def _first_incomplete(self):
        for c in CELL_ORDER:
            if len(self.kept[c]) < self.target[c]:
                return c
        return None

    def _refresh_status(self) -> None:
        total_kept = sum(len(self.kept[c]) for c in CELL_ORDER)
        self.overall_var.set(f"{total_kept} / {sum(self.target.values())} images kept")
        for c in CELL_ORDER:
            k = len(self.kept[c])
            self.cell_title_vars[c].set(f"{c[0]} · {c[1]}   {k}/{self.target[c]}")
            self.cell_bars[c]["value"] = k

    def _update_info(self, row: dict) -> None:
        if row.get("label") == "REAL":
            self.badge_label.configure(text="REAL", bg=COLORS["real_bg"], fg=COLORS["real"])
        else:
            self.badge_label.configure(text="FAKE", bg=COLORS["fake_bg"], fg=COLORS["fake"])

        hint = self.hint_of.get(row.get("image_id", ""), "")
        if hint:
            diff_key = hint.lower()
            text = f"Suggested: {hint}   (press 1 / 2 / 3 to set difficulty)"
        else:
            diff_key = "medium"
            text = "Assign difficulty:   1 Easy   ·   2 Medium   ·   3 Hard"
        self.diff_label.configure(text=text,
                                  bg=COLORS[f"{diff_key}_bg"], fg=COLORS[diff_key])
        for _label, key in self.meta_fields:
            self.meta_vars[key].set(str(row.get(key, "")))

    def _label_incomplete(self, label: str) -> bool:
        return any(len(self.kept[c]) < self.target[c]
                   for c in CELL_ORDER if c[0] == label)

    def _next_label(self) -> str | None:
        """Pick the label to show next: the one that still needs candidates and
        has the fewest total keeps (to keep the six cells filling evenly)."""
        labels = [l for l in common.LABELS if self._label_incomplete(l)]
        if not labels:
            return None
        labels.sort(key=lambda l: sum(len(self.kept[c])
                                      for c in CELL_ORDER if c[0] == l))
        return labels[0]

    @staticmethod
    def _remove_identity(lst: list, row) -> bool:
        for i, r in enumerate(lst):
            if r is row:
                del lst[i]
                return True
        return False

    def _log(self, row: dict, label: str, decision: str,
             difficulty: str | None = None) -> None:
        self.log_entries.append({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "image_id": row["image_id"],
            "label": label,
            "difficulty": difficulty or "",
            "decision": decision,
            "confidence_score": row.get("confidence_score", ""),
            "source": row.get("source", ""),
            "source_dataset": row.get("source_dataset", ""),
            "gender": row.get("gender", ""),
            "age_group": row.get("age_group", ""),
        })

    def _auto_axe(self, reason: str) -> None:
        """Reject the current candidate without showing it (bad download/face)."""
        label = self.current_label
        row = self.current_row
        self.axed[label].append(row)
        self.history.append(("axe", label, row, None))
        self._log(row, label, reason)
        self.current_row = None
        self._show_next()

    # ---- actions ----------------------------------------------------------
    def keep_hint(self) -> None:
        """Keep using the confidence-derived hint (default Medium)."""
        if self.current_row is None:
            return
        hint = self.hint_of.get(self.current_row.get("image_id", ""), "Medium")
        self.keep_as(hint)

    def keep_as(self, difficulty: str) -> None:
        if self.done or self.current_row is None:
            return
        label = self.current_label
        cell = (label, difficulty)
        if len(self.kept[cell]) >= self.target[cell]:
            messagebox.showwarning(
                "Cell full",
                f"{label} · {difficulty} already has {self.target[cell]} images.\n"
                "Pick a different difficulty or axe this image.")
            return
        row = self.current_row
        row["derived_difficulty"] = difficulty
        self.kept[cell].append(row)
        self.history.append(("keep", label, row, difficulty))
        self._log(row, label, "keep", difficulty)
        self.current_row = None
        self._show_next()

    def axe(self) -> None:
        if self.done or self.current_row is None:
            return
        label = self.current_label
        row = self.current_row
        self.axed[label].append(row)
        self.history.append(("axe", label, row, None))
        self._log(row, label, "axe")
        self.current_row = None
        self._show_next()

    def undo(self) -> None:
        if self.done or not self.history:
            return
        action, label, row, difficulty = self.history.pop()
        if action == "keep":
            cell = (label, difficulty)
            self._remove_identity(self.kept[cell], row)
            row.pop("derived_difficulty", None)
        else:
            self._remove_identity(self.axed[label], row)
        # put the undone row back at the front of its label pool and re-show it
        self.pool[label].insert(0, row)
        self.current_label = label
        self.current_row = row
        self._display(row)
        self._refresh_status()

    def _show_next(self) -> None:
        while True:
            if self._first_incomplete() is None:
                self.finish()
                return
            label = self._next_label()
            if not self.pool[label]:
                # Ran out of candidates for this label: accept what we have
                # for its cells, then continue with the other label.
                for c in CELL_ORDER:
                    if c[0] == label:
                        self.target[c] = min(self.target[c], len(self.kept[c]))
                        print(f"[warn] {c} ran out of candidates "
                              f"({len(self.kept[c])} kept, target {self.target[c]})")
                if all(not self.pool[l] for l in common.LABELS):
                    self.finish()
                    return
                continue
            self.current_label = label
            self.current_row = self.pool[label].pop(0)
            self._display(self.current_row)
            return

    def _display(self, row: dict) -> None:
        self._refresh_status()
        self.image_label.configure(image="", text="Loading image…")
        self._update_info(row)
        self.root.update_idletasks()
        self.root.update()  # repaint so "Loading image…" shows during the fetch

        img = common.fetch_image(row, thumbnail=True)
        if img is None:
            # broken/undownloadable image -> treat as an axe
            self._auto_axe("download_failed")
            return

        if self.auto_filter:
            reason = None
            if str(row.get("is_photo", "")) == "0":
                reason = "drawing"
            elif str(row.get("face_ok", "")) == "0":
                reason = "bad_face"
            if reason:
                self._auto_axe(reason)
                return

        if self.face_filter:
            check = common.face_check(
                img,
                min_face_fraction=self.min_face_fraction,
                max_faces=MAX_FACES,
            )
            if not check["ok"]:
                # not a single, prominent face -> auto-axe and pull a fresh one
                self._auto_axe(check["reason"])
                return

        img.thumbnail(self.DISPLAY, Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self.photo, text="")

    # ---- finish -----------------------------------------------------------
    def finish(self) -> None:
        if self.done:
            return
        incomplete = [c for c in CELL_ORDER if len(self.kept[c]) < self.target[c]]
        if incomplete:
            msg = ("Some cells are not full yet:\n"
                   + "\n".join(f"  {c[0]} {c[1]}: {len(self.kept[c])}/{self.target[c]}" for c in incomplete)
                   + "\n\nFinish anyway and save what you have?")
            if not messagebox.askyesno("Finish early?", msg):
                return
        self._finalize()
        self.root.quit()

    def _finalize(self) -> None:
        final_rows: list[dict] = []
        for cell in CELL_ORDER:
            for row in self.kept[cell][: self.target[cell]]:
                out = dict(row)
                out["derived_difficulty"] = cell[1]
                final_rows.append(out)
        write_review_log(self.log_entries)
        if not final_rows:
            print("[done] no images kept — nothing to save.")
            return
        save_outputs(final_rows)
        self.done = True

    def _quit(self) -> None:
        # save the decision log on quit, even mid-review
        write_review_log(self.log_entries)
        self.root.quit()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Curate the Study 1 stimulus set.")
    parser.add_argument("--csv", default=str(common.DATASET_CSV),
                        help="primary source CSV (default: Dataset/FINAL_DATASET.csv)")
    parser.add_argument("--extra-csv", default=str(HERE.parent / "manifests" / "DFR60K_manifest.csv"),
                        help="second source CSV to mix in (default: Dataset/DFR60K_manifest.csv)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and print the candidate pool, then exit (no GUI)")
    parser.add_argument("--trials", action="store_true",
                        help="also write Diss/trials.csv after curation")
    parser.add_argument("--resume", action="store_true",
                        help="load curated.csv + review_log.csv and continue where you left off")
    parser.add_argument("--face-filter", action="store_true",
                        help="auto-axe candidates that aren't a single, large face (live check)")
    parser.add_argument("--auto-filter", action="store_true",
                        help="auto-axe drawings + non-single-face images using image_metadata.csv")
    parser.add_argument("--min-face-fraction", type=float, default=MIN_FACE_FRACTION,
                        help="min face-box fraction with --face-filter (default 0.1)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--candidates-per-label", type=int, default=CANDIDATES_PER_LABEL,
                        help="optional hard cap per label (default: no cap — use all rows)")
    parser.add_argument("--mix-ratio", type=float, default=MIX_RATIO,
                        help="fraction of candidates from the primary CSV (default 0.5)")
    args = parser.parse_args(argv)

    rows = common.load_rows(args.csv)
    print(f"[loaded] {len(rows)} rows from {args.csv}")
    extra = Path(args.extra_csv)
    if extra.exists():
        extra_rows = common.load_rows(extra)
        rows.extend(extra_rows)
        print(f"[loaded] {len(extra_rows)} rows from {extra}")
    else:
        print(f"[warn] extra CSV not found: {extra} (skipping)")

    if METADATA_CSV.exists():
        meta = load_metadata(METADATA_CSV)
        for r in rows:
            m = meta.get(r["image_id"])
            if m:
                for col in ("is_photo", "photo_score", "num_faces",
                            "max_face_fraction", "face_ok", "ethnicity"):
                    r[col] = m.get(col, "")
                if not r.get("gender") and m.get("gender"):
                    r["gender"] = m["gender"]
                if not r.get("age_group") and m.get("age_group"):
                    r["age_group"] = m["age_group"]
        print(f"[metadata] joined {len(meta)} enriched rows from {METADATA_CSV}")

    hint_of, label_pools = build_pools(rows, seed=args.seed,
                                       candidates_per_label=args.candidates_per_label,
                                       mix_ratio=args.mix_ratio)

    kept_seed: dict = {}
    prior_log: list[dict] = []
    if args.resume:
        kept_seed = {c: [] for c in CELL_ORDER}
        decided_ids: set[str] = set()
        if REVIEW_LOG.exists():
            with open(REVIEW_LOG, encoding="utf-8-sig", newline="") as f:
                prior_log = list(csv.DictReader(f))
            decided_ids |= {r["image_id"] for r in prior_log if r.get("image_id")}
        if OUT_CSV.exists():
            with open(OUT_CSV, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    r.setdefault("source_dataset", "final")
                    cell_str = r.get("cell", "")
                    if "|" in cell_str:
                        label, diff = cell_str.split("|", 1)
                        if (label, diff) in CELL_ORDER:
                            kept_seed[(label, diff)].append(r)
        for label in common.LABELS:
            label_pools[label] = [r for r in label_pools[label]
                                  if r["image_id"] not in decided_ids]
        n_kept = sum(len(v) for v in kept_seed.values())
        print(f"[resume] {n_kept} kept from curated.csv, "
              f"{len(decided_ids)} already-decided images filtered out")

    print_pool_summary(label_pools)

    if args.dry_run:
        print("---- first candidates per label (preview) ----")
        for label in common.LABELS:
            preview = label_pools[label][: 10]
            shown = [f"{r['image_id']}({r.get('source_dataset', 'final')})" for r in preview]
            print(f"  {label:5s}: {', '.join(shown)}")
        return 0

    if not _HAS_TK:
        print("[error] tkinter is not available in this environment; "
              "run with --dry-run to inspect the pool instead.")
        return 1

    if args.face_filter:
        try:
            import cv2  # noqa: F401
            print(f"[face-filter] enabled (min face fraction {args.min_face_fraction:.2f})")
        except Exception:
            print("[warn] --face-filter requested but OpenCV is unavailable; filter disabled.")
            args.face_filter = False

    root = tk.Tk()
    app = ReviewApp(root, rows, hint_of, label_pools,
                    face_filter=args.face_filter,
                    auto_filter=args.auto_filter,
                    min_face_fraction=args.min_face_fraction,
                    kept_seed=kept_seed or None,
                    prior_log=prior_log or None)
    root.mainloop()

    # After the GUI closes, optionally write trials.csv from the kept set.
    if args.trials:
        final = []
        for cell in CELL_ORDER:
            for row in app.kept[cell][: app.target[cell]]:
                out = dict(row)
                out["derived_difficulty"] = cell[1]
                final.append(out)
        if final:
            write_trials_csv(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
