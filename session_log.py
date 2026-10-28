"""Versioned, append-only display history and atomic session metadata."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time


class SessionLog:
    def __init__(self, base_name, info, win_size, tracker_size, config, trials, practice):
        self.path = Path(base_name + "_session.json")
        versions = {}
        for package in ("psychopy", "numpy", "Pillow", "pygame", "opencv-python"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        stimuli = {}
        for trial in trials + practice:
            p = Path(trial["stimulus"])
            stimuli[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.metadata = dict(schema_version=1, participant=info, status="running",
                             started_ns=time.time_ns(), python=platform.python_version(),
                             project_root=str(Path.cwd()),
                             versions=versions, window_size=list(map(int, win_size)),
                             tracker_size=list(map(int, tracker_size)), config=config,
                             stimuli_sha256=stimuli, planned_trials=trials, practice=practice,
                             clock="epoch_ns; webcam timestamp after camera read, not measured exposure",
                             coordinates="original image pixels, top-left origin; +x right, +y down")
        self.save()
        self.events = open(base_name + "_display.jsonl", "x", encoding="utf-8", buffering=1)
        self.window_size = tuple(map(int, win_size))
        self.tracker_size = tuple(map(int, tracker_size))

    def save(self):
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def present(self, frame):
        row = dict(frame, kind="present",
                   window_width=self.window_size[0], window_height=self.window_size[1],
                   tracker_width=self.tracker_size[0], tracker_height=self.tracker_size[1])
        self.events.write(json.dumps(row) + "\n")

    def end_view(self, trigger, interrupted=False, timestamp_ns=None):
        self.events.write(json.dumps(dict(kind="offset", timestamp_ns=timestamp_ns or time.time_ns(),
                                         trigger=trigger, interrupted=interrupted)) + "\n")

    def finish(self, status):
        self.events.close()
        self.metadata.update(status=status, ended_ns=time.time_ns())
        self.save()


def export_image_gaze(base_name):
    """Write a convenient, SDK-independent image-coordinate CSV at shutdown."""
    from analysis.ingest import align_samples, load_display, write_csv
    events = load_display(base_name + "_display.jsonl")
    rows = list(align_samples(base_name + "_gaze_raw.csv", events))
    write_csv(base_name + "_gaze_image.csv", rows,
              list(rows[0]) if rows else ["timestamp", "trigger", "image_x", "image_y", "usable"])
