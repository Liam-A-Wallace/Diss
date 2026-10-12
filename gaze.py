"""Eye-tracking wrapper — GazeFollower integration.

GazeFollower (Gancheng Zhu et al., 2025) is licensed under CC BY-NC-SA 4.0.
Use is non-commercial (academic research only). See README for attribution.

The camera samples continuously for the whole session. Each image's on-screen
window (flip-accurate onset/offset in epoch ns, the same clock the worker uses
for sample timestamps) is recorded, and the raw stream is trimmed to those
windows when the session ends.
"""

import collections
import csv
import math
import os
import statistics
import sys
import time

import pygame

# Import GazeFollower from the local checkout (not any pip-installed copy) so
# the model weights shipped in this repo are the ones actually loaded.
_WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GAZEFOLLOWER_DIR = os.path.join(_WORKSPACE_ROOT, "GazeFollower")
if _GAZEFOLLOWER_DIR not in sys.path:
    sys.path.insert(0, _GAZEFOLLOWER_DIR)

from config import (CALIBRATION_POINTS, CAMERA_EXPOSURE, ENABLE_FACE_FILTER,
                    GAZE_FILTER, LOCK_CAMERA_EXPOSURE)
from gazefollower import GazeFollower
from gazefollower.camera import WebCamCamera
from gazefollower.filter import HeuristicFilter, OneEuroFilter
from gazefollower.misc import DefaultConfig

import gazefollower_patch
gazefollower_patch.apply()

# The worker appends rows with no header, so this is pre-written to its temp
# sample file before sampling starts.
SAMPLE_HEADER = (
    "timestamp,raw_gaze_position_x,raw_gaze_position_y,"
    "calibrated_gaze_position_x,calibrated_gaze_position_y,"
    "filtered_gaze_position_x,filtered_gaze_position_y,"
    "left_eye_openness,right_eye_openness,"
    "tracking_status,status,event,trigger\n"
)

WINDOW_FIELDS = ["trial", "trigger", "is_practice", "condition", "stimulus",
                 "onset_ns", "offset_ns"]

# column index of the per-sample "status" flag in SAMPLE_HEADER
_STATUS_COL = 10
_TAIL_BYTES = 1_000_000


def load_windows(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["onset_ns"] = int(row["onset_ns"])
        row["offset_ns"] = int(row["offset_ns"])
    return rows


def trim_gaze_file(raw_path, windows, out_path):
    """Write only the raw samples inside an image window, tagged by trial.

    Also usable after a crash: pair the raw copy with the windows sidecar.
    Returns the number of rows written.
    """
    windows = sorted(windows, key=lambda w: w["onset_ns"])
    written = 0
    with open(raw_path, encoding="utf-8", newline="") as src:
        reader = csv.DictReader(src)
        fields = reader.fieldnames or []
        if "timestamp" not in fields:
            raise ValueError(f"'{raw_path}' has no header row")
        extra = ["trial", "is_practice", "condition", "t_rel_ms"]
        with open(out_path, "w", encoding="utf-8", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fields + extra)
            writer.writeheader()
            idx = 0
            for row in reader:
                try:
                    ts = int(row["timestamp"])
                except (TypeError, ValueError):
                    continue
                while idx < len(windows) and ts > windows[idx]["offset_ns"]:
                    idx += 1
                if idx == len(windows):
                    break
                win = windows[idx]
                if ts < win["onset_ns"]:
                    continue
                row["trial"] = win["trial"]
                row["is_practice"] = win["is_practice"]
                row["condition"] = win["condition"]
                row["t_rel_ms"] = round((ts - win["onset_ns"]) / 1e6, 2)
                writer.writerow(row)
                written += 1
    return written


class GazeRecorder:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.output_path = None   # trimmed, image-window-only gaze
        self.raw_path = None      # everything sampled this session
        self.windows_path = None  # per-trial onset/offset sidecar
        self.tracker = None
        self._windows = []
        self._sampling = False
        # monotonic time of the most recent *new* valid sample
        self._last_valid_time = None
        self._last_info_ts = None
        # recent (monotonic, x, y) samples for the drift check
        self._recent = collections.deque(maxlen=90)
        # resolution (width, height) of the coordinate space GazeFollower
        # reports calibrated gaze in; captured once after the tracker is built
        self.screen_size = None

    def init(self):
        # build the eye tracker (loads the gaze models) before the experiment
        # window opens; preview and calibration are run later, in
        # GazeFollower's own display, after the PsychoPy window is closed.
        #
        # The number of calibration points must be set on the config BEFORE
        # GazeFollower is built: GazeFollower copies config.cali_mode into its
        # CalibrationController at construction time, so changing it later has
        # no effect.
        cfg = DefaultConfig()
        cfg.cali_mode = CALIBRATION_POINTS
        cfg.enable_face_filter = ENABLE_FACE_FILTER
        gaze_filter = (
            OneEuroFilter() if GAZE_FILTER == "one_euro" else HeuristicFilter()
        )
        self.tracker = GazeFollower(
            camera=WebCamCamera(
                webcam_id=self.camera_index,
                lock_exposure=LOCK_CAMERA_EXPOSURE,
                exposure=CAMERA_EXPOSURE,
            ),
            gaze_filter=gaze_filter,
            config=cfg)
        self.screen_size = tuple(
            int(v) for v in self.tracker.config.screen_size)
        self._write_sample_header()

    def set_output(self, base_name):
        self.output_path = base_name + "_gaze.csv"
        self.raw_path = base_name + "_gaze_raw.csv"
        self.windows_path = base_name + "_windows.csv"

    def calibrate(self):
        # let GazeFollower create and manage its own display, exactly as its
        # documented API expects (preview() / calibrate() with no surface)
        if not self.tracker:
            return

        print("[GazeFollower] Launching camera preview...")
        self.tracker.preview()
        print("[GazeFollower] Starting calibration...")
        self.tracker.calibrate()

        # GazeFollower's preview/calibrate initialise pygame, which makes
        # PsychoPy's Mouse read pygame instead of the real window and breaks
        # mouse clicks. Tear pygame down before the PsychoPy window opens.
        pygame.quit()

    def recalibrate(self, win):
        # Re-run calibration on the existing PsychoPy window. Passing the
        # window makes GazeFollower use its psychopy backend, so no pygame
        # fullscreen window is opened mid-experiment.
        if not self.tracker:
            return

        # the camera cannot calibrate while sampling, so pause and resume
        was_sampling = self._sampling
        self.stop_sampling()
        print("[GazeFollower] Recalibrating on PsychoPy window...")
        try:
            self.tracker.calibrate(win=win)
        finally:
            if was_sampling:
                self.start_sampling()

        # the old calibration is stale; wait for fresh post-calibration samples
        self._last_valid_time = None
        self._last_info_ts = None
        self._recent.clear()

    def start_sampling(self):
        if self.tracker and not self._sampling:
            self.tracker.start_sampling()
            self._sampling = True

    def stop_sampling(self):
        if self.tracker and self._sampling:
            self.tracker.stop_sampling()
            self._sampling = False

    def mark_trial(self, trigger):
        # cross-check marker in the raw stream; trimming uses the flip times
        if self.tracker:
            self.tracker.send_trigger(int(trigger))

    def poll(self):
        if not self.tracker:
            return None

        info = self.tracker.get_gaze_info()

        if info is None or not getattr(info, "status", False):
            return None

        coords = getattr(info, "filtered_gaze_coordinates", None)
        if coords is None or len(coords) < 2:
            # fall back to the raw calibrated point if the filter hasn't
            # produced a value yet
            coords = getattr(info, "calibrated_gaze_coordinates", None)

        if coords is None or len(coords) < 2:
            return None

        result = (float(coords[0]), float(coords[1]))

        # get_gaze_info keeps returning the last sample when the queue is
        # empty, so only a changed timestamp counts as a fresh sample
        ts = getattr(info, "timestamp", None)
        if ts is None or ts != self._last_info_ts:
            self._last_info_ts = ts
            now = time.monotonic()
            self._last_valid_time = now
            self._recent.append((now, result[0], result[1]))

        return result

    def recent_gaze(self, within):
        if self._last_valid_time is None:
            return False
        return (time.monotonic() - self._last_valid_time) <= within

    def seconds_since_valid(self):
        if self._last_valid_time is None:
            return math.inf
        return time.monotonic() - self._last_valid_time

    def drift_px(self, window_s):
        # median distance of recent gaze from the screen centre (the cross)
        if not self.screen_size:
            return None
        cx, cy = self.screen_size[0] / 2, self.screen_size[1] / 2
        cutoff = time.monotonic() - window_s
        dists = [math.hypot(x - cx, y - cy)
                 for t, x, y in self._recent if t >= cutoff]
        if len(dists) < 3:
            return None
        return statistics.median(dists)

    def record_window(self, trial, trigger, is_practice, condition, stimulus,
                      onset_ns, offset_ns):
        row = {
            "trial": trial, "trigger": trigger,
            "is_practice": int(is_practice), "condition": condition,
            "stimulus": stimulus, "onset_ns": onset_ns, "offset_ns": offset_ns,
        }
        self._windows.append(row)
        if not self.windows_path:
            return
        new_file = not os.path.exists(self.windows_path)
        with open(self.windows_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=WINDOW_FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    def window_quality(self, onset_ns, offset_ns):
        # (n_samples, fraction_valid) for one image window, read from the
        # tail of the live sample file
        path = getattr(self.tracker, "_tmp_sample_path", None)
        if not path or not os.path.exists(str(path)):
            return 0, 0.0
        try:
            with open(str(path), "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - _TAIL_BYTES))
                lines = f.read().decode("utf-8", "ignore").splitlines()[1:]
        except OSError:
            return 0, 0.0
        n = valid = 0
        for line in lines:
            parts = line.split(",")
            if len(parts) <= _STATUS_COL:
                continue
            try:
                ts = int(parts[0])
            except ValueError:
                continue
            if onset_ns <= ts <= offset_ns:
                n += 1
                valid += parts[_STATUS_COL].strip() == "1"
        return n, (valid / n if n else 0.0)

    def checkpoint(self):
        # copy the live sample stream out so a crash cannot lose it
        if self.tracker and self.raw_path:
            try:
                self.tracker.save_data(self.raw_path)
            except OSError as err:
                print(f"[GazeFollower] checkpoint failed: {err}")

    def _write_sample_header(self):
        path = getattr(self.tracker, "_tmp_sample_path", None)
        if not path:
            return
        path = str(path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(SAMPLE_HEADER)

    def stop_session(self):
        if not self.tracker:
            return
        try:
            # stop first so the file is never copied while still being written
            self.stop_sampling()
            if self.raw_path:
                os.makedirs(os.path.dirname(self.raw_path) or ".",
                            exist_ok=True)
                self.tracker.save_data(self.raw_path)
                print(f"[GazeFollower] Raw gaze data saved to {self.raw_path}")
                if self._windows and os.path.exists(self.raw_path):
                    n = trim_gaze_file(self.raw_path, self._windows,
                                       self.output_path)
                    print(f"[GazeFollower] {n} image-window samples "
                          f"saved to {self.output_path}")
        except Exception as err:
            print(f"[GazeFollower] saving gaze data failed: {err}",
                  file=sys.stderr)
        finally:
            self.tracker.release()
            self.tracker = None
