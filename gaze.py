"""Eye-tracking wrapper — GazeFollower integration.

GazeFollower (Gancheng Zhu et al., 2025) is licensed under CC BY-NC-SA 4.0.
Use is non-commercial (academic research only). See README for attribution.
"""

import csv
import os
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
                    GAZE_FILTER, LOCK_CAMERA_EXPOSURE, WARMUP_TRIGGER)
from gazefollower import GazeFollower
from gazefollower.camera import WebCamCamera
from gazefollower.filter import HeuristicFilter, OneEuroFilter
from gazefollower.misc import DefaultConfig

import gazefollower_patch
gazefollower_patch.apply()

# Header the sample CSV must carry so _forward_fill_trial_column can add a
# 'trial' column later. The worker writes data rows without a header, so we
# write this to the worker's temp sample file before sampling starts.
SAMPLE_HEADER = (
    "timestamp,raw_gaze_position_x,raw_gaze_position_y,"
    "calibrated_gaze_position_x,calibrated_gaze_position_y,"
    "filtered_gaze_position_x,filtered_gaze_position_y,"
    "left_eye_openness,right_eye_openness,"
    "tracking_status,status,event,trigger\n"
)


class GazeRecorder:
    def __init__(self, output_path=None, camera_index=0):
        self.output_path = output_path
        self.camera_index = camera_index
        self.tracker = None
        # monotonic timestamp of the most recent valid calibrated sample
        self._last_valid_time = None
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

        self.stop_trial()
        print("[GazeFollower] Recalibrating on PsychoPy window...")
        self.tracker.calibrate(win=win)

        # The old calibration is stale: clear the recent-gaze timestamp so
        # the ready gate waits for fresh, post-calibration samples.
        self._last_valid_time = None

    def warm_up(self):
        # start the camera sampling so the pipeline is warm by the time the
        # face appears; samples here are stamped WARMUP_TRIGGER and filtered
        if self.tracker:
            # arm the trigger first so the very first warm-up sample is tagged
            self.tracker.send_trigger(WARMUP_TRIGGER)
            self.tracker.start_sampling()

    def mark_trial(self, trigger):
        # stamp this trial's number onto the next sample (onset marker)
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

        if coords is not None and len(coords) >= 2:
            result = (float(coords[0]), float(coords[1]))
            self._last_valid_time = time.monotonic()

            # if not hasattr(self, "_debug_count"):
            #     self._debug_count = 0
            #
            # self._debug_count += 1
            #
            # if self._debug_count % 20 == 0:
            #     print(
            #         f"[GAZE {self._debug_count}] "
            #         f"calibrated=({result[0]:.1f}, {result[1]:.1f})"
            #     )

            return result

        return None

    def recent_gaze(self, within=0.3):
        # True only if a valid calibrated sample arrived within the last
        # `within` seconds — i.e. the tracker is *currently* locked on the
        # participant, not just that it produced one sample earlier.
        if self._last_valid_time is None:
            return False
        return (time.monotonic() - self._last_valid_time) <= within

    def stop_trial(self):
        if self.tracker:
            self.tracker.stop_sampling()

    def _write_sample_header(self):
        # The worker appends rows to its temp sample file with no header
        # row. Pre-write the header once so the CSV stays parseable and
        # _forward_fill_trial_column can add the 'trial' column.
        path = getattr(self.tracker, "_tmp_sample_path", None)
        if not path:
            return
        path = str(path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(SAMPLE_HEADER)

    def _forward_fill_trial_column(self):
        # GazeFollower stamps the trial number on a single onset row and
        # leaves every other row 0. Carry that number forward into a 'trial'
        # column so each row can be matched to its stimulus later.
        if not self.output_path or not os.path.exists(self.output_path):
            return

        rows = []
        with open(self.output_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Only run on a genuine header. Older recordings (or files written
            # before the worker wrote a header) lack these columns, and treating
            # a data row as the header would corrupt the file. Also skip if the
            # trial column has already been added.
            if (not fieldnames or "timestamp" not in fieldnames
                    or "trigger" not in fieldnames or "trial" in fieldnames):
                return

            current = 0
            for row in reader:
                try:
                    trig = int(float(row.get("trigger", "").strip() or 0))
                except (TypeError, ValueError):
                    trig = 0
                if trig != 0:
                    current = trig
                row["trial"] = current
                rows.append(row)

        if not fieldnames:
            return

        with open(self.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames + ["trial"])
            writer.writeheader()
            writer.writerows(rows)

    def stop_session(self):
        if self.tracker:
            # stop any in-flight sampling first, so the file is never saved
            # while the camera thread is still writing (e.g. ESC mid-trial)
            self.stop_trial()
            if self.output_path:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                self.tracker.save_data(self.output_path)
                print(f"[GazeFollower] Gaze data saved to {self.output_path}")
                self._forward_fill_trial_column()
            self.tracker.release()
            self.tracker = None
