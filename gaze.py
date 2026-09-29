"""Eye-tracking wrapper — GazeFollower integration.

GazeFollower (Gancheng Zhu et al., 2025) is licensed under CC BY-NC-SA 4.0.
Use is non-commercial (academic research only). See README for attribution.
"""

import csv
import os

import pygame

from config import WARMUP_TRIGGER
from gazefollower import GazeFollower
from gazefollower.camera import WebCamCamera


class GazeRecorder:
    def __init__(self, output_path=None, camera_index=0):
        self.output_path = output_path
        self.camera_index = camera_index
        self.tracker = None

    def init(self):
        # build the tracker (loads the gaze models); camera preview and
        # calibration happen later, inside the PsychoPy window
        self.tracker = GazeFollower(
            camera=WebCamCamera(webcam_id=self.camera_index))

    def calibrate(self):
        # run GazeFollower's own fast standalone preview + calibration in a
        # borderless pygame window. NOFRAME (rather than fullscreen) avoids
        # the display-mode handoff that leaves a black screen after the
        # PsychoPy window closes.
        if not self.tracker:
            return

        pygame.quit()          # clean slate after the PsychoPy window closed
        pygame.init()
        size = self.tracker.screen_size.tolist()
        win = pygame.display.set_mode(size, pygame.NOFRAME)
        pygame.display.set_caption("GazeFollower Calibration")

        print("[GazeFollower] Launching camera preview...")
        self.tracker.preview(win=win)
        print("[GazeFollower] Starting calibration...")
        self.tracker.calibrate(win=win)

        # the UI leaves pygame initialised; tear it down so PsychoPy's
        # Mouse keeps reading the real window
        pygame.quit()

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
        # live gaze point in screen pixels, for the debug overlay
        if not self.tracker:
            return None
        info = self.tracker.get_gaze_info()
        if info is None or not getattr(info, "status", False):
            return None
        # prefer the smoothed stream so the demo dot is less jittery
        for attr in ("filtered_gaze_coordinates",
                     "calibrated_gaze_coordinates",
                     "raw_gaze_coordinates"):
            coords = getattr(info, attr, None)
            if coords is not None and len(coords) >= 2:
                return (float(coords[0]), float(coords[1]))
        return None

    def stop_trial(self):
        if self.tracker:
            self.tracker.stop_sampling()

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
