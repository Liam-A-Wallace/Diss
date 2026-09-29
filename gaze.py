"""Eye-tracking wrapper — GazeFollower integration.

GazeFollower (Gancheng Zhu et al., 2025) is licensed under CC BY-NC-SA 4.0.
Use is non-commercial (academic research only). See README for attribution.
"""

import os

import pygame

from gazefollower import GazeFollower
from gazefollower.camera import WebCamCamera


class GazeRecorder:
    def __init__(self, output_path=None, camera_index=0):
        self.output_path = output_path
        self.camera_index = camera_index
        self.tracker = None

    def start_session(self):
        # init + preview + calibrate before the PsychoPy window takes focus
        self.tracker = GazeFollower(
            camera=WebCamCamera(webcam_id=self.camera_index))
        print("[GazeFollower] Launching camera preview...")
        self.tracker.preview()
        print("[GazeFollower] Starting calibration...")
        self.tracker.calibrate()

        # GazeFollower's preview/calibrate leave pygame initialised, which
        # makes PsychoPy's Mouse read pygame instead of the real window,
        # breaking mouse clicks. Tear pygame down before PsychoPy opens.
        pygame.quit()

    def start_trial(self, trigger):
        if self.tracker:
            self.tracker.start_sampling()
            self.tracker.send_trigger(int(trigger))

    def poll(self):
        # live gaze point in screen pixels, for the debug overlay
        if not self.tracker:
            return None
        info = self.tracker.get_gaze_info()
        if info is None or not getattr(info, "status", False):
            return None
        for attr in ("calibrated_gaze_coordinates",
                     "filtered_gaze_coordinates",
                     "raw_gaze_coordinates"):
            coords = getattr(info, attr, None)
            if coords is not None and len(coords) >= 2:
                return (float(coords[0]), float(coords[1]))
        return None

    def stop_trial(self):
        if self.tracker:
            self.tracker.stop_sampling()

    def stop_session(self):
        if self.tracker:
            # stop any in-flight sampling first, so the file is never saved
            # while the camera thread is still writing (e.g. ESC mid-trial)
            self.stop_trial()
            if self.output_path:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                self.tracker.save_data(self.output_path)
                print(f"[GazeFollower] Gaze data saved to {self.output_path}")
            self.tracker.release()
            self.tracker = None
