"""Prepare the SDK camera preview while participants read Welcome.

Only camera/model work runs in the background. SDK UI construction and drawing
stay on the PsychoPy thread; the upstream checkout remains untouched.
"""
import threading
import time

from errors import QuitExperiment


class PreparedPreview:
    def __init__(self, tracker, win):
        from sdk_rendering import apply
        apply()
        from gazefollower.ui import CameraPreviewerUI

        owner = self

        class PreviewUI(CameraPreviewerUI):
            def draw_table(self, data, start_pos):
                owner.check_ready()
                with owner._lock:
                    packet = owner._latest
                if packet is not None:
                    images, face_info = packet
                    # The full camera image is refreshed independently of
                    # detection; only the diagnostic crops wait for inference.
                    self.update_images(None, *images[1:])
                    self.face_info_dict = face_info
                super().draw_table(self.face_info_dict, start_pos)

        self._lock = threading.Lock()
        self._latest = None
        self._raw_frame = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._error = None
        self._started_at = time.monotonic()
        self._config = dict(tracker.config_dict)
        # The SDK caches the window units for mouse conversion at construction.
        units = win.units
        try:
            win.units = "pix"
            self.ui = PreviewUI(win=win, backend_name=tracker.backend_name(win))
        finally:
            win.units = units
        # The SDK only handles SPACE and consumes all other keys. Check Escape
        # first so quitting the preview follows the experiment's cleanup path.
        listen_event = self.ui.backend.listen_event

        def listen_with_escape(host, *args, **kwargs):
            from psychopy import event
            if "escape" in event.getKeys(keyList=["escape"]):
                raise QuitExperiment()
            return listen_event(host, *args, **kwargs)

        self.ui.backend.listen_event = listen_with_escape
        before_draw = self.ui.backend.before_draw

        def refresh_camera_frame():
            before_draw()
            with self._lock:
                frame = self._raw_frame
            if frame is not None:
                self.ui.frame_image = frame

        self.ui.backend.before_draw = refresh_camera_frame
        self._thread = threading.Thread(target=self._prepare, daemon=True,
                                        name="camera-preview-preparation")
        self._thread.start()

    def _prepare(self):
        camera = alignment = None
        try:
            from gazefollower.camera import WebCamCamera
            from gazefollower.face_alignment import BlazeFaceAlignment, MediaPipeFaceAlignment

            cfg = self._config
            factory = (BlazeFaceAlignment if cfg.get("face_alignment_type") == "blazeface"
                       else MediaPipeFaceAlignment)
            alignment = factory(enable_filter=cfg.get("enable_face_filter", True),
                                filter_min_cutoff=cfg.get("filter_min_cutoff", 1.0),
                                filter_beta=cfg.get("filter_beta", 0.01))
            if self._stop.is_set():
                return
            camera = WebCamCamera(webcam_id=cfg.get("webcam_id", 0))
            if self._stop.is_set():
                return
            camera.set_on_image_callback(
                lambda state, timestamp, frame: self._on_frame(alignment, timestamp, frame))
            camera._preview_frame_callback = self._on_raw_frame
            camera.start_previewing()
            if not camera._cap.isOpened():
                raise RuntimeError("The webcam could not be opened")
            self._stop.wait()
        except Exception as err:
            self._error = err
            self._ready.set()
        finally:
            try:
                if camera is not None:
                    camera.release()
                if alignment is not None:
                    alignment.release()
            except Exception as err:
                # Preserve the error for the UI; never hand the camera to
                # calibration or release a model still used by a callback.
                self._error = err
                self._ready.set()

    def _on_raw_frame(self, timestamp, frame):
        import cv2
        # Capture owns BGR frames; publish an independent RGB image so face
        # annotations and detection cannot mutate the displayed frame.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._raw_frame = image

    def _on_frame(self, alignment, timestamp, frame):
        import cv2
        from gazefollower.misc import clip_patch

        try:
            info = alignment.detect(timestamp, frame)
            face = left = right = None
            if info.status and info.can_gaze_estimation:
                # Copy patches before annotating the full frame.
                face = clip_patch(frame, info.face_rect)
                left = clip_patch(frame, info.left_rect)
                right = clip_patch(frame, info.right_rect)
                for rect, colour in ((info.left_rect, (255, 0, 0)),
                                     (info.right_rect, (0, 0, 255))):
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                    if face is not None:
                        fx, fy, _, _ = info.face_rect
                        cv2.rectangle(face, (x - fx, y - fy),
                                      (x - fx + w, y - fy + h), colour, 2)
            with self._lock:
                self._latest = ((frame, face, left, right), info.to_dict())
            self._ready.set()
        except Exception as err:
            self._error = err
            self._ready.set()

    def check_ready(self):
        if self._error is not None:
            raise RuntimeError("Camera preview preparation failed") from self._error
        if not self._ready.is_set() and time.monotonic() - self._started_at > 30:
            raise RuntimeError("Timed out waiting for the first camera preview frame")
        return self._ready.is_set()

    def show(self):
        try:
            self.ui.draw()
        except BaseException:
            self.close()
            raise
        if not self.close():
            # Do not let calibration open a second camera while the first
            # one is still shutting down.
            raise RuntimeError("Camera preview did not stop within two seconds")

    def close(self, timeout=2.0):
        """Request cleanup without trapping the UI in a driver/model wait."""
        self._stop.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive() and self._error is None
