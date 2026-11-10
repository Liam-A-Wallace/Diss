"""Runtime patches applied on top of the *unmodified* GazeFollower checkout.

GazeFollower is kept as a pristine upstream checkout (the gaze model is
selected via config.GAZE_MODEL). The exposure mode and optional fixed value
are selected in config.py and must be checked against the webcam and room
lighting. Automatic mode explicitly restores the driver setting when selected.


This module patches ``gazefollower.camera.WebCamCamera`` in-place at import
time. Because ``gaze.py`` is imported from the experiment's main module and the
multiprocessing worker is started with the "spawn" method (which re-executes
the main module in the child process before the worker entrypoint is
unpickled), the same patch is applied in the gaze-worker process too.
"""

from config import CAMERA_EXPOSURE, LOCK_CAMERA_EXPOSURE
from gazefollower.camera import WebCamCamera as _WebCamCamera
from gazefollower.logger import Log

_PATCH_FLAG = "_diss_exposure_lock_patched"


def apply():
    """Give ``WebCamCamera`` optional exposure locking.

    The upstream constructor is extended with ``lock_exposure`` and
    ``exposure`` keyword arguments. When either is left as ``None`` the values
    from ``config`` are used, so cameras created inside the worker process
    (which only passes ``webcam_id``) still honour the experiment's settings.
    """
    if getattr(_WebCamCamera, _PATCH_FLAG, False):
        return

    _upstream_init = _WebCamCamera.__init__
    _upstream_create_thread = _WebCamCamera._create_capture_thread

    def __init__(
        self,
        webcam_id=0,
        img_height=480,
        img_width=640,
        cam_fps=30,
        lock_exposure=None,
        exposure=None,
    ):
        if lock_exposure is None:
            lock_exposure = LOCK_CAMERA_EXPOSURE
        if exposure is None:
            exposure = CAMERA_EXPOSURE
        self.lock_exposure = lock_exposure
        self.exposure = exposure
        _upstream_init(self, webcam_id, img_height, img_width, cam_fps)

    def configure_capture(self):
        # Upstream calls this hook after opening and before starting capture.
        if not self._cap.isOpened():
            return _upstream_create_thread(self)
        import cv2

        for prop, value in (
            (cv2.CAP_PROP_FRAME_WIDTH, self.img_width),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.img_height),
            (cv2.CAP_PROP_FPS, self.cam_fps),
            (cv2.CAP_PROP_BUFFERSIZE, 1),
        ):
            self._cap.set(prop, value)

        # Windows backends use 0/1; V4L2 uses quarter-scale menu values.
        backend = int(self._cap.get(cv2.CAP_PROP_BACKEND))
        if self.lock_exposure:
            auto_value = 0.25 if backend == cv2.CAP_V4L2 else 0
            auto_off = self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)
            set_exp = False
            if self.exposure is not None:
                set_exp = self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            Log.i(
                f"Camera exposure lock requested: auto_off={auto_off}, "
                f"exposure={self.exposure} set={set_exp}"
            )
        else:
            # Restore auto exposure because manual settings can persist in the
            # camera driver between runs, even after changing the config.
            auto_value = 0.75 if backend == cv2.CAP_V4L2 else 1
            enabled = self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)
            Log.i(f"Camera automatic exposure requested: set={enabled}")
            if not enabled:
                Log.w(
                    "The webcam did not accept automatic exposure. "
                    "Check its exposure setting in the camera controls."
                )
        Log.i(
            f"Camera negotiated: {self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)} x "
            f"{self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)} at "
            f"{self._cap.get(cv2.CAP_PROP_FPS)} fps"
        )
        return _upstream_create_thread(self)

    _WebCamCamera.__init__ = __init__
    _WebCamCamera._create_capture_thread = configure_capture
    from camera_capture import capture_latest, stop_camera

    _WebCamCamera.close = stop_camera
    _WebCamCamera.release = stop_camera
    _WebCamCamera.capture = capture_latest
    setattr(_WebCamCamera, _PATCH_FLAG, True)
