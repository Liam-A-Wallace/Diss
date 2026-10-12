"""Runtime patches applied on top of the *unmodified* GazeFollower checkout.

GazeFollower is kept as a pristine upstream checkout (only the model weights
``res/model_weights/base.mnn`` are swapped). The experiment needs one behaviour
that upstream GazeFollower does not provide: camera exposure locking
(``config.LOCK_CAMERA_EXPOSURE`` / ``config.CAMERA_EXPOSURE``), so a bright
calibration cross followed by a dark stimulus does not bias the gaze estimate
downward.

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
    _upstream_open = _WebCamCamera.open

    def __init__(self, webcam_id=0, img_height=480, img_width=640, cam_fps=30,
                 lock_exposure=None, exposure=None):
        if lock_exposure is None:
            lock_exposure = LOCK_CAMERA_EXPOSURE
        if exposure is None:
            exposure = CAMERA_EXPOSURE
        self.lock_exposure = lock_exposure
        self.exposure = exposure
        _upstream_init(self, webcam_id, img_height, img_width, cam_fps)

    def open(self):
        _upstream_open(self)
        if self.lock_exposure and self._cap.isOpened():
            import cv2

            auto_off = self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            set_exp = False
            if self.exposure is not None:
                set_exp = self._cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            Log.i(f"Camera exposure locked: auto_off={auto_off}, "
                  f"exposure={self.exposure} set={set_exp}")

    _WebCamCamera.__init__ = __init__
    _WebCamCamera.open = open
    setattr(_WebCamCamera, _PATCH_FLAG, True)
