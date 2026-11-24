"""Exposure settings must recover from the previous manual camera mode."""
import importlib.util
from pathlib import Path
import types
import unittest
from unittest.mock import patch


class ExposureTests(unittest.TestCase):
    def open_camera(self, lock, backend=1400, opened=True, gain=None, with_gain_prop=False):
        settings = []
        capture = types.SimpleNamespace(
            isOpened=lambda: opened,
            get=lambda prop: backend,
            set=lambda prop, value: settings.append((prop, value)) or True)

        class Camera:
            def __init__(self, *args):
                self._cap = capture
                self.img_width, self.img_height, self.cam_fps = 640, 480, 30

            def open(self):
                self._create_capture_thread()

            def _create_capture_thread(self):
                pass

            def close(self):
                pass

        cv2_attrs = dict(CAP_PROP_BACKEND=42, CAP_PROP_AUTO_EXPOSURE=21,
                         CAP_PROP_EXPOSURE=15, CAP_V4L2=200,
                         CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
                         CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=38)
        if with_gain_prop:
            cv2_attrs["CAP_PROP_GAIN"] = 14
        fake_cv2 = types.SimpleNamespace(**cv2_attrs)
        fake_log = types.SimpleNamespace(i=lambda message: None, w=lambda message: None)
        spec = importlib.util.spec_from_file_location("exposure_under_test",
                                                     Path(__file__).parents[1] / "gazefollower_patch.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {
            "gazefollower": types.ModuleType("gazefollower"),
            "gazefollower.camera": types.SimpleNamespace(WebCamCamera=Camera),
            "gazefollower.logger": types.SimpleNamespace(Log=fake_log), "cv2": fake_cv2}):
            spec.loader.exec_module(module)
            module.apply()
            camera = Camera(lock_exposure=lock, exposure=-6, gain=gain)
            camera.open()
        return settings

    def test_auto_mode_restored_without_manual_exposure(self):
        for backend, expected in ((1400, 1), (200, 0.75)):
            with self.subTest(backend=backend):
                self.assertEqual(self.open_camera(False, backend),
                                 [(3, 640), (4, 480), (5, 30), (38, 1), (21, expected)])

    def test_manual_mode_only_when_requested(self):
        for backend, expected in ((1400, 0), (200, 0.25)):
            with self.subTest(backend=backend):
                self.assertEqual(self.open_camera(True, backend),
                                 [(3, 640), (4, 480), (5, 30), (38, 1),
                                  (21, expected), (15, -6)])

    def test_gain_applied_only_in_manual_mode(self):
        self.assertEqual(self.open_camera(True, 1400, gain=150, with_gain_prop=True),
                         [(3, 640), (4, 480), (5, 30), (38, 1), (21, 0), (15, -6), (14, 150)])
        self.assertEqual(self.open_camera(False, 1400, gain=150, with_gain_prop=True),
                         [(3, 640), (4, 480), (5, 30), (38, 1), (21, 1)])

    def test_closed_camera_is_not_configured(self):
        self.assertEqual(self.open_camera(False, opened=False), [])


if __name__ == "__main__":
    unittest.main()
