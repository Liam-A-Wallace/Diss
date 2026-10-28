"""Background preparation, UI thread ownership and webcam cleanup."""
import threading
import types
import unittest
from unittest.mock import patch

from gaze_preview import PreparedPreview


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.loading = threading.Event()
        self.allow_load = threading.Event()
        self.camera_started = threading.Event()
        self.camera_stopped = threading.Event()
        self.main_thread = threading.get_ident()
        owner = self

        class UI:
            def __init__(self, **kwargs):
                owner.calls.append(("ui", threading.get_ident()))
                owner.assertEqual(kwargs["win"].units, "pix")

            def draw(self):
                self.draw_table({}, (0, 0))

            def draw_table(self, data, start_pos):
                owner.calls.append(("draw", threading.get_ident()))

            def update_images(self, *images):
                owner.calls.append(("images", threading.get_ident()))

        class Alignment:
            def __init__(self, **kwargs):
                owner.calls.append(("model", threading.get_ident()))
                owner.loading.set()
                if not owner.allow_load.wait(2):
                    raise RuntimeError("Test did not allow model to load")

            def detect(self, timestamp, frame):
                return types.SimpleNamespace(status=False, to_dict=lambda: {"status": False})

            def release(self):
                owner.calls.append(("release_model", threading.get_ident()))

        class Camera:
            def __init__(self, **kwargs):
                self._cap = types.SimpleNamespace(
                    isOpened=lambda: True,
                    release=lambda: owner.calls.append(("release_capture", threading.get_ident())))
                owner.camera = self

            def set_on_image_callback(self, callback):
                self.callback = callback

            def start_previewing(self):
                owner.camera_started.set()

            def release(self):
                owner.camera_stopped.set()

        modules = {
            "gazefollower": types.ModuleType("gazefollower"),
            "gazefollower.ui": types.SimpleNamespace(CameraPreviewerUI=UI),
            "gazefollower.camera": types.SimpleNamespace(WebCamCamera=Camera),
            "gazefollower.face_alignment": types.SimpleNamespace(
                MediaPipeFaceAlignment=Alignment, BlazeFaceAlignment=Alignment),
            "gazefollower.misc": types.SimpleNamespace(clip_patch=lambda *args: None),
            "cv2": types.SimpleNamespace(),
        }
        self.modules = patch.dict("sys.modules", modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.preview = None
        self.addCleanup(self.cleanup_preview)

    def cleanup_preview(self):
        self.allow_load.set()
        if self.preview is not None:
            self.preview.close()

    def create(self):
        tracker = types.SimpleNamespace(config_dict={"webcam_id": 1},
                                        backend_name=lambda win: "psychopy")
        win = types.SimpleNamespace(units="height")
        self.preview = PreparedPreview(tracker, win)
        self.assertEqual(win.units, "height")
        self.assertTrue(self.loading.wait(1))
        return self.preview

    def test_loads_in_background_and_waits_for_first_processed_frame(self):
        preview = self.create()
        self.assertFalse(preview.check_ready())
        self.allow_load.set()
        self.assertTrue(self.camera_started.wait(1))
        self.assertFalse(preview.check_ready())
        self.camera.callback(None, 123, "frame")
        self.assertTrue(preview.check_ready())
        preview.show()
        self.assertTrue(self.camera_stopped.is_set())
        self.assertIn(("release_capture", preview._thread.ident), self.calls)
        for action, thread in self.calls:
            if action in ("ui", "draw", "images"):
                self.assertEqual(thread, self.main_thread)
            elif action == "model":
                self.assertNotEqual(thread, self.main_thread)

    def test_exit_during_model_load_does_not_open_camera(self):
        preview = self.create()
        # Model loading is allowed to finish, but must not acquire the camera.
        preview._stop.set()
        self.allow_load.set()
        preview.close()
        self.assertFalse(self.camera_started.is_set())
        self.assertTrue(any(action == "release_model" for action, _ in self.calls))

    def test_model_failure_is_reported_to_main_thread(self):
        with patch.dict("sys.modules", {
            "gazefollower.face_alignment": types.SimpleNamespace(
                BlazeFaceAlignment=None,
                MediaPipeFaceAlignment=lambda **kwargs: (_ for _ in ()).throw(ValueError("model failed")))
        }):
            tracker = types.SimpleNamespace(config_dict={}, backend_name=lambda win: "psychopy")
            self.preview = PreparedPreview(tracker, types.SimpleNamespace(units="height"))
            self.preview._thread.join(1)
            with self.assertRaisesRegex(RuntimeError, "preparation failed") as error:
                self.preview.check_ready()
            self.assertIsInstance(error.exception.__cause__, ValueError)

    def test_camera_is_released_when_preview_drawing_fails(self):
        preview = self.create()
        self.allow_load.set()
        self.assertTrue(self.camera_started.wait(1))
        self.camera.callback(None, 123, "frame")
        with patch.object(preview.ui, "draw", side_effect=RuntimeError("display failed")):
            with self.assertRaisesRegex(RuntimeError, "display failed"):
                preview.show()
        self.assertTrue(self.camera_stopped.is_set())

    def test_missing_frame_times_out(self):
        preview = self.create()
        with patch("gaze_preview.time.monotonic", return_value=preview._started_at + 31):
            with self.assertRaisesRegex(RuntimeError, "first camera preview frame"):
                preview.check_ready()


if __name__ == "__main__":
    unittest.main()
