"""Background preparation, UI thread ownership and webcam cleanup."""

import threading
import types
import unittest
from unittest.mock import patch

from gaze_preview import PreparedPreview
from errors import QuitExperiment


class PreviewTests(unittest.TestCase):
    def setUp(self):
        renderer_patch = patch("sdk_rendering.apply")
        renderer_patch.start()
        self.addCleanup(renderer_patch.stop)
        self.calls = []
        self.loading = threading.Event()
        self.allow_load = threading.Event()
        self.camera_started = threading.Event()
        self.camera_stopped = threading.Event()
        self.main_thread = threading.get_ident()
        owner = self

        class UI:
            def __init__(self, **kwargs):
                self.backend = types.SimpleNamespace(
                    listen_event=lambda host: None, before_draw=lambda: None
                )
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
                return types.SimpleNamespace(
                    status=False, to_dict=lambda: {"status": False}
                )

            def release(self):
                owner.calls.append(("release_model", threading.get_ident()))

        class Camera:
            def __init__(self, **kwargs):
                self._cap = types.SimpleNamespace(
                    isOpened=lambda: True,
                    release=lambda: owner.calls.append(
                        ("release_capture", threading.get_ident())
                    ),
                )
                owner.camera = self

            def set_on_image_callback(self, callback):
                self.callback = callback

            def start_previewing(self):
                owner.camera_started.set()

            def release(self):
                owner.camera_stopped.set()
                self._cap.release()

        modules = {
            "gazefollower": types.ModuleType("gazefollower"),
            "gazefollower.ui": types.SimpleNamespace(CameraPreviewerUI=UI),
            "gazefollower.camera": types.SimpleNamespace(WebCamCamera=Camera),
            "gazefollower.face_alignment": types.SimpleNamespace(
                MediaPipeFaceAlignment=Alignment, BlazeFaceAlignment=Alignment
            ),
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
        tracker = types.SimpleNamespace(
            config_dict={"webcam_id": 1}, backend_name=lambda win: "psychopy"
        )
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
        with patch.dict(
            "sys.modules",
            {
                "gazefollower.face_alignment": types.SimpleNamespace(
                    BlazeFaceAlignment=None,
                    MediaPipeFaceAlignment=lambda **kwargs: (_ for _ in ()).throw(
                        ValueError("model failed")
                    ),
                )
            },
        ):
            tracker = types.SimpleNamespace(
                config_dict={}, backend_name=lambda win: "psychopy"
            )
            self.preview = PreparedPreview(
                tracker, types.SimpleNamespace(units="height")
            )
            self.preview._thread.join(1)
            with self.assertRaisesRegex(RuntimeError, "preparation failed") as error:
                self.preview.check_ready()
            self.assertIsInstance(error.exception.__cause__, ValueError)

    def test_camera_is_released_when_preview_drawing_fails(self):
        preview = self.create()
        self.allow_load.set()
        self.assertTrue(self.camera_started.wait(1))
        self.camera.callback(None, 123, "frame")
        with patch.object(
            preview.ui, "draw", side_effect=RuntimeError("display failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "display failed"):
                preview.show()
        self.assertTrue(self.camera_stopped.is_set())

    def test_missing_frame_times_out(self):
        preview = self.create()
        with patch(
            "gaze_preview.time.monotonic", return_value=preview._started_at + 31
        ):
            with self.assertRaisesRegex(RuntimeError, "first camera preview frame"):
                preview.check_ready()

    def test_raw_preview_updates_while_detection_is_blocked(self):
        preview = self.create()
        self.allow_load.set()
        self.assertTrue(self.camera_started.wait(1))
        entered, finish = threading.Event(), threading.Event()

        def slow_detection(alignment, timestamp, frame):
            entered.set()
            finish.wait(2)

        with patch.object(preview, "_on_frame", side_effect=slow_detection):
            worker = threading.Thread(
                target=self.camera.callback, args=(None, 1, "old")
            )
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                with patch.dict(
                    "sys.modules",
                    {
                        "cv2": types.SimpleNamespace(
                            COLOR_BGR2RGB=1, cvtColor=lambda frame, code: "rgb:" + frame
                        )
                    },
                ):
                    self.camera._preview_frame_callback(2, "new")
                preview.ui.backend.before_draw()
                self.assertEqual(preview.ui.frame_image, "rgb:new")
                self.assertFalse(preview.check_ready())
            finally:
                finish.set()
                worker.join(1)

    def test_close_returns_while_model_is_still_loading(self):
        preview = self.create()
        self.assertFalse(preview.close(timeout=0.01))
        self.assertTrue(preview._stop.is_set())
        self.assertTrue(preview._thread.is_alive())
        self.allow_load.set()
        self.assertTrue(preview.close())
        self.assertFalse(self.camera_started.is_set())

    def test_escape_in_sdk_preview_quits_and_releases_camera(self):
        preview = self.create()
        self.allow_load.set()
        self.assertTrue(self.camera_started.wait(1))
        keyboard = types.SimpleNamespace(getKeys=lambda **kwargs: ["escape"])
        with patch.dict(
            "sys.modules", {"psychopy": types.SimpleNamespace(event=keyboard)}
        ):
            with patch.object(
                preview.ui,
                "draw",
                side_effect=lambda: preview.ui.backend.listen_event(preview.ui),
            ):
                with self.assertRaises(QuitExperiment):
                    preview.show()
        self.assertTrue(self.camera_stopped.is_set())

    def test_continuing_requires_camera_cleanup_to_finish(self):
        preview = self.create()
        with patch.object(preview.ui, "draw"), patch.object(
            preview, "close", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                preview.show()


if __name__ == "__main__":
    unittest.main()
