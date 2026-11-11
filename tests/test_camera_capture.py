"""Slow inference must not stall capture or accumulate stale frames."""

import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from camera_capture import capture_latest, session_camera_class, stop_camera


class CameraCaptureTests(unittest.TestCase):
    def test_mode_close_waits_for_callback_and_keeps_hardware_open(self):
        entered = threading.Event()
        resume = threading.Event()
        stopping = threading.Event()
        stopped = threading.Event()
        samples = []

        class Camera:
            def __init__(self):
                self.callback_and_param_lock = threading.Lock()
                self.camera_running_state = "sampling"
                self._cap = types.SimpleNamespace(release=Mock())

            def stop_sampling(self):
                self.close()
                self.camera_running_state = "closing"

        camera = session_camera_class(Camera)()

        def callback():
            with camera.callback_and_param_lock:
                entered.set()
                resume.wait(2)
                samples.append("final sample and trigger")

        def stop():
            stopping.set()
            camera.stop_sampling()
            stopped.set()

        with patch.dict("sys.modules", {
            "gazefollower.logger": types.SimpleNamespace(
                Log=types.SimpleNamespace(i=lambda message: None)
            ),
            "gazefollower.misc": types.SimpleNamespace(
                CameraRunningState=types.SimpleNamespace(CLOSING="closing")
            ),
        }):
            processing = threading.Thread(target=callback)
            closing = threading.Thread(target=stop)
            processing.start()
            try:
                self.assertTrue(entered.wait(2))
                closing.start()
                self.assertTrue(stopping.wait(2))
                self.assertFalse(stopped.wait(0.05))
            finally:
                resume.set()
                processing.join(2)
                closing.join(2)
            self.assertTrue(stopped.is_set())
            camera._cap.release.assert_not_called()
            self.assertEqual(samples, ["final sample and trigger"])
            with camera.callback_and_param_lock:
                self.assertEqual(camera.camera_running_state, "closing")

    def test_stalled_callback_releases_camera_but_blocks_model_cleanup(self):
        entered = threading.Event()
        resume = threading.Event()
        released = threading.Event()

        def callback(*args):
            entered.set()
            resume.wait()

        camera = types.SimpleNamespace(
            _camera_thread_running=True,
            _cap=types.SimpleNamespace(read=lambda: (True, 1), release=released.set),
            img_width=640,
            img_height=480,
            callback_and_param_lock=threading.Lock(),
            camera_running_state="sampling",
            callback_func=callback,
            callback_args=(),
            callback_kwargs={},
        )
        cv2 = types.SimpleNamespace(
            COLOR_BGR2RGB=1, cvtColor=lambda f, c: f, resize=lambda f, size: f
        )
        with patch.dict(
            "sys.modules",
            {
                "cv2": cv2,
                "gazefollower.logger": types.SimpleNamespace(
                    Log=types.SimpleNamespace(e=lambda e: None)
                ),
            },
        ):
            camera._camera_thread = threading.Thread(
                target=capture_latest, args=(camera,)
            )
            camera._camera_thread.start()
            try:
                self.assertTrue(entered.wait(2))
                started = time.monotonic()
                with self.assertRaisesRegex(TimeoutError, "processing"):
                    stop_camera(camera, timeout=0.1)
                self.assertLess(time.monotonic() - started, 1.0)
                self.assertTrue(released.is_set())
                self.assertFalse(camera._camera_thread.is_alive())
                self.assertTrue(camera._processing_thread.is_alive())
            finally:
                resume.set()
                stop_camera(camera)
            self.assertFalse(camera._processing_thread.is_alive())

    def test_slow_callback_skips_backlog_and_stops_cleanly(self):
        entered = threading.Event()
        drained = threading.Event()
        processed = threading.Event()
        seen = []
        previewed = []
        count = 0

        def read():
            nonlocal count
            count += 1
            if count == 2:
                if not entered.wait(2):
                    raise RuntimeError("Processing did not start")
            if count > 10:
                drained.set()
                if not processed.wait(2):
                    raise RuntimeError("Latest frame was not processed")
                camera._camera_thread_running = False
                return False, None
            return True, count

        def callback(state, timestamp, frame):
            seen.append(frame)
            if frame == 1:
                entered.set()
                if not drained.wait(2):
                    raise RuntimeError("Capture blocked on inference")
                self.assertIn(10, previewed)
            if frame == 10:
                processed.set()

        camera = types.SimpleNamespace(
            _camera_thread_running=True,
            _cap=types.SimpleNamespace(read=read, release=lambda: None),
            img_width=640,
            img_height=480,
            callback_and_param_lock=threading.Lock(),
            camera_running_state="sampling",
            callback_func=callback,
            callback_args=(),
            callback_kwargs={},
        )
        camera._preview_frame_callback = lambda timestamp, frame: previewed.append(
            frame
        )
        errors = []
        cv2 = types.SimpleNamespace(
            COLOR_BGR2RGB=1, cvtColor=lambda f, c: f, resize=lambda f, size: f
        )
        with patch.dict(
            "sys.modules",
            {
                "cv2": cv2,
                "gazefollower.logger": types.SimpleNamespace(
                    Log=types.SimpleNamespace(e=errors.append)
                ),
            },
        ):
            capture_latest(camera)
        self.assertEqual(errors, [])
        self.assertEqual(seen, [1, 10])
        self.assertEqual(previewed, list(range(1, 11)))
        self.assertFalse(camera._camera_thread_running)
