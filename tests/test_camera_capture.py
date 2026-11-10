"""Slow inference must not stall capture or accumulate stale frames."""

import threading
import time
import types
import unittest
from unittest.mock import patch

from camera_capture import capture_latest, stop_camera


class CameraCaptureTests(unittest.TestCase):
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
