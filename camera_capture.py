"""Keep webcam capture draining while gaze inference processes the latest frame."""

import threading
import time


def stop_camera(camera, timeout=2.0):
    """Shared SDK close/release adapter; capture owns release while active.

    Never release the native capture concurrently with read(), or destroy an
    alignment model while its callback is running. A timeout fails the handoff
    to calibration rather than opening a second camera.
    """
    deadline = time.monotonic() + timeout
    camera._camera_thread_running = False
    worker = getattr(camera, "_camera_thread", None)
    if worker is not None:
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            raise TimeoutError(
                f"Webcam capture did not stop within {timeout:g} seconds"
            )
    else:
        camera._cap.release()
    # Capture has released the webcam, but a callback may still own the model.
    # Use the remaining budget so callers cannot destroy that model too early.
    processing = getattr(camera, "_processing_thread", None)
    if processing is not None:
        processing.join(max(0.0, deadline - time.monotonic()))
        if processing.is_alive():
            raise TimeoutError(
                f"Webcam processing did not stop within {timeout:g} seconds"
            )


def capture_latest(camera):
    import cv2
    from gazefollower.logger import Log

    condition = threading.Condition()
    latest = None
    finished = False
    captured = processed = 0
    processing_ms = frame_age_ms = 0.0
    report_at = time.monotonic()

    def process():
        nonlocal latest, processed, processing_ms, frame_age_ms
        while True:
            with condition:
                condition.wait_for(lambda: latest is not None or finished)
                if finished:
                    return
                timestamp, frame = latest
                latest = None
            try:
                started = time.monotonic()
                frame_age_ms = (time.time_ns() - timestamp) / 1e6
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (camera.img_width, camera.img_height))
                with camera.callback_and_param_lock:
                    if camera.callback_func is not None:
                        camera.callback_func(
                            camera.camera_running_state,
                            timestamp,
                            frame,
                            *camera.callback_args,
                            **camera.callback_kwargs,
                        )
                processed += 1
                processing_ms = (time.monotonic() - started) * 1000
            except Exception as error:
                Log.e(f"Camera frame processing failed: {error}")

    worker = threading.Thread(target=process, name="latest-camera-frame", daemon=True)
    camera._processing_thread = worker
    worker.start()
    try:
        while camera._camera_thread_running:
            ok, frame = camera._cap.read()
            timestamp = time.time_ns()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            captured += 1
            now = time.monotonic()
            if now - report_at >= 5:
                elapsed = now - report_at
                Log.i(
                    f"Camera throughput: capture={captured / elapsed:.1f} fps, "
                    f"processing={processed / elapsed:.1f} fps, "
                    f"last processing={processing_ms:.0f} ms, "
                    f"frame age before processing={frame_age_ms:.0f} ms"
                )
                captured = processed = 0
                report_at = now
            # Preview must not wait for face/eye inference. This hook only
            # publishes the newest frame; expensive processing stays below.
            preview = getattr(camera, "_preview_frame_callback", None)
            if preview is not None:
                preview(timestamp, frame)
            with condition:
                # Replace an unprocessed frame rather than building a backlog.
                latest = (timestamp, frame)
                condition.notify()
    finally:
        with condition:
            finished = True
            latest = None
            condition.notify()
        # This thread owns capture: release only after read() has returned.
        # stop_camera waits for processing separately, within its timeout.
        camera._cap.release()
