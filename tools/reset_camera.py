"""Diagnose and reset Windows webcam exposure/gain settings on MSMF and DSHOW."""
import cv2
import numpy as np

PROPS = [
    ("AUTO_EXPOSURE", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("EXPOSURE", cv2.CAP_PROP_EXPOSURE),
    ("GAIN", cv2.CAP_PROP_GAIN),
    ("BRIGHTNESS", cv2.CAP_PROP_BRIGHTNESS),
    ("CONTRAST", cv2.CAP_PROP_CONTRAST),
    ("SATURATION", cv2.CAP_PROP_SATURATION),
    ("FPS", cv2.CAP_PROP_FPS),
]


def dump_state(label, cap):
    vals = {name: cap.get(prop) for name, prop in PROPS}
    ok, frame = cap.read()
    if ok and frame is not None:
        b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
        frame_info = (
            f"shape={frame.shape} min={frame.min()} max={frame.max()} "
            f"mean_BGR=({b.mean():.1f}, {g.mean():.1f}, {r.mean():.1f}) "
            f"std={frame.std():.1f}"
        )
    else:
        frame_info = "NO FRAME"
    print(f"[{label}] {vals}")
    print(f"[{label}] frame: {frame_info}")


def main():
    for name, backend in (("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)):
        cap = cv2.VideoCapture(0, backend)
        if not cap.isOpened():
            print(f"[{name}] Could not open camera 0")
            continue
        dump_state(f"{name} before", cap)
        print(f"[{name}] set GAIN=0 -> {cap.set(cv2.CAP_PROP_GAIN, 0)}")
        print(f"[{name}] set AUTO_EXPOSURE=1 -> {cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)}")
        # Grab a few frames so the sensor adapts to auto-exposure.
        for _ in range(10):
            cap.read()
        dump_state(f"{name} after", cap)
        if backend == cv2.CAP_DSHOW:
            cap.set(cv2.CAP_PROP_SETTINGS, 1)
            input("If a camera dialog opened, click Default -> Apply -> OK on both tabs, then press Enter...")
            for _ in range(5):
                cap.read()
            dump_state(f"{name} after dialog", cap)
        cap.release()


if __name__ == "__main__":
    main()
