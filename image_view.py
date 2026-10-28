"""Dependency-free image geometry; pixels have a top-left origin and +y down."""
from dataclasses import dataclass
import math


@dataclass
class ImageView:
    image_width: int
    image_height: int
    viewport: tuple  # left, top, width, height in window pixels
    max_zoom: float = 4.0

    def __post_init__(self):
        if min(self.image_width, self.image_height, *self.viewport[2:]) <= 0:
            raise ValueError("Image and viewport dimensions must be positive")
        self.fit_scale = min(self.viewport[2] / self.image_width,
                             self.viewport[3] / self.image_height)
        self.reset()

    @property
    def zoom(self):
        return self.scale / self.fit_scale

    def reset(self):
        self.scale = self.fit_scale
        self.left = self.viewport[0] + (self.viewport[2] - self.image_width * self.scale) / 2
        self.top = self.viewport[1] + (self.viewport[3] - self.image_height * self.scale) / 2

    def contains(self, x, y):
        l, t, w, h = self.viewport
        return l <= x < l + w and t <= y < t + h

    def _constrain(self):
        for attr, origin, extent, size in (
            ("left", self.viewport[0], self.viewport[2], self.image_width),
            ("top", self.viewport[1], self.viewport[3], self.image_height),
        ):
            span = size * self.scale
            value = (origin + (extent - span) / 2 if span <= extent else
                     min(origin, max(origin + extent - span, getattr(self, attr))))
            setattr(self, attr, value)

    def scroll(self, steps, x, y):
        if not self.contains(x, y) or not math.isfinite(steps):
            return
        new_scale = min(self.fit_scale * self.max_zoom, max(
            self.fit_scale, self.scale * math.exp(max(-4, min(4, steps * 0.15)))))
        factor = new_scale / self.scale
        self.left = x - (x - self.left) * factor
        self.top = y - (y - self.top) * factor
        self.scale = new_scale
        self._constrain()

    def pan(self, dx, dy):
        self.left += dx
        self.top += dy
        self._constrain()

    def snapshot(self):
        return dict(left=self.left, top=self.top, scale=self.scale, zoom=self.zoom,
                    image_width=self.image_width, image_height=self.image_height,
                    viewport=list(self.viewport))


def map_gaze(x, y, frame):
    """Return original image pixels, normalised coordinates and visibility.

    Tracker pixels are normalised to the fullscreen window on the SAME monitor.
    Never clamp gaze: off-screen/off-image data must remain distinguishable.
    """
    x = float(x) * frame["window_width"] / frame["tracker_width"]
    y = float(y) * frame["window_height"] / frame["tracker_height"]
    ix = (x - frame["left"]) / frame["scale"]
    iy = (y - frame["top"]) / frame["scale"]
    u, v = ix / frame["image_width"], iy / frame["image_height"]
    l, t, w, h = frame["viewport"]
    visible = (math.isfinite(u) and math.isfinite(v) and
               0 <= x < frame["window_width"] and 0 <= y < frame["window_height"] and
               l <= x < l + w and t <= y < t + h and 0 <= u < 1 and 0 <= v < 1)
    return ix, iy, u, v, visible
