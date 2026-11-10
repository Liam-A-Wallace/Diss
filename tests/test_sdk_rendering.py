"""SDK preview drawing reuses labels and unchanged image textures."""
import types
import unittest
from sdk_rendering import optimize_backend


class RenderingTests(unittest.TestCase):
    def test_static_labels_and_frames_are_built_once(self):
        created = []
        uploads = []

        class Stim:
            def __init__(self, *args, **kwargs):
                created.append(kwargs)
                self.text = kwargs.get("text")
                self.draws = 0

            def draw(self):
                self.draws += 1

        class Backend:
            visual = types.SimpleNamespace(TextStim=Stim)
            win = None

            def __init__(self):
                self.texture_cache = {}

            def pixel_to_psychopy_coordinate(self, x, y):
                return x, y

            def draw_texture(self, img, rect):
                uploads.append(img)
                self.texture_cache[tuple(rect)] = types.SimpleNamespace(draw=lambda: None)

            def after_draw(self):
                pass

        optimize_backend(Backend)
        backend = Backend()
        frame = types.SimpleNamespace(shape=(100, 100, 3), size=30000)
        for _ in range(60):
            backend.draw_text("Face Image", "Open Sans", 20, (0, 0, 0), (0, 0, 100, 100))
            backend.draw_texture(frame, (0, 0, 100, 100))
        self.assertEqual(len(created), 1)
        self.assertEqual(len(uploads), 1)
        backend.draw_text("Updated", "Open Sans", 20, (0, 0, 0), (0, 0, 100, 100))
        self.assertEqual(len(created), 1)
        self.assertEqual(next(iter(backend._diss_labels.values()))[0].text, "Updated")
        backend.draw_texture(types.SimpleNamespace(shape=(100, 100, 3), size=30000), (0, 0, 100, 100))
        self.assertEqual(len(uploads), 2)


if __name__ == "__main__":
    unittest.main()
