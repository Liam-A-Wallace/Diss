"""Avoid rebuilding SDK text and uploading unchanged preview frames each flip."""
import time


def optimize_backend(backend_class):
    if getattr(backend_class, "_diss_render_cache", False):
        return
    original_texture = backend_class.draw_texture
    original_after_draw = backend_class.after_draw

    def after_draw(self):
        original_after_draw(self)
        now = time.monotonic()
        started, count = getattr(self, "_diss_draw_rate", (now, 0))
        count += 1
        if now - started >= 5:
            from gazefollower.logger import Log
            Log.i(f"SDK display throughput: {count / (now - started):.1f} fps")
            started, count = now, 0
        self._diss_draw_rate = (started, count)

    def draw_text(self, text, font_name, font_size, text_color, rect, align="center"):
        cache = getattr(self, "_diss_labels", None)
        if cache is None:
            cache = self._diss_labels = {}
        key = (tuple(rect), font_name, font_size, align)
        entry = cache.get(key)
        colour = tuple(text_color)
        if entry is None:
            stim = self.visual.TextStim(
                self.win, text=str(text), font=font_name, height=font_size,
                color=colour, colorSpace="rgb255", units="pix",
                pos=self.pixel_to_psychopy_coordinate(rect[0] + rect[2] // 2,
                                                      rect[1] + rect[3] // 2),
                alignText=align, autoLog=False)
            cache[key] = (stim, str(text), colour)
        else:
            stim, previous, previous_colour = entry
            if str(text) != previous:
                stim.text = str(text)
            if colour != previous_colour:
                stim.color = colour
            cache[key] = (stim, str(text), colour)
        stim.draw()

    def draw_texture(self, img, rect):
        # SDK frame packets are replaced, never updated in place by our UI.
        # Retain the object itself, so Python cannot recycle its id underneath us.
        if hasattr(img, "shape") and len(img.shape) == 3 and img.size:
            h, w = img.shape[:2]
            x, y, tw, th = rect
            if tw / th > w / h:
                sh, sw = int(th), int(int(th) * w / h)
            else:
                sw, sh = int(tw), int(int(tw) / (w / h))
            key = (x + (tw - sw) // 2, y + (th - sh) // 2, sw, sh)
            sources = getattr(self, "_diss_texture_sources", None)
            if sources is None:
                sources = self._diss_texture_sources = {}
            if sources.get(key) is img and key in self.texture_cache:
                self.texture_cache[key].draw()
                return
            original_texture(self, img, rect)
            sources[key] = img
            return
        original_texture(self, img, rect)

    backend_class.draw_text = draw_text
    backend_class.draw_texture = draw_texture
    backend_class.after_draw = after_draw
    backend_class._diss_render_cache = True


def apply():
    from gazefollower.ui.UIBackend import PsychoPyUIBackend
    optimize_backend(PsychoPyUIBackend)
