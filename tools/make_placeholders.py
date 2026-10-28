"""Generate placeholder faces + trials.csv until a real dataset is chosen."""
import csv
import os

from PIL import Image, ImageDraw, ImageFilter, ImageFont

STIM_DIR = "stimuli"
SIZE = (512, 512)
N_REAL = 6
FAKES_PER_DIFFICULTY = 2

DIFFICULTIES = ["easy", "medium", "hard"]


def _font(size=28):
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _face_image(label, skin, artifact=None, light=True):
    # Light backgrounds match the cross screen; dark ones reintroduce the
    # luminance swing for A/B testing of the exposure/lighting effect.
    if light:
        base = (240, 240, 245)
        top, bottom = (248, 248, 252), (218, 218, 228)
        outline = (170, 170, 185)
        label_fill = (60, 60, 70)
    else:
        base = (30, 30, 40)
        top, bottom = (70, 70, 90), (20, 20, 30)
        outline = (230, 230, 230)
        label_fill = (255, 255, 255)

    img = Image.new("RGB", SIZE, base)

    # top-to-bottom gradient behind the face
    px = img.load()
    for y in range(SIZE[1]):
        t = y / (SIZE[1] - 1)
        row = tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(0, SIZE[0], 4):
            for dx in range(4):
                if x + dx < SIZE[0]:
                    px[x + dx, y] = row

    draw = ImageDraw.Draw(img)
    draw.ellipse([156, 96, 356, 396], fill=skin,
                 outline=outline, width=4)
    draw.ellipse([206, 176, 236, 206], fill=(40, 40, 40))
    draw.ellipse([276, 176, 306, 206], fill=(40, 40, 40))
    draw.line([256, 220, 248, 260], fill=(120, 90, 80), width=4)
    draw.line([248, 260, 258, 262], fill=(120, 90, 80), width=4)
    draw.arc([220, 270, 292, 322], start=20, end=160,
             fill=(80, 40, 40), width=5)

    # blur a patch of the face to stand in for an AI artefact
    if artifact:
        box, radius = artifact
        crop = img.crop(box).filter(ImageFilter.GaussianBlur(radius))
        img.paste(crop, box)

    draw.text((12, 12), label, fill=label_fill, font=_font(24))
    return img


def generate(csv_path="trials.csv"):
    """Generate placeholder face images and write the trial list to csv_path.

    Backgrounds alternate light/dark so the A/B lighting test can be run.
    Returns the path to the CSV listing the generated images.
    """
    os.makedirs(os.path.join(STIM_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(STIM_DIR, "fake"), exist_ok=True)

    rows = []
    light = True  # flip after every image: light, dark, light, dark, ...

    # real faces get a plain skin tone, fakes get a blur patch
    for i in range(1, N_REAL + 1):
        name = f"real_{i:03d}.png"
        path = os.path.join(STIM_DIR, "real", name)
        _face_image(f"REAL {i:03d}", skin=(210, 165, 140),
                    light=light).save(path)
        rows.append({"stimulus": path, "condition": "real",
                     "difficulty": "na",
                     "background": "light" if light else "dark"})
        light = not light

    # more visible blur = easier to spot
    artifact_specs = {
        "easy": ((150, 210, 380, 370), 14),
        "medium": ((190, 240, 330, 340), 6),
        "hard": ((240, 300, 280, 330), 2),
    }
    idx = 1
    for difficulty in DIFFICULTIES:
        box, radius = artifact_specs[difficulty]
        for _ in range(FAKES_PER_DIFFICULTY):
            name = f"fake_{idx:03d}.png"
            path = os.path.join(STIM_DIR, "fake", name)
            _face_image(f"FAKE {idx:03d}", skin=(200, 155, 130),
                        artifact=(box, radius), light=light).save(path)
            rows.append({"stimulus": path, "condition": "fake",
                         "difficulty": difficulty,
                         "background": "light" if light else "dark"})
            idx += 1
            light = not light

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stimulus", "condition", "difficulty",
                           "background"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def main():
    # never overwrite the real trials.csv
    csv_path = generate("trials_placeholder.csv")
    n_fake = FAKES_PER_DIFFICULTY * len(DIFFICULTIES)
    print(f"Created {N_REAL + n_fake} placeholder images (alternating "
          f"light/dark) and wrote '{csv_path}' "
          f"({N_REAL} real, {n_fake} fake).")


if __name__ == "__main__":
    main()
