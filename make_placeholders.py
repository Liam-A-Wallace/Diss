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


def _face_image(label, skin, artifact=None):
    img = Image.new("RGB", SIZE, (30, 30, 40))

    top, bottom = (70, 70, 90), (20, 20, 30)
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
                 outline=(230, 230, 230), width=4)
    draw.ellipse([206, 176, 236, 206], fill=(40, 40, 40))
    draw.ellipse([276, 176, 306, 206], fill=(40, 40, 40))
    draw.line([256, 220, 248, 260], fill=(120, 90, 80), width=4)
    draw.line([248, 260, 258, 262], fill=(120, 90, 80), width=4)
    draw.arc([220, 270, 292, 322], start=20, end=160,
             fill=(80, 40, 40), width=5)

    if artifact:
        box, radius = artifact
        crop = img.crop(box).filter(ImageFilter.GaussianBlur(radius))
        img.paste(crop, box)

    draw.text((12, 12), label, fill=(255, 255, 255), font=_font(24))
    return img


def main():
    os.makedirs(os.path.join(STIM_DIR, "real"), exist_ok=True)
    os.makedirs(os.path.join(STIM_DIR, "fake"), exist_ok=True)

    rows = []

    for i in range(1, N_REAL + 1):
        name = f"real_{i:03d}.png"
        path = os.path.join(STIM_DIR, "real", name)
        _face_image(f"REAL {i:03d}", skin=(210, 165, 140)).save(path)
        rows.append({"stimulus": path, "condition": "real", "difficulty": "na"})

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
                        artifact=(box, radius)).save(path)
            rows.append({"stimulus": path, "condition": "fake",
                         "difficulty": difficulty})
            idx += 1

    with open("trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["stimulus", "condition", "difficulty"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created {len(rows)} placeholder images and trials.csv "
          f"({N_REAL} real, {idx - 1} fake).")


if __name__ == "__main__":
    main()
