"""Local facial-landmark preview for the five-photo demo; no simulated gaze."""
import csv
from pathlib import Path
import os
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    from demo import prepare
    from analyse import gallery
    import mediapipe as mp
    from PIL import Image, ImageDraw
    trial_list = prepare()
    output = ROOT / 'processed' / 'demo_landmarks'
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT.parent / 'GazeFollower/gazefollower/res/model_weights/face_landmarker.task'
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE, num_faces=2)
    items, records = [], []
    with trial_list.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    with vision.FaceLandmarker.create_from_options(options) as detector:
        for index, row in enumerate(rows, 1):
            path = ROOT / row['stimulus']
            with Image.open(path) as source:
                image = source.convert('RGB')
            import numpy as np
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                              data=np.array(image)))
            draw = ImageDraw.Draw(image)
            note = 'Detected faces: ' + str(len(result.face_landmarks))
            if len(result.face_landmarks) == 1:
                for landmark_id, point in enumerate(result.face_landmarks[0]):
                    x, y = point.x * image.width, point.y * image.height
                    draw.ellipse((x-1, y-1, x+1, y+1), fill='cyan')
                    records.append(dict(stimulus=row['stimulus'], landmark_id=landmark_id,
                                        image_x=x, image_y=y, image_u=point.x, image_v=point.y,
                                        relative_z=point.z))
            else:
                note += '; requires review; no landmarks exported.'
            preview = output / ('photo_' + str(index) + '.png')
            image.save(preview)
            items.append(('Photo ' + str(index), preview, True, note))
    fields = ['stimulus', 'landmark_id', 'image_x', 'image_y', 'image_u', 'image_v', 'relative_z']
    with (output / 'landmarks.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    report = gallery(output / 'index.html', 'Five-photo facial landmarks',
                     'Cyan dots are detected landmarks on stimulus faces. Coordinates use original image pixels. '
                     'These are not gaze measurements or reviewed AOI masks. Record your viewing with demo.py, '
                     'then run analyse.py --test for measured scanpaths and heatmaps.', items)
    print('Landmark report:', report)
    webbrowser.open(report.resolve().as_uri())


if __name__ == '__main__':
    main()
