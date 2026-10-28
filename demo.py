"""Five-photo pilot: python demo.py [--prepare] [--gaze-dot]."""
import argparse
import csv
from pathlib import Path
import os
import sys
import shutil

ROOT = Path(__file__).resolve().parent


def prepare():
    metadata = ROOT / 'dataset/outputs/picked.csv'
    with metadata.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    pools = {label: [r for r in rows if r['label'].lower() == label]
             for label in ('real', 'fake')}
    chosen, seen = [], set()
    destination = ROOT / 'stimuli' / 'demo'
    destination.mkdir(parents=True, exist_ok=True)
    for label in ('real', 'fake', 'real', 'fake', 'real'):
        for row in pools[label]:
            source = ROOT.parent / 'Dataset' / row['picked_file']
            key = str(source.resolve()).casefold()
            if key not in seen and source.is_file():
                target_image = destination / (label + '_' + source.name)
                shutil.copy2(source, target_image)
                chosen.append(dict(stimulus=target_image.relative_to(ROOT).as_posix(),
                                   condition=label, difficulty=''))
                seen.add(key)
                break
        else:
            raise ValueError('Demo needs three distinct real and two distinct fake curated photos.')
    target = ROOT / 'data' / 'demo_trials.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)
    print('Five-photo list:', target)
    for row in chosen:
        print(' ', row['stimulus'])
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='prepare list without opening camera')
    parser.add_argument('--gaze-dot', action='store_true', help='show live gaze dot (can influence viewing)')
    args = parser.parse_args()
    os.chdir(ROOT)
    target = prepare()
    if args.prepare:
        return
    import config
    config.TRIALS_CSV = str(target)
    config.PRACTICE_TRIALS = 0
    config.BREAK_AT_FRACTION = None
    sys.argv = ['experiment.py', '--test'] + (['--demo'] if args.gaze_dot else [])
    import experiment
    experiment.main()


if __name__ == '__main__':
    main()

