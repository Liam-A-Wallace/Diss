"""Five-photo pilot: python demo.py [--prepare] [--gaze-dot]."""
import argparse
import csv
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent


DEMO_SOURCES = [
    ('stimuli/real/real_021.png', 'real', 'real'),
    ('stimuli/quality_xhigh/pair_012_A.png', 'fake', 'openai_xhigh_A'),
    ('stimuli/real/real_022.png', 'real', 'real'),
    ('outputs/sd35-medium-v4-framed/img_seed_000046.png', 'fake', 'sd35'),
    ('stimuli/real/real_023.png', 'real', 'real'),
]


def prepare():
    chosen, seen = [], set()
    for rel_path, label, difficulty in DEMO_SOURCES:
        source = ROOT / rel_path
        key = str(source.resolve()).casefold()
        if key in seen or not source.is_file():
            raise ValueError(f'Demo stimulus missing or duplicated: {source}')
        chosen.append(dict(stimulus=source.relative_to(ROOT).as_posix(),
                           condition=label, difficulty=difficulty))
        seen.add(key)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='prepare list without opening camera')
    parser.add_argument('--gaze-dot', action='store_true', help='show live gaze dot (can influence viewing)')
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    target = prepare()
    if args.prepare:
        return
    import experiment
    experiment.main(['--test', '--trials', str(target), '--practice-trials', '0',
                     '--no-break'] + (['--demo'] if args.gaze_dot else []))


if __name__ == '__main__':
    main()

