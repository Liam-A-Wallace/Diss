"""Five-photo pilot: python demo.py [--prepare] [--gaze-dot]."""
import argparse
import csv
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent


def prepare():
    metadata = ROOT / 'trials.csv'
    with metadata.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    pools = {label: [r for r in rows if r['condition'].lower() == label
                    and (label != 'fake' or Path(r['stimulus']).as_posix().startswith('stimuli/openai_shared_style/'))]
             for label in ('real', 'fake')}
    chosen, seen = [], set()
    for label in ('real', 'fake', 'real', 'fake', 'real'):
        for row in pools[label]:
            source = ROOT / row['stimulus']
            key = str(source.resolve()).casefold()
            if key not in seen and source.is_file():
                chosen.append(dict(stimulus=row['stimulus'],
                                   condition=label, difficulty=''))
                seen.add(key)
                break
        else:
            raise ValueError('Demo needs three real images and two OpenAI fakes in trials.csv.')
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

