import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import demo


class DemoSourceTests(unittest.TestCase):
    def test_demo_uses_current_openai_sources_without_copying(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = []
            for label, paths in [('real', ['real/a.png', 'real/b.png', 'real/c.png']),
                                 ('fake', ['legacy/old.png', 'stimuli/openai_shared_style/a.png', 'stimuli/openai_shared_style/b.png'])]:
                for path in paths:
                    source = root / path
                    source.parent.mkdir(parents=True, exist_ok=True)
                    source.touch()
                    rows.append(dict(stimulus=path, condition=label, difficulty=''))
            with (root / 'trials.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(demo, 'ROOT', root):
                output = demo.prepare()
            with output.open() as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual([r['condition'] for r in selected], ['real', 'fake', 'real', 'fake', 'real'])
            self.assertTrue(all(r['stimulus'].startswith('stimuli/openai_shared_style/') for r in selected if r['condition']=='fake'))
            self.assertFalse((root / 'stimuli/demo').exists())
