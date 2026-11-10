"""Regression checks for deferred I/O, streaming and camera shutdown."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from analysis.ingest import write_csv
from camera_capture import stop_camera
from session_log import SessionLog, export_image_gaze


class EfficiencyTests(unittest.TestCase):
    def test_logging_callback_returns_while_disk_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            base = str(Path(folder) / "session")
            log = SessionLog(base, {}, (10, 10), (10, 10), {}, [], [])
            entered, resume = threading.Event(), threading.Event()
            original = log.events

            class SlowFile:
                def write(self, value):
                    entered.set()
                    if not resume.wait(2):
                        raise RuntimeError("Test failed to unblock writer")
                    original.write(value)

                flush = original.flush
                close = original.close

            log.events = SlowFile()
            try:
                log.present({"timestamp_ns": 1})
                self.assertTrue(entered.wait(1))
                log.end_view(1, timestamp_ns=2)
            finally:
                resume.set()
                log.finish("complete")
            import json
            rows = [json.loads(line) for line in Path(base + "_display.jsonl").read_text().splitlines()]
            self.assertEqual([r["kind"] for r in rows], ["present", "offset"])

    def test_writer_failure_reaches_caller(self):
        with tempfile.TemporaryDirectory() as folder:
            log = SessionLog(str(Path(folder) / "s"), {}, (1, 1), (1, 1), {}, [], [])
            log.events.close()
            log.present({"timestamp_ns": 1})
            with self.assertRaisesRegex(RuntimeError, "Display log"):
                log.finish("error")

    def test_explicit_csv_fields_stream_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rows.csv"

            def rows():
                self.assertTrue(path.exists())
                yield {"value": 1}
                yield {"value": 2}

            write_csv(path, rows(), ["value"])
            with path.open() as source:
                self.assertEqual(list(csv.DictReader(source)), [{"value": "1"}, {"value": "2"}])

    def test_optional_export_keeps_first_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            base = str(Path(folder) / "s")
            with patch("analysis.ingest.load_display", return_value=[]), patch(
                    "analysis.ingest.align_samples", return_value=iter([{"timestamp": 1}, {"timestamp": 2}])):
                export_image_gaze(base)
            with open(base + "_gaze_image.csv") as source:
                self.assertEqual(list(csv.DictReader(source)), [{"timestamp": "1"}, {"timestamp": "2"}])

    def test_shutdown_never_releases_capture_while_read_is_blocked(self):
        released = []
        camera = types.SimpleNamespace(
            _camera_thread=types.SimpleNamespace(join=lambda timeout: None, is_alive=lambda: True),
            _cap=types.SimpleNamespace(release=lambda: released.append(True)))
        with self.assertRaises(TimeoutError):
            stop_camera(camera, timeout=0.01)
        self.assertFalse(camera._camera_thread_running)
        self.assertEqual(released, [])

    def test_demo_passes_options_without_mutating_argv(self):
        import demo
        import sys
        original = list(sys.argv)
        calls = []
        with patch.object(demo, "prepare", return_value=Path("demo.csv")), patch.object(demo.os, "chdir"), patch.dict(
                "sys.modules", {"experiment": types.SimpleNamespace(main=calls.append)}):
            demo.main(["--gaze-dot"])
        self.assertEqual(sys.argv, original)
        self.assertIn("--no-break", calls[0])
        self.assertIn("--demo", calls[0])

    def test_manifest_commands_produce_identical_rows(self):
        root = Path(__file__).parents[1]
        modules = []
        for relative in ("dataset/scripts/build_60k_manifest.py", "stimulus-curation/build_manifest.py"):
            spec = importlib.util.spec_from_file_location("manifest_test", root / relative)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            real, fake = folder / "Real", folder / "Fake"
            real.mkdir()
            fake.mkdir()
            for i in range(3):
                (real / f"{i}.png").touch()
                (fake / f"{i}.jpg").touch()
            outputs = []
            for i, module in enumerate(modules):
                output = folder / f"{i}.csv"
                module.main(["--real-dir", str(real), "--fake-dir", str(fake),
                             "--sample", "2", "--out", str(output)])
                outputs.append(output.read_bytes())
            self.assertEqual(*outputs)


if __name__ == "__main__":
    unittest.main()
