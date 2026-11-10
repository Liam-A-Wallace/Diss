"""Exercise generation recovery without making API requests."""

import base64
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import generate_images


class FakeRateLimitError(Exception):
    code = "credit_balance_exhausted"


class GenerateImagesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.out = Path(self.temporary.name)
        self.args = ["--count", "2", "--out", str(self.out), "--seed", "20261105"]
        self.client = MagicMock()
        self.response = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(
                    b64_json=base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode()
                )
            ]
        )
        self.client.images.generate.return_value = self.response
        self.factory = MagicMock()
        self.factory.return_value.__enter__.return_value = self.client
        fake_sdk = types.SimpleNamespace(
            OpenAI=self.factory, RateLimitError=FakeRateLimitError
        )
        sdk_patch = patch.dict("sys.modules", {"openai": fake_sdk})
        sdk_patch.start()
        self.addCleanup(sdk_patch.stop)
        env_patch = patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def run_generator(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return generate_images.main(self.args)

    def read_log(self):
        return json.loads((self.out / "prompts.json").read_text(encoding="utf-8"))

    def test_random_default_seed_is_saved_and_reused(self):
        self.args = self.args[:-2]
        with patch("generate_images.random.SystemRandom") as rng:
            rng.return_value.randrange.return_value = 12345
            self.assertEqual(self.run_generator(), 0)
            self.assertEqual(self.read_log()["settings"]["seed"], 12345)
            self.assertEqual(self.run_generator(), 0)
            rng.return_value.randrange.assert_called_once()
        self.assertEqual(self.client.images.generate.call_count, 2)

    def test_quota_rejection_can_resume_without_repeating_saved_images(self):
        response = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(
                    b64_json=base64.b64encode(b"\x89PNG\r\n\x1a\ntest").decode()
                )
            ]
        )
        self.client.images.generate.side_effect = [
            response,
            FakeRateLimitError("No credits remaining"),
        ]
        self.assertEqual(self.run_generator(), 1)
        log = self.read_log()
        self.assertEqual([row["status"] for row in log["images"]], ["saved", "pending"])
        self.assertEqual(
            log["images"][1]["last_error"]["code"], "credit_balance_exhausted"
        )
        saved_path = self.out / f"{log['images'][0]['id']}.png"
        saved_bytes = saved_path.read_bytes()

        self.client.images.generate.reset_mock()
        self.client.images.generate.side_effect = None
        self.client.images.generate.return_value = response
        self.assertEqual(self.run_generator(), 0)
        self.client.images.generate.assert_called_once()
        self.assertEqual(saved_path.read_bytes(), saved_bytes)
        self.assertTrue(
            all(row["status"] == "saved" for row in self.read_log()["images"])
        )
        self.factory.assert_called_with(timeout=600, max_retries=0)

    def test_timeout_remains_blocked_without_another_api_call(self):
        self.client.images.generate.side_effect = TimeoutError("Request timed out")
        self.assertEqual(self.run_generator(), 1)
        self.assertEqual(self.read_log()["images"][0]["status"], "started")
        self.client.images.generate.reset_mock()
        self.assertEqual(self.run_generator(), 1)
        self.client.images.generate.assert_not_called()

    def test_pending_image_on_disk_blocks_request(self):
        rows = generate_images.make_prompts(2, 20261105)
        (self.out / f"{rows[0]['id']}.png").write_bytes(b"existing image")
        self.assertEqual(self.run_generator(), 1)
        self.client.images.generate.assert_not_called()

    def test_extend_folder_preserves_images_prompts_and_numbering(self):
        self.assertEqual(self.run_generator(), 0)
        original = self.read_log()["images"]
        originals = {p.name: p.read_bytes() for p in self.out.glob("*.png")}
        self.args[1] = "6"
        self.client.images.generate.reset_mock()
        self.assertEqual(self.run_generator(), 0)
        self.assertEqual(self.client.images.generate.call_count, 4)
        log = self.read_log()
        self.assertEqual(log["images"][:2], original)
        self.assertEqual(log["prompt_batches"], [2, 4])
        self.assertEqual(
            {r["id"] for r in log["images"]},
            {f"pair_{pair:03d}_{group}" for pair in range(1, 4) for group in "AB"},
        )
        for name, content in originals.items():
            self.assertEqual((self.out / name).read_bytes(), content)
        self.client.images.generate.reset_mock()
        self.assertEqual(self.run_generator(), 0)
        self.client.images.generate.assert_not_called()
        self.args[1] = "8"
        self.assertEqual(self.run_generator(), 0)
        self.assertEqual(self.read_log()["prompt_batches"], [2, 4, 2])
        self.assertEqual(self.client.images.generate.call_count, 2)

    def test_legacy_log_can_extend_and_resume_limited_generation(self):
        self.assertEqual(self.run_generator(), 0)
        legacy = self.read_log()
        del legacy["prompt_batches"]
        generate_images.save_log(self.out / "prompts.json", legacy)
        self.args[1] = "6"
        self.args.extend(["--limit", "1"])
        self.assertEqual(self.run_generator(), 0)
        log = self.read_log()
        self.assertEqual(log["images"][:2], legacy["images"])
        self.assertEqual(sum(r["status"] == "saved" for r in log["images"]), 3)
        self.args = self.args[:-2]
        self.client.images.generate.reset_mock()
        self.assertEqual(self.run_generator(), 0)
        self.assertEqual(self.client.images.generate.call_count, 3)

    def test_smaller_target_and_changed_settings_are_rejected(self):
        self.args[1] = "4"
        self.assertEqual(self.run_generator(), 0)
        original = (self.out / "prompts.json").read_bytes()
        for args in (["--count", "2"], ["--count", "6", "--quality", "high"]):
            with self.subTest(args=args):
                self.args = [*args, "--out", str(self.out)]
                self.client.images.generate.reset_mock()
                self.assertEqual(self.run_generator(), 1)
                self.client.images.generate.assert_not_called()
                self.assertEqual((self.out / "prompts.json").read_bytes(), original)

    def test_changed_templates_block_extension(self):
        self.assertEqual(self.run_generator(), 0)
        self.args[1] = "4"
        original = (self.out / "prompts.json").read_bytes()
        make_prompts = generate_images.make_prompts

        def changed_prompts(*args):
            rows = make_prompts(*args)
            rows[0]["prompt"] += "changed"
            return rows

        self.client.images.generate.reset_mock()
        with patch.object(generate_images, "make_prompts", side_effect=changed_prompts):
            self.assertEqual(self.run_generator(), 1)
        self.client.images.generate.assert_not_called()
        self.assertEqual((self.out / "prompts.json").read_bytes(), original)

    def test_dry_run_previews_extension_without_changing_log(self):
        self.assertEqual(self.run_generator(), 0)
        original = (self.out / "prompts.json").read_bytes()
        self.args[1] = "4"
        self.args.append("--dry-run")
        self.client.images.generate.reset_mock()
        output = io.StringIO()
        with patch.dict(
            "os.environ", {"OPENAI_API_KEY": ""}
        ), contextlib.redirect_stdout(output):
            self.assertEqual(generate_images.main(self.args), 0)
        self.assertIn("pair_002_A (pending)", output.getvalue())
        self.assertEqual((self.out / "prompts.json").read_bytes(), original)
        self.client.images.generate.assert_not_called()

    def test_existing_randomised_plan_generates_in_pair_order(self):
        rows = generate_images.make_prompts(40, 20261105)
        settings = dict(
            seed=20261105,
            model="gpt-image-2.6-sunburst",
            quality="medium",
            size="1024x1024",
        )
        # Simulate an existing log from before generation order was fixed.
        log = dict(settings=settings, images=[dict(r, status="pending") for r in rows])
        generate_images.save_log(self.out / "prompts.json", log)
        self.args[1] = "40"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(generate_images.main(self.args), 0)
        expected_ids = [
            f"pair_{pair:03d}_{group}" for pair in range(1, 21) for group in "AB"
        ]
        self.assertEqual(
            [
                line.removeprefix("Generating ").removesuffix("...")
                for line in output.getvalue().splitlines()
                if line.startswith("Generating ")
            ],
            expected_ids,
        )
        prompts_by_id = {row["id"]: row["prompt"] for row in rows}
        self.assertEqual(
            [
                call.kwargs["prompt"]
                for call in self.client.images.generate.call_args_list
            ],
            [prompts_by_id[image_id] for image_id in expected_ids],
        )
        self.assertEqual(
            [
                {k: row[k] for k in ("id", "prompt")}
                for row in self.read_log()["images"]
            ],
            rows,
        )

    def test_new_and_extended_plans_generate_a_then_b(self):
        for count, first_pair in ((4, 1), (8, 3)):
            with self.subTest(count=count):
                self.args[1] = str(count)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(generate_images.main(self.args), 0)
                self.assertEqual(
                    [
                        line
                        for line in output.getvalue().splitlines()
                        if line.startswith("Generating ")
                    ],
                    [
                        f"Generating pair_{pair:03d}_{group}..."
                        for pair in range(first_pair, count // 2 + 1)
                        for group in "AB"
                    ],
                )


if __name__ == "__main__":
    unittest.main()
