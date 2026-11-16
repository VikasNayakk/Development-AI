import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from core.train import _training_settings_match, main as train_main


class TrainingResumeTests(unittest.TestCase):
    def test_training_cli_rejects_zero_gradient_accumulation(self):
        with patch("sys.argv", [
            "core.train",
            "--gradient-accumulation-steps", "0",
        ]):
            with self.assertRaises(SystemExit) as raised:
                train_main()
        self.assertEqual(raised.exception.code, 2)

    def test_legacy_resume_settings_are_compatible_only_with_accumulation_one(self):
        legacy = {
            "seed": 42,
            "batch_size": 8,
            "learning_rate": 0.001,
            "block_size": 128,
            "validation_fraction": 0.1,
            "eval_interval": 100,
            "eval_iters": 20,
            "device_type": "cpu",
            "cuda_device_count": 0,
        }
        expected = {
            **legacy,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 8,
        }
        self.assertTrue(_training_settings_match(legacy, expected))

        incompatible = {
            **expected,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 16,
        }
        self.assertFalse(_training_settings_match(legacy, incompatible))

    def _run_training(self, arguments):
        stdout = io.StringIO()
        with patch("sys.argv", ["core.train", *arguments]), contextlib.redirect_stdout(stdout):
            train_main()
        return stdout.getvalue()

    def _common_arguments(self, data_path):
        return [
            "--data", str(data_path),
            "--batch-size", "2",
            "--learning-rate", "0.001",
            "--block-size", "8",
            "--eval-interval", "2",
            "--eval-iters", "1",
            "--validation-fraction", "0.2",
            "--resume-interval", "2",
            "--gradient-accumulation-steps", "2",
            "--seed", "123",
        ]

    def test_interrupted_resume_matches_uninterrupted_cpu_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.txt"
            data_path.write_text(
                ("Development-AI learns from reviewed examples. नमस्ते दुनिया! "
                 "This is a deterministic training fixture.\\n") * 80,
                encoding="utf-8",
            )
            common = self._common_arguments(data_path)

            split_best = root / "split.pt"
            split_resume = root / "split.resume.pt"
            self._run_training([
                *common, "--steps", "2", "--output", str(split_best),
            ])
            self.assertTrue(split_best.is_file())
            self.assertTrue(split_resume.is_file())

            resumed_output = self._run_training([
                *common,
                "--steps", "4",
                "--output", str(split_best),
                "--resume-from", str(split_resume),
            ])
            self.assertIn("Resuming from step 2", resumed_output)

            full_best = root / "full.pt"
            full_resume = root / "full.resume.pt"
            self._run_training([
                *common, "--steps", "4", "--output", str(full_best),
            ])

            resumed = torch.load(split_resume, map_location="cpu", weights_only=True)
            uninterrupted = torch.load(full_resume, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["training"]["steps_completed"], 4)
            self.assertEqual(
                resumed["training"]["settings"]["gradient_accumulation_steps"], 2
            )
            self.assertEqual(resumed["training"]["settings"]["effective_batch_size"], 4)
            self.assertEqual(resumed["history"][-1]["effective_batch_size"], 4)
            self.assertEqual(
                [entry["step"] for entry in resumed["history"]],
                [1, 2, 4],
            )
            self.assertEqual(resumed["history"], uninterrupted["history"])
            self.assertEqual(resumed["training"], uninterrupted["training"])

            for name, tensor in uninterrupted["model_state"].items():
                self.assertTrue(
                    torch.equal(resumed["model_state"][name], tensor),
                    f"Model parameter differs after resume: {name}",
                )

            metrics = json.loads(split_best.with_suffix(".metrics.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["step"] for entry in metrics], [1, 2, 4])

            # The public inference checkpoint remains the existing version-1 format.
            best = torch.load(split_best, map_location="cpu", weights_only=True)
            self.assertEqual(best["format_version"], 1)
            self.assertEqual(best["tokenizer"], "utf8-byte-v1")

    def test_resume_refuses_changed_training_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.txt"
            data_path.write_text(("A small training corpus for resume validation.\\n") * 80,
                                 encoding="utf-8")
            best_path = root / "model.pt"
            resume_path = root / "model.resume.pt"
            common = self._common_arguments(data_path)
            common[common.index("--eval-interval") + 1] = "1"
            common[common.index("--resume-interval") + 1] = "1"

            self._run_training([
                *common, "--steps", "1", "--output", str(best_path),
            ])
            original_resume = resume_path.read_bytes()
            data_path.write_text(("A changed training corpus must not be accepted.\\n") * 80,
                                 encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                self._run_training([
                    *common,
                    "--steps", "2",
                    "--output", str(best_path),
                    "--resume-from", str(resume_path),
                ])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(resume_path.read_bytes(), original_resume)

    def test_resume_requires_a_larger_total_step_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "train.txt"
            data_path.write_text(("Step target validation corpus.\\n") * 80, encoding="utf-8")
            best_path = root / "model.pt"
            resume_path = root / "model.resume.pt"
            common = self._common_arguments(data_path)
            common[common.index("--eval-interval") + 1] = "1"
            common[common.index("--resume-interval") + 1] = "1"

            self._run_training([
                *common, "--steps", "1", "--output", str(best_path),
            ])
            with self.assertRaises(SystemExit) as raised:
                self._run_training([
                    *common,
                    "--steps", "1",
                    "--output", str(best_path),
                    "--resume-from", str(resume_path),
                ])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
