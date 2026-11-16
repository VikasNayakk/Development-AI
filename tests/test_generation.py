import contextlib
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from core.config import ModelConfig
from core.generate import main as generate_main
from core.model import DevelopmentAIModel
from core.tokenizer import ByteTokenizer


class GenerationValidationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = DevelopmentAIModel(
            ModelConfig(block_size=8, d_model=16, n_heads=2, n_layers=1)
        )
        self.prompt = torch.tensor([[65, 66]], dtype=torch.long)

    def test_greedy_generation_is_independent_of_random_seed(self):
        torch.manual_seed(1)
        first = self.model.generate(self.prompt.clone(), 5, do_sample=False)
        torch.manual_seed(999)
        second = self.model.generate(self.prompt.clone(), 5, do_sample=False)
        self.assertTrue(torch.equal(first, second))

    def test_sampled_generation_is_repeatable_with_same_torch_seed(self):
        torch.manual_seed(1234)
        first = self.model.generate(self.prompt.clone(), 5, temperature=0.8)
        torch.manual_seed(1234)
        second = self.model.generate(self.prompt.clone(), 5, temperature=0.8)
        self.assertTrue(torch.equal(first, second))

    def test_generation_preserves_prefix_and_restores_training_mode(self):
        self.model.train()
        output = self.model.generate(self.prompt.clone(), 3, do_sample=False)
        self.assertTrue(torch.equal(output[:, :2], self.prompt))
        self.assertTrue(self.model.training)

    def test_zero_new_tokens_returns_prompt_unchanged(self):
        output = self.model.generate(self.prompt.clone(), 0, do_sample=False)
        self.assertTrue(torch.equal(output, self.prompt))

    def test_long_prompt_uses_last_context_window_and_preserves_original_prefix(self):
        long_prompt = torch.arange(10, dtype=torch.long).remainder(256).unsqueeze(0)
        output = self.model.generate(long_prompt.clone(), 1, do_sample=False)
        self.assertTrue(torch.equal(output[:, :10], long_prompt))
        self.assertEqual(tuple(output.shape), (1, 11))

    def test_rejects_invalid_prompt_types_shapes_and_dtypes(self):
        with self.assertRaises(TypeError):
            self.model.generate([[65, 66]], 1)
        with self.assertRaises(ValueError):
            self.model.generate(torch.empty((0, 2), dtype=torch.long), 1)
        with self.assertRaises(ValueError):
            self.model.generate(torch.empty((1, 0), dtype=torch.long), 1)
        with self.assertRaises(TypeError):
            self.model.generate(torch.tensor([[65.0]]), 1)

    def test_rejects_out_of_range_token_ids(self):
        for token_id in (-1, 256):
            with self.subTest(token_id=token_id):
                with self.assertRaises(ValueError):
                    self.model.generate(torch.tensor([[token_id]]), 1)

    def test_rejects_invalid_generation_limits_and_temperature(self):
        with self.assertRaises(TypeError):
            self.model.generate(self.prompt.clone(), True)
        with self.assertRaises(ValueError):
            self.model.generate(self.prompt.clone(), -1)
        for temperature in (0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(temperature=temperature):
                with self.assertRaises(ValueError):
                    self.model.generate(self.prompt.clone(), 1, temperature)
        with self.assertRaises(TypeError):
            self.model.generate(self.prompt.clone(), 1, True)

    def test_rejects_non_boolean_sampling_flag(self):
        with self.assertRaises(TypeError):
            self.model.generate(self.prompt.clone(), 1, do_sample=1)

    def test_cli_seed_reproduces_sampled_output_and_greedy_option_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "model.pt"
            torch.save(
                {
                    "format_version": 1,
                    "tokenizer": ByteTokenizer.name,
                    "config": self.model.config.to_dict(),
                    "model_state": self.model.state_dict(),
                },
                checkpoint_path,
            )
            common = [
                "core.generate",
                "--checkpoint", str(checkpoint_path),
                "--prompt", "Hi",
                "--max-new-tokens", "4",
                "--seed", "321",
            ]
            outputs = []
            for _ in range(2):
                stdout = io.StringIO()
                with patch("sys.argv", common), contextlib.redirect_stdout(stdout):
                    generate_main()
                outputs.append(stdout.getvalue())
            self.assertEqual(outputs[0], outputs[1])
            self.assertTrue(outputs[0].startswith("Hi"))

            stdout = io.StringIO()
            with patch("sys.argv", [*common, "--greedy"]), contextlib.redirect_stdout(stdout):
                generate_main()
            self.assertTrue(stdout.getvalue().startswith("Hi"))

    def test_cli_rejects_non_finite_temperature_before_loading_checkpoint(self):
        with patch("sys.argv", [
            "core.generate", "--prompt", "Hello", "--temperature", "nan",
        ]):
            with self.assertRaises(SystemExit) as raised:
                generate_main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
