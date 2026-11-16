import unittest

import torch

from core.benchmark_training import run_benchmark
from core.config import ModelConfig


class TrainingBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            block_size=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
        )

    def test_report_counts_only_measured_supervised_tokens(self):
        report = run_benchmark(
            config=self.config,
            device=torch.device("cpu"),
            warmup_steps=1,
            measured_steps=2,
            batch_size=2,
            gradient_accumulation_steps=3,
            generation_tokens=2,
            seed=123,
        )
        training = report["training"]
        self.assertEqual(training["supervised_tokens_processed"], 2 * 2 * 3 * 8)
        self.assertEqual(training["effective_batch_size"], 6)
        self.assertEqual(training["measured_optimizer_updates"], 2)
        self.assertEqual(len(training["per_update_seconds"]), 2)
        self.assertGreater(training["supervised_tokens_per_second"], 0.0)
        self.assertEqual(report["generation"]["generated_token_count"], 2)
        self.assertEqual(report["generation"]["kv_cache"], False)
        self.assertEqual(report["model"]["unique_parameter_count"],
                         sum(p.numel() for p in run_benchmark_model(self.config).parameters()))

    def test_rejects_invalid_measurement_arguments(self):
        base = dict(
            config=self.config,
            device=torch.device("cpu"),
            warmup_steps=0,
            measured_steps=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            generation_tokens=1,
        )
        for field, value in (
            ("warmup_steps", -1),
            ("measured_steps", 0),
            ("batch_size", True),
            ("gradient_accumulation_steps", 0),
            ("generation_tokens", 0),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    run_benchmark(**{**base, field: value})

    def test_report_marks_synthetic_metrics_as_not_quality(self):
        report = run_benchmark(
            config=self.config,
            device=torch.device("cpu"),
            warmup_steps=0,
            measured_steps=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            generation_tokens=1,
        )
        self.assertIn("Synthetic benchmark only", report["quality_warning"])
        self.assertEqual(report["benchmark_kind"],
                         "synthetic_training_and_greedy_generation")


def run_benchmark_model(config):
    from core.model import DevelopmentAIModel
    return DevelopmentAIModel(config)


if __name__ == "__main__":
    unittest.main()
