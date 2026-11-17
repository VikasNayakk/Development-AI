import unittest

import torch

from core.benchmark_inference import run_inference_benchmark
from core.config import ModelConfig


class InferenceBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(
            block_size=8,
            d_model=16,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
        )

    def test_report_contains_repeated_latency_and_throughput_metrics(self):
        report = run_inference_benchmark(
            config=self.config,
            device=torch.device("cpu"),
            warmup_runs=1,
            measured_runs=3,
            batch_size=2,
            prompt_tokens=4,
            generation_tokens=2,
            seed=123,
        )
        inference = report["inference"]
        self.assertEqual(report["benchmark_kind"], "synthetic_greedy_inference_latency")
        self.assertEqual(inference["generated_tokens_total"], 12)
        self.assertEqual(len(inference["per_run_seconds"]), 3)
        self.assertGreater(inference["mean_request_latency_seconds"], 0.0)
        self.assertGreater(inference["median_request_latency_seconds"], 0.0)
        self.assertGreater(inference["p95_request_latency_seconds"], 0.0)
        self.assertGreater(inference["aggregate_generated_tokens_per_second"], 0.0)
        self.assertIsNone(inference["peak_cuda_allocated_bytes"])
        self.assertFalse(inference["kv_cache"])
        self.assertIn("not language understanding", report["quality_warning"])

    def test_same_seed_and_shape_produce_same_model_metadata(self):
        kwargs = dict(
            config=self.config,
            device=torch.device("cpu"),
            warmup_runs=0,
            measured_runs=1,
            batch_size=1,
            prompt_tokens=3,
            generation_tokens=1,
            seed=99,
        )
        first = run_inference_benchmark(**kwargs)
        second = run_inference_benchmark(**kwargs)
        self.assertEqual(first["model"], second["model"])
        self.assertEqual(
            first["inference"]["generated_tokens_total"],
            second["inference"]["generated_tokens_total"],
        )

    def test_rejects_invalid_measurement_arguments(self):
        base = dict(
            config=self.config,
            device=torch.device("cpu"),
            warmup_runs=0,
            measured_runs=2,
            batch_size=1,
            prompt_tokens=4,
            generation_tokens=2,
        )
        invalid = (
            {"warmup_runs": -1},
            {"measured_runs": 0},
            {"batch_size": True},
            {"prompt_tokens": 0},
            {"prompt_tokens": 9},
            {"generation_tokens": 0},
            {"seed": -1},
        )
        for override in invalid:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    run_inference_benchmark(**{**base, **override})


if __name__ == "__main__":
    unittest.main()
