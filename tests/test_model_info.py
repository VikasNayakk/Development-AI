import contextlib
import io
import json
import unittest
from unittest.mock import patch

from core.config import ModelConfig
from core.model_info import build_model_report, main as model_info_main


class ModelInfoTests(unittest.TestCase):
    def test_default_parameter_count_is_exact_and_tied_weight_is_counted_once(self):
        report = build_model_report()

        self.assertEqual(report["parameters"]["unique_total"], 842496)
        self.assertEqual(report["parameters"]["trainable"], 842496)
        self.assertEqual(
            sum(report["parameters"]["groups"].values()),
            report["parameters"]["unique_total"],
        )
        self.assertTrue(report["weight_tying"]["enabled"])
        self.assertTrue(report["weight_tying"]["counted_once"])
        self.assertEqual(
            report["parameters"]["storage_estimate_bytes"]["fp32"],
            842496 * 4,
        )

    def test_custom_configuration_is_reflected_in_report(self):
        config = ModelConfig(
            block_size=16,
            d_model=32,
            n_heads=4,
            n_layers=2,
            dropout=0.0,
        )
        report = build_model_report(config)

        self.assertEqual(report["config"], config.to_dict())
        self.assertGreater(report["parameters"]["unique_total"], 0)
        self.assertEqual(
            sum(report["parameters"]["groups"].values()),
            report["parameters"]["unique_total"],
        )

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model_report(ModelConfig(d_model=32, n_heads=3))

    def test_json_cli_emits_machine_readable_report(self):
        stdout = io.StringIO()
        arguments = [
            "core.model_info",
            "--block-size", "16",
            "--d-model", "32",
            "--n-heads", "4",
            "--n-layers", "2",
            "--dropout", "0",
            "--json",
        ]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(stdout):
            model_info_main()

        report = json.loads(stdout.getvalue())
        self.assertEqual(report["model_name"], "Development-AI")
        self.assertEqual(report["training_status"], "untrained_initialization")
        self.assertEqual(report["config"]["d_model"], 32)


if __name__ == "__main__":
    unittest.main()
