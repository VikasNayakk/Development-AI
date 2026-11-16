import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from core.config import ModelConfig
from core.evaluate import _create_untrained_baseline, main as evaluate_main
from core.evaluation import (
    EvaluationManifestError,
    evaluate_manifest,
    load_evaluation_manifest,
    score_documents,
)
from core.model import DevelopmentAIModel


class HeldOutEvaluationManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "evaluation.json"
        self.sources = self.root / "sources"
        self.sources.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, datasets):
        self.manifest_path.write_text(
            json.dumps({"format_version": 1, "datasets": datasets}, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_source(self, filename, text):
        path = self.sources / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
        return str(path.relative_to(self.root))

    def entry(self, dataset_id, path, language="english", split="validation", rights=True):
        return {
            "dataset_id": dataset_id,
            "path": path,
            "language": language,
            "split": split,
            "rights_confirmed": rights,
        }

    def complete_fixture(self):
        samples = {
            ("hindi", "validation"): "यह हिंदी validation का अलग दस्तावेज़ है।\n\nयह दूसरा वाक्य है।",
            ("hindi", "test"): "यह हिंदी test के लिए अलग सामग्री है।\n\nयह एक और वाक्य है।",
            ("english", "validation"): "An independent English validation document.\n\nA second English sentence follows.",
            ("english", "test"): "A separate English test document for byte scoring.\n\nThis is another held-out sentence.",
            ("hinglish", "validation"): "Yeh Hinglish validation document alag rakha gaya hai.\n\nIsme English technical words hain.",
            ("hinglish", "test"): "Hinglish test ke liye yeh doosra unique document hai.\n\nModel is text ko byte level par score karega.",
        }
        entries = []
        for index, ((language, split), text) in enumerate(samples.items()):
            relative = self.add_source(f"{language}_{split}.txt", text)
            entries.append(self.entry(f"{language}-{split}-{index}", relative, language, split))
        return entries, samples

    def test_loads_six_language_split_corpora_without_exposing_text(self):
        entries, samples = self.complete_fixture()
        self.write_manifest(entries)

        datasets, manifest_sha = load_evaluation_manifest(self.manifest_path)

        self.assertEqual(len(datasets), 6)
        self.assertEqual(len(manifest_sha), 64)
        self.assertEqual(
            {(item.language, item.split) for item in datasets},
            set(samples),
        )
        for dataset in datasets:
            self.assertGreater(dataset.tokens_scored, 0)
            self.assertEqual(len(dataset.sha256), 64)
        serialized = json.dumps([
            {
                "dataset_id": item.dataset_id,
                "filename": item.filename,
                "sha256": item.sha256,
            }
            for item in datasets
        ])
        self.assertNotIn("independent English validation document", serialized)
        self.assertNotIn("हिंदी validation", serialized)

    def test_evaluation_returns_separate_weighted_language_and_split_metrics(self):
        entries, samples = self.complete_fixture()
        self.write_manifest(entries)
        torch.manual_seed(11)
        model = DevelopmentAIModel(
            ModelConfig(block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.1)
        )
        model.train()

        result = evaluate_manifest(
            model, self.manifest_path, device=torch.device("cpu")
        )

        self.assertEqual(len(result["datasets"]), 6)
        self.assertEqual(len(result["language_split_metrics"]), 6)
        self.assertEqual(result["missing_language_split_combinations"], [])
        self.assertTrue(model.training)
        for dataset in result["datasets"]:
            self.assertGreater(dataset["tokens_scored"], 0)
            self.assertTrue(torch.isfinite(torch.tensor(dataset["loss"])).item())
            self.assertGreater(dataset["perplexity"], 0)
        for group in result["language_split_metrics"]:
            self.assertGreater(group["tokens_scored"], 0)
            self.assertGreater(group["perplexity"], 0)
        encoded_target_count = sum(
            len(document.encode("utf-8")) - 1
            for text in samples.values()
            for document in text.split("\n\n")
        )
        self.assertEqual(
            sum(item["tokens_scored"] for item in result["datasets"]),
            encoded_target_count,
        )

    def test_untrained_baseline_uses_same_documents_and_token_weighted_groups(self):
        entries, _ = self.complete_fixture()
        self.write_manifest(entries)
        torch.manual_seed(11)
        model = DevelopmentAIModel(
            ModelConfig(block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.1)
        )
        baseline = copy.deepcopy(model)

        result = evaluate_manifest(
            model,
            self.manifest_path,
            device=torch.device("cpu"),
            baseline_model=baseline,
            baseline_seed=77,
        )

        comparison = result["baseline_comparison"]
        self.assertEqual(comparison["baseline_type"], "randomly_initialized_same_architecture")
        self.assertEqual(comparison["baseline_seed"], 77)
        self.assertEqual(len(comparison["datasets"]), 6)
        self.assertEqual(len(comparison["language_split_metrics"]), 6)
        for item in comparison["datasets"]:
            self.assertAlmostEqual(item["loss_improvement"], 0.0, places=7)
            self.assertEqual(item["tokens_scored"], next(
                dataset["tokens_scored"]
                for dataset in result["datasets"]
                if dataset["dataset_id"] == item["dataset_id"]
            ))
        for group in comparison["language_split_metrics"]:
            self.assertAlmostEqual(group["loss_improvement"], 0.0, places=7)

    def test_baseline_rejects_same_instance_mismatched_config_and_invalid_seed(self):
        entries, _ = self.complete_fixture()
        self.write_manifest(entries)
        config = ModelConfig(block_size=8, d_model=16, n_heads=2, n_layers=1)
        model = DevelopmentAIModel(config)
        different_config = DevelopmentAIModel(
            ModelConfig(block_size=9, d_model=16, n_heads=2, n_layers=1)
        )
        with self.assertRaisesRegex(ValueError, "separate model instance"):
            evaluate_manifest(
                model, self.manifest_path, device=torch.device("cpu"),
                baseline_model=model, baseline_seed=1,
            )
        with self.assertRaisesRegex(ValueError, "same configuration"):
            evaluate_manifest(
                model, self.manifest_path, device=torch.device("cpu"),
                baseline_model=different_config, baseline_seed=1,
            )
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            evaluate_manifest(
                model, self.manifest_path, device=torch.device("cpu"),
                baseline_model=copy.deepcopy(model), baseline_seed=-1,
            )

    def test_evaluation_is_deterministic_for_same_model_and_corpus(self):
        entries, _ = self.complete_fixture()
        self.write_manifest(entries)
        model = DevelopmentAIModel(
            ModelConfig(block_size=8, d_model=16, n_heads=2, n_layers=1, dropout=0.2)
        )
        first = evaluate_manifest(model, self.manifest_path, device=torch.device("cpu"))
        second = evaluate_manifest(model, self.manifest_path, device=torch.device("cpu"))
        self.assertEqual(first, second)

    def test_report_identifies_missing_language_split_combinations(self):
        relative = self.add_source("only_en_val.txt", "Only one English validation corpus.")
        self.write_manifest([self.entry("en-val", relative)])
        model = DevelopmentAIModel(
            ModelConfig(block_size=4, d_model=16, n_heads=2, n_layers=1)
        )

        result = evaluate_manifest(model, self.manifest_path, device=torch.device("cpu"))

        self.assertEqual(len(result["language_split_metrics"]), 1)
        self.assertEqual(len(result["missing_language_split_combinations"]), 5)

    def test_rejects_training_split_and_unconfirmed_rights(self):
        relative = self.add_source("data.txt", "An evaluation document that is not for training.")
        for entry in (
            self.entry("training", relative, split="training"),
            self.entry("unknown-rights", relative, rights=False),
        ):
            with self.subTest(entry=entry):
                self.write_manifest([entry])
                with self.assertRaises(EvaluationManifestError):
                    load_evaluation_manifest(self.manifest_path)

    def test_rejects_duplicate_ids_and_duplicate_documents_across_splits(self):
        relative_a = self.add_source("one.txt", "This sentence is duplicated in two evaluation sets.")
        relative_b = self.add_source("two.txt", "This sentence is duplicated in two evaluation sets.")
        self.write_manifest([
            self.entry("same-id", relative_a),
            self.entry("same-id", relative_b, split="test"),
        ])
        with self.assertRaisesRegex(EvaluationManifestError, "Duplicate evaluation dataset_id"):
            load_evaluation_manifest(self.manifest_path)

        self.write_manifest([
            self.entry("validation-doc", relative_a),
            self.entry("test-doc", relative_b, split="test"),
        ])
        with self.assertRaisesRegex(EvaluationManifestError, "Exact duplicate document"):
            load_evaluation_manifest(self.manifest_path)

    def test_rejects_path_traversal_absolute_paths_and_wrong_extensions(self):
        outside = self.root.parent / (self.root.name + "-outside.txt")
        outside.write_text("Outside evaluation content.", encoding="utf-8")
        try:
            for relative in ("../" + outside.name, str(outside), "C:\\temp\\private.txt"):
                with self.subTest(path=relative):
                    self.write_manifest([self.entry("escape", relative)])
                    with self.assertRaises(EvaluationManifestError):
                        load_evaluation_manifest(self.manifest_path)
        finally:
            outside.unlink(missing_ok=True)

        self.write_manifest([self.entry("wrong-type", "sources/data.csv")])
        with self.assertRaises(EvaluationManifestError):
            load_evaluation_manifest(self.manifest_path)

    def test_rejects_invalid_utf8_too_short_documents_and_privacy_matches(self):
        invalid = self.sources / "invalid.txt"
        invalid.write_bytes(b"\xff\xfe")
        self.write_manifest([self.entry("invalid", str(invalid.relative_to(self.root)))])
        with self.assertRaisesRegex(EvaluationManifestError, "valid UTF-8"):
            load_evaluation_manifest(self.manifest_path)

        short = self.add_source("short.txt", "x")
        self.write_manifest([self.entry("short", short)])
        with self.assertRaisesRegex(EvaluationManifestError, "at least two UTF-8 bytes"):
            load_evaluation_manifest(self.manifest_path)

        private = self.add_source(
            "private.txt",
            "Contact private-test@example.com for this evaluation document.",
        )
        self.write_manifest([self.entry("private", private)])
        with self.assertRaisesRegex(EvaluationManifestError, "privacy checks"):
            load_evaluation_manifest(self.manifest_path)

    def test_rejects_exact_training_corpus_even_when_newline_style_differs(self):
        training_text = "Training corpus first line.\nTraining corpus second line.\n"
        evaluation_file = self.sources / "same.txt"
        evaluation_file.write_bytes(training_text.replace("\n", "\r\n").encode("utf-8"))
        training_hash = hashlib.sha256(training_text.encode("utf-8")).hexdigest()
        self.write_manifest([self.entry("same-as-training", str(evaluation_file.relative_to(self.root)))])
        with self.assertRaisesRegex(EvaluationManifestError, "matches the training corpus"):
            load_evaluation_manifest(
                self.manifest_path,
                training_corpus_sha256=training_hash,
            )

    def test_rejects_malformed_language_types_without_leaking_type_errors(self):
        relative = self.add_source("language.txt", "A valid two-token evaluation source.")
        for language in ([], {}, 5):
            with self.subTest(language=language):
                self.write_manifest([self.entry("invalid-language", relative, language=language)])
                with self.assertRaises(EvaluationManifestError):
                    load_evaluation_manifest(self.manifest_path)

    def test_rejects_invalid_checkpoint_training_hash(self):
        relative = self.add_source("hash.txt", "A valid two-token evaluation source.")
        self.write_manifest([self.entry("invalid-hash", relative)])
        with self.assertRaisesRegex(EvaluationManifestError, "invalid training corpus SHA-256"):
            load_evaluation_manifest(
                self.manifest_path,
                training_corpus_sha256="not-a-sha256",
            )



class EvaluationCliTests(unittest.TestCase):
    def test_seeded_untrained_baseline_is_reproducible_and_preserves_rng(self):
        config = ModelConfig(block_size=8, d_model=16, n_heads=2, n_layers=1)
        torch.manual_seed(999)
        rng_before = torch.get_rng_state().clone()

        first = _create_untrained_baseline(config, 42, torch.device("cpu"))
        rng_after = torch.get_rng_state().clone()
        second = _create_untrained_baseline(config, 42, torch.device("cpu"))

        self.assertTrue(torch.equal(rng_before, rng_after))
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.state_dict()[name]), name)

    def test_manifest_cli_writes_machine_readable_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "model.pt"
            manifest_path = root / "evaluation.json"
            source_path = root / "english_validation.txt"
            model = DevelopmentAIModel(
                ModelConfig(block_size=8, d_model=16, n_heads=2, n_layers=1)
            )
            torch.save({
                "format_version": 1,
                "model_name": "Development-AI",
                "tokenizer": "utf8-byte-v1",
                "config": model.config.to_dict(),
                "model_state": model.state_dict(),
                "training": {
                    "corpus_sha256": hashlib.sha256(b"separate training data").hexdigest()
                },
            }, checkpoint_path)
            source_path.write_text(
                "This independent English evaluation sentence is private local input.",
                encoding="utf-8",
            )
            manifest_path.write_text(json.dumps({
                "format_version": 1,
                "datasets": [{
                    "dataset_id": "english-validation",
                    "path": source_path.name,
                    "language": "english",
                    "split": "validation",
                    "rights_confirmed": True,
                }],
            }), encoding="utf-8")

            stdout = io.StringIO()
            argv = [
                "core.evaluate",
                "--checkpoint", str(checkpoint_path),
                "--manifest", str(manifest_path),
                "--device", "cpu",
                "--compare-untrained",
                "--baseline-seed", "7",
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(stdout):
                evaluate_main()

            result = json.loads(stdout.getvalue())
            evaluation = result["evaluation"]
            self.assertEqual(result["model"], "Development-AI")
            self.assertTrue(evaluation["training_corpus_hash_available"])
            self.assertEqual(len(evaluation["datasets"]), 1)
            self.assertEqual(evaluation["datasets"][0]["language"], "english")
            self.assertEqual(
                evaluation["baseline_comparison"]["baseline_seed"], 7
            )
            self.assertEqual(len(evaluation["baseline_comparison"]["datasets"]), 1)
            self.assertEqual(
                len(evaluation["baseline_comparison"]["language_split_metrics"]), 1
            )

            default_stdout = io.StringIO()
            default_argv = [
                "core.evaluate",
                "--checkpoint", str(checkpoint_path),
                "--manifest", str(manifest_path),
                "--device", "cpu",
            ]
            with patch("sys.argv", default_argv), contextlib.redirect_stdout(default_stdout):
                evaluate_main()
            default_result = json.loads(default_stdout.getvalue())
            self.assertNotIn("baseline_comparison", default_result["evaluation"])
            self.assertNotIn("private local input", stdout.getvalue())
            self.assertNotIn("private local input", default_stdout.getvalue())

    def test_cli_requires_manifest_for_untrained_comparison(self):
        with patch("sys.argv", [
            "core.evaluate",
            "--compare-untrained",
        ]):
            with self.assertRaises(SystemExit) as raised:
                evaluate_main()
        self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_manifest_and_legacy_data_together(self):
        with patch("sys.argv", [
            "core.evaluate",
            "--manifest", "unused.json",
            "--data", "unused.txt",
        ]):
            with self.assertRaises(SystemExit) as raised:
                evaluate_main()
        self.assertEqual(raised.exception.code, 2)


class DeterministicScoringTests(unittest.TestCase):
    def test_scores_every_next_byte_once_and_restores_training_mode(self):
        model = DevelopmentAIModel(
            ModelConfig(block_size=4, d_model=16, n_heads=2, n_layers=1)
        )
        model.train()
        documents = [
            tuple("abcdefghij".encode("utf-8")),
            tuple("नमस्ते".encode("utf-8")),
        ]

        metrics = score_documents(model, documents, device=torch.device("cpu"))

        self.assertEqual(
            metrics["tokens_scored"],
            sum(len(document) - 1 for document in documents),
        )
        self.assertEqual(metrics["document_count"], 2)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
