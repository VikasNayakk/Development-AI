import json
import contextlib
import io
from collections import Counter
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from core.bpe_tokenizer import ByteLevelBPETokenizer
from core.benchmark_tokenizer import main as benchmark_main


def reference_train(documents, max_vocab_size, min_frequency):
    sequences = [list(text.encode("utf-8")) for text in documents]
    merges = []
    while 256 + len(merges) < max_vocab_size:
        counts = Counter(
            (sequence[index], sequence[index + 1])
            for sequence in sequences
            for index in range(len(sequence) - 1)
        )
        if not counts:
            break
        pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
        if frequency < min_frequency:
            break
        updated = []
        for sequence in sequences:
            result = []
            index = 0
            while index < len(sequence):
                if (
                    index + 1 < len(sequence)
                    and (sequence[index], sequence[index + 1]) == pair
                ):
                    result.append(256 + len(merges))
                    index += 2
                else:
                    result.append(sequence[index])
                    index += 1
            updated.append(result)
        sequences = updated
        merges.append(pair)
    return tuple(merges)


def reference_encode(text, merges):
    tokens = list(text.encode("utf-8"))
    for rank, pair in enumerate(merges):
        result = []
        index = 0
        while index < len(tokens):
            if (
                index + 1 < len(tokens)
                and (tokens[index], tokens[index + 1]) == pair
            ):
                result.append(256 + rank)
                index += 2
            else:
                result.append(tokens[index])
                index += 1
        tokens = result
    return tokens


class ByteLevelBPETokenizerTests(unittest.TestCase):
    def test_learns_merges_and_reduces_repeated_text_tokens(self):
        text = "hello world " * 30
        tokenizer = ByteLevelBPETokenizer.train(text, max_vocab_size=280)

        self.assertGreater(tokenizer.vocab_size, 256)
        self.assertLess(len(tokenizer.encode(text)), len(text.encode("utf-8")))
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_round_trip_mixed_hindi_english_emoji_and_symbols(self):
        text = "नमस्ते, Development-AI! 🚀 café 中文 @#$ 123"
        tokenizer = ByteLevelBPETokenizer.train(text * 3, max_vocab_size=300)

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_trainer_matches_simple_reference_implementation(self):
        documents = ["abababab xyz " * 3, "xyz xyz नमस्ते " * 2, "aaaaab"]
        expected = reference_train(documents, 290, 2)
        actual = ByteLevelBPETokenizer.train(
            documents, max_vocab_size=290, min_frequency=2
        )
        self.assertEqual(actual.merges, expected)

    def test_encoder_matches_sequential_reference_merges(self):
        documents = ["abracadabra " * 4, "नमस्ते नमस्ते " * 3]
        tokenizer = ByteLevelBPETokenizer.train(documents, max_vocab_size=290)
        samples = ["abracadabra", "नमस्ते 🚀", "aaaaab", "unknown symbols!?"]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertEqual(
                    tokenizer.encode(sample),
                    reference_encode(sample, tokenizer.merges),
                )

    def test_training_is_deterministic(self):
        documents = ["alpha beta " * 8, "नमस्ते दुनिया " * 5]
        first = ByteLevelBPETokenizer.train(documents, max_vocab_size=280)
        second = ByteLevelBPETokenizer.train(documents, max_vocab_size=280)

        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_does_not_merge_across_document_boundaries(self):
        tokenizer = ByteLevelBPETokenizer.train(
            ["a", "b"], max_vocab_size=257, min_frequency=1
        )
        self.assertEqual(tokenizer.vocab_size, 256)

    def test_equal_frequency_tie_break_is_lexicographic(self):
        tokenizer = ByteLevelBPETokenizer.train(
            ["ab", "cd"], max_vocab_size=257, min_frequency=1
        )
        self.assertEqual(tokenizer.merges[0], (ord("a"), ord("b")))

    def test_overlapping_pairs_merge_left_to_right_without_overlap(self):
        tokenizer = ByteLevelBPETokenizer.train(
            "aaaa", max_vocab_size=257, min_frequency=2
        )
        self.assertEqual(tokenizer.encode("aaaa"), [256, 256])
        self.assertEqual(tokenizer.decode([256, 256]), "aaaa")

    def test_min_frequency_is_respected(self):
        tokenizer = ByteLevelBPETokenizer.train(
            "abc", max_vocab_size=300, min_frequency=2
        )
        self.assertEqual(tokenizer.vocab_size, 256)

    def test_empty_text_and_empty_documents(self):
        tokenizer = ByteLevelBPETokenizer.train(["", ""], max_vocab_size=300)
        self.assertEqual(tokenizer.encode(""), [])
        self.assertEqual(tokenizer.vocab_size, 256)

    def test_rejects_invalid_training_arguments(self):
        with self.assertRaises(ValueError):
            ByteLevelBPETokenizer.train("abc", max_vocab_size=255)
        with self.assertRaises(ValueError):
            ByteLevelBPETokenizer.train("abc", min_frequency=0)
        with self.assertRaises(TypeError):
            ByteLevelBPETokenizer.train(["valid", 123])

    def test_decode_rejects_invalid_token_ids(self):
        tokenizer = ByteLevelBPETokenizer()
        with self.assertRaises(TypeError):
            tokenizer.decode([True])
        with self.assertRaises(TypeError):
            tokenizer.decode([1.5])
        with self.assertRaises(ValueError):
            tokenizer.decode([256])

    def test_encode_rejects_non_string(self):
        tokenizer = ByteLevelBPETokenizer()
        with self.assertRaises(TypeError):
            tokenizer.encode(b"bytes")

    def test_artifact_round_trip_preserves_encoding(self):
        text = "BPE artifact integrity नमस्ते " * 4
        tokenizer = ByteLevelBPETokenizer.train(text, max_vocab_size=280)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            loaded = ByteLevelBPETokenizer.load(path)

        self.assertEqual(loaded.to_dict(), tokenizer.to_dict())
        self.assertEqual(loaded.encode(text), tokenizer.encode(text))
        self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_rejects_tampered_artifact_hash(self):
        tokenizer = ByteLevelBPETokenizer.train("repeat repeat " * 5, max_vocab_size=270)
        payload = tokenizer.to_dict()
        payload["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            ByteLevelBPETokenizer.from_dict(payload)

    def test_rejects_non_hex_artifact_digest(self):
        payload = ByteLevelBPETokenizer().to_dict()
        payload["sha256"] = "é" * 64
        with self.assertRaisesRegex(ValueError, "metadata"):
            ByteLevelBPETokenizer.from_dict(payload)

    def test_rejects_artifact_with_invalid_merge_reference(self):
        payload = {
            "format_version": 1,
            "tokenizer": "utf8-byte-bpe-v1",
            "base_vocab_size": 256,
            "vocab_size": 257,
            "merges": [[256, 65]],
            "sha256": "0" * 64,
        }
        with self.assertRaises(ValueError):
            ByteLevelBPETokenizer.from_dict(payload)

    def test_artifact_schema_rejects_unknown_fields(self):
        payload = ByteLevelBPETokenizer().to_dict()
        payload["unexpected"] = "field"
        with self.assertRaises(ValueError):
            ByteLevelBPETokenizer.from_dict(payload)

    def test_artifact_json_is_valid_and_deterministic(self):
        tokenizer = ByteLevelBPETokenizer.train("deterministic data " * 5, max_vocab_size=270)
        encoded = json.dumps(tokenizer.to_dict(), sort_keys=True)
        self.assertEqual(encoded, json.dumps(tokenizer.to_dict(), sort_keys=True))

    def test_benchmark_cli_writes_artifact_and_hashed_report(self):
        text = "benchmark data नमस्ते " * 5
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.txt"
            output = root / "results"
            corpus.write_text(text, encoding="utf-8")
            arguments = [
                "benchmark_tokenizer",
                "--data", str(corpus),
                "--output-dir", str(output),
                "--vocab-sizes", "270",
            ]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                benchmark_main()

            report = json.loads((output / "benchmark-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["corpus_utf8_bytes"], len(text.encode("utf-8")))
            self.assertEqual(len(report["corpus_sha256"]), 64)
            self.assertTrue((output / "byte-bpe-target-270.json").is_file())
            self.assertGreater(report["results"][0]["actual_vocab_size"], 256)
            self.assertLessEqual(report["results"][0]["actual_vocab_size"], 270)

    def test_multilingual_benchmark_compares_each_language_and_combined(self):
        samples = {
            "hindi": "मुझे अपना AI model बनाना है।\n\nअगला काम quality के साथ करो।",
            "english": "Build an independent local AI model.\n\nRun tests before merging code.",
            "hinglish": "Mujhe apna AI model banana hai.\n\nNext kaam quality ke sath karo.",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for language, sample in samples.items():
                path = root / f"{language}.txt"
                path.write_text(sample, encoding="utf-8")
                paths[language] = path

            output = root / "results"
            arguments = [
                "benchmark_tokenizer",
                "--data-hindi", str(paths["hindi"]),
                "--data-english", str(paths["english"]),
                "--data-hinglish", str(paths["hinglish"]),
                "--output-dir", str(output),
                "--vocab-sizes", "270",
                "--max-training-bytes", "5000",
            ]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                benchmark_main()

            report = json.loads(
                (output / "benchmark-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(report["mode"], "multilingual")
            self.assertEqual(
                set(report["corpora"]), {"hindi", "english", "hinglish"}
            )
            self.assertEqual(len(report["models"]), 4)
            self.assertEqual(
                {model["trained_on"] for model in report["models"]},
                {"hindi", "english", "hinglish", "combined"},
            )
            for model in report["models"]:
                self.assertEqual(
                    set(model["evaluations"]),
                    {"hindi", "english", "hinglish", "combined"},
                )
                self.assertTrue(
                    all(result["round_trip_exact"] for result in model["evaluations"].values())
                )
                self.assertTrue((output / Path(model["artifact"]).name).is_file())

    def test_multilingual_benchmark_requires_all_three_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hindi = root / "hindi.txt"
            hindi.write_text("नमस्ते", encoding="utf-8")
            arguments = [
                "benchmark_tokenizer",
                "--data-hindi", str(hindi),
                "--vocab-sizes", "256",
            ]
            with patch("sys.argv", arguments), self.assertRaises(SystemExit):
                benchmark_main()

    def test_benchmark_cli_rejects_oversized_corpus_before_full_read(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "large.txt"
            corpus.write_bytes(b"x" * 100)
            arguments = [
                "benchmark_tokenizer",
                "--data", str(corpus),
                "--output-dir", str(root / "results"),
                "--vocab-sizes", "256",
                "--max-training-bytes", "10",
            ]
            with patch("sys.argv", arguments), self.assertRaises(SystemExit):
                benchmark_main()


if __name__ == "__main__":
    unittest.main()
