import json
import tempfile
import unittest
from pathlib import Path

from core.bpe_tokenizer import ByteLevelBPETokenizer


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
        payload["vocab_size"] += 1
        with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
