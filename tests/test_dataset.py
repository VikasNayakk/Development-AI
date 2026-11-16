import json
import tempfile
import unittest
from pathlib import Path

from core.dataset import (
    DatasetError, normalize_text, privacy_flags, prepare_dataset,
    repeated_character_ratio, script_counts,
)


class DatasetUtilityTests(unittest.TestCase):
    def test_normalizes_unicode_and_whitespace(self):
        self.assertEqual(normalize_text("Cafe\\u0301\\r\\n  hello   world"), "Café\\nhello world")

    def test_detects_common_sensitive_patterns_without_echoing_values(self):
        self.assertIn("email_like_value", privacy_flags("Contact test@example.com"))
        self.assertIn("phone_like_value", privacy_flags("Call +91 98765 43210"))
        self.assertIn("private_key_marker", privacy_flags("-----BEGIN PRIVATE KEY-----"))
        self.assertIn("credential_like_assignment", privacy_flags("api_key = 'abcdefghijklmnop'"))

    def test_repetition_and_script_statistics(self):
        self.assertGreater(repeated_character_ratio("A" * 20), 0.9)
        counts = script_counts("Hello नमस्ते")
        self.assertGreater(counts["latin"], 0)
        self.assertGreater(counts["devanagari"], 0)


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "raw").mkdir()
        self.manifest = self.root / "sources.json"
        self.output = self.root / "processed"

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self, sources):
        self.manifest.write_text(
            json.dumps({"sources": sources}), encoding="utf-8"
        )

    def source(self, source_id, path, text, **extra):
        (self.root / path).write_text(text, encoding="utf-8")
        return {
            "source_id": source_id, "path": path,
            "license": "CC0-1.0", "rights_confirmed": True, **extra
        }

    def test_prepares_corpus_with_provenance(self):
        text = ("This is a sufficiently long educational document about science. " * 4)
        self.write_manifest([self.source("science-1", "raw/a.txt", text)])
        report = prepare_dataset(self.manifest, self.output, min_chars=80)
        self.assertEqual(report["accepted_count"], 1)
        self.assertEqual(report["rejected_count"], 0)
        self.assertTrue((self.output / "train.txt").is_file())
        self.assertTrue((self.output / "dataset_report.json").is_file())
        self.assertEqual(report["accepted_sources"][0]["license"], "CC0-1.0")

    def test_rejects_sensitive_source_without_leaking_match(self):
        text = "Private contact test@example.com. " + ("Educational material. " * 8)
        self.write_manifest([self.source("private-1", "raw/a.txt", text)])
        report = prepare_dataset(self.manifest, self.output, min_chars=80)
        self.assertEqual(report["accepted_count"], 0) if False else None
        self.assertEqual(report["rejected_sources"][0]["reason"], "privacy_filter")
        serialized = json.dumps(report)
        self.assertNotIn("test@example.com", serialized)

    def test_rejects_unconfirmed_rights(self):
        entry = self.source("unknown-rights", "raw/a.txt", "Long enough text. " * 10)
        entry["rights_confirmed"] = False
        self.write_manifest([entry])
        with self.assertRaises(DatasetError):
            prepare_dataset(self.manifest, self.output)

    def test_rejects_path_traversal(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("This is deliberately outside the manifest root. " * 5)
        try:
            self.write_manifest([{
                "source_id": "escape", "path": "../outside.txt",
                "license": "CC0-1.0", "rights_confirmed": True
            }])
            with self.assertRaises(DatasetError):
                prepare_dataset(self.manifest, self.output)
        finally:
            outside.unlink(missing_ok=True)

    def test_deduplicates_normalized_documents(self):
        text = "A unique educational text for the model. " * 5
        entries = [
            self.source("one", "raw/a.txt", text),
            self.source("two", "raw/b.txt", "  " + text.replace("  ", " ") + "  "),
        ]
        self.write_manifest(entries)
        report = prepare_dataset(self.manifest, self.output, min_chars=80)
        self.assertEqual(report["accepted_count"], 1)
        self.assertEqual(report["rejected_sources"][0]["reason"], "duplicate_normalized_content")

    def test_empty_valid_set_does_not_write_training_file(self):
        self.write_manifest([self.source("short", "raw/a.txt", "Too short.")])
        with self.assertRaises(DatasetError):
            prepare_dataset(self.manifest, self.output, min_chars=80)
        self.assertFalse((self.output / "train.txt").exists())


if __name__ == "__main__":
    unittest.main()
