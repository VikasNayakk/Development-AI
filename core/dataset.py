"""Conservative local text dataset preparation and provenance reporting."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

EMAIL_RE = re.compile(r"(?i)(?<![\\w.+-])[\\w.+-]+@[\\w-]+(?:\\.[\\w-]+)+")
PHONE_RE = re.compile(r"(?<!\\w)(?:\\+?\\d[\\d .()/-]{8,}\\d)(?!\\w)")
PRIVATE_KEY_RE = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
SECRET_RE = re.compile(
    r"(?i)\\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\\b"
    r"\\s*[:=]\\s*['\\"]?[A-Za-z0-9_./+\\-=]{12,}"
)


class DatasetError(ValueError):
    """Raised when a dataset manifest or source violates the input contract."""


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\\r\\n", "\\n").replace("\\r", "\\n"))
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if char in "\\n\\t" or category[0] != "C":
            cleaned.append(char)
    lines = [" ".join(line.split()) for line in "".join(cleaned).split("\\n")]
    normalized = "\\n".join(lines)
    normalized = re.sub(r"\\n{3,}", "\\n\\n", normalized)
    return normalized.strip()


def privacy_flags(text: str) -> list[str]:
    flags = []
    if EMAIL_RE.search(text):
        flags.append("email_like_value")
    if PHONE_RE.search(text):
        flags.append("phone_like_value")
    if PRIVATE_KEY_RE.search(text):
        flags.append("private_key_marker")
    if SECRET_RE.search(text):
        flags.append("credential_like_assignment")
    return flags


def repeated_character_ratio(text: str) -> float:
    if not text:
        return 1.0
    repeated = sum(len(match.group(0)) for match in re.finditer(r"(.)\\1{7,}", text))
    return repeated / len(text)


def script_counts(text: str) -> dict[str, int]:
    counts = Counter()
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            counts["devanagari"] += 1
        elif ("LATIN" in unicodedata.name(char, "")):
            counts["latin"] += 1
        elif char.isalpha():
            counts["other"] += 1
    return dict(sorted(counts.items()))


def _safe_source_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise DatasetError(f"Source path escapes manifest directory: {relative_path}") from error
    if candidate.suffix.lower() not in {".txt", ".md"}:
        raise DatasetError(f"Only .txt and .md sources are supported: {relative_path}")
    if not candidate.is_file():
        raise DatasetError(f"Source file does not exist: {relative_path}")
    return candidate


def prepare_dataset(manifest_path: Path, output_dir: Path, *, min_chars: int = 80,
                    max_repeated_ratio: float = 0.30) -> dict:
    if min_chars < 1:
        raise DatasetError("min_chars must be positive.")
    if not 0.0 <= max_repeated_ratio < 1.0:
        raise DatasetError("max_repeated_ratio must be in [0, 1).")
    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetError(f"Cannot read valid UTF-8 JSON manifest: {error}") from error
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    if not isinstance(sources, list) or not sources:
        raise DatasetError("Manifest must contain a non-empty 'sources' array.")

    seen_ids, seen_hashes = set(), set()
    accepted, rejected, output_parts = [], [], []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise DatasetError(f"Source entry {index} must be an object.")
        source_id = source.get("source_id")
        relative_path = source.get("path")
        license_name = source.get("license")
        if not all(isinstance(value, str) and value.strip()
                   for value in (source_id, relative_path, license_name)):
            raise DatasetError(f"Source entry {index} requires source_id, path and license.")
        if source_id in seen_ids:
            raise DatasetError(f"Duplicate source_id: {source_id}")
        seen_ids.add(source_id)
        if source.get("rights_confirmed") is not True:
            raise DatasetError(f"Rights must be explicitly confirmed for source: {source_id}")

        path = _safe_source_path(root, relative_path)
        raw = path.read_bytes()
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            rejected.append({"source_id": source_id, "reason": "invalid_utf8"})
            continue
        text = normalize_text(decoded)
        flags = privacy_flags(text)
        if flags:
            rejected.append({"source_id": source_id, "reason": "privacy_filter", "flags": flags})
            continue
        if len(text) < min_chars:
            rejected.append({"source_id": source_id, "reason": "below_minimum_characters"})
            continue
        repeat_ratio = repeated_character_ratio(text)
        if repeat_ratio > max_repeated_ratio:
            rejected.append({"source_id": source_id, "reason": "excessive_character_repetition"})
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            rejected.append({"source_id": source_id, "reason": "duplicate_normalized_content"})
            continue
        seen_hashes.add(digest)
        output_parts.append(text)
        accepted.append({
            "source_id": source_id,
            "relative_path": relative_path,
            "license": license_name,
            "sha256_raw": hashlib.sha256(raw).hexdigest(),
            "sha256_normalized": digest,
            "raw_bytes": len(raw),
            "normalized_characters": len(text),
            "word_count_whitespace_approx": len(text.split()),
            "script_character_counts": script_counts(text),
        })

    if not accepted:
        raise DatasetError("No documents passed validation; no training corpus was written.")

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = "\\n\\n".join(output_parts) + "\\n"
    report = {
        "format_version": 1,
        "manifest_file": manifest_path.name,
        "normalization": "Unicode NFC; line-ending and whitespace normalization",
        "privacy_policy": "Reject entire source document when conservative email, phone, private-key or credential patterns match; this is not a complete PII detector.",
        "deduplication": "Exact SHA-256 of normalized full document; near-duplicate detection is not implemented.",
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_sources": accepted,
        "rejected_sources": rejected,
        "corpus_sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        "corpus_characters": len(corpus),
    }
    (output_dir / "train.txt").write_text(corpus, encoding="utf-8", newline="\\n")
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8"
    )
    return report
