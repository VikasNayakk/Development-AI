"""Rights-aware, deterministic evaluation on separate held-out language corpora."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import torch
import torch.nn.functional as F

from .dataset import normalize_text, privacy_flags
from .tokenizer import ByteTokenizer


SUPPORTED_LANGUAGES = {"hindi", "english", "hinglish"}
SUPPORTED_SPLITS = {"validation", "test"}
MANIFEST_FORMAT_VERSION = 1
_REQUIRED_MANIFEST_KEYS = {"format_version", "datasets"}
_REQUIRED_DATASET_KEYS = {
    "dataset_id", "path", "language", "split", "rights_confirmed"
}


class EvaluationManifestError(ValueError):
    """Raised when an evaluation manifest or held-out source violates its contract."""


@dataclass(frozen=True)
class EvaluationDataset:
    dataset_id: str
    filename: str
    language: str
    split: str
    sha256: str
    file_bytes: int
    documents: tuple[tuple[int, ...], ...]

    @property
    def tokens_scored(self) -> int:
        return sum(len(document) - 1 for document in self.documents)


def _safe_evaluation_path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise EvaluationManifestError("Each dataset requires a non-empty relative path.")
    supplied = Path(relative_path)
    windows_path = PureWindowsPath(relative_path)
    if supplied.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise EvaluationManifestError("Evaluation source paths must be relative to the manifest.")
    candidate = (root / supplied).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise EvaluationManifestError(
            f"Evaluation source path escapes manifest directory: {relative_path}"
        ) from error
    if candidate.suffix.lower() not in {".txt", ".md"}:
        raise EvaluationManifestError("Evaluation sources must use .txt or .md files.")
    if not candidate.is_file():
        raise EvaluationManifestError(f"Evaluation source file does not exist: {relative_path}")
    return candidate


def _split_documents(text: str) -> list[str]:
    normalized_lines = text.replace("\r\n", "\n").replace("\r", "\n")
    return [
        document.strip()
        for document in re.split(r"\n[ \t]*\n+", normalized_lines)
        if document.strip()
    ]


def load_evaluation_manifest(
    manifest_path: Path,
    *,
    training_corpus_sha256: str | None = None,
) -> tuple[list[EvaluationDataset], str]:
    """Load private local validation/test corpora without exposing their text in reports."""
    manifest_path = Path(manifest_path).resolve()
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8", errors="strict"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationManifestError(
            f"Cannot read a valid UTF-8 evaluation manifest: {error}"
        ) from error

    if not isinstance(manifest, dict) or set(manifest) != _REQUIRED_MANIFEST_KEYS:
        raise EvaluationManifestError("Evaluation manifest has an invalid top-level schema.")
    if type(manifest.get("format_version")) is not int or manifest["format_version"] != MANIFEST_FORMAT_VERSION:
        raise EvaluationManifestError("Unsupported evaluation manifest format_version.")
    entries = manifest.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise EvaluationManifestError("Evaluation manifest must contain a non-empty datasets array.")

    if training_corpus_sha256 is not None and (
        not isinstance(training_corpus_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", training_corpus_sha256) is None
    ):
        raise EvaluationManifestError("Checkpoint contains an invalid training corpus SHA-256.")

    root = manifest_path.parent
    seen_ids: set[str] = set()
    seen_documents: set[str] = set()
    loaded: list[EvaluationDataset] = []

    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != _REQUIRED_DATASET_KEYS:
            raise EvaluationManifestError(f"Dataset entry {index} has an invalid schema.")

        dataset_id = entry.get("dataset_id")
        if (
            not isinstance(dataset_id, str)
            or not dataset_id.strip()
            or len(dataset_id) > 128
            or any(ord(char) < 32 for char in dataset_id)
        ):
            raise EvaluationManifestError(f"Dataset entry {index} requires a valid dataset_id.")
        if dataset_id in seen_ids:
            raise EvaluationManifestError(f"Duplicate evaluation dataset_id: {dataset_id}")
        seen_ids.add(dataset_id)

        language = entry.get("language")
        split = entry.get("split")
        if not isinstance(language, str) or language not in SUPPORTED_LANGUAGES:
            raise EvaluationManifestError(
                f"Dataset {dataset_id} language must be one of: {', '.join(sorted(SUPPORTED_LANGUAGES))}."
            )
        if not isinstance(split, str) or split not in SUPPORTED_SPLITS:
            raise EvaluationManifestError(
                f"Dataset {dataset_id} split must be 'validation' or 'test'; training data is not allowed."
            )
        if entry.get("rights_confirmed") is not True:
            raise EvaluationManifestError(
                f"Dataset {dataset_id} requires explicit rights_confirmed: true."
            )

        source_path = _safe_evaluation_path(root, entry.get("path"))
        raw = source_path.read_bytes()
        file_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise EvaluationManifestError(
                f"Dataset {dataset_id} is not valid UTF-8."
            ) from error
        # core.train hashes the UTF-8 text after universal newline translation.
        canonical_text = text.replace("\\r\\n", "\\n").replace("\\r", "\\n")
        training_comparison_sha256 = hashlib.sha256(
            canonical_text.encode("utf-8")
        ).hexdigest()
        if training_corpus_sha256 and training_comparison_sha256 == training_corpus_sha256:
            raise EvaluationManifestError(
                f"Dataset {dataset_id} exactly matches the training corpus; held-out data is required."
            )

        documents = _split_documents(canonical_text)
        if not documents:
            raise EvaluationManifestError(f"Dataset {dataset_id} contains no documents.")

        token_documents = []
        for document_index, document in enumerate(documents):
            flags = privacy_flags(document)
            if flags:
                raise EvaluationManifestError(
                    f"Dataset {dataset_id} document {document_index + 1} failed privacy checks: "
                    + ", ".join(flags)
                )
            token_ids = tuple(ByteTokenizer().encode(document))
            if len(token_ids) < 2:
                raise EvaluationManifestError(
                    f"Dataset {dataset_id} document {document_index + 1} must contain at least two UTF-8 bytes."
                )
            document_digest = hashlib.sha256(
                normalize_text(document).encode("utf-8")
            ).hexdigest()
            if document_digest in seen_documents:
                raise EvaluationManifestError(
                    "Exact duplicate document detected across evaluation corpora; "
                    "validation and test documents must be disjoint."
                )
            seen_documents.add(document_digest)
            token_documents.append(token_ids)

        loaded.append(EvaluationDataset(
            dataset_id=dataset_id,
            filename=source_path.name,
            language=language,
            split=split,
            sha256=file_sha256,
            file_bytes=len(raw),
            documents=tuple(token_documents),
        ))

    if not loaded:
        raise EvaluationManifestError("Evaluation manifest contains no usable datasets.")

    return loaded, hashlib.sha256(manifest_bytes).hexdigest()


@torch.no_grad()
def score_documents(model, documents, *, device: torch.device) -> dict:
    """Score every next-byte target once, using non-overlapping block-sized windows."""
    block_size = model.config.block_size
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("Model block_size must be a positive integer.")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_targets = 0
    document_count = 0
    byte_count = 0

    try:
        for token_ids in documents:
            if len(token_ids) < 2:
                raise ValueError("Every evaluation document must contain at least two tokens.")
            document_count += 1
            byte_count += len(token_ids)
            tokens = torch.tensor(token_ids, dtype=torch.long, device=device)
            # Adjacent windows share the previous window's final target byte as
            # the next window's first input byte. No next-byte target is omitted
            # or counted twice; each window's context is at most block_size.
            for start in range(0, len(token_ids) - 1, block_size):
                window = tokens[start : start + block_size + 1]
                if window.numel() < 2:
                    continue
                inputs = window[:-1].unsqueeze(0)
                targets = window[1:].unsqueeze(0)
                logits, _ = model(inputs)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, model.config.vocab_size),
                    targets.reshape(-1),
                    reduction="sum",
                )
                if not torch.isfinite(loss_sum):
                    raise ValueError("Model produced a non-finite evaluation loss.")
                total_loss += float(loss_sum.item())
                total_targets += int(targets.numel())
    finally:
        model.train(was_training)

    if total_targets == 0:
        raise ValueError("Evaluation corpus contains no next-token targets.")
    mean_loss = total_loss / total_targets
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = None
    return {
        "document_count": document_count,
        "byte_count": byte_count,
        "tokens_scored": total_targets,
        "loss": mean_loss,
        "perplexity": perplexity,
    }


def evaluate_manifest(
    model,
    manifest_path: Path,
    *,
    device: torch.device,
    training_corpus_sha256: str | None = None,
) -> dict:
    """Evaluate separate language/split datasets and report token-weighted groups."""
    datasets, manifest_sha256 = load_evaluation_manifest(
        manifest_path,
        training_corpus_sha256=training_corpus_sha256,
    )

    dataset_results = []
    grouped: dict[tuple[str, str], dict] = {}
    for dataset in datasets:
        metrics = score_documents(model, dataset.documents, device=device)
        result = {
            "dataset_id": dataset.dataset_id,
            "filename": dataset.filename,
            "language": dataset.language,
            "split": dataset.split,
            "corpus_sha256": dataset.sha256,
            "file_bytes": dataset.file_bytes,
            **metrics,
        }
        dataset_results.append(result)

        key = (dataset.language, dataset.split)
        group = grouped.setdefault(key, {
            "language": dataset.language,
            "split": dataset.split,
            "dataset_count": 0,
            "document_count": 0,
            "file_bytes": 0,
            "tokens_scored": 0,
            "_total_loss": 0.0,
        })
        group["dataset_count"] += 1
        group["document_count"] += metrics["document_count"]
        group["file_bytes"] += dataset.file_bytes
        group["tokens_scored"] += metrics["tokens_scored"]
        group["_total_loss"] += metrics["loss"] * metrics["tokens_scored"]

    group_results = []
    for key in sorted(grouped):
        group = grouped[key]
        token_count = group["tokens_scored"]
        mean_loss = group.pop("_total_loss") / token_count
        try:
            perplexity = math.exp(mean_loss)
        except OverflowError:
            perplexity = None
        group["loss"] = mean_loss
        group["perplexity"] = perplexity
        group_results.append(group)

    expected = {
        (language, split)
        for language in sorted(SUPPORTED_LANGUAGES)
        for split in sorted(SUPPORTED_SPLITS)
    }
    present = set(grouped)
    return {
        "format_version": 1,
        "manifest_sha256": manifest_sha256,
        "training_corpus_hash_available": training_corpus_sha256 is not None,
        "evaluation_protocol": (
            "UTF-8 byte next-token loss; each document is evaluated independently; "
            "non-overlapping input windows of at most model block_size; every next-byte "
            "target is scored exactly once; context resets at window boundaries."
        ),
        "datasets": dataset_results,
        "language_split_metrics": group_results,
        "missing_language_split_combinations": [
            {"language": language, "split": split}
            for language, split in sorted(expected - present)
        ],
        "warning": (
            "Byte-token loss/perplexity measures next-byte prediction only. It does not "
            "measure factuality, instruction following, reasoning or answer quality. "
            "Exact-duplicate checks do not detect paraphrases or near-duplicates. "
            + (
                "This checkpoint has no training corpus hash, so exact whole-file "
                "train/evaluation equality could not be checked."
                if training_corpus_sha256 is None
                else "The evaluation files were checked against the checkpoint training corpus hash."
            )
        ),
    }
