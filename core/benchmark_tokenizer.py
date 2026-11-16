"""Benchmark byte-level BPE on separate and combined language corpora.

Each non-empty paragraph (separated by a blank line) is treated as a document,
so merges cannot cross prompt/document boundaries.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

from .bpe_tokenizer import ByteLevelBPETokenizer
from .tokenizer import ByteTokenizer


LANGUAGE_ARGUMENTS = {
    "hindi": "--data-hindi",
    "english": "--data-english",
    "hinglish": "--data-hinglish",
}


def _read_corpus(path: str, label: str, byte_limit: int) -> dict:
    corpus_path = Path(path)
    if not corpus_path.is_file():
        raise ValueError(f"{label} corpus file not found: {corpus_path}")
    try:
        with corpus_path.open("rb") as handle:
            raw = handle.read(byte_limit + 1)
    except OSError as error:
        raise ValueError(f"Cannot read {label} corpus: {error}") from error

    if len(raw) > byte_limit:
        raise ValueError(
            f"{label} corpus exceeds the {byte_limit}-byte total input limit."
        )
    if not raw:
        raise ValueError(f"{label} corpus must not be empty.")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} corpus is not valid UTF-8: {error}") from error

    documents = [part.strip() for part in re.split(r"\r?\n[ \t]*\r?\n", text) if part.strip()]
    if not documents:
        raise ValueError(f"{label} corpus contains no non-empty documents.")

    return {
        "path": corpus_path.name,
        "raw_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "documents": documents,
    }


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _evaluate(tokenizer, documents: list[str]) -> dict:
    byte_tokenizer = ByteTokenizer()
    byte_token_count = sum(len(byte_tokenizer.encode(doc)) for doc in documents)
    started = time.perf_counter()
    encoded_documents = [tokenizer.encode(doc) for doc in documents]
    encoding_seconds = time.perf_counter() - started
    bpe_token_count = sum(len(tokens) for tokens in encoded_documents)
    return {
        "document_count": len(documents),
        "utf8_bytes": sum(len(doc.encode("utf-8")) for doc in documents),
        "byte_token_count": byte_token_count,
        "bpe_token_count": bpe_token_count,
        "token_reduction_percent": (
            100.0 * (1.0 - bpe_token_count / byte_token_count)
            if byte_token_count else 0.0
        ),
        "byte_to_bpe_compression_ratio": (
            byte_token_count / bpe_token_count if bpe_token_count else 0.0
        ),
        "encoding_seconds": encoding_seconds,
        "encoding_tokens_per_second": (
            bpe_token_count / encoding_seconds if encoding_seconds else None
        ),
        "round_trip_exact": all(
            tokenizer.decode(tokenizer.encode(doc)) == doc for doc in documents
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        help="Single UTF-8 corpus (legacy mode); use language-specific options for multilingual tests",
    )
    for language, option in LANGUAGE_ARGUMENTS.items():
        parser.add_argument(option, dest=f"data_{language}")
    parser.add_argument("--output-dir", default="artifacts/tokenizer-benchmarks")
    parser.add_argument(
        "--vocab-sizes",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192],
        help="Target vocabulary sizes; each experiment is trained independently",
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--max-training-bytes",
        type=int,
        default=50000,
        help="Maximum total UTF-8 bytes across all supplied corpora",
    )
    args = parser.parse_args()

    if not args.vocab_sizes or any(size < 256 for size in args.vocab_sizes):
        parser.error("--vocab-sizes must contain values >= 256")
    if len(set(args.vocab_sizes)) != len(args.vocab_sizes):
        parser.error("--vocab-sizes must not contain duplicates")
    if args.min_frequency < 1:
        parser.error("--min-frequency must be positive")
    if args.max_training_bytes < 1:
        parser.error("--max-training-bytes must be positive")

    supplied_languages = {
        language: getattr(args, f"data_{language}")
        for language in LANGUAGE_ARGUMENTS
    }
    language_mode = any(supplied_languages.values())
    if args.data and language_mode:
        parser.error("Use either --data or all three language-specific corpus options, not both.")
    if language_mode and not all(supplied_languages.values()):
        parser.error(
            "Multilingual mode requires --data-hindi, --data-english and --data-hinglish."
        )
    if not args.data and not language_mode:
        parser.error("Provide --data or all three language-specific corpus options.")

    try:
        corpora = {}
        remaining_bytes = args.max_training_bytes
        if language_mode:
            for language, path in supplied_languages.items():
                item = _read_corpus(path, language, remaining_bytes)
                corpora[language] = item
                remaining_bytes -= item["raw_bytes"]
                if remaining_bytes < 0:
                    raise ValueError(
                        f"Combined corpus size exceeds the {args.max_training_bytes}-byte total input limit."
                    )
        else:
            corpora["custom"] = _read_corpus(
                args.data, "custom", args.max_training_bytes
            )
    except ValueError as error:
        parser.error(str(error))

    total_bytes = sum(item["raw_bytes"] for item in corpora.values())
    if total_bytes > args.max_training_bytes:
        parser.error(
            f"Combined corpus size is {total_bytes} bytes; the total input limit is "
            f"{args.max_training_bytes}. Increase the limit only after checking memory."
        )

    # In multilingual mode, train one tokenizer per language and one shared tokenizer.
    # Documents remain separate, so merges cannot cross prompt boundaries.
    training_sets = dict(corpora)
    if language_mode:
        training_sets["combined"] = {
            "documents": [
                document
                for language in ("hindi", "english", "hinglish")
                for document in corpora[language]["documents"]
            ]
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format_version": 2,
        "experiment": "utf8-byte-bpe-v1",
        "mode": "multilingual" if language_mode else "single-corpus",
        "total_corpus_utf8_bytes": total_bytes,
        "max_training_bytes": args.max_training_bytes,
        "vocab_sizes_requested": args.vocab_sizes,
        "min_frequency": args.min_frequency,
        "corpora": {
            language: {
                "path": item["path"],
                "utf8_bytes": item["raw_bytes"],
                "sha256": item["sha256"],
                "document_count": len(item["documents"]),
            }
            for language, item in corpora.items()
        },
        "models": [],
        "note": (
            "Token reduction is a corpus-specific compression measure, not a language "
            "quality or reasoning score. Compare each tokenizer on every language. "
            "The production model remains on utf8-byte-v1 with vocab_size=256."
        ),
    }

    evaluation_sets = {
        language: item["documents"] for language, item in corpora.items()
    }
    if language_mode:
        evaluation_sets["combined"] = training_sets["combined"]["documents"]

    for training_name, training_data in training_sets.items():
        for target_vocab_size in args.vocab_sizes:
            started = time.perf_counter()
            tokenizer = ByteLevelBPETokenizer.train(
                training_data["documents"],
                max_vocab_size=target_vocab_size,
                min_frequency=args.min_frequency,
            )
            training_seconds = time.perf_counter() - started

            artifact_name = (
                f"byte-bpe-target-{target_vocab_size}.json"
                if training_name == "custom"
                else f"{training_name}-target-{target_vocab_size}.json"
            )
            artifact_path = output_dir / artifact_name
            tokenizer.save(artifact_path)
            artifact = tokenizer.to_dict()

            evaluations = {
                evaluation_name: _evaluate(tokenizer, documents)
                for evaluation_name, documents in evaluation_sets.items()
            }
            model_result = {
                "trained_on": training_name,
                "target_vocab_size": target_vocab_size,
                "actual_vocab_size": tokenizer.vocab_size,
                "merge_count": len(tokenizer.merges),
                "training_seconds": training_seconds,
                "artifact": artifact_name,
                "artifact_sha256": artifact["sha256"],
                "evaluations": evaluations,
            }
            report["models"].append(model_result)
            print(
                f"trained_on={training_name} target={target_vocab_size} "
                f"actual={tokenizer.vocab_size} "
                + " ".join(
                    f"{name}_reduction={result['token_reduction_percent']:.2f}%"
                    for name, result in evaluations.items()
                )
            )

    if not language_mode:
        custom = corpora["custom"]
        report["corpus_file"] = custom["path"]
        report["corpus_utf8_bytes"] = custom["raw_bytes"]
        report["corpus_sha256"] = custom["sha256"]
        report["results"] = [
            {
                "target_vocab_size": model["target_vocab_size"],
                "actual_vocab_size": model["actual_vocab_size"],
                "merge_count": model["merge_count"],
                "min_frequency": args.min_frequency,
                "corpus_utf8_bytes": custom["raw_bytes"],
                "byte_token_count": model["evaluations"]["custom"]["byte_token_count"],
                "bpe_token_count": model["evaluations"]["custom"]["bpe_token_count"],
                "token_reduction_percent": model["evaluations"]["custom"]["token_reduction_percent"],
                "byte_to_bpe_compression_ratio": model["evaluations"]["custom"]["byte_to_bpe_compression_ratio"],
                "training_seconds": model["training_seconds"],
                "encoding_seconds": model["evaluations"]["custom"]["encoding_seconds"],
                "encoding_tokens_per_second": model["evaluations"]["custom"]["encoding_tokens_per_second"],
                "artifact": model["artifact"],
                "artifact_sha256": model["artifact_sha256"],
            }
            for model in report["models"]
        ]

    report_path = output_dir / "benchmark-report.json"
    _atomic_write_json(report_path, report)
    print(f"Benchmark report saved to: {report_path}")


if __name__ == "__main__":
    main()
