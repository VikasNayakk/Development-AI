"""Train and benchmark experimental byte-level BPE vocabularies."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from .bpe_tokenizer import ByteLevelBPETokenizer
from .tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="UTF-8 text corpus used only for this experiment")
    parser.add_argument("--output-dir", default="artifacts/tokenizer-benchmarks")
    parser.add_argument(
        "--vocab-sizes", nargs="+", type=int, default=[2048, 4096, 8192],
        help="Target vocabulary sizes; each experiment is trained independently",
    )
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--max-training-bytes", type=int, default=50000,
        help="Safety limit for this in-memory experimental trainer",
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

    corpus_path = Path(args.data)
    if not corpus_path.is_file():
        parser.error(f"Corpus file not found: {corpus_path}")
    try:
        text = corpus_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        parser.error(f"Cannot read UTF-8 corpus: {error}")

    corpus_bytes = len(text.encode("utf-8"))
    if corpus_bytes == 0:
        parser.error("Corpus must not be empty")
    if corpus_bytes > args.max_training_bytes:
        parser.error(
            f"Corpus is {corpus_bytes} bytes; the experimental in-memory trainer "
            f"limit is {args.max_training_bytes}. Use a reviewed representative sample "
            "or explicitly raise --max-training-bytes after checking available memory."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    byte_token_count = len(ByteTokenizer().encode(text))
    results = []

    for target_vocab_size in args.vocab_sizes:
        started = time.perf_counter()
        tokenizer = ByteLevelBPETokenizer.train(
            text,
            max_vocab_size=target_vocab_size,
            min_frequency=args.min_frequency,
        )
        training_seconds = time.perf_counter() - started

        started = time.perf_counter()
        encoded = tokenizer.encode(text)
        encoding_seconds = time.perf_counter() - started
        token_count = len(encoded)

        artifact_path = output_dir / f"byte-bpe-target-{target_vocab_size}.json"
        tokenizer.save(artifact_path)
        artifact = tokenizer.to_dict()
        results.append({
            "target_vocab_size": target_vocab_size,
            "actual_vocab_size": tokenizer.vocab_size,
            "merge_count": len(tokenizer.merges),
            "min_frequency": args.min_frequency,
            "corpus_utf8_bytes": corpus_bytes,
            "byte_token_count": byte_token_count,
            "bpe_token_count": token_count,
            "token_reduction_percent": (
                100.0 * (1.0 - token_count / byte_token_count)
                if byte_token_count else 0.0
            ),
            "byte_to_bpe_compression_ratio": (
                byte_token_count / token_count if token_count else 0.0
            ),
            "training_seconds": training_seconds,
            "encoding_seconds": encoding_seconds,
            "encoding_tokens_per_second": (
                token_count / encoding_seconds if encoding_seconds else None
            ),
            "artifact": str(artifact_path),
            "artifact_sha256": artifact["sha256"],
        })
        print(
            f"target={target_vocab_size} actual={tokenizer.vocab_size} "
            f"tokens={token_count}/{byte_token_count} "
            f"reduction={results[-1]['token_reduction_percent']:.2f}%"
        )

    report = {
        "format_version": 1,
        "experiment": "utf8-byte-bpe-v1",
        "corpus_file": str(corpus_path),
        "corpus_utf8_bytes": corpus_bytes,
        "note": (
            "Experimental results on this corpus only. Token reduction is not a "
            "quality metric; tokenizer/model quality and downstream training must "
            "be evaluated separately. The production model still uses utf8-byte-v1."
        ),
        "results": results,
    }
    report_path = output_dir / "benchmark-report.json"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n",
            prefix=".benchmark-report.", suffix=".tmp",
            dir=output_dir, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, report_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Benchmark report saved to: {report_path}")


if __name__ == "__main__":
    main()
