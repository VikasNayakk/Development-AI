"""Validate and prepare an explicitly rights-confirmed local text corpus."""
import argparse
import json
from pathlib import Path

from .dataset import DatasetError, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/sources.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-repeated-ratio", type=float, default=0.30)
    args = parser.parse_args()
    try:
        report = prepare_dataset(
            Path(args.manifest), Path(args.output_dir),
            min_chars=args.min_chars,
            max_repeated_ratio=args.max_repeated_ratio,
        )
    except (DatasetError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({
        "accepted": report["accepted_count"],
        "rejected": report["rejected_count"],
        "corpus_characters": report["corpus_characters"],
        "report": str(Path(args.output_dir) / "dataset_report.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
