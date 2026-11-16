"""Evaluate a locally trained Development-AI checkpoint on held-out corpora."""
import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
from .evaluation import EvaluationManifestError, evaluate_manifest
from .model import DevelopmentAIModel
from .tokenizer import ByteTokenizer
from .train import get_batch, split_corpus


@torch.no_grad()
def evaluate(model, data, *, batch_size, eval_iters, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, model.config.block_size, device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "perplexity": float(torch.exp(torch.tensor(mean_loss)).item())}


def _load_inference_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")
    if checkpoint.get("format_version") != 1 or checkpoint.get("tokenizer") != ByteTokenizer.name:
        raise ValueError("Unsupported checkpoint format or tokenizer.")
    config_data = checkpoint.get("config")
    model_state = checkpoint.get("model_state")
    if not isinstance(config_data, dict) or not isinstance(model_state, dict):
        raise ValueError("Checkpoint is missing model configuration or weights.")
    config = ModelConfig(**config_data)
    model = DevelopmentAIModel(config).to(device)
    model.load_state_dict(model_state)
    model.eval()
    return checkpoint, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/latest.pt")
    parser.add_argument(
        "--data", default=None,
        help="Legacy single-corpus input; split internally using --validation-fraction.",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Evaluate separate local validation/test corpora by language using a JSON manifest.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.manifest and args.data is not None:
        parser.error("Use either --manifest or --data, not both.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    checkpoint_path = Path(args.checkpoint)
    device = torch.device(args.device)
    try:
        checkpoint, model = _load_inference_model(checkpoint_path, device)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        parser.error(f"Cannot load a valid inference checkpoint: {error}")

    if args.manifest:
        training = checkpoint.get("training")
        training_hash = (
            training.get("corpus_sha256")
            if isinstance(training, dict)
            else None
        )
        try:
            metrics = evaluate_manifest(
                model,
                Path(args.manifest),
                device=device,
                training_corpus_sha256=training_hash,
            )
        except (EvaluationManifestError, OSError, RuntimeError, ValueError) as error:
            parser.error(f"Cannot evaluate held-out manifest: {error}")
        print(json.dumps({
            "model": checkpoint.get("model_name", "Development-AI"),
            "checkpoint": str(checkpoint_path),
            "evaluation": metrics,
        }, ensure_ascii=False, indent=2))
        return

    if args.batch_size <= 0 or args.eval_iters <= 0:
        parser.error("--batch-size and --eval-iters must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")

    corpus_path = Path(args.data or "data/train.txt")
    if not corpus_path.is_file():
        parser.error(f"Evaluation corpus not found: {corpus_path}")
    try:
        tokens = torch.tensor(
            ByteTokenizer().encode(corpus_path.read_text(encoding="utf-8")),
            dtype=torch.long,
        )
        _, validation_data = split_corpus(
            tokens, args.validation_fraction, model.config.block_size
        )
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(f"Cannot prepare evaluation corpus: {error}")

    metrics = evaluate(
        model, validation_data, batch_size=args.batch_size,
        eval_iters=args.eval_iters, device=device
    )
    print(json.dumps({
        "model": checkpoint.get("model_name", "Development-AI"),
        "checkpoint": str(checkpoint_path),
        "evaluation": metrics,
        "warning": (
            "Legacy random-batch byte-token perplexity is corpus-specific; "
            "it is not a measure of factuality, reasoning or answer quality."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
