"""Evaluate a locally trained Development-AI checkpoint on a held-out corpus."""
import argparse
import json
from pathlib import Path

import torch

from .config import ModelConfig
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/latest.pt")
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.eval_iters <= 0:
        parser.error("--batch-size and --eval-iters must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    checkpoint_path, corpus_path = Path(args.checkpoint), Path(args.data)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    if not corpus_path.is_file():
        parser.error(f"Evaluation corpus not found: {corpus_path}")

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("tokenizer") != "utf8-byte-v1":
        parser.error("Unsupported checkpoint format or tokenizer.")
    config = ModelConfig(**checkpoint["config"])
    model = DevelopmentAIModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    tokens = torch.tensor(ByteTokenizer().encode(corpus_path.read_text(encoding="utf-8")))
    try:
        _, validation_data = split_corpus(tokens, args.validation_fraction, config.block_size)
    except ValueError as error:
        parser.error(str(error))
    metrics = evaluate(
        model, validation_data, batch_size=args.batch_size,
        eval_iters=args.eval_iters, device=device
    )
    print(json.dumps({
        "model": checkpoint["model_name"],
        "checkpoint": str(checkpoint_path),
        "evaluation": metrics,
        "warning": "Perplexity is byte-token perplexity; compare only with the same tokenizer and evaluation corpus."
    }, indent=2))


if __name__ == "__main__":
    main()
