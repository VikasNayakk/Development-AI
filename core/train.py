"""Reproducible training with disjoint train/validation byte streams."""
import argparse
import json
import random
import tempfile
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DevelopmentAIModel
from .tokenizer import ByteTokenizer


def split_corpus(data: torch.Tensor, validation_fraction: float, block_size: int):
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    split_at = int(len(data) * (1.0 - validation_fraction))
    train_data, validation_data = data[:split_at], data[split_at:]
    if len(train_data) <= block_size or len(validation_data) <= block_size:
        raise ValueError(
            "Corpus is too small for this block size and validation fraction; "
            "add data or reduce --block-size."
        )
    return train_data, validation_data


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if data.ndim != 1 or len(data) <= block_size:
        raise ValueError("A one-dimensional token stream longer than block_size is required.")
    # high is exclusive; this includes the final valid start index.
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[int(i) : int(i) + block_size] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, validation_data, *, batch_size, block_size,
                  eval_iters, device):
    was_training = model.training
    model.eval()
    results = {}
    try:
        for name, data in (("train", train_data), ("validation", validation_data)):
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(data, batch_size, block_size, device)
                _, loss = model(x, y)
                losses.append(float(loss.item()))
            results[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return results


def atomic_save_checkpoint(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".tmp", prefix=f".{destination.name}.",
            dir=destination.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/latest.pt")
    args = parser.parse_args()

    for name in ("steps", "batch_size", "block_size", "eval_interval", "eval_iters"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    corpus_path = Path(args.data)
    if not corpus_path.is_file():
        parser.error(f"Training file not found: {corpus_path}")

    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode(corpus_path.read_text(encoding="utf-8"))
    data = torch.tensor(encoded, dtype=torch.long)
    try:
        train_data, validation_data = split_corpus(
            data, args.validation_fraction, args.block_size
        )
    except ValueError as error:
        parser.error(str(error))

    config = ModelConfig(block_size=args.block_size)
    model = DevelopmentAIModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    history = []

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            metrics = estimate_loss(
                model, train_data, validation_data,
                batch_size=args.batch_size, block_size=config.block_size,
                eval_iters=args.eval_iters, device=device
            )
            record = {"step": step, **metrics}
            history.append(record)
            print(
                f"step {step}/{args.steps} | "
                f"train_loss {metrics['train']:.4f} | "
                f"validation_loss {metrics['validation']:.4f}"
            )
            if metrics["validation"] < best_validation_loss:
                best_validation_loss = metrics["validation"]
                payload = {
                    "format_version": 1,
                    "model_name": "Development-AI",
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "tokenizer": "utf8-byte-v1",
                    "training": {
                        "steps_completed": step,
                        "seed": args.seed,
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "validation_fraction": args.validation_fraction,
                        "best_validation_loss": best_validation_loss,
                        "corpus_bytes": len(data),
                    },
                }
                atomic_save_checkpoint(payload, Path(args.output))

    history_path = Path(args.output).with_suffix(".metrics.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint saved to: {args.output}")
    print(f"Metrics saved to: {history_path}")


if __name__ == "__main__":
    main()
