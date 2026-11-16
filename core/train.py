"""Train Development-AI from random initialization using a local UTF-8 text corpus."""
import argparse
import random
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DevelopmentAIModel
from .tokenizer import ByteTokenizer


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str):
    if len(data) <= block_size + 1:
        raise ValueError("Training corpus is too small for the selected block_size.")
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/latest.pt")
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("--steps and --batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    corpus_path = Path(args.data)
    if not corpus_path.is_file():
        parser.error(f"Training file not found: {corpus_path}")

    tokenizer = ByteTokenizer()
    encoded = tokenizer.encode(corpus_path.read_text(encoding="utf-8"))
    if len(encoded) < args.block_size + 2:
        parser.error("Corpus is too short. Add more UTF-8 text or reduce --block-size.")
    data = torch.tensor(encoded, dtype=torch.long)

    config = ModelConfig(block_size=args.block_size)
    model = DevelopmentAIModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()

    for step in range(1, args.steps + 1):
        x, y = get_batch(data, args.batch_size, config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            print(f"step {step}/{args.steps} | loss {loss.item():.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": config.to_dict(), "model_state": model.state_dict(),
         "tokenizer": "utf8-byte-v1", "training": {"steps": args.steps, "seed": args.seed}},
        output,
    )
    print(f"Saved project-trained checkpoint: {output}")


if __name__ == "__main__":
    main()
