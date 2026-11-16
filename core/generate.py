"""Run CPU or CUDA inference using a checkpoint trained by core.train."""
import argparse
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DevelopmentAIModel
from .tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/latest.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}. Train the model first.")
    device = torch.device(args.device)
    # Checkpoints are locally generated project artifacts; do not load untrusted files.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("tokenizer") != "utf8-byte-v1":
        parser.error("Unsupported tokenizer metadata.")
    config = ModelConfig(**checkpoint["config"])
    model = DevelopmentAIModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tokenizer = ByteTokenizer()
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    if input_ids.numel() == 0:
        parser.error("Prompt must not be empty.")
    output = model.generate(input_ids, args.max_new_tokens, args.temperature)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
