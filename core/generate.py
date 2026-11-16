"""Run CPU or CUDA inference using a checkpoint produced by core.train."""
import argparse
import math
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DevelopmentAIModel
from .tokenizer import ByteTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="artifacts/latest.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--greedy", action="store_true",
                        help="Use deterministic greedy decoding instead of sampling.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional non-negative seed for reproducible sampled generation.")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens cannot be negative")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and greater than zero")
    if args.seed is not None and not 0 <= args.seed <= (2**64 - 1):
        parser.error("--seed must be between 0 and 2**64 - 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}. Train the model first.")

    device = torch.device(args.device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a dictionary.")
        if (
            checkpoint.get("format_version") != 1
            or checkpoint.get("tokenizer") != ByteTokenizer.name
        ):
            raise ValueError("Unsupported checkpoint format or tokenizer.")
        config_data = checkpoint.get("config")
        model_state = checkpoint.get("model_state")
        if not isinstance(config_data, dict) or not isinstance(model_state, dict):
            raise ValueError("Checkpoint is missing model configuration or weights.")
        config = ModelConfig(**config_data)
        model = DevelopmentAIModel(config).to(device)
        model.load_state_dict(model_state)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        parser.error(f"Cannot load a valid inference checkpoint: {error}")

    model.eval()
    tokenizer = ByteTokenizer()
    prompt_tokens = tokenizer.encode(args.prompt)
    if not prompt_tokens:
        parser.error("Prompt must not be empty.")
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    try:
        output = model.generate(
            input_ids,
            args.max_new_tokens,
            args.temperature,
            do_sample=not args.greedy,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        parser.error(f"Generation failed: {error}")
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
