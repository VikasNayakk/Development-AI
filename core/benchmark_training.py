"""Reproducible synthetic benchmark for local training and generation performance.

This benchmark measures software/hardware throughput, not model quality. It uses
synthetic byte-token batches and never reads or uploads a user's training corpus.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DevelopmentAIModel
from .train import optimizer_update


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    *,
    config: ModelConfig,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    generation_tokens: int,
    seed: int = 42,
) -> dict:
    """Run isolated synthetic training and greedy-generation timing."""
    for name, value in (
        ("warmup_steps", warmup_steps),
        ("measured_steps", measured_steps),
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("generation_tokens", generation_tokens),
    ):
        if type(value) is not int or value < (0 if name == "warmup_steps" else 1):
            raise ValueError(f"{name} has an invalid value.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    config.validate()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA benchmark devices are supported.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = DevelopmentAIModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # The synthetic stream is deterministic for a given seed and shape.
    stream_length = max(4096, (warmup_steps + measured_steps + 2) *
                        batch_size * gradient_accumulation_steps * config.block_size)
    synthetic_data = torch.randint(0, config.vocab_size, (stream_length,), dtype=torch.long)

    for _ in range(warmup_steps):
        optimizer_update(
            model, optimizer, synthetic_data, batch_size=batch_size,
            block_size=config.block_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    model.train()
    for _ in range(measured_steps):
        _synchronize(device)
        started = time.perf_counter()
        optimizer_update(
            model, optimizer, synthetic_data, batch_size=batch_size,
            block_size=config.block_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
        )
        _synchronize(device)
        durations.append(time.perf_counter() - started)

    total_seconds = sum(durations)
    supervised_tokens = (
        measured_steps * batch_size * gradient_accumulation_steps * config.block_size
    )
    if total_seconds <= 0.0:
        raise RuntimeError("Measured training duration was not positive.")
    average_update_seconds = total_seconds / measured_steps
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else None
    )

    prompt_length = min(config.block_size, 32)
    prompt = torch.randint(0, config.vocab_size, (1, prompt_length), device=device)
    was_training = model.training
    model.eval()
    _synchronize(device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            prompt, generation_tokens, do_sample=False
        )
    _synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    model.train(was_training)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" else platform.processor() or platform.machine()
    )

    return {
        "schema_version": 1,
        "benchmark_kind": "synthetic_training_and_greedy_generation",
        "quality_warning": (
            "Synthetic benchmark only. Throughput is not a measure of model quality "
            "and may differ from real-corpus training."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "device_type": device.type,
            "device_name": device_name,
        },
        "model": {
            "config": config.to_dict(),
            "unique_parameter_count": parameter_count,
        },
        "training": {
            "warmup_optimizer_updates": warmup_steps,
            "measured_optimizer_updates": measured_steps,
            "micro_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "block_size": config.block_size,
            "supervised_tokens_processed": supervised_tokens,
            "total_update_seconds": total_seconds,
            "average_optimizer_update_seconds": average_update_seconds,
            "optimizer_updates_per_second": measured_steps / total_seconds,
            "supervised_tokens_per_second": supervised_tokens / total_seconds,
            "per_update_seconds": durations,
            "peak_cuda_allocated_bytes": peak_memory_bytes,
        },
        "generation": {
            "prompt_tokens": prompt_length,
            "new_tokens": generation_tokens,
            "total_seconds": generation_seconds,
            "milliseconds_per_generated_token": (
                generation_seconds * 1000.0 / generation_tokens
            ),
            "generated_token_count": int(generated.shape[1] - prompt_length),
            "decoding": "greedy",
            "kv_cache": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10,
                        help="Measured optimizer updates, excluding warmup.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Optional JSON report path; stdout is always printed.")
    args = parser.parse_args()

    try:
        report = run_benchmark(
            config=ModelConfig(),
            device=torch.device(args.device),
            warmup_steps=args.warmup_steps,
            measured_steps=args.steps,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            generation_tokens=args.generation_tokens,
            seed=args.seed,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))

    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
