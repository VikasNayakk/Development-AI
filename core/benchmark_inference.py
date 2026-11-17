"""Reproducible synthetic inference latency benchmark for Development-AI.

Measures greedy autoregressive generation on synthetic byte-token IDs. It does
not load checkpoints, read user prompts or datasets, or measure model quality.
"""
import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DevelopmentAIModel


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def run_inference_benchmark(
    *,
    config: ModelConfig,
    device: torch.device,
    warmup_runs: int = 2,
    measured_runs: int = 5,
    batch_size: int = 1,
    prompt_tokens: int = 32,
    generation_tokens: int = 16,
    seed: int = 42,
) -> dict:
    """Benchmark greedy generation with warmup and repeated wall-clock samples."""
    config.validate()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA benchmark devices are supported.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    if type(warmup_runs) is not int or warmup_runs < 0:
        raise ValueError("warmup_runs must be a non-negative integer.")
    _positive_int(measured_runs, "measured_runs")
    _positive_int(batch_size, "batch_size")
    _positive_int(prompt_tokens, "prompt_tokens")
    _positive_int(generation_tokens, "generation_tokens")
    if prompt_tokens > config.block_size:
        raise ValueError("prompt_tokens cannot exceed config.block_size.")
    if type(seed) is not int or not 0 <= seed <= (2**64) - 1:
        raise ValueError("seed must be an integer between 0 and 2^64 - 1.")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = DevelopmentAIModel(config).to(device)
    model.eval()
    prompt = torch.randint(
        0, config.vocab_size, (batch_size, prompt_tokens),
        dtype=torch.long, device=device,
    )

    def generate_once():
        with torch.inference_mode():
            return model.generate(prompt, generation_tokens, do_sample=False)

    for _ in range(warmup_runs):
        generate_once()
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations = []
    generated_token_count = 0
    for _ in range(measured_runs):
        _synchronize(device)
        started = time.perf_counter()
        output = generate_once()
        _synchronize(device)
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise RuntimeError("Inference benchmark measured a non-positive duration.")
        actual_new_tokens = int(output.shape[1] - prompt_tokens)
        if actual_new_tokens != generation_tokens:
            raise RuntimeError("Model generated an unexpected number of tokens.")
        durations.append(elapsed)
        generated_token_count += actual_new_tokens * batch_size

    total_seconds = sum(durations)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda" else platform.processor() or platform.machine()
    )
    return {
        "schema_version": 1,
        "benchmark_kind": "synthetic_greedy_inference_latency",
        "quality_warning": (
            "Synthetic random-token benchmark only. It measures runtime performance, "
            "not language understanding, answer quality, or real-user serving latency."
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
        "inference": {
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "batch_size": batch_size,
            "prompt_tokens_per_sequence": prompt_tokens,
            "new_tokens_per_sequence": generation_tokens,
            "generated_tokens_total": generated_token_count,
            "total_measured_seconds": total_seconds,
            "mean_request_latency_seconds": statistics.mean(durations),
            "median_request_latency_seconds": statistics.median(durations),
            "p95_request_latency_seconds": _percentile(durations, 0.95),
            "mean_milliseconds_per_generated_token_per_sequence": (
                statistics.mean(durations) * 1000.0 / generation_tokens
            ),
            "aggregate_generated_tokens_per_second": (
                generated_token_count / total_seconds
            ),
            "per_run_seconds": durations,
            "decoding": "greedy",
            "kv_cache": False,
            "peak_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5,
                        help="Measured generation calls, excluding warmup.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--output", default=None,
                        help="Optional JSON report path; stdout is always printed.")
    args = parser.parse_args()

    config = ModelConfig(
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=0.0,
    )
    try:
        report = run_inference_benchmark(
            config=config,
            device=torch.device(args.device),
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
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
