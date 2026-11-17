# Inference latency benchmark

## Purpose

Run a repeatable, local synthetic benchmark for the current greedy autoregressive generation path. It reports warmup and measured call counts, request latency distribution, aggregate generated-token throughput, model configuration, software/hardware context, and CUDA peak allocated memory when applicable.

This is a performance diagnostic, not a model-quality evaluation or a production serving benchmark. The current decoder recomputes the context for every generated token and does not use a KV cache.

## Run

```bash
python -m core.benchmark_inference \
  --device cpu \
  --warmup-runs 2 \
  --runs 5 \
  --batch-size 1 \
  --prompt-tokens 32 \
  --generation-tokens 16 \
  --seed 42 \
  --output artifacts/inference-baseline.json
```

Use `--device cuda` only when CUDA is available. The command fails clearly rather than silently falling back to CPU. Model dimensions can be adjusted with `--block-size`, `--d-model`, `--n-heads`, and `--n-layers`; record those values when comparing runs.

## Metric definitions

- `mean_request_latency_seconds`, `median_request_latency_seconds`, and `p95_request_latency_seconds` summarize full generation-call wall time across measured runs. The p95 uses the nearest-rank method.
- `mean_milliseconds_per_generated_token_per_sequence` divides mean call latency by the number of new tokens in each output sequence. It is not a per-batch token latency.
- `aggregate_generated_tokens_per_second` counts generated tokens across all batch sequences and measured runs, divided by total measured wall time.
- `per_run_seconds` preserves individual samples so run-to-run variability is visible.
- `peak_cuda_allocated_bytes` is PyTorch peak allocated tensor memory after warmup; it is null on CPU and is not total process or device memory.

Warmup calls are excluded from measured samples. CUDA synchronization brackets every timed call. The report records environment and model configuration; CPU results can still vary with system load and thread scheduling.

## Privacy and scope

The benchmark creates synthetic token IDs in memory. It does not load model checkpoints, user prompts, or datasets and does not upload information. Reports under `artifacts/` are local and ignored by Git.

Do not use synthetic throughput as evidence of Hindi, English, or Hinglish understanding, answer accuracy, or real-user API latency. Those require representative rights-cleared data, held-out evaluation, human review, and separate end-to-end serving measurements.
