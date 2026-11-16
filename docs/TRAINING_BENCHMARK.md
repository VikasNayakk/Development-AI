# Training performance benchmark

## Purpose

Run a repeatable, local synthetic benchmark for the current training update and greedy generation paths. It reports hardware/software context, optimizer-update latency, supervised-token throughput, generation latency and CUDA peak allocated memory when applicable.

This is a performance diagnostic, not a model-quality evaluation. Synthetic random byte IDs do not represent a real corpus, and results must not be compared across different hardware, PyTorch versions, model configurations, batch sizes or accumulation settings without recording those differences.

## Run

```bash
python -m core.benchmark_training \
  --device cpu \
  --warmup-steps 2 \
  --steps 10 \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --generation-tokens 16 \
  --seed 42 \
  --output artifacts/performance-baseline.json
```

Use `--device cuda` only when CUDA is available. The benchmark fails clearly rather than silently falling back to CPU.

## Metric definitions

- `supervised_tokens_processed`: measured optimizer updates × micro-batch size × accumulation steps × block size. Warmup tokens are excluded.
- `supervised_tokens_per_second`: the above supervised target-token count divided by measured optimizer-update wall time. Evaluation, checkpoint I/O and data loading are not included.
- `optimizer_updates_per_second`: measured optimizer updates divided by the same training-only duration.
- `average_optimizer_update_seconds`: arithmetic mean of measured update durations.
- `milliseconds_per_generated_token`: greedy-generation wall time divided by the requested number of new tokens. This implementation has no KV cache; it is not a production serving benchmark.
- `peak_cuda_allocated_bytes`: PyTorch's peak allocated CUDA tensor memory during measured training. It is null on CPU and does not represent total process or device memory.

The benchmark uses a deterministic synthetic token stream, fixed seed, warmup updates and explicit CUDA synchronization around timed sections. Each report includes environment and model configuration. CPU timing can still vary with system load and thread scheduling.

## Privacy and scope

The command does not read a dataset, checkpoint or user prompt. It creates synthetic token IDs in memory and writes only aggregate metrics and environment metadata. Reports under `artifacts/` are local and ignored by Git.

Do not treat this report as proof that the model understands Hindi, English or Hinglish, or that it can answer questions accurately. Those require representative rights-cleared corpora, held-out evaluation and human review.
