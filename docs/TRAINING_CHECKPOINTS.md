# Resumable training checkpoints

## Output files

A training run writes three separate local files:

- The `--output` file (default `artifacts/latest.pt`) is the best-validation inference checkpoint. It remains format version 1 and can still be loaded by `core.generate` and `core.evaluate`.
- The resume file (by default `artifacts/latest.resume.pt`) is the latest resumable training state. It contains model weights, AdamW optimizer state, completed step, best validation loss, metric history, corpus fingerprint, training settings, cumulative optimizer-compute time, and Python/CPU/CUDA random-number-generator states.
- The metrics file (for example `artifacts/latest.metrics.json`) contains the metric history.

Resume checkpoints use a separate type marker and format version. They are not inference checkpoints and must not be passed to `core.generate`.

## Start a new run

```bash
python -m core.train \
  --data data/processed/train.txt \
  --steps 1000 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --device cpu \
  --output artifacts/latest.pt
```

A resumable checkpoint is saved every 100 steps by default and at the final step. Change this with `--resume-interval` when a different recovery-point interval is required. Checkpoints are written to a temporary file and atomically replaced after serialization.

## Gradient accumulation

- `--batch-size` is the micro-batch size used by each forward/backward pass.
- `--gradient-accumulation-steps` is the number of micro-batches accumulated before one optimizer update (default: `1`).
- Effective batch size is `batch_size × gradient_accumulation_steps`. For example, micro-batch 8 with accumulation 4 gives effective batch size 32.
- Gradients are averaged across the equally sized micro-batches. Gradient clipping is applied once to the combined gradient, immediately before the optimizer update.
- `--steps`, `--eval-interval` and `--resume-interval` continue to count optimizer updates, not individual micro-batches. One optimizer step consumes `gradient_accumulation_steps` micro-batches.
- Accumulation lowers activation memory per forward pass, but it does not reduce the memory required by model parameters or optimizer state. It also does not guarantee the exact throughput of a physical large batch.
- For a reliable comparison, keep optimizer-update count, effective batch size, learning rate, data, model and other settings fixed.

The default accumulation value of 1 preserves the previous single-batch update behavior. Earlier resumable checkpoints that predate the accumulation setting can resume only with accumulation 1. New resume checkpoints store the accumulation count and effective batch size; changing these when resuming is rejected.

## Training telemetry

At evaluation intervals, the metrics JSON and training log also record:

- Cumulative supervised target-byte positions processed: optimizer updates × micro-batch size × accumulation steps × block size.
- Time for the latest optimizer update.
- Cumulative optimizer-compute time across the run, including previously checkpointed time when resuming.
- Training tokens per second and optimizer updates per second, calculated from cumulative optimizer-compute time.
- The active learning rate.

CUDA updates are synchronized immediately before and after timing so asynchronous kernels are included. These durations include batch sampling and host-to-device transfer inside the optimizer update, but exclude evaluation, checkpoint serialization and other idle/wall-clock time. Treat them as optimizer-pipeline throughput, not end-to-end job throughput.

Timing values are measurements, not deterministic training state. A resumed run should reproduce its model weights and deterministic metrics, but elapsed times naturally differ from an uninterrupted run. Older resume checkpoints without timing metadata resume with a cumulative optimizer-compute time of zero.

## Resume a run

```bash
python -m core.train \
  --data data/processed/train.txt \
  --steps 2000 \
  --device cpu \
  --output artifacts/latest.pt \
  --resume-from artifacts/latest.resume.pt
```

**`--steps` is the total target step count, not the number of additional steps.** The example continues a run saved at step 1,000 through step 2,000.

Keep the same training corpus, seed, micro-batch size, gradient accumulation steps, learning rate, block size, validation fraction, evaluation interval, evaluation iterations, device type and CUDA device count. The trainer validates the corpus SHA-256, model configuration and run settings before loading state. A mismatch stops the run rather than silently creating a different experiment.

Keep the best inference checkpoint and its paired resume checkpoint together. The trainer checks that both belong to the same corpus and configuration. The resume output path can be changed explicitly with `--resume-output`.

## Compatibility and privacy

- The existing version-1 inference checkpoint format is preserved.
- Old inference checkpoints without the required corpus fingerprint cannot be used as resumable training checkpoints.
- BPE remains experimental and is not connected to training.
- Checkpoints can contain optimizer state and training data fingerprints; keep them local. The repository's ignore rules exclude `artifacts/`, `*.pt` and `*.pth`.
- A checkpoint is a recovery mechanism, not a substitute for backups of the original dataset.

## Validation

The test suite compares an interrupted-and-resumed CPU training run against an uninterrupted run with the same settings. It also checks that changed corpus content and invalid step targets are rejected.
