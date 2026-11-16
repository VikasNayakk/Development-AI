# Resumable training checkpoints

## Output files

A training run writes three separate local files:

- The `--output` file (default `artifacts/latest.pt`) is the best-validation inference checkpoint. It remains format version 1 and can still be loaded by `core.generate` and `core.evaluate`.
- The resume file (by default `artifacts/latest.resume.pt`) is the latest resumable training state. It contains model weights, AdamW optimizer state, completed step, best validation loss, metric history, corpus fingerprint, training settings, and Python/CPU/CUDA random-number-generator states.
- The metrics file (for example `artifacts/latest.metrics.json`) contains the metric history.

Resume checkpoints use a separate type marker and format version. They are not inference checkpoints and must not be passed to `core.generate`.

## Start a new run

```bash
python -m core.train \
  --data data/processed/train.txt \
  --steps 1000 \
  --device cpu \
  --output artifacts/latest.pt
```

A resumable checkpoint is saved every 100 steps by default and at the final step. Change this with `--resume-interval` when a different recovery-point interval is required. Checkpoints are written to a temporary file and atomically replaced after serialization.

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

Keep the same training corpus, seed, batch size, learning rate, block size, validation fraction, evaluation interval, evaluation iterations, device type and CUDA device count. The trainer validates the corpus SHA-256, model configuration and run settings before loading state. A mismatch stops the run rather than silently creating a different experiment.

Keep the best inference checkpoint and its paired resume checkpoint together. The trainer checks that both belong to the same corpus and configuration. The resume output path can be changed explicitly with `--resume-output`.

## Compatibility and privacy

- The existing version-1 inference checkpoint format is preserved.
- Old inference checkpoints without the required corpus fingerprint cannot be used as resumable training checkpoints.
- BPE remains experimental and is not connected to training.
- Checkpoints can contain optimizer state and training data fingerprints; keep them local. The repository's ignore rules exclude `artifacts/`, `*.pt` and `*.pth`.
- A checkpoint is a recovery mechanism, not a substitute for backups of the original dataset.

## Validation

The test suite compares an interrupted-and-resumed CPU training run against an uninterrupted run with the same settings. It also checks that changed corpus content and invalid step targets are rejected.
