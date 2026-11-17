# Development-AI model architecture and implementation plan

## Status

This document records the verified baseline and the proposed sequence for improving it. It is an engineering plan, not a claim that the model is already capable of general-purpose question answering.

The project must remain independent at the model layer:
- No pretrained model weights or pretrained language model.
- No hosted AI inference API and no third-party model runtime.
- PyTorch may be used as numerical and training infrastructure for now.
- The model, tokenizer implementation, training policy, checkpoint format and eventual inference runtime remain project-owned.
- Private prompts, credentials and private datasets must never be committed to the public repository.

## Verified implementation baseline

The current production path is:
`core.tokenizer.ByteTokenizer` -> `core.model.DevelopmentAIModel` -> `core.train` checkpoint -> `core.generate` / `core.evaluate`.

Current default configuration:
- Tokenizer: `utf8-byte-v1`, fixed 256-token UTF-8 byte vocabulary.
- Context length: 128 tokens.
- Model width: 128.
- Attention heads: 4.
- Transformer blocks: 4.
- Feed-forward width: 4 x model width (512).
- Dropout: 0.1.
- Decoder-only causal self-attention.
- Pre-layer normalization within each block, GELU feed-forward network, learned absolute position embeddings.
- Input and output token embeddings are weight-tied.
- Weights are initialized randomly; there is no pretrained intelligence.

With the current default dimensions and tied token/output embeddings, the model has approximately 842,496 trainable parameters. This is a small experimental language model, not a production assistant.

Run the parameter/configuration report at any time:

```bash
python -m core.model_info --json
```

The report instantiates the current architecture, counts unique parameters (so the tied output embedding is counted once), and gives a component breakdown plus FP32/FP16-or-BF16 parameter-storage estimates. Those estimates cover parameter tensors only; they do not include gradients, optimizer state, activations or runtime workspaces. The report describes the randomly initialized architecture and does not claim that a trained checkpoint is capable.

The experimental `utf8-byte-bpe-v1` tokenizer is deliberately not wired into model training. The production vocabulary size and checkpoint format remain unchanged until the multilingual benchmark and a versioned migration are reviewed.

## Design principles

1. **Reproducibility:** every training run records the seed, model configuration, tokenizer identity, data provenance and evaluation metrics.
2. **No silent compatibility breaks:** tokenizer or model vocabulary changes require a new checkpoint format/version and explicit loader validation.
3. **Data quality before scale:** rights, provenance, privacy filtering, deduplication and held-out evaluation are mandatory.
4. **Language coverage:** Hindi (Devanagari), English and Hinglish (Romanized Hindi with English technical terms) must be evaluated separately as well as together.
5. **Measured changes:** one major variable should change per experiment; retain baselines for comparison.
6. **Honest capability claims:** lower token loss/perplexity is not proof of factuality, reasoning, instruction following or safe tool use.

## Proposed implementation sequence

### Stage A — Establish the baseline (current code)

- Preserve the current byte tokenizer and checkpoint format.
- Run deterministic smoke training on synthetic data in CI only.
- Completed: add a local synthetic benchmark for optimizer-update throughput, supervised tokens per second, greedy-generation latency, environment metadata and CUDA peak allocated memory. This measures runtime performance only, not answer quality.
- Still pending: record real-corpus training throughput and per-language validation loss on documented hardware.
- Keep synthetic CI data separate from real training data.

### Stage B — Data and evaluation readiness

Before training on real user-facing material:
- Build a rights-cleared corpus from sources whose licenses and permissions are explicitly confirmed.
- Keep raw sources local/private; commit only schemas, tests and synthetic fixtures.
- Maintain separate train, validation and test sets. The test set must not be used for training or tuning.
- Completed: local evaluation manifests validate rights, language/split labels, safe file paths, UTF-8, privacy patterns, exact duplicate documents and training-corpus whole-file fingerprints when available.
- Completed: deterministic byte-level next-token evaluation reports separate token-weighted metrics for Hindi, English and Hinglish validation/test groups; missing language/split groups are made explicit.
- Add task-specific evaluation sets for next-token modeling, prompt continuation, instruction response format, factual recall and refusal/safety behavior.
- Record corpus hashes and source identifiers without exposing private text.
- Manually review privacy-filter output; heuristic filters are not a guarantee that all personal data has been removed.

A tiny prompt collection is useful for tokenizer smoke tests, but is not enough to train a useful general-purpose language model.

### Stage C — Tokenizer decision gate

The byte tokenizer remains the production tokenizer until all of the following are true:
- Representative, rights-cleared Hindi, English and Hinglish corpora have been benchmarked.
- Exact encode/decode round trips pass for all evaluated text.
- Per-language compression and encoding throughput are reviewed, not just combined averages.
- The selected vocabulary size is justified by measured trade-offs.
- Checkpoint metadata and a backward-compatible migration/rollback plan are implemented and tested.

Tokenizer compression is a storage/sequence-length metric, not a language-understanding score. Do not change `ModelConfig.vocab_size`, training, generation or checkpoint loading as part of the benchmark-only stage.

### Stage D — Training-system hardening

Before a meaningful long run:
- Add explicit train/validation/test manifest support with leakage checks.
- Completed: add micro-batch gradient accumulation, average gradients across micro-batches, clip once per optimizer update, and preserve optimizer-step semantics; numerical parity tests are included. Mixed precision remains pending and requires separate numerical tests.
- Completed: resumable checkpoints preserve model, AdamW optimizer, Python/CPU/CUDA RNG state, completed step and best-validation metadata.
- Completed: best-validation inference checkpoint is kept separate from the latest resumable training checkpoint.
- Still pending: scheduler state when a learning-rate scheduler is introduced; no scheduler is currently used.
- Completed: record optimizer-update duration, cumulative optimizer-compute time, supervised target positions, derived training throughput and active learning rate; CUDA timing includes explicit synchronization. This excludes evaluation/checkpoint time and is not end-to-end job wall time.
- Completed: per-language validation/test losses are reported by the separate held-out evaluation manifest path.
- Still pending: early stopping or a documented fixed-budget policy.
- Make data-loader and sampling behavior deterministic where practical.
- Test corrupted, incompatible and incomplete checkpoints.

### Stage E — Model scaling experiments

Do not scale dimensions by guesswork. Establish a measured small-model baseline first. Then run controlled experiments varying one dimension at a time:
- Context length.
- Model width and attention heads.
- Number of decoder blocks.
- Feed-forward multiplier.
- Dropout and regularization.
- Tokenizer vocabulary only after Stage C.

Choose model size based on available compute, memory, tokens processed and held-out metrics. A larger randomly initialized model trained on insufficient data can perform worse than a smaller well-trained model.

### Stage F — Local inference, API and Studio UI

Only after training and evaluation are reliable:

- Completed: add a loopback-only local HTTP API for checkpoint status, prompt generation, training jobs, evaluation jobs and cancellation.
- Completed: add a responsive local Studio UI with Overview, Playground, Training and Evaluation screens, connected to the repository's existing model, trainer and evaluator.
- Completed: validate generation tensor type/shape/device, token-ID range, finite temperature and output-token limits; expose deterministic greedy decoding and optional seeded sampling.
- Completed: validate local data/manifest and artifact paths; reject traversal and symlink escapes; limit request sizes; execute subprocesses without a shell; reject non-local Host/Origin requests; bind only to `127.0.0.1`.
- Pending: formal versioning of the inference API contract, generation timeouts/rate limits, and production-grade authentication/multi-user access. The Studio is a developer workstation UI, not an authenticated application.
- A project-owned optimized CPU runtime is a later optimization, not a prerequisite for the first correct local inference interface.

## Acceptance criteria before calling it a question-answering model

The model should not be described as ready for normal user Q&A until:
- A representative training corpus has passed rights and privacy review.
- Training runs reproducibly and can resume after interruption.
- Held-out metrics improve over a random/untrained baseline.
- Hindi, English and Hinglish task evaluations meet documented minimum thresholds.
- Human-reviewed answers are coherent and follow simple instructions across all three language settings.
- Hallucination, unsafe output and refusal behavior have been evaluated and documented.
- A local inference interface passes functional, load and failure-mode tests.

Passing unit tests proves software behavior, not answer quality.

## Immediate next engineering tasks

1. Run the multilingual tokenizer benchmark on representative private data when the user supplies the completed corpus template.
2. Run the synthetic benchmark on the target hardware and preserve the report as a baseline; later compare it with real-corpus training measurements.
3. Add mixed precision only after numerical stability and resume behavior have separate tests.

Completed foundation items include the reproducible model-parameter/configuration report, resumable training checkpoints with optimizer and RNG state, generation input validation with deterministic decoding controls, held-out per-language validation/test evaluation, tested gradient accumulation with effective-batch reporting, and a synthetic training/generation performance benchmark.

Each task should be implemented on a feature branch, covered by tests, and reviewed through a pull request. Do not merge an unverified tokenizer migration or commit private prompt data.
