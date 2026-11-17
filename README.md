# Development-AI

An independent, local-first AI research project. Its primary product goal is a project-owned AI brain that powers the browser extension and coordinates other connected automation tools—not merely a standalone chat/Q&A model.

The AI is intended to understand goals, discover tool capabilities, plan work, support decisions, dispatch permission-checked operations, verify results and explain outcomes. The browser extension is one adapter; other automation systems must connect through a shared, versioned tool contract. The extension/automation platform is a planned capability and is not yet implemented. See [docs/EXTENSION_AUTOMATION_ARCHITECTURE.md](docs/EXTENSION_AUTOMATION_ARCHITECTURE.md).

## Independence
- No required hosted AI API, pretrained model, or external model runtime.
- Model weights are initialized randomly and learned by this repository's training code.
- CPU is the baseline for small experiments and local inference.
- PyTorch is currently numerical/training infrastructure, not a source of pretrained intelligence. A project-owned optimized CPU inference runtime is a later milestone.

## Current capabilities
- Byte-level decoder-only Transformer, local training and CPU/CUDA generation.
- Experimental project-owned byte-level BPE tokenizer and corpus benchmark CLI; not yet connected to model training.
- Disjoint train/validation evaluation and best-validation checkpointing.
- Local dataset preparation with explicit rights manifest, provenance hashes, conservative privacy filtering, normalization and exact deduplication.

This is a small research model, not a capable general-purpose assistant. Initial output will be poor. Capability requires substantial high-quality data, compute and evaluation.

## Setup
Python 3.10+ is recommended:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare data
Create `data/sources.json`:
```json
{
  "sources": [
    {
      "source_id": "my-original-notes",
      "path": "raw/notes.txt",
      "license": "CC0-1.0",
      "rights_confirmed": true
    }
  ]
}
```
Place your UTF-8 text at `data/raw/notes.txt`, then run:
```bash
python -m core.prepare_data --manifest data/sources.json --output-dir data/processed
```
Review `data/processed/dataset_report.json` before training. Privacy checks are heuristic and cannot guarantee detection of all sensitive information. Exact duplicate detection does not detect near-duplicates.

## Inspect the model
```bash
python -m core.model_info --json
```

This reports the active model configuration, unique parameter count, component breakdown, tied-weight accounting and parameter-storage estimates. It reports the randomly initialized architecture; it does not load or measure a trained checkpoint.

## Train and evaluate
```bash
python -m core.train --data data/processed/train.txt --steps 1000 --batch-size 8 --gradient-accumulation-steps 4 --device cpu
python -m core.evaluate --checkpoint artifacts/latest.pt --data data/processed/train.txt --device cpu
python -m core.generate --checkpoint artifacts/latest.pt --prompt "Hello" --max-new-tokens 120 --device cpu
# Deterministic greedy decoding:
python -m core.generate --checkpoint artifacts/latest.pt --prompt "Hello" --max-new-tokens 120 --device cpu --greedy
# Reproducible sampled decoding (same seed, model and device):
python -m core.generate --checkpoint artifacts/latest.pt --prompt "Hello" --max-new-tokens 120 --device cpu --seed 42
```

Generation validates prompt token IDs and sampling settings. Sampling remains the default; `--greedy` selects deterministic argmax decoding.

Training saves a best-validation inference checkpoint plus a separate resumable checkpoint (including optimizer and RNG state). Use `--batch-size` for the micro-batch and `--gradient-accumulation-steps` to set how many micro-batches form one optimizer update. For example, 8 × 4 = effective batch 32; `--steps` still counts optimizer updates. To continue an interrupted run, increase the total `--steps` target and pass `--resume-from artifacts/latest.resume.pt`. Keep the corpus and training settings unchanged. See [docs/TRAINING_CHECKPOINTS.md](docs/TRAINING_CHECKPOINTS.md) for details.

## Held-out multilingual evaluation
Keep validation and test sources separate from training data. Create a local manifest following [docs/EVALUATION_MANIFEST.md](docs/EVALUATION_MANIFEST.md), then run:

```bash
python -m core.evaluate --checkpoint artifacts/latest.pt --manifest data/evaluation/manifest.json --device cpu
```

The report gives separate Hindi, English and Hinglish metrics for validation and test. Raw evaluation text and manifest files stay local and must not be committed.

## Performance benchmark
Run a local synthetic benchmark for optimizer-update throughput, supervised tokens per second, greedy-generation latency and CUDA peak allocated memory:

```bash
python -m core.benchmark_training --device cpu --warmup-steps 2 --steps 10 --batch-size 2 --gradient-accumulation-steps 1 --generation-tokens 16 --output artifacts/performance-baseline.json
```

The report records hardware/software context. Synthetic throughput is not a model-quality metric and may differ from real-corpus training. See [docs/TRAINING_BENCHMARK.md](docs/TRAINING_BENCHMARK.md) for metric definitions and limitations.

## Local web UI

Start the loopback-only Studio interface from the repository root:

```bash
python -m core.server
```

Open `http://127.0.0.1:8765` in your browser. The Studio includes:
- **Overview:** model configuration, parameter count, checkpoint status and recent local jobs.
- **Playground:** prompt generation from `artifacts/latest.pt`.
- **Training:** local training configuration, progress/log monitoring, stop control and resume support.
- **Evaluation:** held-out manifest runs with optional untrained-baseline comparison.

The server binds only to `127.0.0.1`; it deliberately does not expose the API to a LAN or public network. Training files and evaluation manifests must be inside the local `data/` directory. Checkpoints and logs remain local under `artifacts/` and in process memory. The Studio does not upload prompts or datasets.

The current model starts from random weights. Do not expect coherent answers until a suitable rights-cleared dataset has been prepared and the model has been trained and independently evaluated. See [docs/LOCAL_UI.md](docs/LOCAL_UI.md) for the complete interface guide and security scope.

## Quality checks
```bash
python -m compileall -q core tests
python -m unittest discover -s tests -v
```

## Data governance
Only use data with appropriate rights. Keep private datasets and checkpoints out of Git. Review all source material manually before training.
