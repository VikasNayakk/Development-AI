# Development-AI

An independent, local-first AI research project. The long-term objective is to own the model architecture, training pipeline, trained weights, inference runtime, memory and tool-decision layers.

## Independence
- No required hosted AI API, pretrained model, or external model runtime.
- Model weights are initialized randomly and learned by this repository's training code.
- CPU is the baseline for small experiments and local inference.
- PyTorch is currently numerical/training infrastructure, not a source of pretrained intelligence. A project-owned optimized CPU inference runtime is a later milestone.

## Current capabilities
- Byte-level decoder-only Transformer, local training and CPU/CUDA generation.
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

## Train and evaluate
```bash
python -m core.train --data data/processed/train.txt --steps 1000 --device cpu
python -m core.evaluate --checkpoint artifacts/latest.pt --data data/processed/train.txt --device cpu
python -m core.generate --checkpoint artifacts/latest.pt --prompt "Hello" --max-new-tokens 120 --device cpu
```

## Quality checks
```bash
python -m compileall -q core tests
python -m unittest discover -s tests -v
```

## Data governance
Only use data with appropriate rights. Keep private datasets and checkpoints out of Git. Review all source material manually before training.
