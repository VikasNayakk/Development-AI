# Development-AI

An independent, local-first AI research project. The long-term objective is to own the model architecture, training pipeline, trained weights, inference runtime, memory and tool-decision layers.

## Independence requirements
- No required hosted AI API.
- No required pretrained model or external model runtime.
- Model weights are initialized randomly and learned by this repository's training code.
- CPU is the supported baseline for small experiments and local inference.
- Compatible applications will connect to a local API in a later milestone.

Standard numerical libraries are permitted as implementation infrastructure. They do not supply pretrained intelligence or model weights.

## Current milestone
A byte-level decoder-only Transformer, random initialization, local-corpus training, disjoint validation, best-validation checkpointing, metrics export, and CPU/CUDA text generation.

This is a small research model, not a useful general-purpose assistant yet. Early output will likely be incoherent. Capability requires substantial high-quality data, compute, evaluation, and iteration.

## Setup
Python 3.10+ is recommended. Install PyTorch for your platform, then:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Train from scratch
Place an owned/licensed UTF-8 text corpus at `data/train.txt`. Keep private datasets out of Git.
```bash
python -m core.train --data data/train.txt --steps 1000 --device cpu
```
The script splits the corpus sequentially into training and held-out validation portions, reports both losses, and saves the best validation checkpoint to `artifacts/latest.pt`. Metrics are written to `artifacts/latest.metrics.json`.

## Evaluate a checkpoint
```bash
python -m core.evaluate --checkpoint artifacts/latest.pt --data data/train.txt --device cpu
```
The reported perplexity is byte-token perplexity and should only be compared using the same tokenizer and evaluation corpus.

## Generate
```bash
python -m core.generate --checkpoint artifacts/latest.pt --prompt "Hello" --max-new-tokens 120 --device cpu
```

## Quality checks
```bash
python -m unittest discover -s tests -v
python -m compileall -q core tests
```

## Roadmap
1. From-scratch model and training/evaluation baseline (current).
2. Reproducible experiments, dataset manifests, and benchmark suite.
3. Tokenizer and data engineering improvements.
4. CPU inference optimization and stable local API.
5. Instruction tuning on owned/licensed data, memory, planning, decisions, and tool adapters.
6. Safety, reliability, documentation, and release engineering.

## Data and security
Use only data you have rights to use. Do not train on credentials, secrets, or sensitive personal data. Checkpoints and datasets are ignored by Git. Do not expose a future local API to a network without authentication, authorization, rate limits, and audit logs.
