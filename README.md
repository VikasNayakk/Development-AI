# Development-AI

An independent, local-first AI research and development project. The long-term goal is to build and train our own model architecture, own model weights, and own inference/runtime layer so compatible tools can connect to a local machine through a documented API.

## Non-negotiable design goals
- No required cloud AI API.
- No required third-party pretrained model or model runtime.
- Model weights must be produced by our own training pipeline from randomly initialized parameters.
- Local CPU inference is a supported baseline; performance depends on hardware and model size.
- External tools connect through a local, documented interface.
- Training data is supplied and controlled by the project owner.

## Current milestone: from-scratch model foundation
The `core/` directory contains:
- A byte-level tokenizer (fixed 256-token vocabulary).
- A decoder-only causal Transformer implemented in PyTorch.
- A training script that initializes weights randomly and trains only on a local text file.
- A CPU-capable inference script that loads this project's own checkpoint.
- Basic unit tests.

This is a small research prototype, not a capable general-purpose assistant yet. A byte-level model trained on a small private corpus will initially produce weak and often incoherent text. Capability must be earned through data quality, compute, training, evaluation, and iterative architecture work.

## Requirements
- Python 3.10+
- PyTorch (CPU build is sufficient to start)

Install:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Train from scratch
1. Put your own UTF-8 plain-text training corpus at `data/train.txt` (do not commit private or sensitive data).
2. Run:
```bash
python -m core.train --data data/train.txt --steps 1000 --device cpu
```
3. The training process creates a checkpoint under `artifacts/` (ignored by Git).

## Generate text
```bash
python -m core.generate --checkpoint artifacts/latest.pt --prompt "Hello" --max-new-tokens 120 --device cpu
```

The checkpoint must be created by this repository's own training code. No pretrained weights are downloaded or loaded.

## Important limits
- This milestone is a minimal language-model research baseline, not a complete reasoning/decision agent.
- CPU training may be slow. Start with a small corpus and modest step count.
- The current model has no tool execution, long-term memory, planning, safety policy, or external-tool API yet.
- Never train on secrets, credentials, private third-party data, or data without appropriate rights.
- Do not expose future local APIs to a network until authentication, authorization, request limits, and audit logging are implemented.

## Roadmap
1. From-scratch model + CPU inference foundation (current).
2. Reproducible training, checkpoint metadata, evaluation and experiment tracking.
3. Tokenizer/data pipeline improvements and larger context handling.
4. Local inference service with authentication and stable API contract.
5. Memory, planning, decision policies, and tool adapter framework.
6. Curated datasets, instruction tuning, safety evaluation, and model improvements.
