# Development-AI Studio — local UI guide

## Start the interface

From the repository root, install the project requirements and launch:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m core.server
```

Open **http://127.0.0.1:8765**. To select another local port:

```bash
python -m core.server --port 9000
```

The server accepts only the `127.0.0.1` bind address. It does not support `0.0.0.0`, LAN sharing or public deployment. Stop it with Ctrl+C.

## Screens

### Overview

Shows the active model configuration, unique trainable parameter count, tokenizer vocabulary, checkpoint readiness, recorded training steps and recent jobs. A checkpoint is only labelled trained when its training metadata records a positive completed-step count.

### Playground

Uses `artifacts/latest.pt` and the current UTF-8 byte tokenizer. It supports:
- Hindi, English and Hinglish text prompts.
- Greedy decoding or sampling.
- Temperature from 0.05 to 2.0.
- A maximum of 256 generated tokens per request.
- Optional reproducible sampling seed.
- CPU or CUDA where available.

The UI blocks generation when no checkpoint is present or the checkpoint is not verified as trained. The current model is a small from-scratch research model; a completed training run does not guarantee coherent responses.

### Training

The training file path is relative to `data/`, for example `processed/train.txt`. All input paths are resolved and checked to ensure they remain inside the project's local `data/` directory.

The screen controls optimizer steps, micro-batch size, gradient accumulation, context length, learning rate, seed, validation fraction, evaluation interval, evaluation iterations, early-stopping patience/minimum validation-loss improvement, and CPU/CUDA device. Early stopping is disabled when patience is `0`; when enabled, it stops after the configured number of consecutive evaluation checks without the required validation-loss improvement. Its state is preserved in resumable checkpoints. It starts the existing `core.train` implementation; it does not introduce a separate training algorithm.

The UI writes to:
- `artifacts/latest.pt`: best-validation inference checkpoint.
- `artifacts/latest.resume.pt`: resumable training state.
- `artifacts/latest.metrics.json`: metric history.

The resume option requires the paired best and resume checkpoint files. Keep the same corpus and compatible training settings when resuming. A running job can be stopped from the monitor panel. A stop request can leave a resumable checkpoint from the last configured save point; inspect the job log before resuming.

### Evaluation

The manifest path is relative to `data/`, for example `evaluation/manifest.json`. This launches the existing held-out evaluation CLI. The optional comparison mode evaluates the same held-out text with the trained model and a seeded random initialization of the same architecture.

Use validation results for experiments. Keep the test split untouched until final reporting. Byte-level cross-entropy and perplexity measure next-byte prediction, not instruction following, factuality or reasoning.

## Local API

The browser interface talks only to the local server:

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Local server health |
| GET | `/api/status` | Model, checkpoint and runtime information |
| GET | `/api/jobs` | Recent jobs |
| GET | `/api/jobs/{id}` | Job status, progress, log tail and evaluation result |
| POST | `/api/chat/generate` | Generate text from a trained checkpoint |
| POST | `/api/training` | Start a local training process |
| POST | `/api/evaluation` | Start a local held-out evaluation process |
| POST | `/api/jobs/{id}/stop` | Request cancellation of a running process |

JSON request bodies are size-limited. The local HTTP server handles at most 16 concurrent request workers by default; excess requests receive HTTP 503 and can be retried. Incomplete or idle socket I/O is bounded by a 15-second request timeout. At most one training/evaluation subprocess is active at a time. Commands are launched as argument arrays with `shell=False`; shell command strings are never constructed from browser input.

## Privacy and security boundaries

- The HTTP server binds to loopback only and rejects non-loopback Host/Origin values.
- Data and manifest paths are restricted to files under the repository's `data/` directory. Path traversal, absolute paths, unsupported extensions and symlinks escaping that directory are rejected.
- The browser UI uses no external JavaScript libraries, remote font downloads or hosted AI API.
- Prompts are sent to the local process only. Request bodies are not written to access logs. Job output is retained in local process memory and is exposed only through the loopback API.
- Training corpora, evaluation documents, prompts and model checkpoints are not uploaded or committed by the UI.
- This is a developer workstation interface, not an authenticated multi-user web application. Do not proxy it onto a network, expose its port publicly or run it on a shared untrusted machine.
- Privacy heuristics and local-only binding do not replace manual data-rights review or endpoint security.

## Verification

The server tests cover local path confinement, training/evaluation command construction, resume-pair requirements, one-job-at-a-time execution, progress parsing, cancellation, loopback binding and HTTP health/UI responses.

```bash
python -m compileall -q core tests
python -m unittest discover -s tests -v
```
