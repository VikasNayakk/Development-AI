# Held-out evaluation manifest

## Purpose

Use a separate local manifest to evaluate the model on rights-cleared validation and test data for Hindi, English and Hinglish. This mode does not change training data, checkpoints, the production tokenizer or the legacy single-file evaluation mode.

## Manifest format

Create a local JSON file such as `data/evaluation/manifest.json`:

```json
{
  "format_version": 1,
  "datasets": [
    {
      "dataset_id": "hindi-validation",
      "path": "hindi/validation.txt",
      "language": "hindi",
      "split": "validation",
      "rights_confirmed": true
    },
    {
      "dataset_id": "hindi-test",
      "path": "hindi/test.txt",
      "language": "hindi",
      "split": "test",
      "rights_confirmed": true
    },
    {
      "dataset_id": "english-validation",
      "path": "english/validation.txt",
      "language": "english",
      "split": "validation",
      "rights_confirmed": true
    },
    {
      "dataset_id": "english-test",
      "path": "english/test.txt",
      "language": "english",
      "split": "test",
      "rights_confirmed": true
    },
    {
      "dataset_id": "hinglish-validation",
      "path": "hinglish/validation.txt",
      "language": "hinglish",
      "split": "validation",
      "rights_confirmed": true
    },
    {
      "dataset_id": "hinglish-test",
      "path": "hinglish/test.txt",
      "language": "hinglish",
      "split": "test",
      "rights_confirmed": true
    }
  ]
}
```

Each path is relative to the manifest directory and must point to a local UTF-8 `.txt` or `.md` file. Split documents with a blank line. Each document is evaluated independently.

Allowed language values are exactly `hindi`, `english`, and `hinglish`. Allowed split values are exactly `validation` and `test`; a training split is not accepted. The manifest requires explicit `rights_confirmed: true`.

Include all six language/split combinations for a complete report. Partial manifests are allowed while a corpus is being prepared; the report lists missing combinations instead of silently treating them as zero.

## Run evaluation

```bash
python -m core.evaluate \
  --checkpoint artifacts/latest.pt \
  --manifest data/evaluation/manifest.json \
  --device cpu
```

The existing single-file command remains available:

```bash
python -m core.evaluate \
  --checkpoint artifacts/latest.pt \
  --data data/processed/train.txt \
  --device cpu
```

Use either `--manifest` or `--data`, not both.

## Metrics and protocol

For each dataset and each language/split group, the report includes:
- Number of documents and file bytes.
- Number of next-byte targets scored.
- Token-weighted mean byte cross-entropy loss and byte perplexity.
- Corpus and manifest SHA-256 fingerprints.

Each document is scored with windows of at most the model's `block_size`. Every next-byte target is scored exactly once. At window boundaries, the previous window's final target byte becomes the next window's first input byte; context is reset at the boundary. Documents never share context.

Language/split group loss is weighted by the number of scored target bytes, not by averaging dataset averages. Validation and test groups are kept separate. Compare results only using the same tokenizer, model configuration and evaluation protocol.

Byte loss/perplexity measures next-byte prediction only. It is not a measure of answer quality, factuality, reasoning, instruction following or safety.

## Leakage, privacy and rights safeguards

- Exact whole-file equality with the training corpus is rejected when the inference checkpoint contains the training corpus SHA-256. Older checkpoints without that fingerprint are explicitly identified in the report; use independently held-out sources in that case.
- Exact duplicate normalized documents across evaluation files are rejected to reduce validation/test leakage. This does not detect paraphrases or near-duplicates.
- Common email, phone, private-key and credential-like patterns are flagged using the same conservative checks used by local dataset preparation. These heuristics are not a complete personal-data detector; manually review source content.
- The manifest reports file names and hashes, not the actual prompt/document text.
- Keep the manifest, evaluation corpora and checkpoints local. Do not commit private evaluation prompts or raw corpus data to GitHub.
- Only include sources when the source rights/licence permit the intended evaluation use.
