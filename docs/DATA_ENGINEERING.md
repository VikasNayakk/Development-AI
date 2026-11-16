# Data engineering contract

## Source manifest
Create `data/sources.json` (this file and all raw data are local and ignored by Git):
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
Paths are relative to the manifest directory and must resolve inside it. Only UTF-8 .txt and .md files are accepted. Each source requires a unique ID, a license/rights label and explicit rights confirmation. The flag is the user's assertion; software cannot independently establish copyright ownership or license validity.

## Preparation
```bash
python -m core.prepare_data --manifest data/sources.json --output-dir data/processed
``
The pipeline applies Unicode NFC normalization, line-ending/whitespace normalization, conservative sensitive-pattern rejection, minimum length and repeated-character checks, and exact normalized-document deduplication. It writes `train.txt` and `dataset_report.json`.

## Privacy limitations
Documents matching email-like, phone-like, private-key-marker or credential-assignment patterns are rejected as a whole. The report records only reason codes and pattern categories, not matched values. This is a conservative heuristic, **not** a complete PII detector: it can miss sensitive information and can produce false positives. Human review remains necessary before training.

## Deduplication and language statistics
Only exact full-document duplicates after normalization are removed. Near-duplicate detection is not implemented. Script counts (Latin, Devanagari and other alphabetic characters) are descriptive Unicode-script statistics, not reliable language identification.

## Integrity and provenance
The report records raw and normalized SHA-256 hashes, source ID, relative path, declared license, byte/character counts and script counts. It also records a hash of the assembled corpus. Hashes detect content changes; they do not prove ownership or authenticity.

## Safe use
Do not include credentials, private records, secrets or data without appropriate rights. Inspect the report and source material before training. The pipeline does not upload or transmit data.
