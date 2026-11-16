# Own-model development plan

## Independence definition
The architecture, training code, model weights and inference implementation are maintained by this project. We do not load pretrained weights, call hosted AI APIs, or require Ollama/another model runtime. PyTorch is currently a numerical/training dependency; replacing it with a custom optimized CPU runtime is a later engineering milestone.

## Milestones
1. **From-scratch baseline:** byte tokenizer, decoder-only Transformer, random initialization, local training and CPU generation.
2. **Training quality (current):** sequential held-out validation split, train/validation loss estimation, best-validation checkpoint, metrics JSON, checkpoint format version and unit tests.
3. **Data engineering:** dataset provenance, rights/licensing, deduplication, quality filters, privacy controls and language balance.
4. **Model engineering:** tokenizer experiments, context scaling, parameter sizing, profiling and CPU memory optimization.
5. **Independent inference runtime:** optimized CPU kernels, streaming, stable API, model format and integrity checks.
6. **Assistant capabilities:** instruction tuning, structured output, memory retrieval, planning, tool schemas and decision benchmarks.
7. **Release readiness:** reproducible benchmarks, safety evaluation, model card, licensing review and signed releases.

## Evaluation rules
- Keep training and validation segments disjoint.
- Compare checkpoints on a fixed, documented held-out corpus.
- Track both training and validation loss; a widening gap can indicate overfitting.
- Treat byte-token perplexity as a tokenizer-specific metric, not a direct measure of reasoning or helpfulness.
- Never claim model quality from training loss alone.
- Record seed, model configuration, data size, training steps and evaluation results.

## Current limitations
This model is randomly initialized and intentionally small. It is not a pretrained assistant and will initially generate low-quality text. A capable general-purpose model requires large, carefully governed datasets, substantial compute, and extensive evaluation. CPU is practical for small experiments and appropriately sized inference, not economical for training a frontier-scale model.

## Data governance
Use only data with appropriate rights. Exclude secrets, credentials and sensitive personal records. Keep data and checkpoints out of Git by default.
