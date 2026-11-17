# Own-model development plan

## Independence definition
The architecture, training code, model weights and inference implementation are maintained by this project. We do not load pretrained weights, call hosted AI APIs, or require another model runtime. PyTorch is currently a numerical/training dependency; replacing it with a project-owned optimized CPU runtime is a later engineering milestone.

## Milestones
1. From-scratch byte-level decoder-only Transformer and CPU generation.
2. Reproducible training, held-out evaluation, best-checkpoint saving and automated quality checks.
3. **Data engineering:** source rights manifest, path validation, provenance hashes, Unicode normalization, conservative privacy screening, exact deduplication, quality filters and script statistics.
4. Tokenizer and model engineering, profiling and CPU memory optimization. Current substage: experimental byte-level BPE implementation and benchmark; production model remains on the 256-token byte tokenizer until explicit acceptance gates pass.
5. Project-owned inference runtime and stable local API.
6. Instruction tuning on owned/licensed data, memory, planning, decisions and tool adapters.
7. Safety, reliability, benchmarks and release engineering.\n8. **Local Studio UI:** model status, prompt playground, training controls, job monitoring and held-out evaluation.

## Evaluation and limitations
Keep train/validation data disjoint. Record seed, model configuration, data size, steps and metrics. Byte-token perplexity is tokenizer-specific and is not a direct measure of reasoning. This randomly initialized small model is not a pretrained assistant. Frontier-level capability requires extensive data, compute and evaluation; CPU is suitable for small experiments, not economical frontier-scale training.

## Data governance
Use data with appropriate rights. Exclude secrets, credentials and sensitive personal records. Keep datasets and checkpoints out of Git. Heuristic privacy checks are not complete PII protection; manual review is required.
