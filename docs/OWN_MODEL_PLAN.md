# Own-model development plan

## Product purpose

Development-AI is being built as the central decision-support and coordination brain for the user's browser extension and other connected automation tools. It is not primarily a standalone chat/Q&A application. It must understand requests, discover available tool capabilities, plan multi-step work, help evaluate decisions, invoke approved operations through adapters, verify outcomes and report failures or incomplete work. The browser extension is one client/adapter; additional automation tools must use the same versioned tool contract.

The detailed system boundaries, request lifecycle, security requirements and integration acceptance gates are maintained in [EXTENSION_AUTOMATION_ARCHITECTURE.md](EXTENSION_AUTOMATION_ARCHITECTURE.md).

## Independence definition

The architecture, training code, model weights and inference implementation are maintained by this project. We do not load pretrained weights, call hosted AI APIs, or require another model runtime. PyTorch is currently a numerical/training dependency; replacing it with a project-owned optimized CPU runtime is a later engineering milestone.

## Milestones and current status

1. **Complete — From-scratch model foundation:** byte-level decoder-only Transformer and CPU generation.
2. **Complete — Training and evaluation foundation:** reproducible training, held-out evaluation, best-checkpoint saving and automated quality checks.
3. **Complete — Data engineering foundation:** source rights manifest, path validation, provenance hashes, Unicode normalization, conservative privacy screening, exact deduplication, quality filters and script statistics.
4. **Partial — Tokenizer/model engineering and profiling:** experimental project-owned byte-level BPE implementation and benchmark; production model remains on the 256-token byte tokenizer until representative multilingual corpora, measured trade-offs and migration/rollback gates are accepted. Representative corpus benchmarking and CPU memory/performance optimization remain open.
5. **Partial — Inference runtime and local API:** local generation API and Studio integration exist. Synthetic inference latency benchmarking is now available. Real target-hardware baselines, KV-cache correctness/optimization, a project-owned optimized runtime and formal API versioning remain open.
6. **Pending — Model intelligence:** instruction tuning on owned/licensed data, reliable task understanding, structured planning and decision quality, scoped working memory and durable user-approved memory, plus dedicated Hindi/English/Hinglish task evaluations.
7. **Partial — Extension and automation agent platform:** initial versioned tool manifests/contracts, strict schema validation, explicit registry, permission checks, exact single-use approval grants and a guarded in-process dispatcher are implemented with fake-adapter tests. Persistent task state, approval storage, audit events, authenticated extension transport, browser extension adapter, other automation adapters and verified multi-tool execution remain open. This is a core product milestone, not an optional add-on.
8. **Partial — Safety, reliability and release engineering:** local API hardening, bounded request/generation concurrency and benchmark foundations exist. End-to-end tool safety, adversarial prompt-injection evaluation, permission-boundary tests, recovery tests and release acceptance remain open.
9. **Partial — Local Studio and extension experience:** the local Studio provides model status, prompt playground, training controls, job monitoring and held-out evaluation. Visual acceptance and the user-facing extension workflow remain unverified/not implemented.

The current overall meter is **3 of 9 major milestones complete (33.3%)**. The denominator has been expanded from the earlier eight-milestone plan to separately track the now-explicit extension/automation platform. This changes the scope measurement; it does not mean previously completed work was lost. Partial milestones do not count as complete.

## Evaluation and limitations

Keep train/validation data disjoint. Record seed, model configuration, data size, steps and metrics. Byte-token perplexity is tokenizer-specific and is not a direct measure of reasoning. This randomly initialized small model is not a pretrained assistant. It must not be described as ready for normal Q&A or autonomous automation until representative training, held-out task evaluation, human review, safety tests and real adapter integration acceptance gates pass.

A passing software test suite establishes only the tested software behavior. Synthetic inference benchmarks measure runtime performance, not language understanding, decision quality or successful real-world automation.

## Data governance

Use data with appropriate rights. Exclude secrets, credentials and sensitive personal records. Keep datasets, prompts, checkpoints, cookies, access tokens and private automation outputs out of Git. Heuristic privacy checks are not complete PII protection; manual review is required. Conversation/history training must be a separate explicit data-governance choice, never a silent side effect of memory.

## Engineering workflow

Each task should be implemented on a feature branch, covered by focused tests and reviewed through a pull request. Merge only when the latest PR head has successful required CI and the head SHA is verified. Do not merge an unverified tokenizer migration, model change or automation permission change. For integrations, use fake adapters first, then real adapters under explicit permissions and end-to-end tests.
