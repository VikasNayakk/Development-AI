# Own-model development plan

## Product definition
Build a local-first AI system with its own model architecture, training pipeline, trained weights, inference implementation, memory and decision/tool layers. Compatible applications should connect to a local service. Cloud AI and third-party pretrained weights are not required dependencies.

## What “independent” means
- The model architecture and training/inference code are maintained in this repository.
- Weights are initialized randomly and produced by our training runs.
- No pretrained checkpoint, hosted model API, Ollama runtime, or third-party model is loaded.
- Standard numerical/ML libraries (initially PyTorch) may be used as implementation infrastructure. If the goal later includes removing framework dependencies too, that is a separate compiler/runtime project.

## Milestones
1. **Research baseline (implemented):** byte tokenizer, decoder-only Transformer, random initialization, local-corpus training, local checkpoint, CPU generation.
2. **Reproducibility:** train/validation split, deterministic experiment metadata, checkpoint versioning, loss curves and evaluation scripts.
3. **Data engineering:** corpus ingestion, deduplication, quality filters, language balance, licensing/provenance manifest and privacy filters.
4. **Model engineering:** tokenizer alternatives, context scaling, parameter sizing, profiling and memory optimization for target CPUs.
5. **Inference runtime:** optimized CPU kernels, streaming generation, batching, model loading safeguards and stable local HTTP API.
6. **Assistant capabilities:** instruction tuning on owned/licensed data, structured outputs, tool schemas, planning, memory retrieval and decision evaluation.
7. **Release readiness:** benchmark suite, safety tests, model card, license review, reproducible builds and signed releases.

## Honest capability limits
The first model is deliberately tiny and randomly initialized. It will not behave like a pretrained assistant. It must be trained on a substantial, high-quality corpus and evaluated. CPU-only training of a capable general-purpose language model is not realistic for an individual workstation; CPU is supported for small experiments and inference of appropriately sized models. Larger training may require GPUs and substantial compute.

## Data governance
Use only data the project owner has rights to use. Keep personal, confidential, credential, and sensitive records out of training corpora unless there is a documented lawful basis and appropriate safeguards. Keep datasets and model checkpoints out of Git by default.
