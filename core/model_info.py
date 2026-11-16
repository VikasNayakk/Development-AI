"""Inspect the project-owned model configuration and unique parameter count."""
import argparse
import json

from .config import ModelConfig
from .model import DevelopmentAIModel
from .tokenizer import ByteTokenizer


def _parameter_group(name: str) -> str:
    if name.startswith("token_embedding."):
        return "token_embedding"
    if name.startswith("position_embedding."):
        return "position_embedding"
    if ".attention." in name:
        return "attention"
    if ".feed_forward." in name:
        return "feed_forward"
    if "norm" in name:
        return "normalization"
    return "other"


def build_model_report(config: ModelConfig | None = None) -> dict:
    """Build a parameter report from the instantiated architecture, counting tied weights once."""
    selected_config = config or ModelConfig()
    model = DevelopmentAIModel(selected_config)

    groups = {
        "token_embedding": 0,
        "position_embedding": 0,
        "attention": 0,
        "feed_forward": 0,
        "normalization": 0,
        "other": 0,
    }
    total_parameters = 0
    trainable_parameters = 0

    # Module.named_parameters() removes duplicate objects by default. This is
    # important because language_head.weight is tied to token_embedding.weight.
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        if parameter.requires_grad:
            trainable_parameters += count
        groups[_parameter_group(name)] += count

    if sum(groups.values()) != total_parameters:
        raise RuntimeError("Parameter group totals do not match the unique parameter count.")

    tied_output_embedding = model.language_head.weight is model.token_embedding.weight
    return {
        "report_version": 1,
        "model_name": "Development-AI",
        "architecture": "decoder-only causal Transformer",
        "training_status": "untrained_initialization",
        "tokenizer": {
            "name": ByteTokenizer.name,
            "vocab_size": ByteTokenizer.vocab_size,
        },
        "config": selected_config.to_dict(),
        "parameters": {
            "unique_total": total_parameters,
            "trainable": trainable_parameters,
            "non_trainable": total_parameters - trainable_parameters,
            "groups": groups,
            "storage_estimate_bytes": {
                "fp32": total_parameters * 4,
                "fp16_or_bf16": total_parameters * 2,
            },
            "storage_estimate_scope": (
                "Parameter tensors only; excludes gradients, optimizer state, "
                "activations, temporary workspaces and runtime overhead."
            ),
        },
        "weight_tying": {
            "enabled": tied_output_embedding,
            "output_weight": "language_head.weight",
            "shared_with": "token_embedding.weight" if tied_output_embedding else None,
            "counted_once": tied_output_embedding,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args()

    config = ModelConfig(
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    try:
        report = build_model_report(config)
    except ValueError as error:
        parser.error(str(error))

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return

    parameters = report["parameters"]
    print(f"{report['model_name']} — {report['architecture']}")
    print(f"Status: {report['training_status']}")
    print(f"Tokenizer: {report['tokenizer']['name']} ({report['tokenizer']['vocab_size']} tokens)")
    print(f"Configuration: {json.dumps(report['config'], sort_keys=True)}")
    print(f"Unique parameters: {parameters['unique_total']:,}")
    print(f"Trainable parameters: {parameters['trainable']:,}")
    print("Parameter groups:")
    for name, count in parameters["groups"].items():
        print(f"  {name}: {count:,}")
    print(f"FP32 parameter storage: {parameters['storage_estimate_bytes']['fp32']:,} bytes")
    print(f"Weight tying enabled: {report['weight_tying']['enabled']}")
    print("Note: storage estimate excludes gradients, optimizer state and activations.")


if __name__ == "__main__":
    main()
