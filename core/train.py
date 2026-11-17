"""Reproducible training with disjoint train/validation byte streams."""
import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import torch

from .config import ModelConfig
from .model import DevelopmentAIModel
from .tokenizer import ByteTokenizer


RESUME_FORMAT_VERSION = 1
RESUME_CHECKPOINT_TYPE = "development-ai-training-resume"


def _synchronize_device(device: torch.device) -> None:
    """Wait for queued CUDA work before measuring optimizer update duration."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_elapsed_seconds(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Resume checkpoint has invalid optimizer compute time.")
    try:
        is_finite = math.isfinite(value)
    except OverflowError:
        is_finite = False
    if not is_finite or value < 0:
        raise ValueError("Resume checkpoint has invalid optimizer compute time.")
    return float(value)


def _validate_non_negative_integer(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Resume checkpoint has invalid {label}.")
    return value


def _validate_non_negative_finite_float(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {label}; expected a finite non-negative number.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0:
        raise ValueError(f"Invalid {label}; expected a finite non-negative number.")
    return float(value)


def _validate_early_stopping_settings(settings: dict) -> tuple[int, float]:
    patience = settings.get("early_stopping_patience")
    if type(patience) is not int or patience < 0:
        raise ValueError("Training settings have an invalid early-stopping patience.")
    min_delta = _validate_non_negative_finite_float(
        settings.get("early_stopping_min_delta"), "early-stopping min_delta"
    )
    if patience == 0 and min_delta != 0.0:
        raise ValueError("early-stopping min_delta requires a positive patience.")
    return patience, min_delta


def _validate_telemetry_counters(
    *,
    optimizer_compute_seconds,
    optimizer_compute_tokens_processed,
    timed_optimizer_updates,
    effective_batch_size: int,
    block_size: int,
) -> tuple[float, int, int]:
    seconds = _validate_elapsed_seconds(optimizer_compute_seconds)
    tokens = _validate_non_negative_integer(
        optimizer_compute_tokens_processed, "optimizer compute token count"
    )
    updates = _validate_non_negative_integer(
        timed_optimizer_updates, "timed optimizer update count"
    )
    expected_tokens = updates * effective_batch_size * block_size
    if tokens != expected_tokens:
        raise ValueError("Resume checkpoint optimizer telemetry counters do not match.")
    if (updates == 0 and seconds != 0.0) or (updates > 0 and seconds <= 0.0):
        raise ValueError("Resume checkpoint optimizer timing and update counters do not match.")
    return seconds, tokens, updates


def split_corpus(data: torch.Tensor, validation_fraction: float, block_size: int):
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5.")
    split_at = int(len(data) * (1.0 - validation_fraction))
    train_data, validation_data = data[:split_at], data[split_at:]
    if len(train_data) <= block_size or len(validation_data) <= block_size:
        raise ValueError(
            "Corpus is too small for this block size and validation fraction; "
            "add data or reduce --block-size."
        )
    return train_data, validation_data


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    if data.ndim != 1 or len(data) <= block_size:
        raise ValueError("A one-dimensional token stream longer than block_size is required.")
    # high is exclusive; this includes the final valid start index.
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[int(i) : int(i) + block_size] for i in starts])
    y = torch.stack([data[int(i) + 1 : int(i) + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, validation_data, *, batch_size, block_size,
                  eval_iters, device):
    was_training = model.training
    model.eval()
    results = {}
    try:
        for name, data in (("train", train_data), ("validation", validation_data)):
            losses = []
            for _ in range(eval_iters):
                x, y = get_batch(data, batch_size, block_size, device)
                _, loss = model(x, y)
                losses.append(float(loss.item()))
            results[name] = sum(losses) / len(losses)
    finally:
        model.train(was_training)
    return results


def optimizer_update(
    model,
    optimizer,
    train_data: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    gradient_accumulation_steps: int,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> dict:
    """Perform one optimizer update using equally sized micro-batches.

    gradient_accumulation_steps controls the number of forward/backward
    micro-batches per optimizer update. Loss is averaged over micro-batches,
    gradients are clipped once after accumulation, and the optimizer steps once.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if type(gradient_accumulation_steps) is not int or gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer.")
    try:
        grad_norm_is_finite = (
            not isinstance(max_grad_norm, bool)
            and isinstance(max_grad_norm, (int, float))
            and math.isfinite(max_grad_norm)
        )
    except OverflowError:
        grad_norm_is_finite = False
    if not grad_norm_is_finite or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and greater than zero.")

    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    try:
        for _ in range(gradient_accumulation_steps):
            inputs, targets = get_batch(train_data, batch_size, block_size, device)
            _, loss = model(inputs, targets)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise ValueError("Training model must return a scalar tensor loss.")
            if not bool(torch.isfinite(loss.detach()).item()):
                raise FloatingPointError("Non-finite loss; optimizer update was cancelled.")
            (loss / gradient_accumulation_steps).backward()
            loss_total += float(loss.detach().item())

        # Clip once using the full accumulated gradient, not once per micro-batch.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float(max_grad_norm),
            error_if_nonfinite=True,
        )
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise

    return {
        "loss": loss_total / gradient_accumulation_steps,
        "gradient_norm": float(grad_norm.detach().item()),
        "micro_batches": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
    }


def _training_settings_match(saved_settings: dict, expected_settings: dict) -> bool:
    """Compare settings while allowing pre-accumulation checkpoints to resume at 1x."""
    if not isinstance(saved_settings, dict):
        return False
    normalized = dict(saved_settings)
    # Before gradient accumulation existed, every optimizer update used one batch.
    normalized.setdefault("gradient_accumulation_steps", 1)
    normalized.setdefault("effective_batch_size", normalized.get("batch_size"))
    normalized.setdefault("early_stopping_patience", 0)
    normalized.setdefault("early_stopping_min_delta", 0.0)
    return normalized == expected_settings


def atomic_save_checkpoint(payload: dict, destination: Path) -> None:
    """Write a checkpoint atomically so interruption cannot leave a partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".tmp", prefix=f".{destination.name}.",
            dir=destination.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _as_tuple_tree(value):
    """Restore tuple structure if a safe checkpoint loader returns lists."""
    if isinstance(value, list):
        return tuple(_as_tuple_tree(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_tuple_tree(item) for item in value)
    return value


def capture_rng_state(device: torch.device) -> dict:
    """Capture every RNG stream used by the training loop."""
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA training devices are supported.")
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, device: torch.device) -> None:
    """Restore RNG streams after model and optimizer state have been loaded."""
    expected_keys = {"python", "torch_cpu"}
    if device.type == "cuda":
        expected_keys.add("torch_cuda")
    if not isinstance(state, dict) or set(state) != expected_keys:
        raise ValueError("Resume checkpoint has an invalid RNG-state schema.")

    cpu_state = state["torch_cpu"]
    if not isinstance(cpu_state, torch.Tensor) or cpu_state.dtype != torch.uint8:
        raise ValueError("Resume checkpoint has an invalid CPU RNG state.")

    cuda_states = state.get("torch_cuda")
    if device.type == "cuda":
        if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved training run.")
        if any(
            not isinstance(item, torch.Tensor) or item.dtype != torch.uint8
            for item in cuda_states
        ):
            raise ValueError("Resume checkpoint has an invalid CUDA RNG state.")

    try:
        random.setstate(_as_tuple_tree(state["python"]))
        torch.set_rng_state(cpu_state.cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Resume checkpoint RNG state cannot be restored: {error}") from error


def build_resume_checkpoint(
    model,
    optimizer,
    *,
    config: ModelConfig,
    settings: dict,
    corpus_sha256: str,
    corpus_bytes: int,
    steps_completed: int,
    best_validation_loss: float,
    history: list,
    device: torch.device,
    optimizer_compute_seconds: float = 0.0,
    optimizer_compute_tokens_processed: int = 0,
    timed_optimizer_updates: int = 0,
    early_stopping_best_loss: float | None = None,
    early_stopping_bad_evaluations: int = 0,
) -> dict:
    if type(steps_completed) is not int or steps_completed < 1:
        raise ValueError("steps_completed must be a positive integer.")
    if not math.isfinite(best_validation_loss):
        raise ValueError("best_validation_loss must be finite before saving a resume checkpoint.")
    if not isinstance(settings, dict):
        raise ValueError("Training settings must be a dictionary.")
    effective_batch_size = settings.get("effective_batch_size")
    if type(effective_batch_size) is not int or effective_batch_size <= 0:
        raise ValueError("Training settings have an invalid effective batch size.")
    _validate_early_stopping_settings(settings)
    if early_stopping_best_loss is None:
        early_stopping_best_loss = best_validation_loss
    early_stopping_best_loss = _validate_non_negative_finite_float(
        early_stopping_best_loss, "early-stopping best validation loss"
    )
    early_stopping_bad_evaluations = _validate_non_negative_integer(
        early_stopping_bad_evaluations, "early-stopping bad-evaluation count"
    )
    (
        optimizer_compute_seconds,
        optimizer_compute_tokens_processed,
        timed_optimizer_updates,
    ) = _validate_telemetry_counters(
        optimizer_compute_seconds=optimizer_compute_seconds,
        optimizer_compute_tokens_processed=optimizer_compute_tokens_processed,
        timed_optimizer_updates=timed_optimizer_updates,
        effective_batch_size=effective_batch_size,
        block_size=config.block_size,
    )
    return {
        "format_version": RESUME_FORMAT_VERSION,
        "checkpoint_type": RESUME_CHECKPOINT_TYPE,
        "model_name": "Development-AI",
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "tokenizer": ByteTokenizer.name,
        "training": {
            "steps_completed": steps_completed,
            "best_validation_loss": float(best_validation_loss),
            "corpus_sha256": corpus_sha256,
            "corpus_bytes": corpus_bytes,
            "optimizer_compute_seconds": optimizer_compute_seconds,
            "optimizer_compute_tokens_processed": optimizer_compute_tokens_processed,
            "timed_optimizer_updates": timed_optimizer_updates,
            "early_stopping_best_loss": early_stopping_best_loss,
            "early_stopping_bad_evaluations": early_stopping_bad_evaluations,
            "settings": dict(settings),
        },
        "history": [dict(record) for record in history],
        "rng_state": capture_rng_state(device),
    }


def validate_resume_checkpoint(
    checkpoint: dict,
    *,
    config: ModelConfig,
    settings: dict,
    corpus_sha256: str,
) -> dict:
    """Validate resume metadata before applying any saved state to the live model."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Resume checkpoint must be a dictionary.")
    if (
        checkpoint.get("format_version") != RESUME_FORMAT_VERSION
        or checkpoint.get("checkpoint_type") != RESUME_CHECKPOINT_TYPE
    ):
        raise ValueError("Unsupported training-resume checkpoint format.")
    if checkpoint.get("model_name") != "Development-AI":
        raise ValueError("Resume checkpoint belongs to a different model.")
    if checkpoint.get("tokenizer") != ByteTokenizer.name:
        raise ValueError("Resume checkpoint uses an incompatible tokenizer.")
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("Model configuration differs from the saved training run.")

    training = checkpoint.get("training")
    if not isinstance(training, dict):
        raise ValueError("Resume checkpoint is missing training metadata.")
    if training.get("corpus_sha256") != corpus_sha256:
        raise ValueError("Training corpus changed; refusing an unsafe resume.")
    if not _training_settings_match(training.get("settings"), settings):
        raise ValueError(
            "Training settings differ from the saved run. Keep seed, batch size, "
            "learning rate, block size, gradient accumulation, validation fraction, "
            "evaluation settings and device configuration unchanged when resuming."
        )
    step = training.get("steps_completed")
    best_loss = training.get("best_validation_loss")
    if type(step) is not int or step < 1:
        raise ValueError("Resume checkpoint has an invalid completed-step count.")
    if not isinstance(best_loss, (int, float)) or isinstance(best_loss, bool) or not math.isfinite(best_loss):
        raise ValueError("Resume checkpoint has an invalid best validation loss.")
    _validate_early_stopping_settings(settings)
    _validate_non_negative_finite_float(
        training.get("early_stopping_best_loss", best_loss),
        "early-stopping best validation loss",
    )
    _validate_non_negative_integer(
        training.get("early_stopping_bad_evaluations", 0),
        "early-stopping bad-evaluation count",
    )
    _validate_telemetry_counters(
        optimizer_compute_seconds=training.get("optimizer_compute_seconds", 0.0),
        optimizer_compute_tokens_processed=training.get(
            "optimizer_compute_tokens_processed", 0
        ),
        timed_optimizer_updates=training.get("timed_optimizer_updates", 0),
        effective_batch_size=settings["effective_batch_size"],
        block_size=config.block_size,
    )
    if not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("Resume checkpoint is missing model weights.")
    if not isinstance(checkpoint.get("optimizer_state"), dict):
        raise ValueError("Resume checkpoint is missing optimizer state.")
    if not isinstance(checkpoint.get("history"), list):
        raise ValueError("Resume checkpoint has invalid metric history.")
    if not isinstance(checkpoint.get("rng_state"), dict):
        raise ValueError("Resume checkpoint is missing RNG state.")
    return training


def validate_best_checkpoint(
    checkpoint: dict,
    *,
    config: ModelConfig,
    settings: dict,
    corpus_sha256: str,
) -> float:
    """Confirm the inference checkpoint paired with a resume file is from this run."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Best checkpoint must be a dictionary.")
    if checkpoint.get("format_version") != 1 or checkpoint.get("tokenizer") != ByteTokenizer.name:
        raise ValueError("Best checkpoint has an unsupported format or tokenizer.")
    if checkpoint.get("model_name") != "Development-AI":
        raise ValueError("Best checkpoint belongs to a different model.")
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("Best checkpoint model configuration does not match.")
    training = checkpoint.get("training")
    if not isinstance(training, dict) or training.get("corpus_sha256") != corpus_sha256:
        raise ValueError("Best checkpoint does not match the current training corpus.")
    if not _training_settings_match(training.get("settings"), settings):
        raise ValueError("Best checkpoint training settings do not match.")
    best_loss = training.get("best_validation_loss")
    if not isinstance(best_loss, (int, float)) or not math.isfinite(best_loss):
        raise ValueError("Best checkpoint has an invalid validation loss.")
    return float(best_loss)


def _default_resume_path(output_path: Path) -> Path:
    suffix = output_path.suffix or ".pt"
    return output_path.with_name(f"{output_path.stem}.resume{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/train.txt")
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Total target step count, including steps already completed in a resumed run.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Micro-batch size per forward/backward pass.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1,
        help="Micro-batches per optimizer update; --steps still counts optimizer updates.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many consecutive validation checks without a min_delta improvement; 0 disables it.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                        help="Minimum absolute validation-loss decrease that resets early-stopping patience.")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="artifacts/latest.pt",
                        help="Best-validation inference checkpoint path.")
    parser.add_argument("--resume-from", default=None,
                        help="Resume from a training checkpoint created by this trainer.")
    parser.add_argument("--resume-output", default=None,
                        help="Latest resumable checkpoint path; defaults beside --output.")
    parser.add_argument("--resume-interval", type=int, default=100,
                        help="Save resumable state every N steps (and at the final step).")
    args = parser.parse_args()

    for name in (
        "steps", "batch_size", "gradient_accumulation_steps", "block_size",
        "eval_interval", "eval_iters", "resume_interval"
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        parser.error("--learning-rate must be finite and positive")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative")
    try:
        early_stopping_min_delta = _validate_non_negative_finite_float(
            args.early_stopping_min_delta, "early-stopping min_delta"
        )
    except ValueError as error:
        parser.error(str(error))
    if args.early_stopping_patience == 0 and early_stopping_min_delta != 0.0:
        parser.error("--early-stopping-min-delta requires --early-stopping-patience greater than zero")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    device = torch.device(args.device)
    output_path = Path(args.output)
    resume_from_path = Path(args.resume_from) if args.resume_from else None
    if args.resume_output:
        resume_output_path = Path(args.resume_output)
    elif resume_from_path is not None:
        resume_output_path = resume_from_path
    else:
        resume_output_path = _default_resume_path(output_path)

    if output_path.resolve() == resume_output_path.resolve():
        parser.error("--output and --resume-output must be different files.")
    if resume_from_path is not None:
        if not resume_from_path.is_file():
            parser.error(f"Resume checkpoint not found: {resume_from_path}")
        if resume_from_path.resolve() == output_path.resolve():
            parser.error("--resume-from must point to a training-resume checkpoint, not --output.")
        if (
            resume_output_path.resolve() != resume_from_path.resolve()
            and resume_output_path.exists()
        ):
            parser.error(f"Resume output already exists: {resume_output_path}")
    else:
        if output_path.exists():
            parser.error(f"Best checkpoint already exists: {output_path}. Use --resume-from or a new --output path.")
        if resume_output_path.exists():
            parser.error(
                f"Resume checkpoint already exists: {resume_output_path}. "
                "Use --resume-from or choose a new --output path."
            )

    corpus_path = Path(args.data)
    if not corpus_path.is_file():
        parser.error(f"Training file not found: {corpus_path}")

    tokenizer = ByteTokenizer()
    try:
        corpus_text = corpus_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        parser.error(f"Cannot read UTF-8 training file: {error}")
    encoded = tokenizer.encode(corpus_text)
    data = torch.tensor(encoded, dtype=torch.long)
    corpus_sha256 = hashlib.sha256(bytes(encoded)).hexdigest()
    try:
        train_data, validation_data = split_corpus(
            data, args.validation_fraction, args.block_size
        )
    except ValueError as error:
        parser.error(str(error))

    config = ModelConfig(block_size=args.block_size)
    settings = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "block_size": args.block_size,
        "validation_fraction": args.validation_fraction,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "device_type": device.type,
        "cuda_device_count": torch.cuda.device_count() if device.type == "cuda" else 0,
    }

    if resume_from_path is None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    model = DevelopmentAIModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_validation_loss = float("inf")
    history = []
    optimizer_compute_seconds = 0.0
    optimizer_compute_tokens_processed = 0
    timed_optimizer_updates = 0
    early_stopping_best_loss = float("inf")
    early_stopping_bad_evaluations = 0
    start_step = 1

    if resume_from_path is not None:
        try:
            resume_checkpoint = torch.load(
                resume_from_path, map_location="cpu", weights_only=True
            )
            resume_training = validate_resume_checkpoint(
                resume_checkpoint,
                config=config,
                settings=settings,
                corpus_sha256=corpus_sha256,
            )
            completed_steps = resume_training["steps_completed"]
            if completed_steps >= args.steps:
                raise ValueError(
                    f"Checkpoint already completed {completed_steps} steps; "
                    "--steps must be a larger total target."
                )
            if not output_path.is_file():
                raise ValueError(
                    f"Paired best checkpoint not found: {output_path}. "
                    "Keep the best and resume checkpoint files together."
                )
            best_checkpoint = torch.load(
                output_path, map_location="cpu", weights_only=True
            )
            best_file_loss = validate_best_checkpoint(
                best_checkpoint,
                config=config,
                settings=settings,
                corpus_sha256=corpus_sha256,
            )
            saved_best_loss = float(resume_training["best_validation_loss"])
            if best_file_loss > saved_best_loss + 1e-12:
                raise ValueError(
                    "Best checkpoint is older than the resume state; "
                    "restore the matching best checkpoint before resuming."
                )

            model.load_state_dict(resume_checkpoint["model_state"])
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            restore_rng_state(resume_checkpoint["rng_state"], device)
            best_validation_loss = min(saved_best_loss, best_file_loss)
            history = [dict(record) for record in resume_checkpoint["history"]]
            optimizer_compute_seconds = _validate_elapsed_seconds(
                resume_training.get("optimizer_compute_seconds", 0.0)
            )
            optimizer_compute_tokens_processed = _validate_non_negative_integer(
                resume_training.get("optimizer_compute_tokens_processed", 0),
                "optimizer compute token count",
            )
            timed_optimizer_updates = _validate_non_negative_integer(
                resume_training.get("timed_optimizer_updates", 0),
                "timed optimizer update count",
            )
            early_stopping_best_loss = _validate_non_negative_finite_float(
                resume_training.get("early_stopping_best_loss", saved_best_loss),
                "early-stopping best validation loss",
            )
            early_stopping_bad_evaluations = _validate_non_negative_integer(
                resume_training.get("early_stopping_bad_evaluations", 0),
                "early-stopping bad-evaluation count",
            )
            if (
                args.early_stopping_patience > 0
                and early_stopping_bad_evaluations >= args.early_stopping_patience
            ):
                print(
                    "Continuing a previously early-stopped run; "
                    "resetting the consecutive non-improvement counter."
                )
                early_stopping_bad_evaluations = 0
            start_step = completed_steps + 1
            print(f"Resuming from step {completed_steps}; target is {args.steps} steps.")
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
            parser.error(f"Cannot safely resume training: {error}")

    for step in range(start_step, args.steps + 1):
        early_stopping_triggered = False
        model.train()
        _synchronize_device(device)
        update_started = time.perf_counter()
        update_metrics = optimizer_update(
            model,
            optimizer,
            train_data,
            batch_size=args.batch_size,
            block_size=config.block_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device,
            max_grad_norm=1.0,
        )
        _synchronize_device(device)
        update_seconds = time.perf_counter() - update_started
        if not math.isfinite(update_seconds) or update_seconds <= 0.0:
            raise RuntimeError("Optimizer update timing was not positive and finite.")
        optimizer_compute_seconds += update_seconds
        timed_optimizer_updates += 1
        update_token_count = (
            args.batch_size * args.gradient_accumulation_steps * config.block_size
        )
        tokens_processed = step * update_token_count
        optimizer_compute_tokens_processed += update_token_count
        optimizer_compute_tokens_per_second = (
            optimizer_compute_tokens_processed / optimizer_compute_seconds
        )
        optimizer_updates_per_second = timed_optimizer_updates / optimizer_compute_seconds
        learning_rate = float(optimizer.param_groups[0]["lr"])

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            metrics = estimate_loss(
                model, train_data, validation_data,
                batch_size=args.batch_size, block_size=config.block_size,
                eval_iters=args.eval_iters, device=device
            )
            validation_loss = metrics["validation"]
            if validation_loss < early_stopping_best_loss - early_stopping_min_delta:
                early_stopping_best_loss = validation_loss
                early_stopping_bad_evaluations = 0
            elif args.early_stopping_patience > 0:
                early_stopping_bad_evaluations += 1
            else:
                early_stopping_best_loss = min(early_stopping_best_loss, validation_loss)
            early_stopping_triggered = (
                args.early_stopping_patience > 0
                and early_stopping_bad_evaluations >= args.early_stopping_patience
            )
            record = {
                "step": step,
                **metrics,
                "micro_batch_loss": update_metrics["loss"],
                "gradient_norm_before_clipping": update_metrics["gradient_norm"],
                "micro_batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "training_tokens_processed": tokens_processed,
                "optimizer_update_seconds": update_seconds,
                "optimizer_compute_seconds": optimizer_compute_seconds,
                "optimizer_compute_tokens_processed": optimizer_compute_tokens_processed,
                "timed_optimizer_updates": timed_optimizer_updates,
                "optimizer_compute_tokens_per_second": optimizer_compute_tokens_per_second,
                "optimizer_updates_per_second": optimizer_updates_per_second,
                "early_stopping_best_validation_loss": early_stopping_best_loss,
                "early_stopping_bad_evaluations": early_stopping_bad_evaluations,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "early_stopping_triggered": early_stopping_triggered,
            }
            history.append(record)
            print(
                f"step {step}/{args.steps} | "
                f"train_loss {metrics['train']:.4f} | "
                f"validation_loss {metrics['validation']:.4f} | "
                f"micro_batch_loss {update_metrics['loss']:.4f} | "
                f"gradient_norm {update_metrics['gradient_norm']:.4f} | "
                f"micro_batch {args.batch_size} x {args.gradient_accumulation_steps} "
                f"= effective_batch {args.batch_size * args.gradient_accumulation_steps} | "
                f"update_seconds {update_seconds:.4f} | "
                f"measured_tokens_per_second {optimizer_compute_tokens_per_second:.2f} | "
                f"learning_rate {learning_rate:.6g} | "
                f"early_stop_bad_checks {early_stopping_bad_evaluations}/"
                f"{args.early_stopping_patience if args.early_stopping_patience else 'disabled'}"
            )
            if early_stopping_triggered:
                print(
                    f"Early stopping triggered at step {step}: validation loss "
                    f"did not improve by at least {early_stopping_min_delta} "
                    f"for {args.early_stopping_patience} consecutive evaluation checks."
                )
            if metrics["validation"] < best_validation_loss:
                best_validation_loss = metrics["validation"]
                payload = {
                    "format_version": 1,
                    "model_name": "Development-AI",
                    "config": config.to_dict(),
                    "model_state": model.state_dict(),
                    "tokenizer": ByteTokenizer.name,
                    "training": {
                        "steps_completed": step,
                        **settings,
                        "settings": dict(settings),
                        "best_validation_loss": best_validation_loss,
                        "corpus_bytes": len(data),
                        "corpus_sha256": corpus_sha256,
                        "optimizer_compute_seconds": optimizer_compute_seconds,
                        "optimizer_compute_tokens_processed": optimizer_compute_tokens_processed,
                        "timed_optimizer_updates": timed_optimizer_updates,
                        "early_stopping_best_loss": early_stopping_best_loss,
                        "early_stopping_bad_evaluations": early_stopping_bad_evaluations,
                    },
                }
                atomic_save_checkpoint(payload, output_path)

        if (
            step % args.resume_interval == 0
            or step == args.steps
            or early_stopping_triggered
        ):
            resume_payload = build_resume_checkpoint(
                model,
                optimizer,
                config=config,
                settings=settings,
                corpus_sha256=corpus_sha256,
                corpus_bytes=len(data),
                steps_completed=step,
                best_validation_loss=best_validation_loss,
                history=history,
                device=device,
                optimizer_compute_seconds=optimizer_compute_seconds,
                optimizer_compute_tokens_processed=optimizer_compute_tokens_processed,
                timed_optimizer_updates=timed_optimizer_updates,
                early_stopping_best_loss=early_stopping_best_loss,
                early_stopping_bad_evaluations=early_stopping_bad_evaluations,
            )
            atomic_save_checkpoint(resume_payload, resume_output_path)

        if early_stopping_triggered:
            break

    history_path = output_path.with_suffix(".metrics.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint saved to: {output_path}")
    print(f"Resume checkpoint saved to: {resume_output_path}")
    print(f"Metrics saved to: {history_path}")
    print(f"Training tokens processed (all steps): {tokens_processed:,}")
    print(f"Optimizer-compute tokens measured: {optimizer_compute_tokens_processed:,}")
    print(f"Optimizer compute seconds: {optimizer_compute_seconds:.4f}")
    print(f"Measured optimizer tokens/second: {optimizer_compute_tokens_per_second:.2f}")
    print(
        "Training stop reason: "
        + (
            "validation-based early stopping"
            if history and history[-1].get("early_stopping_triggered")
            else "maximum optimizer-step target reached"
        )
    )


if __name__ == "__main__":
    main()
