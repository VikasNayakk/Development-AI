"""Loopback-only local web UI and API for Development-AI."""
from __future__ import annotations

import argparse
import json
import math
import re
import secrets
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

import torch

from .evaluate import _load_inference_model
from .model_info import build_model_report
from .tokenizer import ByteTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_ROOT = PROJECT_ROOT / "ui"
MAX_REQUEST_BYTES = 64 * 1024
MAX_PROMPT_CHARS = 12_000
MAX_GENERATION_TOKENS = 256
MAX_LOG_LINES = 400
MAX_CAPTURED_OUTPUT = 8 * 1024 * 1024
_GENERATION_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bounded_int(payload: dict, key: str, default: int, minimum: int, maximum: int) -> int:
    value = payload.get(key, default)
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{key} must be an integer between {minimum} and {maximum}.")
    return value


def _bounded_float(payload: dict, key: str, default: float, minimum: float, maximum: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number between {minimum} and {maximum}.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < minimum or value > maximum:
        raise ValueError(f"{key} must be a finite number between {minimum} and {maximum}.")
    return float(value)


def _validate_device(value: object) -> str:
    if not isinstance(value, str) or value not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but CUDA is not available on this computer.")
    return str(value)


def _resolve_data_file(root: Path, relative_path: object, suffixes: set[str]) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("A relative file path inside the data directory is required.")
    windows_path = PureWindowsPath(relative_path)
    supplied = Path(relative_path)
    if supplied.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError("Absolute paths are not allowed; choose a file inside the data directory.")
    declared_data_root = root / "data"
    if declared_data_root.is_symlink():
        raise ValueError("The project's data directory cannot be a symlink.")
    data_root = declared_data_root.resolve()
    try:
        data_root.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("The project's data directory must remain inside the repository.") from error
    candidate = (data_root / supplied).resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError as error:
        raise ValueError("The selected file must remain inside the project's data directory.") from error
    if candidate.suffix.lower() not in suffixes:
        allowed = ", ".join(sorted(suffixes))
        raise ValueError(f"File extension must be one of: {allowed}.")
    if not candidate.is_file():
        raise ValueError(f"Selected data file does not exist: {relative_path}")
    return candidate


def _resolve_artifact_file(root: Path, filename: str) -> Path:
    if filename not in {"latest.pt", "latest.resume.pt", "latest.metrics.json"}:
        raise ValueError("Unsupported local artifact path.")
    artifacts_root = root / "artifacts"
    if artifacts_root.is_symlink():
        raise ValueError("The artifacts directory cannot be a symlink.")
    if artifacts_root.exists() and not artifacts_root.is_dir():
        raise ValueError("The artifacts path must be a directory.")
    candidate = artifacts_root / filename
    if candidate.is_symlink():
        raise ValueError("Checkpoint and artifact files cannot be symlinks.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Artifacts must remain inside the project directory.") from error
    return candidate


def build_training_command(root: Path, payload: dict) -> list[str]:
    data_path = _resolve_data_file(root, payload.get("data_path", "processed/train.txt"), {".txt"})
    steps = _bounded_int(payload, "steps", 1000, 1, 10_000_000)
    batch_size = _bounded_int(payload, "batch_size", 8, 1, 1024)
    accumulation = _bounded_int(payload, "gradient_accumulation_steps", 1, 1, 1024)
    block_size = _bounded_int(payload, "block_size", 128, 2, 4096)
    eval_interval = _bounded_int(payload, "eval_interval", 100, 1, 10_000_000)
    eval_iters = _bounded_int(payload, "eval_iters", 20, 1, 100_000)
    resume_interval = _bounded_int(payload, "resume_interval", 100, 1, 10_000_000)
    seed = _bounded_int(payload, "seed", 42, 0, (2**64) - 1)
    learning_rate = _bounded_float(payload, "learning_rate", 3e-4, 1e-8, 1.0)
    validation_fraction = _bounded_float(payload, "validation_fraction", 0.1, 0.001, 0.499999)
    device = _validate_device(payload.get("device", "cpu"))

    output_path = _resolve_artifact_file(root, "latest.pt")
    resume_path = _resolve_artifact_file(root, "latest.resume.pt")
    resume = payload.get("resume", False)
    if not isinstance(resume, bool):
        raise ValueError("resume must be true or false.")
    if resume:
        if not output_path.is_file() or not resume_path.is_file():
            raise ValueError("Resume requires both artifacts/latest.pt and artifacts/latest.resume.pt.")
    elif output_path.exists() or resume_path.exists():
        raise ValueError(
            "A checkpoint already exists. Enable 'Resume previous run' or move the old artifacts first."
        )

    command = [
        sys.executable, "-u", "-m", "core.train",
        "--data", str(data_path),
        "--steps", str(steps),
        "--batch-size", str(batch_size),
        "--gradient-accumulation-steps", str(accumulation),
        "--learning-rate", str(learning_rate),
        "--block-size", str(block_size),
        "--eval-interval", str(eval_interval),
        "--eval-iters", str(eval_iters),
        "--resume-interval", str(resume_interval),
        "--validation-fraction", str(validation_fraction),
        "--resume-interval", str(resume_interval),
        "--device", device,
        "--seed", str(seed),
        "--output", str(output_path),
    ]
    if resume:
        command.extend(["--resume-from", str(resume_path)])
    return command


def build_evaluation_command(root: Path, payload: dict) -> list[str]:
    manifest = _resolve_data_file(
        root, payload.get("manifest_path", "evaluation/manifest.json"), {".json"}
    )
    checkpoint = _resolve_artifact_file(root, "latest.pt")
    if not checkpoint.is_file():
        raise ValueError("No inference checkpoint found. Train the model before evaluation.")
    device = _validate_device(payload.get("device", "cpu"))
    command = [
        sys.executable, "-u", "-m", "core.evaluate",
        "--checkpoint", str(checkpoint),
        "--manifest", str(manifest),
        "--device", device,
    ]
    compare = payload.get("compare_untrained", False)
    if not isinstance(compare, bool):
        raise ValueError("compare_untrained must be true or false.")
    if compare:
        seed = _bounded_int(payload, "baseline_seed", 42, 0, (2**64) - 1)
        command.extend(["--compare-untrained", "--baseline-seed", str(seed)])
    return command


class LocalJobManager:
    """Runs one local training or evaluation process at a time and retains safe logs."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._active_job_id: str | None = None

    def start(self, kind: str, command: list[str]) -> dict:
        with self._lock:
            if self._active_job_id is not None:
                active = self._jobs.get(self._active_job_id, {})
                if active.get("status") in {"queued", "running", "stopping"}:
                    raise RuntimeError("Another training or evaluation job is already running.")
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "kind": kind,
                "status": "queued",
                "created_at": _utc_now(),
                "finished_at": None,
                "step": None,
                "target_steps": None,
                "return_code": None,
                "logs": [],
                "result": None,
                "error": None,
                "_command": list(command),
                "_process": None,
                "_stop_requested": False,
            }
            self._jobs[job_id] = job
            self._active_job_id = job_id
            worker = threading.Thread(target=self._run, args=(job_id,), daemon=True)
            worker.start()
            return self._public_job(job)

    def _append_log(self, job: dict, line: str) -> None:
        line = line.rstrip("\r\n")
        if not line:
            return
        job["logs"].append(line[:2500])
        if len(job["logs"]) > MAX_LOG_LINES:
            del job["logs"][:-MAX_LOG_LINES]
        match = re.search(r"step\s+(\d+)\s*/\s*(\d+)", line)
        if match:
            job["step"] = int(match.group(1))
            job["target_steps"] = int(match.group(2))

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job["_stop_requested"]:
                job["status"] = "cancelled"
                job["finished_at"] = _utc_now()
                self._active_job_id = None
                return
            job["status"] = "running"

        captured: list[str] = []
        try:
            process = subprocess.Popen(
                job["_command"],
                cwd=self.root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
            )
            with self._lock:
                job["_process"] = process
                stop_requested = job["_stop_requested"]
            if stop_requested:
                process.terminate()

            assert process.stdout is not None
            for line in process.stdout:
                with self._lock:
                    self._append_log(job, line)
                if sum(len(part) for part in captured) < MAX_CAPTURED_OUTPUT:
                    captured.append(line[:8000])
            return_code = process.wait()
            output = "".join(captured)
            with self._lock:
                job["return_code"] = return_code
                if job["_stop_requested"]:
                    job["status"] = "cancelled"
                elif return_code == 0:
                    job["status"] = "succeeded"
                    if job["kind"] == "evaluation":
                        try:
                            job["result"] = json.loads(output)
                        except json.JSONDecodeError:
                            job["error"] = "Evaluation completed but returned invalid JSON."
                            job["status"] = "failed"
                else:
                    job["status"] = "failed"
                    job["error"] = f"Process exited with code {return_code}."
        except OSError as error:
            with self._lock:
                job["status"] = "failed"
                job["error"] = f"Could not start local process: {error}"
        finally:
            with self._lock:
                job["finished_at"] = _utc_now()
                job["_process"] = None
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def stop(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Job not found.")
            if job["status"] not in {"queued", "running", "stopping"}:
                return self._public_job(job)
            job["_stop_requested"] = True
            process = job.get("_process")
            if process is None:
                job["status"] = "stopping"
            else:
                job["status"] = "stopping"
        if process is not None and process.poll() is None:
            process.terminate()
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Job not found.")
            return self._public_job(job)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item["created_at"], reverse=True)
            summary_keys = (
                "id", "kind", "status", "created_at", "finished_at",
                "step", "target_steps", "return_code", "error",
            )
            return [
                {key: job[key] for key in summary_keys}
                for job in jobs[:20]
            ]

    @staticmethod
    def _public_job(job: dict) -> dict:
        return {
            key: job[key]
            for key in (
                "id", "kind", "status", "created_at", "finished_at",
                "step", "target_steps", "return_code", "logs", "result", "error",
            )
        }


def build_status(root: Path, jobs: LocalJobManager) -> dict:
    report = build_model_report()
    checkpoint_path = _resolve_artifact_file(root, "latest.pt")
    resume_path = _resolve_artifact_file(root, "latest.resume.pt")
    checkpoint_info = {
        "exists": checkpoint_path.is_file(),
        "resume_exists": resume_path.is_file(),
        "trained": False,
        "steps_completed": None,
        "best_validation_loss": None,
        "status": "missing",
    }
    if checkpoint_path.is_file():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            valid_structure = (
                isinstance(checkpoint, dict)
                and checkpoint.get("format_version") == 1
                and checkpoint.get("tokenizer") == ByteTokenizer.name
                and isinstance(checkpoint.get("config"), dict)
                and isinstance(checkpoint.get("model_state"), dict)
            )
            if not valid_structure:
                checkpoint_info["status"] = "invalid"
            else:
                training = checkpoint.get("training", {})
                step = training.get("steps_completed") if isinstance(training, dict) else None
                best_loss = (
                    training.get("best_validation_loss")
                    if isinstance(training, dict) else None
                )
                valid_best_loss = (
                    type(best_loss) in {int, float}
                    and math.isfinite(best_loss)
                )
                checkpoint_info.update({
                    "trained": type(step) is int and step > 0,
                    "steps_completed": step if type(step) is int else None,
                    "best_validation_loss": float(best_loss) if valid_best_loss else None,
                    "status": "ready",
                })
        except (OSError, RuntimeError, TypeError, ValueError):
            checkpoint_info["status"] = "invalid"
    return {
        "project": "Development-AI",
        "server_time": _utc_now(),
        "model": report,
        "checkpoint": checkpoint_info,
        "runtime": {
            "default_device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": torch.cuda.is_available(),
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
        },
        "jobs": jobs.list_jobs(),
    }


def _generate_response(root: Path, payload: dict) -> dict:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Enter a prompt before generating.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must be at most {MAX_PROMPT_CHARS} characters.")
    max_new_tokens = _bounded_int(payload, "max_new_tokens", 120, 1, MAX_GENERATION_TOKENS)
    temperature = _bounded_float(payload, "temperature", 0.8, 0.05, 2.0)
    greedy = payload.get("greedy", False)
    if not isinstance(greedy, bool):
        raise ValueError("greedy must be true or false.")
    seed = payload.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= (2**64) - 1):
        raise ValueError("seed must be null or an integer between 0 and 2^64 - 1.")
    device_name = _validate_device(payload.get("device", "cpu"))
    checkpoint_path = _resolve_artifact_file(root, "latest.pt")
    if not checkpoint_path.is_file():
        raise ValueError("No model checkpoint is available. Complete a training run first.")

    device = torch.device(device_name)
    with _GENERATION_LOCK:
        checkpoint, model = _load_inference_model(checkpoint_path, device)
        if not isinstance(checkpoint, dict):
            raise ValueError("Model checkpoint is invalid.")
        token_ids = ByteTokenizer().encode(prompt)
        if not token_ids:
            raise ValueError("Prompt contains no encodable text.")
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed if seed is not None else secrets.randbits(64))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed if seed is not None else secrets.randbits(64))
            output = model.generate(
                input_ids,
                max_new_tokens,
                temperature,
                do_sample=not greedy,
            )
        return {
            "text": ByteTokenizer().decode(output[0].tolist()),
            "prompt_characters": len(prompt),
            "generated_tokens": max_new_tokens,
            "decoding": "greedy" if greedy else "sampling",
            "seed": seed,
            "checkpoint_trained": bool(
                isinstance(checkpoint.get("training"), dict)
                and type(checkpoint["training"].get("steps_completed")) is int
                and checkpoint["training"]["steps_completed"] > 0
            ),
        }


def create_server(host: str = "127.0.0.1", port: int = 8765, root: Path = PROJECT_ROOT):
    if host != "127.0.0.1":
        raise ValueError("Development-AI UI is loopback-only; host must be 127.0.0.1.")
    root = Path(root).resolve()
    manager = LocalJobManager(root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "DevelopmentAI-LocalUI/1"

        def log_message(self, format_string, *args):
            # Avoid logging request bodies, prompt text or local dataset paths.
            super().log_message(format_string, *args)

        def _send_json(self, status: int, payload: dict):
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(encoded)

        def _check_local_request(self) -> bool:
            host_header = self.headers.get("Host", "")
            parsed_host = urlsplit("//" + host_header)
            if parsed_host.hostname not in {"127.0.0.1", "localhost"}:
                self._send_json(403, {"error": "Local requests only."})
                return False
            origin = self.headers.get("Origin")
            if origin:
                parsed_origin = urlsplit(origin)
                try:
                    origin_port = parsed_origin.port
                except ValueError:
                    origin_port = -1
                if (
                    parsed_origin.scheme != "http"
                    or parsed_origin.hostname not in {"127.0.0.1", "localhost"}
                    or origin_port not in {None, self.server.server_address[1]}
                ):
                    self._send_json(403, {"error": "Cross-origin requests are not allowed."})
                    return False
            return True

        def _read_json(self) -> dict:
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ValueError("Content-Type must be application/json.")
            raw_length = self.headers.get("Content-Length")
            if raw_length is None or not raw_length.isdigit():
                raise ValueError("A valid Content-Length header is required.")
            length = int(raw_length)
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError(f"Request body must be between 1 and {MAX_REQUEST_BYTES} bytes.")
            raw = self.rfile.read(length)
            try:
                value = json.loads(raw.decode("utf-8", errors="strict"))
            except (UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("Request body must be valid UTF-8 JSON.") from error
            if not isinstance(value, dict):
                raise ValueError("Request body must be a JSON object.")
            return value

        def do_GET(self):
            path = urlsplit(self.path).path
            if path.startswith("/api/") and not self._check_local_request():
                return
            if path == "/api/health":
                self._send_json(200, {"status": "ok", "project": "Development-AI"})
                return
            if path == "/api/status":
                try:
                    self._send_json(200, build_status(root, manager))
                except (OSError, RuntimeError, ValueError) as error:
                    self._send_json(500, {"error": str(error)})
                return
            if path == "/api/jobs":
                self._send_json(200, {"jobs": manager.list_jobs()})
                return
            if path.startswith("/api/jobs/"):
                try:
                    job_id = path.rsplit("/", 1)[-1]
                    self._send_json(200, manager.get(job_id))
                except KeyError as error:
                    self._send_json(404, {"error": str(error)})
                return
            assets = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/app.css": ("app.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            }
            asset = assets.get(path)
            if asset is None:
                self._send_json(404, {"error": "Resource not found."})
                return
            file_path = UI_ROOT / asset[0]
            try:
                content = file_path.read_bytes()
            except OSError:
                self._send_json(500, {"error": "UI asset is missing from the project."})
                return
            self.send_response(200)
            self.send_header("Content-Type", asset[1])
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if not self._check_local_request():
                return
            path = urlsplit(self.path).path
            try:
                payload = self._read_json()
                if path == "/api/chat/generate":
                    self._send_json(200, _generate_response(root, payload))
                    return
                if path == "/api/training":
                    command = build_training_command(root, payload)
                    job = manager.start("training", command)
                    self._send_json(202, {"job": job})
                    return
                if path == "/api/evaluation":
                    command = build_evaluation_command(root, payload)
                    job = manager.start("evaluation", command)
                    self._send_json(202, {"job": job})
                    return
                if path.startswith("/api/jobs/") and path.endswith("/stop"):
                    job_id = path.split("/")[-2]
                    try:
                        self._send_json(200, manager.stop(job_id))
                    except KeyError as error:
                        self._send_json(404, {"error": str(error)})
                    return
                self._send_json(404, {"error": "API route not found."})
            except (ValueError, RuntimeError) as error:
                self._send_json(400, {"error": str(error)})
            except (OSError, RuntimeError, TypeError, KeyError, ValueError) as error:
                self._send_json(500, {"error": f"Local operation failed: {error}"})

        def log_error(self, format_string, *args):
            super().log_error(format_string, *args)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.job_manager = manager
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Loopback only. Remote network binding is deliberately disabled.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        server = create_server(args.host, args.port)
    except OSError as error:
        parser.error(f"Cannot start local UI server: {error}")
    except ValueError as error:
        parser.error(str(error))
    print(f"Development-AI Studio: http://127.0.0.1:{server.server_address[1]}")
    print("Local-only server. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Development-AI Studio.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
