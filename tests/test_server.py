import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from core.server import (
    LocalJobManager,
    _resolve_data_file,
    build_evaluation_command,
    build_training_command,
    create_server,
)


class LocalPathValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data" / "processed").mkdir(parents=True)
        (self.root / "data" / "evaluation").mkdir(parents=True)
        (self.root / "artifacts").mkdir()
        (self.root / "data" / "processed" / "train.txt").write_text(
            "A small local training fixture.", encoding="utf-8"
        )
        (self.root / "data" / "evaluation" / "manifest.json").write_text(
            json.dumps({"format_version": 1, "datasets": []}), encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_resolves_only_existing_files_inside_data_directory(self):
        path = _resolve_data_file(self.root, "processed/train.txt", {".txt"})
        self.assertEqual(path, (self.root / "data" / "processed" / "train.txt").resolve())

        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        for value in (
            "../outside.txt",
            str(outside),
            "C:\\private\\train.txt",
            "missing.txt",
            "evaluation/manifest.json",
        ):
            with self.subTest(path=value):
                with self.assertRaises(ValueError):
                    _resolve_data_file(self.root, value, {".txt"})

    def test_rejects_data_symlink_that_escapes_root(self):
        outside = self.root / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        link = self.root / "data" / "linked.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks are unavailable on this runner.")
        with self.assertRaises(ValueError):
            _resolve_data_file(self.root, "linked.txt", {".txt"})

    def test_builds_training_command_with_explicit_local_settings(self):
        command = build_training_command(self.root, {
            "data_path": "processed/train.txt",
            "steps": 25,
            "batch_size": 2,
            "gradient_accumulation_steps": 3,
            "learning_rate": 0.001,
            "block_size": 64,
            "eval_interval": 5,
            "eval_iters": 2,
            "validation_fraction": 0.2,
            "device": "cpu",
            "seed": 7,
        })
        self.assertEqual(command[:4], [sys.executable, "-u", "-m", "core.train"])
        self.assertIn("--gradient-accumulation-steps", command)
        self.assertEqual(command[command.index("--steps") + 1], "25")
        self.assertEqual(command[command.index("--batch-size") + 1], "2")
        self.assertEqual(command[command.index("--resume-interval") + 1], "100")
        self.assertFalse(any("shell" in item for item in command))

    def test_training_command_rejects_invalid_values_and_overwrite(self):
        for payload in (
            {"steps": True},
            {"batch_size": 0},
            {"device": []},
            {"learning_rate": float("nan")},
            {"validation_fraction": 0.5},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    build_training_command(self.root, payload)

        (self.root / "artifacts" / "latest.pt").write_bytes(b"existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            build_training_command(self.root, {})

    def test_resume_requires_paired_best_and_resume_files(self):
        (self.root / "artifacts" / "latest.pt").write_bytes(b"best")
        with self.assertRaisesRegex(ValueError, "both artifacts"):
            build_training_command(self.root, {"resume": True})
        (self.root / "artifacts" / "latest.resume.pt").write_bytes(b"resume")
        command = build_training_command(self.root, {"resume": True})
        self.assertIn("--resume-from", command)
        self.assertEqual(command[command.index("--resume-from") + 1],
                         str(self.root / "artifacts" / "latest.resume.pt"))

    def test_evaluation_command_requires_checkpoint_and_valid_manifest(self):
        with self.assertRaisesRegex(ValueError, "No inference checkpoint"):
            build_evaluation_command(self.root, {})

        (self.root / "artifacts" / "latest.pt").write_bytes(b"checkpoint")
        command = build_evaluation_command(self.root, {
            "manifest_path": "evaluation/manifest.json",
            "compare_untrained": True,
            "baseline_seed": 123,
            "device": "cpu",
        })
        self.assertIn("--compare-untrained", command)
        self.assertEqual(command[command.index("--baseline-seed") + 1], "123")

        with self.assertRaises(ValueError):
            build_evaluation_command(self.root, {"manifest_path": "../outside.json"})

    def test_cuda_is_rejected_when_not_available(self):
        with patch("core.server.torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                build_training_command(self.root, {"device": "cuda"})


class LocalJobManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manager = LocalJobManager(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_job_captures_progress_logs_and_completion_state(self):
        command = [
            sys.executable, "-u", "-c",
            "print('step 1/2 | validation_loss 1.2345', flush=True)",
        ]
        job = self.manager.start("training", command)
        job_id = job["id"]
        deadline = time.time() + 5
        current = self.manager.get(job_id)
        while current["status"] not in {"succeeded", "failed", "cancelled"} and time.time() < deadline:
            time.sleep(0.02)
            current = self.manager.get(job_id)

        self.assertEqual(current["status"], "succeeded")
        self.assertEqual(current["step"], 1)
        self.assertEqual(current["target_steps"], 2)
        self.assertIn("validation_loss 1.2345", "\n".join(current["logs"]))
        self.assertEqual(current["return_code"], 0)
        self.assertIsNotNone(current["finished_at"])

    def test_only_one_active_job_can_run(self):
        job = self.manager.start("training", [
            sys.executable, "-u", "-c",
            "import time; print('starting', flush=True); time.sleep(0.4)",
        ])
        with self.assertRaisesRegex(RuntimeError, "already running"):
            self.manager.start("evaluation", [sys.executable, "-c", "print('second')"])
        self.assertIn(self.manager.get(job["id"])["status"], {"queued", "running"})

    def test_stop_marks_a_running_job_as_cancelled(self):
        job = self.manager.start("training", [
            sys.executable, "-u", "-c",
            "import time; print('started', flush=True); time.sleep(10)",
        ])
        job_id = job["id"]
        deadline = time.time() + 3
        while self.manager.get(job_id)["status"] == "queued" and time.time() < deadline:
            time.sleep(0.01)
        self.manager.stop(job_id)
        deadline = time.time() + 5
        current = self.manager.get(job_id)
        while current["status"] not in {"cancelled", "failed", "succeeded"} and time.time() < deadline:
            time.sleep(0.02)
            current = self.manager.get(job_id)
        self.assertEqual(current["status"], "cancelled")


class LocalHttpServerTests(unittest.TestCase):
    def test_ui_assets_are_local_and_all_primary_screens_are_present(self):
        from core.server import UI_ROOT

        html = (UI_ROOT / "index.html").read_text(encoding="utf-8")
        css = (UI_ROOT / "app.css").read_text(encoding="utf-8")
        javascript = (UI_ROOT / "app.js").read_text(encoding="utf-8")
        for screen in ('id="view-overview"', 'id="view-playground"',
                       'id="view-training"', 'id="view-evaluation"'):
            self.assertIn(screen, html)
        self.assertIn("/api/chat/generate", javascript)
        self.assertIn("/api/training", javascript)
        self.assertIn("/api/evaluation", javascript)
        self.assertNotIn("fonts.googleapis.com", css)
        self.assertNotIn("cdn.jsdelivr.net", html + javascript + css)

    def test_server_refuses_non_loopback_binding(self):
        with self.assertRaisesRegex(ValueError, "loopback-only"):
            create_server("0.0.0.0", 0)

    def test_health_and_ui_are_served_on_loopback(self):
        server = create_server("127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=3) as response:
                self.assertEqual(json.loads(response.read())["status"], "ok")
            with urlopen(f"http://127.0.0.1:{port}/", timeout=3) as response:
                page = response.read().decode("utf-8")
                self.assertIn("Development-AI Studio", page)
                self.assertEqual(response.headers.get_content_type(), "text/html")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

    def test_training_api_rejects_path_traversal(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        (root / "data").mkdir()
        server = create_server("127.0.0.1", 0, root=root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            body = json.dumps({"data_path": "../private.txt"}).encode("utf-8")
            request = Request(
                f"http://127.0.0.1:{port}/api/training",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=3)
            self.assertEqual(raised.exception.code, 400)
            result = json.loads(raised.exception.read())
            self.assertIn("inside the project's data directory", result["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
