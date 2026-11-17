import json
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path

from core.agent.sqlite_task_state import (
    SQLiteTaskStateStore,
    TaskPersistenceError,
    TaskStoreIntegrityError,
)

from core.agent.sqlite_approval_store import SQLiteApprovalStore
from core.agent.contracts import ApprovalGrant
from core.agent.task_state import (
    PlanStep,
    StaleTaskRevisionError,
    StepStatus,
    TaskStateError,
)


def plan():
    return (
        PlanStep("inspect", "Inspect the selected page."),
        PlanStep("prepare", "Prepare the requested change.", ("inspect",)),
        PlanStep("verify", "Verify the resulting page.", ("prepare",)),
    )


class SQLiteTaskStateStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "tasks.sqlite3"

    def open_store(self):
        store = SQLiteTaskStateStore(self.path, clock=lambda: 100.0)
        self.addCleanup(store.close)
        return store

    def create_task(self, store):
        return store.create_task(
            task_id="task-1",
            request_id="request-1",
            goal="Inspect and verify a page.",
            steps=plan(),
        )

    def test_task_and_event_history_survive_store_restart(self):
        store = self.open_store()
        created = self.create_task(store)
        started = store.start_step("task-1", "inspect", expected_revision=created.revision)
        completed = store.complete_step(
            "task-1", "inspect", result_ref="artifact://inspection",
            expected_revision=started.revision,
        )
        expected_events = completed.events
        store.close()

        reopened = SQLiteTaskStateStore(self.path, clock=lambda: 200.0)
        self.addCleanup(reopened.close)
        restored = reopened.get_task("task-1")
        self.assertEqual(restored, completed)
        self.assertEqual(restored.events, expected_events)
        self.assertEqual(
            [event.sequence for event in restored.events],
            list(range(1, restored.revision + 1)),
        )
        self.assertEqual(restored.get_step("inspect").status, StepStatus.SUCCEEDED)

    def test_approval_reference_survives_restart_with_its_durable_grant(self):
        tasks = self.open_store()
        tasks.create_task(
            task_id="task-approval",
            request_id="request-approval",
            goal="Submit an explicitly approved form.",
            steps=(
                PlanStep(
                    "submit",
                    "Submit the reviewed form.",
                    tool_id="browser.test",
                    operation="submit_form",
                ),
            ),
        )
        approvals = SQLiteApprovalStore(self.path, clock=lambda: 100.0)
        self.addCleanup(approvals.close)
        grant = ApprovalGrant(
            approval_id="_opaque-random-approval-id",
            request_id="request-approval",
            task_id="task-approval",
            step_id="submit",
            tool_id="browser.test",
            tool_version="1.0.0",
            operation="submit_form",
            arguments_sha256="a" * 64,
            idempotency_key="submit-001",
            expires_at_unix=200.0,
            user_confirmed=True,
        )
        approvals.issue(grant)
        waiting = tasks.require_approval(
            "task-approval", "submit", approval_id=grant.approval_id
        )
        self.assertEqual(waiting.get_step("submit").status, StepStatus.WAITING_APPROVAL)
        self.assertEqual(waiting.get_step("submit").approval_id, grant.approval_id)
        tasks.close()
        approvals.close()

        reopened_tasks = SQLiteTaskStateStore(self.path, clock=lambda: 101.0)
        self.addCleanup(reopened_tasks.close)
        reopened_approvals = SQLiteApprovalStore(self.path, clock=lambda: 101.0)
        self.addCleanup(reopened_approvals.close)
        restored = reopened_tasks.get_task("task-approval")
        restored_step = restored.get_step("submit")
        self.assertEqual(restored_step.status, StepStatus.WAITING_APPROVAL)
        self.assertEqual(restored_step.approval_id, grant.approval_id)
        self.assertEqual(reopened_approvals.get(restored_step.approval_id), grant)

    def test_legacy_snapshot_without_approval_reference_still_loads(self):
        store = self.open_store()
        original = self.create_task(store)
        row = store._connection.execute(
            "SELECT snapshot_json FROM tasks WHERE task_id = ?", ("task-1",)
        ).fetchone()
        payload = json.loads(row["snapshot_json"])
        payload.pop("principal_id", None)
        payload.pop("session_id", None)
        for step in payload["steps"]:
            step.pop("approval_id", None)
        store._connection.execute(
            "UPDATE tasks SET snapshot_json = ? WHERE task_id = ?",
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "task-1"),
        )
        restored = store.get_task("task-1")
        self.assertEqual(restored, original)
        self.assertIsNone(restored.get_step("inspect").approval_id)

    def test_two_store_instances_reject_stale_revision(self):
        first = self.open_store()
        created = self.create_task(first)
        second = self.open_store()

        updated = second.start_step("task-1", "inspect", expected_revision=created.revision)
        with self.assertRaises(StaleTaskRevisionError):
            first.start_step("task-1", "prepare", expected_revision=created.revision)
        self.assertEqual(first.get_task("task-1").revision, updated.revision)
        self.assertEqual(first.get_task("task-1").get_step("prepare").status, StepStatus.PENDING)

    def test_concurrent_writers_cannot_overwrite_each_other(self):
        first = self.open_store()
        created = first.create_task(
            task_id="task-1",
            request_id="request-1",
            goal="Start two independent actions concurrently.",
            steps=(
                PlanStep("inspect", "Inspect the selected page."),
                PlanStep("prepare", "Prepare the requested change."),
            ),
        )
        second = self.open_store()
        barrier = threading.Barrier(2)
        outcomes = []
        outcome_lock = threading.Lock()

        def writer(store, step_id):
            barrier.wait(timeout=3)
            try:
                store.start_step("task-1", step_id, expected_revision=created.revision)
                result = "updated"
            except StaleTaskRevisionError:
                result = "stale"
            with outcome_lock:
                outcomes.append(result)

        threads = [
            threading.Thread(target=writer, args=(first, "inspect")),
            threading.Thread(target=writer, args=(second, "prepare")),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertCountEqual(outcomes, ["updated", "stale"])
        snapshot = first.get_task("task-1")
        self.assertEqual(snapshot.revision, 3)
        self.assertEqual(
            sum(step.status == StepStatus.RUNNING for step in snapshot.steps), 1
        )

    def test_list_tasks_uses_one_consistent_read_snapshot(self):
        first = self.open_store()
        self.create_task(first)
        first.create_task(
            task_id="task-2",
            request_id="request-2",
            goal="Inspect a second page.",
            steps=plan(),
        )
        second = self.open_store()
        original_load = first._load_task
        mutation_performed = False

        def load_and_update(task_id):
            nonlocal mutation_performed
            snapshot = original_load(task_id)
            if task_id == "task-1" and not mutation_performed:
                mutation_performed = True
                second.start_step("task-2", "inspect")
            return snapshot

        with patch.object(first, "_load_task", side_effect=load_and_update):
            listed = first.list_tasks()

        self.assertTrue(mutation_performed)
        listed_task_2 = next(task for task in listed if task.task_id == "task-2")
        self.assertEqual(
            listed_task_2.get_step("inspect").status, StepStatus.PENDING
        )
        self.assertEqual(
            second.get_task("task-2").get_step("inspect").status, StepStatus.RUNNING
        )

    def test_get_task_uses_consistent_row_and_event_snapshot(self):
        first = self.open_store()
        created = self.create_task(first)
        second = self.open_store()
        mutation_performed = False

        def update_between_reads(statement):
            nonlocal mutation_performed
            if (
                "SELECT sequence, event_json FROM task_events" in statement
                and not mutation_performed
            ):
                mutation_performed = True
                second.start_step("task-1", "inspect")

        first._connection.set_trace_callback(update_between_reads)
        try:
            snapshot = first.get_task("task-1")
        finally:
            first._connection.set_trace_callback(None)

        self.assertTrue(mutation_performed)
        self.assertEqual(snapshot.revision, created.revision)
        self.assertEqual(snapshot.get_step("inspect").status, StepStatus.PENDING)
        latest = second.get_task("task-1")
        self.assertEqual(latest.revision, 3)
        self.assertEqual(latest.get_step("inspect").status, StepStatus.RUNNING)

    def test_duplicate_task_id_is_rejected_without_overwrite(self):
        store = self.open_store()
        original = self.create_task(store)
        with self.assertRaises(TaskStateError):
            self.create_task(store)
        self.assertEqual(store.get_task("task-1"), original)

    def test_event_rows_are_append_only(self):
        store = self.open_store()
        self.create_task(store)
        with self.assertRaises(sqlite3.IntegrityError):
            store._connection.execute(
                "UPDATE task_events SET event_json = '{}' WHERE task_id = ? AND sequence = 1",
                ("task-1",),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            store._connection.execute(
                "DELETE FROM task_events WHERE task_id = ? AND sequence = 1",
                ("task-1",),
            )

    def test_malformed_event_history_is_rejected(self):
        store = self.open_store()
        self.create_task(store)
        store._connection.execute("DROP TRIGGER task_events_no_update")
        store._connection.execute(
            "UPDATE task_events SET event_json = '{}' WHERE task_id = ? AND sequence = 1",
            ("task-1",),
        )
        with self.assertRaises(TaskStoreIntegrityError):
            store.get_task("task-1")

    def test_malformed_approval_reference_in_snapshot_is_rejected(self):
        store = self.open_store()
        task = self.create_task(store)
        waiting = store.require_approval(
            task.task_id, "inspect", approval_id="approval-valid-123"
        )
        row = store._connection.execute(
            "SELECT snapshot_json FROM tasks WHERE task_id = ?", (task.task_id,)
        ).fetchone()
        payload = json.loads(row["snapshot_json"])
        payload["steps"][0]["approval_id"] = "invalid approval id"
        store._connection.execute(
            "UPDATE tasks SET snapshot_json = ? WHERE task_id = ?",
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")), task.task_id),
        )
        self.assertGreater(waiting.revision, task.revision)
        with self.assertRaises(TaskStoreIntegrityError):
            store.get_task(task.task_id)

    def test_future_schema_version_is_rejected(self):
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA user_version = 99")
        connection.close()
        with self.assertRaises(TaskPersistenceError):
            SQLiteTaskStateStore(self.path)

    def test_closed_store_rejects_operations(self):
        store = self.open_store()
        store.close()
        with self.assertRaises(TaskPersistenceError):
            store.list_tasks()


if __name__ == "__main__":
    unittest.main()
