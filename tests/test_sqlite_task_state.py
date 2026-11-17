import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from core.agent.sqlite_task_state import (
    SQLiteTaskStateStore,
    TaskPersistenceError,
    TaskStoreIntegrityError,
)
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
        created = self.create_task(first)
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
