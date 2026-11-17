import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from core.agent import (
    ApprovalGrant,
    ResultStatus,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolDispatcher,
    ToolManifest,
    ToolOperation,
    ToolRegistry,
    ToolResult,
    SQLiteApprovalStore,
    ApprovalStoreError,
    ApprovalStoreIntegrityError,
    arguments_digest,
)


INPUT_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string", "minLength": 1}},
    "required": ["url"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"submitted": {"type": "boolean"}},
    "required": ["submitted"],
    "additionalProperties": False,
}


def make_call(**overrides):
    values = {
        "request_id": "request-1",
        "task_id": "task-1",
        "step_id": "step-1",
        "tool_id": "browser.test",
        "tool_version": "1.0.0",
        "operation": "submit_form",
        "arguments": {"url": "https://example.test/form"},
        "idempotency_key": "idem-1",
    }
    values.update(overrides)
    return ToolCall(**values)


def make_grant(call=None, *, approval_id="approval-1", expires=200.0, confirmed=True):
    call = call or make_call()
    return ApprovalGrant(
        approval_id=approval_id,
        request_id=call.request_id,
        task_id=call.task_id,
        step_id=call.step_id,
        tool_id=call.tool_id,
        tool_version=call.tool_version,
        operation=call.operation,
        arguments_sha256=arguments_digest(call.arguments),
        idempotency_key=call.idempotency_key,
        expires_at_unix=expires,
        user_confirmed=confirmed,
    )


class FakeAdapter:
    def __init__(self):
        self.manifest = ToolManifest(
            tool_id="browser.test",
            display_name="Test Browser",
            description="Deterministic approval-store test adapter.",
            version="1.0.0",
            operations=(
                ToolOperation(
                    name="submit_form",
                    description="Submit the explicitly approved test form.",
                    input_schema=INPUT_SCHEMA,
                    output_schema=OUTPUT_SCHEMA,
                    required_permissions=frozenset({"browser.write"}),
                    risk_level=RiskLevel.HIGH,
                    side_effect=SideEffect.IRREVERSIBLE,
                    requires_approval=True,
                ),
            ),
        )
        self.calls = []

    def execute(self, call, operation, context):
        self.calls.append(call)
        return ToolResult(status=ResultStatus.SUCCESS, output={"submitted": True})


class SQLiteApprovalStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "approvals.sqlite3"

    def open_store(self, *, clock=lambda: 100.0):
        store = SQLiteApprovalStore(self.path, clock=clock)
        self.addCleanup(store.close)
        return store

    def test_issues_only_explicitly_confirmed_unexpired_grants(self):
        store = self.open_store()
        with self.assertRaises(ValueError):
            store.issue(make_grant(confirmed=False))
        with self.assertRaises(ValueError):
            store.issue(make_grant(expires=100.0))
        self.assertFalse(
            store._connection.execute("SELECT 1 FROM approval_grants").fetchone()
        )

    def test_approval_lifetime_is_bounded(self):
        store = self.open_store(clock=lambda: 100.0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            store.issue(make_grant(expires=1001.0))

    def test_grant_survives_restart_and_dispatcher_consumes_it_once(self):
        original_store = self.open_store()
        call = make_call()
        grant = make_grant(call)
        approval_id = grant.approval_id
        original_store.issue(grant)
        grant = None
        original_store.close()

        reopened = SQLiteApprovalStore(self.path, clock=lambda: 101.0)
        self.addCleanup(reopened.close)
        grant = reopened.get(approval_id)
        self.assertIsNotNone(grant)
        adapter = FakeAdapter()
        registry = ToolRegistry(approval_consumer=reopened.consume)
        registry.register(adapter)
        dispatcher = ToolDispatcher(registry)

        first = dispatcher.dispatch(
            call,
            granted_permissions={"browser.write"},
            approval=grant,
            wall_clock=lambda: 101.0,
        )
        self.assertEqual(first.status, ResultStatus.SUCCESS)
        self.assertEqual(len(adapter.calls), 1)

        second_adapter = FakeAdapter()
        second_registry = ToolRegistry(approval_consumer=reopened.consume)
        second_registry.register(second_adapter)
        second = ToolDispatcher(second_registry).dispatch(
            call,
            granted_permissions={"browser.write"},
            approval=grant,
            wall_clock=lambda: 101.0,
        )
        self.assertEqual(second.status, ResultStatus.DENIED)
        self.assertEqual(second.error_code, "approval_unavailable")
        self.assertEqual(len(second_adapter.calls), 0)

    def test_scope_mismatch_does_not_consume_the_original_grant(self):
        store = self.open_store()
        call = make_call()
        grant = make_grant(call)
        store.issue(grant)
        adapter = FakeAdapter()
        registry = ToolRegistry(approval_consumer=store.consume)
        registry.register(adapter)
        dispatcher = ToolDispatcher(registry)

        changed = make_call(arguments={"url": "https://different.test"})
        mismatch = dispatcher.dispatch(
            changed,
            granted_permissions={"browser.write"},
            approval=grant,
            wall_clock=lambda: 100.0,
        )
        self.assertEqual(mismatch.error_code, "approval_scope_mismatch")
        self.assertEqual(len(adapter.calls), 0)

        accepted = dispatcher.dispatch(
            call,
            granted_permissions={"browser.write"},
            approval=grant,
            wall_clock=lambda: 100.0,
        )
        self.assertEqual(accepted.status, ResultStatus.SUCCESS)
        self.assertEqual(len(adapter.calls), 1)

    def test_expired_grant_is_not_consumed(self):
        issuer = self.open_store(clock=lambda: 100.0)
        grant = make_grant(expires=150.0)
        issuer.issue(grant)
        expired_store = self.open_store(clock=lambda: 200.0)
        self.assertFalse(expired_store.consume(grant))
        row = expired_store._connection.execute(
            "SELECT consumed_at_unix FROM approval_grants WHERE approval_id = ?",
            (grant.approval_id,),
        ).fetchone()
        self.assertIsNone(row["consumed_at_unix"])

    def test_revoked_grant_cannot_be_consumed(self):
        store = self.open_store()
        grant = make_grant()
        store.issue(grant)
        self.assertTrue(store.revoke(grant.approval_id))
        self.assertFalse(store.revoke(grant.approval_id))
        self.assertIsNone(store.get(grant.approval_id))
        self.assertFalse(store.consume(grant))

    def test_two_independent_connections_can_consume_only_once(self):
        first = self.open_store()
        grant = make_grant()
        first.issue(grant)
        second = self.open_store()
        barrier = threading.Barrier(2)
        outcomes = []
        errors = []
        lock = threading.Lock()

        def consume(store):
            try:
                barrier.wait(timeout=3)
                outcome = store.consume(grant)
                with lock:
                    outcomes.append(outcome)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=consume, args=(first,)),
            threading.Thread(target=consume, args=(second,)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertCountEqual(outcomes, [True, False])

    def test_grant_scope_is_immutable_and_rows_cannot_be_deleted(self):
        store = self.open_store()
        grant = make_grant()
        store.issue(grant)
        with self.assertRaises(sqlite3.IntegrityError):
            store._connection.execute(
                "UPDATE approval_grants SET operation = ? WHERE approval_id = ?",
                ("delete_account", grant.approval_id),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            store._connection.execute(
                "DELETE FROM approval_grants WHERE approval_id = ?",
                (grant.approval_id,),
            )

    def test_duplicate_approval_id_is_rejected(self):
        store = self.open_store()
        grant = make_grant()
        store.issue(grant)
        with self.assertRaisesRegex(ValueError, "already exists"):
            store.issue(grant)

    def test_database_does_not_store_raw_arguments(self):
        store = self.open_store()
        store.issue(make_grant())
        columns = {
            row["name"]
            for row in store._connection.execute("PRAGMA table_info(approval_grants)")
        }
        self.assertIn("arguments_sha256", columns)
        self.assertNotIn("arguments_json", columns)
        self.assertNotIn("arguments", columns)

    def test_future_schema_version_is_rejected(self):
        store = self.open_store()
        store.close()
        connection = sqlite3.connect(self.path)
        connection.execute(
            "UPDATE approval_store_metadata SET schema_version = 99 WHERE singleton = 1"
        )
        connection.close()
        with self.assertRaises(ApprovalStoreError):
            SQLiteApprovalStore(self.path)

    def test_incomplete_schema_is_rejected(self):
        store = self.open_store()
        store.close()
        connection = sqlite3.connect(self.path)
        connection.execute("DROP TRIGGER approval_grants_no_delete")
        connection.close()
        with self.assertRaises(ApprovalStoreIntegrityError):
            SQLiteApprovalStore(self.path)

    def test_closed_store_rejects_operations(self):
        store = self.open_store()
        store.close()
        with self.assertRaises(ApprovalStoreError):
            store.issue(make_grant())
        with self.assertRaises(ApprovalStoreError):
            store.consume(make_grant())
        with self.assertRaises(ApprovalStoreError):
            store.revoke("approval-1")


if __name__ == "__main__":
    unittest.main()
