import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path

from core.agent import (
    ApprovalGrant,
    RuntimeIdentity,
    ResultStatus,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolDispatcher,
    ToolManifest,
    ToolOperation,
    ToolRegistry,
    ToolResult,
    PlanStep,
    SQLiteTaskStateStore,
    SQLiteApprovalStore,
    SQLiteSessionStore,
    ApprovalStoreError,
    ApprovalStoreIntegrityError,
)
from core.agent.contracts import arguments_digest


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


def make_grant(call=None, *, approval_id="approval-1", expires=200.0, confirmed=True, identity=None):
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
        principal_id=None if identity is None else identity.principal_id,
        session_id=None if identity is None else identity.session_id,
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

    def test_identity_binding_is_persisted_and_part_of_approval_scope(self):
        store = self.open_store()
        identity = RuntimeIdentity("owner-1", "session-1")
        sessions = SQLiteSessionStore(self.path, clock=lambda: 100.0)
        self.addCleanup(sessions.close)
        sessions.issue(identity, expires_at_unix=200.0)
        grant = make_grant(identity=identity)
        store.issue(grant)
        self.assertEqual(store.get(grant.approval_id), grant)

        wrong_session = make_grant(identity=RuntimeIdentity("owner-1", "session-2"))
        self.assertFalse(store.consume(wrong_session))
        self.assertTrue(store.consume(grant))

    def test_revoked_session_invalidates_bound_approval_before_consumption(self):
        store = self.open_store()
        sessions = SQLiteSessionStore(self.path, clock=lambda: 100.0)
        self.addCleanup(sessions.close)
        identity = RuntimeIdentity("owner-1", "session-revoked")
        sessions.issue(identity, expires_at_unix=200.0)
        grant = make_grant(identity=identity, approval_id="revoked-session-grant")
        store.issue(grant)

        self.assertTrue(sessions.revoke(identity))
        self.assertIsNone(store.get(grant.approval_id))
        self.assertFalse(store.consume(grant))
        row = store._connection.execute(
            "SELECT consumed_at_unix FROM approval_grants WHERE approval_id = ?",
            (grant.approval_id,),
        ).fetchone()
        self.assertIsNone(row["consumed_at_unix"])

    def test_schema_v1_migrates_to_identity_columns_without_rebinding_old_grants(self):
        connection = sqlite3.connect(self.path)
        connection.executescript("""
            CREATE TABLE approval_store_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
            );
            INSERT INTO approval_store_metadata(singleton, schema_version) VALUES (1, 1);
            CREATE TABLE approval_grants (
                approval_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                tool_version TEXT NOT NULL,
                operation TEXT NOT NULL,
                arguments_sha256 TEXT NOT NULL CHECK (length(arguments_sha256) = 64),
                idempotency_key TEXT,
                expires_at_unix REAL NOT NULL,
                user_confirmed INTEGER NOT NULL CHECK (user_confirmed = 1),
                issued_at_unix REAL NOT NULL,
                consumed_at_unix REAL,
                revoked_at_unix REAL,
                CHECK (expires_at_unix > issued_at_unix),
                CHECK (consumed_at_unix IS NULL OR consumed_at_unix >= issued_at_unix),
                CHECK (revoked_at_unix IS NULL OR revoked_at_unix >= issued_at_unix),
                CHECK (consumed_at_unix IS NULL OR revoked_at_unix IS NULL)
            );
            INSERT INTO approval_grants (
                approval_id, request_id, task_id, step_id, tool_id, tool_version,
                operation, arguments_sha256, idempotency_key, expires_at_unix,
                user_confirmed, issued_at_unix, consumed_at_unix, revoked_at_unix
            ) VALUES (
                'legacy-approval', 'request-1', 'task-1', 'step-1', 'browser.test',
                '1.0.0', 'submit_form',
                '0000000000000000000000000000000000000000000000000000000000000000',
                'idem-1', 200.0, 1, 100.0, NULL, NULL
            );
            CREATE TRIGGER approval_grants_scope_immutable
            BEFORE UPDATE OF approval_id, request_id, task_id, step_id, tool_id,
                tool_version, operation, arguments_sha256, idempotency_key,
                expires_at_unix, user_confirmed, issued_at_unix
            ON approval_grants BEGIN
                SELECT RAISE(ABORT, 'approval grant scope is immutable');
            END;
            CREATE TRIGGER approval_grants_terminal_once
            BEFORE UPDATE OF consumed_at_unix, revoked_at_unix ON approval_grants
            WHEN OLD.consumed_at_unix IS NOT NULL OR OLD.revoked_at_unix IS NOT NULL
              OR (NEW.consumed_at_unix IS NULL AND NEW.revoked_at_unix IS NULL)
              OR (NEW.consumed_at_unix IS NOT NULL AND NEW.revoked_at_unix IS NOT NULL)
            BEGIN SELECT RAISE(ABORT, 'approval grant is already terminal'); END;
            CREATE TRIGGER approval_grants_no_delete
            BEFORE DELETE ON approval_grants BEGIN
                SELECT RAISE(ABORT, 'approval grants cannot be deleted');
            END;
        """)
        connection.commit()
        connection.close()

        migrated = SQLiteApprovalStore(self.path, clock=lambda: 100.0)
        self.addCleanup(migrated.close)
        columns = {
            row["name"]
            for row in migrated._connection.execute("PRAGMA table_info(approval_grants)")
        }
        self.assertIn("principal_id", columns)
        self.assertIn("session_id", columns)
        legacy_grant = migrated.get("legacy-approval")
        self.assertIsNotNone(legacy_grant)
        self.assertIsNone(legacy_grant.principal_id)
        self.assertIsNone(legacy_grant.session_id)
        self.assertEqual(
            migrated._connection.execute(
                "SELECT schema_version FROM approval_store_metadata WHERE singleton = 1"
            ).fetchone()["schema_version"],
            2,
        )

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

    def test_clock_is_sampled_inside_each_approval_transaction(self):
        store = self.open_store()
        connection = store._connection

        def transaction_clock():
            if not connection.in_transaction:
                raise AssertionError("Approval timestamp was read outside its SQLite transaction.")
            return 100.0

        store._clock = transaction_clock
        first = make_grant(approval_id="approval-first")
        store.issue(first)
        self.assertEqual(store.get(first.approval_id), first)
        self.assertTrue(store.consume(first))

        second = make_grant(approval_id="approval-second")
        store.issue(second)
        self.assertTrue(store.revoke(second.approval_id))

    def test_pre_dispatch_cancellation_does_not_consume_approval(self):
        store = self.open_store()
        call = make_call()
        grant = make_grant(call)
        store.issue(grant)
        adapter = FakeAdapter()
        registry = ToolRegistry(approval_consumer=store.consume)
        registry.register(adapter)
        dispatcher = ToolDispatcher(registry)

        cancellation = threading.Event()
        cancellation.set()
        cancelled = dispatcher.dispatch(
            call,
            granted_permissions={"browser.write"},
            approval=grant,
            cancel_event=cancellation,
            wall_clock=lambda: 100.0,
        )
        self.assertEqual(cancelled.status, ResultStatus.CANCELLED)
        self.assertEqual(cancelled.error_code, "cancelled_before_dispatch")
        self.assertEqual(len(adapter.calls), 0)
        self.assertEqual(store.get(grant.approval_id), grant)

        accepted = dispatcher.dispatch(
            call,
            granted_permissions={"browser.write"},
            approval=grant,
            wall_clock=lambda: 100.0,
        )
        self.assertEqual(accepted.status, ResultStatus.SUCCESS)
        self.assertEqual(len(adapter.calls), 1)

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

    def test_consume_checks_expiry_after_acquiring_write_transaction(self):
        transaction_acquired = False

        def clock():
            return 200.0 if transaction_acquired else 100.0

        store = self.open_store(clock=clock)
        grant = make_grant(expires=150.0)
        store.issue(grant)

        def mark_transaction(statement):
            nonlocal transaction_acquired
            if statement.strip().upper() == "BEGIN IMMEDIATE":
                transaction_acquired = True

        store._connection.set_trace_callback(mark_transaction)
        try:
            self.assertFalse(store.consume(grant))
        finally:
            store._connection.set_trace_callback(None)

        row = store._connection.execute(
            "SELECT consumed_at_unix FROM approval_grants WHERE approval_id = ?",
            (grant.approval_id,),
        ).fetchone()
        self.assertIsNone(row["consumed_at_unix"])

    def test_get_checks_expiry_inside_read_transaction(self):
        transaction_started = False

        def clock():
            return 200.0 if transaction_started else 100.0

        store = self.open_store(clock=clock)
        grant = make_grant(expires=150.0)
        store.issue(grant)

        def mark_transaction(statement):
            nonlocal transaction_started
            if statement.strip().upper() == "BEGIN":
                transaction_started = True

        store._connection.set_trace_callback(mark_transaction)
        try:
            self.assertIsNone(store.get(grant.approval_id))
        finally:
            store._connection.set_trace_callback(None)

    def test_future_issued_grant_is_not_recovered_after_clock_rollback(self):
        store = self.open_store(clock=lambda: 100.0)
        grant = make_grant(expires=200.0)
        store.issue(grant)
        store._connection.execute("DROP TRIGGER approval_grants_scope_immutable")
        store._connection.execute(
            "UPDATE approval_grants SET issued_at_unix = 101.0 WHERE approval_id = ?",
            (grant.approval_id,),
        )
        self.assertIsNone(store.get(grant.approval_id))
        self.assertFalse(store.consume(grant))

    def test_expired_grant_is_not_consumed(self):
        issuer = self.open_store(clock=lambda: 100.0)
        grant = make_grant(expires=150.0)
        issuer.issue(grant)
        expired_store = self.open_store(clock=lambda: 200.0)
        self.assertIsNone(expired_store.get(grant.approval_id))
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

    def test_approval_and_task_stores_can_share_one_database_file(self):
        tasks = SQLiteTaskStateStore(self.path, clock=lambda: 100.0)
        self.addCleanup(tasks.close)
        tasks.create_task(
            task_id="task-1",
            request_id="request-1",
            goal="Inspect a page.",
            steps=(PlanStep("inspect", "Inspect the page."),),
        )

        approvals = self.open_store(clock=lambda: 100.0)
        grant = make_grant()
        approvals.issue(grant)

        self.assertEqual(tasks.get_task("task-1").goal, "Inspect a page.")
        self.assertEqual(approvals.get(grant.approval_id), grant)

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
        connection.commit()
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
