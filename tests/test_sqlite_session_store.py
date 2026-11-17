import sqlite3
import tempfile
import unittest
from pathlib import Path

from core.agent import (
    RuntimeIdentity,
    SQLiteSessionStore,
    SessionStoreError,
    SessionStoreIntegrityError,
)


class SQLiteSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sessions.sqlite3"
        self.now = [100.0]
        self.store = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(self.store.close)
        self.identity = RuntimeIdentity("owner-1", "session-1")

    def test_registers_and_recovers_exact_principal_session_pair(self):
        self.store.issue(self.identity, expires_at_unix=200.0)
        self.assertTrue(self.store.is_active(self.identity))
        self.assertFalse(
            self.store.is_active(RuntimeIdentity("other-owner", "session-1"))
        )
        self.store.close()
        reopened = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(reopened.close)
        self.assertTrue(reopened.is_active(self.identity))

    def test_expired_session_is_inactive(self):
        self.store.issue(self.identity, expires_at_unix=150.0)
        self.now[0] = 150.0
        self.assertFalse(self.store.is_active(self.identity))

    def test_future_issued_session_fails_closed_after_clock_rollback(self):
        self.store.issue(self.identity, expires_at_unix=200.0)
        self.store._connection.execute(
            "DROP TRIGGER agent_sessions_scope_immutable"
        )
        self.store._connection.execute(
            "UPDATE agent_sessions SET issued_at_unix = 101.0 WHERE session_id = ?",
            (self.identity.session_id,),
        )
        self.assertFalse(self.store.is_active(self.identity))

    def test_revocation_is_terminal_and_survives_restart(self):
        self.store.issue(self.identity, expires_at_unix=200.0)
        self.assertTrue(self.store.revoke(self.identity))
        self.assertFalse(self.store.revoke(self.identity))
        self.assertFalse(self.store.is_active(self.identity))
        self.store.close()
        reopened = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(reopened.close)
        self.assertFalse(reopened.is_active(self.identity))

    def test_rejects_expired_or_excessively_long_session(self):
        with self.assertRaises(ValueError):
            self.store.issue(self.identity, expires_at_unix=100.0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.store.issue(self.identity, expires_at_unix=100.0 + 24 * 60 * 60 + 1)

    def test_duplicate_session_id_cannot_be_rebound(self):
        self.store.issue(self.identity, expires_at_unix=200.0)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.store.issue(
                RuntimeIdentity("other-owner", "session-1"),
                expires_at_unix=200.0,
            )
        self.assertFalse(
            self.store.is_active(RuntimeIdentity("other-owner", "session-1"))
        )

    def test_scope_is_immutable_and_session_rows_cannot_be_deleted(self):
        self.store.issue(self.identity, expires_at_unix=200.0)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store._connection.execute(
                "UPDATE agent_sessions SET principal_id = ? WHERE session_id = ?",
                ("other-owner", self.identity.session_id),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store._connection.execute(
                "DELETE FROM agent_sessions WHERE session_id = ?",
                (self.identity.session_id,),
            )

    def test_incomplete_schema_fails_closed(self):
        self.store.close()
        connection = sqlite3.connect(self.path)
        connection.execute("DROP TRIGGER agent_sessions_no_delete")
        connection.commit()
        connection.close()
        with self.assertRaises(SessionStoreIntegrityError):
            SQLiteSessionStore(self.path)

    def test_closed_store_rejects_operations(self):
        self.store.close()
        with self.assertRaises(SessionStoreError):
            self.store.issue(self.identity, expires_at_unix=200.0)
        with self.assertRaises(SessionStoreError):
            self.store.is_active(self.identity)
        with self.assertRaises(SessionStoreError):
            self.store.revoke(self.identity)


if __name__ == "__main__":
    unittest.main()
