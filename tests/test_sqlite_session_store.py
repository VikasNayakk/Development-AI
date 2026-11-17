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


CLIENT_ID = "a" * 32
CREDENTIAL = "session-token-" + ("x" * 40)


class SQLiteSessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sessions.sqlite3"
        self.now = [100.0]
        self.store = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(self.store.close)
        self.identity = RuntimeIdentity("owner-1", "session-1")

    def issue(self, identity=None, *, credential=CREDENTIAL, client_id=CLIENT_ID, expires=200.0):
        self.store.issue(
            identity or self.identity,
            credential=credential,
            client_id=client_id,
            expires_at_unix=expires,
        )

    def test_registers_and_recovers_exact_principal_session_pair(self):
        self.issue()
        self.assertTrue(self.store.is_active(self.identity))
        self.assertFalse(
            self.store.is_active(RuntimeIdentity("other-owner", "session-1"))
        )
        self.store.close()
        reopened = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(reopened.close)
        self.assertTrue(reopened.is_active(self.identity))

    def test_raw_bearer_credential_is_never_persisted(self):
        self.issue()
        row = self.store._connection.execute(
            "SELECT client_id, credential_sha256 FROM agent_sessions WHERE session_id = ?",
            (self.identity.session_id,),
        ).fetchone()
        self.assertEqual(row["client_id"], CLIENT_ID)
        self.assertEqual(len(row["credential_sha256"]), 64)
        self.assertNotEqual(row["credential_sha256"], CREDENTIAL)
        self.assertNotIn(CREDENTIAL, tuple(row))

    def test_authenticates_only_matching_session_token_and_extension_id(self):
        self.issue()
        self.assertEqual(
            self.store.authenticate(
                self.identity.session_id,
                credential=CREDENTIAL,
                client_id=CLIENT_ID,
            ),
            self.identity,
        )
        self.assertIsNone(self.store.authenticate(
            self.identity.session_id,
            credential="wrong-token-" + ("z" * 40),
            client_id=CLIENT_ID,
        ))
        self.assertIsNone(self.store.authenticate(
            self.identity.session_id,
            credential=CREDENTIAL,
            client_id="b" * 32,
        ))
        self.assertIsNone(self.store.authenticate(
            "unknown-session",
            credential=CREDENTIAL,
            client_id=CLIENT_ID,
        ))

    def test_expired_session_is_inactive_and_cannot_authenticate(self):
        self.issue(expires=150.0)
        self.now[0] = 150.0
        self.assertFalse(self.store.is_active(self.identity))
        self.assertIsNone(self.store.authenticate(
            self.identity.session_id,
            credential=CREDENTIAL,
            client_id=CLIENT_ID,
        ))

    def test_future_issued_session_fails_closed_after_clock_rollback(self):
        self.issue()
        self.store._connection.execute("DROP TRIGGER agent_sessions_scope_immutable")
        self.store._connection.execute(
            "UPDATE agent_sessions SET issued_at_unix = 101.0 WHERE session_id = ?",
            (self.identity.session_id,),
        )
        self.assertFalse(self.store.is_active(self.identity))
        self.assertIsNone(self.store.authenticate(
            self.identity.session_id,
            credential=CREDENTIAL,
            client_id=CLIENT_ID,
        ))

    def test_revocation_is_terminal_and_survives_restart(self):
        self.issue()
        self.assertTrue(self.store.revoke(self.identity))
        self.assertFalse(self.store.revoke(self.identity))
        self.assertFalse(self.store.is_active(self.identity))
        self.assertIsNone(self.store.authenticate(
            self.identity.session_id,
            credential=CREDENTIAL,
            client_id=CLIENT_ID,
        ))
        self.store.close()
        reopened = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(reopened.close)
        self.assertFalse(reopened.is_active(self.identity))

    def test_rejects_expired_or_excessively_long_session(self):
        with self.assertRaises(ValueError):
            self.issue(expires=100.0)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.issue(expires=100.0 + 24 * 60 * 60 + 1)

    def test_rejects_weak_credentials_and_invalid_extension_ids(self):
        with self.assertRaises(ValueError):
            self.issue(credential="short")
        with self.assertRaisesRegex(ValueError, "Chrome extension ID"):
            self.issue(client_id="not-an-extension-id")

    def test_duplicate_session_id_cannot_be_rebound(self):
        self.issue()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.issue(
                RuntimeIdentity("other-owner", "session-1"),
                credential="other-session-token-" + ("y" * 40),
            )
        self.assertFalse(
            self.store.is_active(RuntimeIdentity("other-owner", "session-1"))
        )

    def test_scope_and_credential_are_immutable_and_rows_cannot_be_deleted(self):
        self.issue()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store._connection.execute(
                "UPDATE agent_sessions SET principal_id = ? WHERE session_id = ?",
                ("other-owner", self.identity.session_id),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store._connection.execute(
                "UPDATE agent_sessions SET credential_sha256 = ? WHERE session_id = ?",
                ("0" * 64, self.identity.session_id),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store._connection.execute(
                "DELETE FROM agent_sessions WHERE session_id = ?",
                (self.identity.session_id,),
            )

    def test_schema_v1_migration_keeps_old_sessions_unpaired(self):
        self.store.close()
        connection = sqlite3.connect(self.path)
        connection.executescript("""
            CREATE TABLE agent_session_store_metadata (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
            );
            INSERT INTO agent_session_store_metadata(singleton, schema_version) VALUES (1, 1);
            CREATE TABLE agent_sessions (
                session_id TEXT PRIMARY KEY,
                principal_id TEXT NOT NULL,
                issued_at_unix REAL NOT NULL,
                expires_at_unix REAL NOT NULL,
                revoked_at_unix REAL,
                CHECK (expires_at_unix > issued_at_unix),
                CHECK (revoked_at_unix IS NULL OR revoked_at_unix >= issued_at_unix)
            );
            INSERT INTO agent_sessions(session_id, principal_id, issued_at_unix, expires_at_unix)
            VALUES ('legacy-session', 'owner-1', 100.0, 200.0);
            CREATE TRIGGER agent_sessions_scope_immutable
            BEFORE UPDATE OF session_id, principal_id, issued_at_unix, expires_at_unix
            ON agent_sessions BEGIN
                SELECT RAISE(ABORT, 'session identity and lifetime are immutable');
            END;
            CREATE TRIGGER agent_sessions_revoke_once
            BEFORE UPDATE OF revoked_at_unix ON agent_sessions
            WHEN OLD.revoked_at_unix IS NOT NULL OR NEW.revoked_at_unix IS NULL
            BEGIN SELECT RAISE(ABORT, 'session revocation is terminal'); END;
            CREATE TRIGGER agent_sessions_no_delete
            BEFORE DELETE ON agent_sessions BEGIN
                SELECT RAISE(ABORT, 'session records cannot be deleted');
            END;
        """)
        connection.commit()
        connection.close()

        migrated = SQLiteSessionStore(self.path, clock=lambda: 100.0)
        self.addCleanup(migrated.close)
        legacy = RuntimeIdentity("owner-1", "legacy-session")
        self.assertFalse(migrated.is_active(legacy))
        self.assertIsNone(migrated.authenticate(
            legacy.session_id,
            credential=CREDENTIAL,
            client_id=CLIENT_ID,
        ))
        self.assertEqual(
            migrated._connection.execute(
                "SELECT schema_version FROM agent_session_store_metadata WHERE singleton = 1"
            ).fetchone()["schema_version"],
            3,
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
            self.issue()
        with self.assertRaises(SessionStoreError):
            self.store.is_active(self.identity)
        with self.assertRaises(SessionStoreError):
            self.store.revoke(self.identity)


if __name__ == "__main__":
    unittest.main()
