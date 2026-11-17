"""Durable active-session registry for authenticated runtime identities.

Session registration is a trusted-host operation. The session identifier is an
opaque lookup key, not a bearer credential; bridge authentication must happen
before host code constructs RuntimeIdentity and registers it here.
"""
from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

from .contracts import RuntimeIdentity

_SCHEMA_VERSION = 1
_MAX_DATABASE_PATH_LENGTH = 4096
_MAX_SESSION_TTL_SECONDS = 24 * 60 * 60


class SessionStoreError(RuntimeError):
    """The durable session store could not safely complete an operation."""


class SessionStoreIntegrityError(SessionStoreError):
    """The session database schema or stored records failed integrity checks."""


class SQLiteSessionStore:
    """Persistent session expiry/revocation registry backed by SQLite."""

    def __init__(
        self,
        database_path: str | os.PathLike[str],
        *,
        clock: Callable[[], float] = time.time,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a positive finite number.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError("timeout_seconds must be greater than 0 and at most 60.")
        raw_path = os.fspath(database_path)
        if not isinstance(raw_path, str) or not raw_path.strip() or len(raw_path) > _MAX_DATABASE_PATH_LENGTH:
            raise ValueError("database_path must be a non-empty path of bounded length.")

        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._database_path = raw_path
        if raw_path != ":memory:":
            path = Path(raw_path).expanduser()
            if path.is_symlink():
                raise ValueError("Session database path must not be a symbolic link.")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not path.is_file():
                raise ValueError("Session database path must refer to a regular file.")
            self._database_path = str(path.resolve())

        try:
            self._connection = sqlite3.connect(
                self._database_path,
                timeout=float(timeout_seconds),
                isolation_level=None,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
            if self._database_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
                self._connection.execute("PRAGMA synchronous = FULL")
                try:
                    os.chmod(self._database_path, 0o600)
                except OSError:
                    pass
            self._initialize_schema()
        except Exception:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            raise

    def _initialize_schema(self) -> None:
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_session_store_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
                )
                """
            )
            metadata = connection.execute(
                "SELECT schema_version FROM agent_session_store_metadata WHERE singleton = 1"
            ).fetchone()
            version = 0 if metadata is None else metadata["schema_version"]
            if type(version) is not int or version < 0 or version > _SCHEMA_VERSION:
                raise SessionStoreError(
                    f"Session schema version {version} is newer than supported version {_SCHEMA_VERSION}."
                )
            if version == 0:
                connection.execute(
                    """
                    CREATE TABLE agent_sessions (
                        session_id TEXT PRIMARY KEY,
                        principal_id TEXT NOT NULL,
                        issued_at_unix REAL NOT NULL,
                        expires_at_unix REAL NOT NULL,
                        revoked_at_unix REAL,
                        CHECK (expires_at_unix > issued_at_unix),
                        CHECK (revoked_at_unix IS NULL OR revoked_at_unix >= issued_at_unix)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER agent_sessions_scope_immutable
                    BEFORE UPDATE OF session_id, principal_id, issued_at_unix, expires_at_unix
                    ON agent_sessions
                    BEGIN
                        SELECT RAISE(ABORT, 'session identity and lifetime are immutable');
                    END
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER agent_sessions_revoke_once
                    BEFORE UPDATE OF revoked_at_unix ON agent_sessions
                    WHEN OLD.revoked_at_unix IS NOT NULL
                      OR NEW.revoked_at_unix IS NULL
                    BEGIN
                        SELECT RAISE(ABORT, 'session revocation is terminal');
                    END
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER agent_sessions_no_delete
                    BEFORE DELETE ON agent_sessions
                    BEGIN
                        SELECT RAISE(ABORT, 'session records cannot be deleted');
                    END
                    """
                )
                connection.execute(
                    "INSERT INTO agent_session_store_metadata(singleton, schema_version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
            else:
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(agent_sessions)").fetchall()
                }
                triggers = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                    ).fetchall()
                }
                metadata_exists = connection.execute(
                    "SELECT 1 FROM agent_session_store_metadata WHERE singleton = 1"
                ).fetchone()
                if metadata_exists is None:
                    raise SessionStoreIntegrityError("Session schema metadata is missing.")
                required_columns = {
                    "session_id", "principal_id", "issued_at_unix",
                    "expires_at_unix", "revoked_at_unix",
                }
                required_triggers = {
                    "agent_sessions_scope_immutable",
                    "agent_sessions_revoke_once",
                    "agent_sessions_no_delete",
                }
                if not required_columns.issubset(columns) or not required_triggers.issubset(triggers):
                    raise SessionStoreIntegrityError("Session database schema is incomplete.")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionStoreError("Session store is closed.")

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise SessionStoreError("Session store clock must return a finite Unix timestamp.")
        return float(value)

    def issue(self, identity: RuntimeIdentity, *, expires_at_unix: float) -> None:
        """Register a session after the host has authenticated its owner."""
        if not isinstance(identity, RuntimeIdentity):
            raise TypeError("identity must be a RuntimeIdentity.")
        if (
            isinstance(expires_at_unix, bool)
            or not isinstance(expires_at_unix, (int, float))
            or not math.isfinite(expires_at_unix)
        ):
            raise ValueError("expires_at_unix must be a finite Unix timestamp.")
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._now()
                if expires_at_unix <= now:
                    raise ValueError("An expired session cannot be registered.")
                if expires_at_unix - now > _MAX_SESSION_TTL_SECONDS:
                    raise ValueError(
                        f"Session lifetime cannot exceed {_MAX_SESSION_TTL_SECONDS} seconds."
                    )
                self._connection.execute(
                    """
                    INSERT INTO agent_sessions (
                        session_id, principal_id, issued_at_unix, expires_at_unix
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (identity.session_id, identity.principal_id, now, float(expires_at_unix)),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                if "UNIQUE constraint failed: agent_sessions.session_id" in str(exc):
                    raise ValueError("Session ID already exists.") from exc
                raise SessionStoreIntegrityError("Session record violates database constraints.") from exc
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise SessionStoreError("Could not register session.") from exc
            except Exception:
                self._connection.rollback()
                raise

    def is_active(self, identity: RuntimeIdentity) -> bool:
        """Return true only for the exact principal/session pair while active."""
        if not isinstance(identity, RuntimeIdentity):
            return False
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN")
                row = self._connection.execute(
                    "SELECT * FROM agent_sessions WHERE session_id = ?",
                    (identity.session_id,),
                ).fetchone()
                now = self._now()
                if row is None:
                    self._connection.commit()
                    return False
                timestamps = (row["issued_at_unix"], row["expires_at_unix"])
                if (
                    row["principal_id"] != identity.principal_id
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in timestamps)
                    or row["expires_at_unix"] <= row["issued_at_unix"]
                    or row["expires_at_unix"] - row["issued_at_unix"] > _MAX_SESSION_TTL_SECONDS
                ):
                    raise SessionStoreIntegrityError("Stored session record is malformed.")
                active = (
                    row["revoked_at_unix"] is None
                    and row["issued_at_unix"] <= now
                    and row["expires_at_unix"] > now
                )
                self._connection.commit()
                return active
            except Exception:
                self._connection.rollback()
                raise

    def revoke(self, identity: RuntimeIdentity) -> bool:
        """Revoke the exact principal/session pair; revocation is irreversible."""
        if not isinstance(identity, RuntimeIdentity):
            raise TypeError("identity must be a RuntimeIdentity.")
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._now()
                cursor = self._connection.execute(
                    """
                    UPDATE agent_sessions SET revoked_at_unix = ?
                    WHERE session_id = ? AND principal_id = ? AND revoked_at_unix IS NULL
                    """,
                    (now, identity.session_id, identity.principal_id),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return False
                self._connection.commit()
                return True
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise SessionStoreError("Could not revoke session.") from exc
            except Exception:
                self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "SQLiteSessionStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
