"""SQLite-backed, single-use approval grant storage.

Only a trusted host should issue a grant after explicit user confirmation.
The store persists the exact operation scope and argument digest, never the
raw arguments or credentials. Consumption and revocation are atomic.
"""
from __future__ import annotations

import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

from .contracts import ApprovalGrant

_SCHEMA_VERSION = 1
_MAX_PATH_LENGTH = 4096
_MAX_APPROVAL_TTL_SECONDS = 15 * 60


class ApprovalStoreError(RuntimeError):
    """The durable approval store could not safely complete an operation."""


class ApprovalStoreIntegrityError(ApprovalStoreError):
    """The approval database schema or stored records failed integrity checks."""


class SQLiteApprovalStore:
    """Durable single-use approval grants backed by standard-library SQLite.

    Use one instance per application process and close it during shutdown.
    Separate instances/processes may safely share the same database file.
    """

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
        if not isinstance(raw_path, str) or not raw_path.strip() or len(raw_path) > _MAX_PATH_LENGTH:
            raise ValueError("database_path must be a non-empty path of bounded length.")

        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._database_path = raw_path
        if raw_path != ":memory:":
            path = Path(raw_path).expanduser()
            if path.is_symlink():
                raise ValueError("Approval database path must not be a symbolic link.")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not path.is_file():
                raise ValueError("Approval database path must refer to a regular file.")
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
                    # Windows ACLs and restricted filesystems may not support chmod.
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
                CREATE TABLE IF NOT EXISTS approval_store_metadata (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    schema_version INTEGER NOT NULL CHECK (schema_version >= 1)
                )
                """
            )
            metadata = connection.execute(
                "SELECT schema_version FROM approval_store_metadata WHERE singleton = 1"
            ).fetchone()
            version = 0 if metadata is None else metadata["schema_version"]
            if type(version) is not int or version < 0 or version > _SCHEMA_VERSION:
                raise ApprovalStoreError(
                    f"Approval schema version {version} is newer than supported "
                    f"version {_SCHEMA_VERSION}."
                )
            if version == 0:
                connection.execute(
                    """
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
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER approval_grants_scope_immutable
                    BEFORE UPDATE OF
                        approval_id, request_id, task_id, step_id, tool_id,
                        tool_version, operation, arguments_sha256,
                        idempotency_key, expires_at_unix, user_confirmed,
                        issued_at_unix
                    ON approval_grants
                    BEGIN
                        SELECT RAISE(ABORT, 'approval grant scope is immutable');
                    END
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER approval_grants_terminal_once
                    BEFORE UPDATE OF consumed_at_unix, revoked_at_unix
                    ON approval_grants
                    WHEN OLD.consumed_at_unix IS NOT NULL
                      OR OLD.revoked_at_unix IS NOT NULL
                      OR (NEW.consumed_at_unix IS NULL AND NEW.revoked_at_unix IS NULL)
                      OR (NEW.consumed_at_unix IS NOT NULL AND NEW.revoked_at_unix IS NOT NULL)
                    BEGIN
                        SELECT RAISE(ABORT, 'approval grant is already terminal');
                    END
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER approval_grants_no_delete
                    BEFORE DELETE ON approval_grants
                    BEGIN
                        SELECT RAISE(ABORT, 'approval grants cannot be deleted');
                    END
                    """
                )
                connection.execute(
                    "INSERT INTO approval_store_metadata(singleton, schema_version) VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
            else:
                required_columns = {
                    "approval_id", "request_id", "task_id", "step_id", "tool_id",
                    "tool_version", "operation", "arguments_sha256", "idempotency_key",
                    "expires_at_unix", "user_confirmed", "issued_at_unix",
                    "consumed_at_unix", "revoked_at_unix",
                }
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(approval_grants)").fetchall()
                }
                triggers = {
                    row["name"]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                    ).fetchall()
                }
                metadata_exists = connection.execute(
                    "SELECT 1 FROM approval_store_metadata WHERE singleton = 1"
                ).fetchone()
                if metadata_exists is None:
                    raise ApprovalStoreIntegrityError("Approval schema metadata is missing.")
                required_triggers = {
                    "approval_grants_scope_immutable",
                    "approval_grants_terminal_once",
                    "approval_grants_no_delete",
                }
                if not required_columns.issubset(columns) or not required_triggers.issubset(triggers):
                    raise ApprovalStoreIntegrityError("Approval database schema is incomplete.")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise ApprovalStoreError("Approval store is closed.")

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ApprovalStoreError("Approval store clock must return a finite Unix timestamp.")
        return float(value)

    @staticmethod
    def _validate_grant(grant: ApprovalGrant) -> None:
        if not isinstance(grant, ApprovalGrant):
            raise TypeError("grant must be an ApprovalGrant.")
        limits = {
            "approval_id": 128,
            "request_id": 128,
            "task_id": 128,
            "step_id": 128,
            "tool_id": 96,
            "tool_version": 128,
            "operation": 64,
        }
        for name, maximum in limits.items():
            value = getattr(grant, name)
            if not isinstance(value, str) or not value.strip() or len(value) > maximum:
                raise ValueError(f"{name} must be non-empty and at most {maximum} characters.")
        if grant.idempotency_key is not None and len(grant.idempotency_key) > 256:
            raise ValueError("idempotency_key cannot exceed 256 characters.")
        if grant.user_confirmed is not True:
            raise ValueError("Only explicitly user-confirmed grants may be issued.")

    @staticmethod
    def _matches(row: sqlite3.Row, grant: ApprovalGrant) -> bool:
        return (
            row["approval_id"] == grant.approval_id
            and row["request_id"] == grant.request_id
            and row["task_id"] == grant.task_id
            and row["step_id"] == grant.step_id
            and row["tool_id"] == grant.tool_id
            and row["tool_version"] == grant.tool_version
            and row["operation"] == grant.operation
            and row["arguments_sha256"] == grant.arguments_sha256
            and row["idempotency_key"] == grant.idempotency_key
            and float(row["expires_at_unix"]) == float(grant.expires_at_unix)
            and row["user_confirmed"] == 1
            and grant.user_confirmed is True
        )

    def issue(self, grant: ApprovalGrant) -> None:
        """Persist a host-created grant after user confirmation."""
        self._validate_grant(grant)
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._now()
                if grant.expires_at_unix <= now:
                    raise ValueError("An expired approval grant cannot be issued.")
                if grant.expires_at_unix - now > _MAX_APPROVAL_TTL_SECONDS:
                    raise ValueError(
                        f"Approval grant lifetime cannot exceed {_MAX_APPROVAL_TTL_SECONDS} seconds."
                    )
                self._connection.execute(
                    """
                    INSERT INTO approval_grants (
                        approval_id, request_id, task_id, step_id, tool_id,
                        tool_version, operation, arguments_sha256,
                        idempotency_key, expires_at_unix, user_confirmed,
                        issued_at_unix
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        grant.approval_id, grant.request_id, grant.task_id,
                        grant.step_id, grant.tool_id, grant.tool_version,
                        grant.operation, grant.arguments_sha256,
                        grant.idempotency_key, float(grant.expires_at_unix), now,
                    ),
                )
                self._connection.commit()
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                if "UNIQUE constraint failed: approval_grants.approval_id" in str(exc):
                    raise ValueError("Approval ID already exists.") from exc
                raise ApprovalStoreIntegrityError("Approval grant violates database constraints.") from exc
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise ApprovalStoreError("Could not persist approval grant.") from exc
            except Exception:
                self._connection.rollback()
                raise

    @staticmethod
    def _grant_from_row(row: sqlite3.Row) -> ApprovalGrant:
        try:
            if type(row["user_confirmed"]) is not int or row["user_confirmed"] != 1:
                raise ValueError("Stored confirmation marker is invalid.")
            issued_at = row["issued_at_unix"]
            expires_at = row["expires_at_unix"]
            if (
                type(issued_at) not in (int, float)
                or not math.isfinite(issued_at)
                or type(expires_at) not in (int, float)
                or not math.isfinite(expires_at)
                or expires_at <= issued_at
                or expires_at - issued_at > _MAX_APPROVAL_TTL_SECONDS
            ):
                raise ValueError("Stored approval timestamps are invalid.")
            return ApprovalGrant(
                approval_id=row["approval_id"],
                request_id=row["request_id"],
                task_id=row["task_id"],
                step_id=row["step_id"],
                tool_id=row["tool_id"],
                tool_version=row["tool_version"],
                operation=row["operation"],
                arguments_sha256=row["arguments_sha256"],
                idempotency_key=row["idempotency_key"],
                expires_at_unix=expires_at,
                user_confirmed=True,
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ApprovalStoreIntegrityError("Stored approval grant is malformed.") from exc

    def get(self, approval_id: str) -> ApprovalGrant | None:
        """Load an active, unexpired grant after an application restart."""
        if not isinstance(approval_id, str) or not approval_id.strip() or len(approval_id) > 128:
            raise ValueError("approval_id must be a non-empty identifier of at most 128 characters.")
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN")
                row = self._connection.execute(
                    "SELECT * FROM approval_grants WHERE approval_id = ?",
                    (approval_id,),
                ).fetchone()
                if row is None:
                    self._connection.commit()
                    return None
                grant = self._grant_from_row(row)
                now = self._now()
                if (
                    row["consumed_at_unix"] is not None
                    or row["revoked_at_unix"] is not None
                    or row["issued_at_unix"] > now
                    or grant.expires_at_unix <= now
                ):
                    self._connection.commit()
                    return None
                self._connection.commit()
                return grant
            except Exception:
                self._connection.rollback()
                raise

    def consume(self, grant: ApprovalGrant) -> bool:
        """Atomically match and consume an unexpired grant exactly once."""
        if not isinstance(grant, ApprovalGrant):
            return False
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT * FROM approval_grants WHERE approval_id = ?",
                    (grant.approval_id,),
                ).fetchone()
                if row is None:
                    self._connection.rollback()
                    return False
                stored_grant = self._grant_from_row(row)
                now = self._now()
                if (
                    not self._matches(row, grant)
                    or row["issued_at_unix"] > now
                    or row["consumed_at_unix"] is not None
                    or row["revoked_at_unix"] is not None
                    or stored_grant.expires_at_unix <= now
                ):
                    self._connection.rollback()
                    return False
                cursor = self._connection.execute(
                    """
                    UPDATE approval_grants
                    SET consumed_at_unix = ?
                    WHERE approval_id = ?
                      AND consumed_at_unix IS NULL
                      AND revoked_at_unix IS NULL
                    """,
                    (now, grant.approval_id),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return False
                self._connection.commit()
                return True
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise ApprovalStoreError("Could not consume approval grant.") from exc
            except Exception:
                self._connection.rollback()
                raise

    def revoke(self, approval_id: str) -> bool:
        """Revoke an unused grant. A consumed or previously revoked grant stays terminal."""
        if not isinstance(approval_id, str) or not approval_id.strip() or len(approval_id) > 128:
            raise ValueError("approval_id must be a non-empty identifier of at most 128 characters.")
        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                now = self._now()
                cursor = self._connection.execute(
                    """
                    UPDATE approval_grants
                    SET revoked_at_unix = ?
                    WHERE approval_id = ?
                      AND consumed_at_unix IS NULL
                      AND revoked_at_unix IS NULL
                    """,
                    (now, approval_id),
                )
                if cursor.rowcount != 1:
                    self._connection.rollback()
                    return False
                self._connection.commit()
                return True
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise ApprovalStoreError("Could not revoke approval grant.") from exc
            except Exception:
                self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "SQLiteApprovalStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
