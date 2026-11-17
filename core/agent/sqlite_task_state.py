"""SQLite-backed durable task ledger.

The store delegates workflow transitions to TaskStateStore, while each write
uses a SQLite IMMEDIATE transaction to serialize read/validate/write across
threads, store instances, and processes. Event rows are append-only.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

from .task_state import (
    PlanStep,
    StaleTaskRevisionError,
    StepSnapshot,
    StepStatus,
    TaskEvent,
    TaskNotFoundError,
    TaskSnapshot,
    TaskStateError,
    TaskStateStore,
    TaskStatus,
)

_SCHEMA_VERSION = 1
_MAX_DATABASE_PATH_LENGTH = 4096


class TaskPersistenceError(RuntimeError):
    """The durable task ledger could not be read or updated safely."""


class TaskStoreIntegrityError(TaskPersistenceError):
    """Persisted task or event data failed integrity validation."""


class SQLiteTaskStateStore:
    """Durable implementation of the task-state store API.

    Use one instance per application process and call close() during shutdown.
    The SQLite connection is serialized by an instance lock; BEGIN IMMEDIATE
    also serializes writers across independent instances/processes. No tool
    execution, approval authorization, or secret storage is performed here.
    """

    def __init__(
        self,
        database_path: str | Path,
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
        raw_path = str(database_path)
        if not raw_path or len(raw_path) > _MAX_DATABASE_PATH_LENGTH:
            raise ValueError("database_path must be a non-empty path of bounded length.")
        self._clock = clock
        self._lock = threading.RLock()
        self._closed = False
        self._database_path = raw_path
        if raw_path != ":memory:":
            path = Path(raw_path).expanduser()
            if path.exists() and path.is_symlink():
                raise ValueError("Database path must not be a symbolic link.")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and not path.is_file():
                raise ValueError("Database path must refer to a regular file.")
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
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > _SCHEMA_VERSION:
                raise TaskPersistenceError(
                    f"Database schema version {version} is newer than supported "
                    f"version {_SCHEMA_VERSION}."
                )
            if version == 0:
                connection.execute(
                    """
                    CREATE TABLE tasks (
                        task_id TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        snapshot_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE task_events (
                        task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE RESTRICT,
                        sequence INTEGER NOT NULL CHECK (sequence >= 1),
                        event_json TEXT NOT NULL,
                        PRIMARY KEY (task_id, sequence)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER task_events_no_update
                    BEFORE UPDATE ON task_events
                    BEGIN SELECT RAISE(ABORT, 'task events are append-only'); END
                    """
                )
                connection.execute(
                    """
                    CREATE TRIGGER task_events_no_delete
                    BEFORE DELETE ON task_events
                    BEGIN SELECT RAISE(ABORT, 'task events are append-only'); END
                    """
                )
                connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            connection.commit()
        except Exception:
            connection.rollback()
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "SQLiteTaskStateStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise TaskPersistenceError("Task store is closed.")

    @staticmethod
    def _event_to_dict(event: TaskEvent) -> dict:
        return {
            "sequence": event.sequence,
            "event_type": event.event_type,
            "step_id": event.step_id,
            "timestamp_unix": event.timestamp_unix,
            "revision": event.revision,
            "error_code": event.error_code,
        }

    @staticmethod
    def _step_to_dict(step: StepSnapshot) -> dict:
        return {
            "plan": {
                "step_id": step.plan.step_id,
                "description": step.plan.description,
                "depends_on": list(step.plan.depends_on),
                "tool_id": step.plan.tool_id,
                "operation": step.plan.operation,
            },
            "status": step.status.value,
            "attempt": step.attempt,
            "result_ref": step.result_ref,
            "error_code": step.error_code,
            "side_effects_may_have_occurred": step.side_effects_may_have_occurred,
        }

    @classmethod
    def _snapshot_to_json(cls, task: TaskSnapshot) -> str:
        payload = {
            "task_id": task.task_id,
            "request_id": task.request_id,
            "goal": task.goal,
            "status": task.status.value,
            "revision": task.revision,
            "created_at_unix": task.created_at_unix,
            "updated_at_unix": task.updated_at_unix,
            "cancel_requested": task.cancel_requested,
            "steps": [cls._step_to_dict(step) for step in task.steps],
        }
        return json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )

    @staticmethod
    def _event_from_json(raw: str, expected_sequence: int) -> TaskEvent:
        try:
            item = json.loads(raw)
            event = TaskEvent(
                sequence=item["sequence"],
                event_type=item["event_type"],
                step_id=item["step_id"],
                timestamp_unix=item["timestamp_unix"],
                revision=item["revision"],
                error_code=item["error_code"],
            )
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise TaskStoreIntegrityError("Stored task event is malformed.") from exc
        if (
            type(event.sequence) is not int
            or event.sequence != expected_sequence
            or type(event.revision) is not int
            or event.revision != expected_sequence
            or not isinstance(event.event_type, str)
            or not event.event_type
            or not isinstance(event.timestamp_unix, (int, float))
            or not math.isfinite(event.timestamp_unix)
            or (event.step_id is not None and not isinstance(event.step_id, str))
            or (event.error_code is not None and not isinstance(event.error_code, str))
        ):
            raise TaskStoreIntegrityError("Stored task event failed integrity checks.")
        return event

    @classmethod
    def _snapshot_from_row(cls, row: sqlite3.Row, event_rows: list[sqlite3.Row]) -> TaskSnapshot:
        try:
            payload = json.loads(row["snapshot_json"])
            steps = []
            for item in payload["steps"]:
                plan_data = item["plan"]
                plan = PlanStep(
                    step_id=plan_data["step_id"],
                    description=plan_data["description"],
                    depends_on=tuple(plan_data["depends_on"]),
                    tool_id=plan_data["tool_id"],
                    operation=plan_data["operation"],
                )
                attempt = item["attempt"]
                if type(attempt) is not int or attempt < 0:
                    raise ValueError("Invalid attempt count.")
                may_have_effects = item["side_effects_may_have_occurred"]
                if type(may_have_effects) is not bool:
                    raise ValueError("Invalid side-effect marker.")
                steps.append(
                    StepSnapshot(
                        plan=plan,
                        status=StepStatus(item["status"]),
                        attempt=attempt,
                        result_ref=item["result_ref"],
                        error_code=item["error_code"],
                        side_effects_may_have_occurred=may_have_effects,
                    )
                )
            if any(event_row["sequence"] != index for index, event_row in enumerate(event_rows, start=1)):
                raise TaskStoreIntegrityError("Stored task event sequence column is inconsistent.")
            events = tuple(
                cls._event_from_json(event_row["event_json"], index)
                for index, event_row in enumerate(event_rows, start=1)
            )
            task = TaskSnapshot(
                task_id=payload["task_id"],
                request_id=payload["request_id"],
                goal=payload["goal"],
                status=TaskStatus(payload["status"]),
                revision=payload["revision"],
                created_at_unix=payload["created_at_unix"],
                updated_at_unix=payload["updated_at_unix"],
                cancel_requested=payload["cancel_requested"],
                steps=tuple(steps),
                events=events,
            )
        except TaskStoreIntegrityError:
            raise
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise TaskStoreIntegrityError("Stored task snapshot is malformed.") from exc

        if (
            not isinstance(task.task_id, str)
            or task.task_id != row["task_id"]
            or not isinstance(task.request_id, str)
            or not isinstance(task.goal, str)
            or type(task.revision) is not int
            or task.revision < 1
            or task.revision != row["revision"]
            or len(task.events) != task.revision
            or not task.events
            or task.events[-1].revision != task.revision
            or not isinstance(task.cancel_requested, bool)
            or not isinstance(task.created_at_unix, (int, float))
            or not math.isfinite(task.created_at_unix)
            or not isinstance(task.updated_at_unix, (int, float))
            or not math.isfinite(task.updated_at_unix)
            or task.created_at_unix > task.updated_at_unix
            or row["created_at"] != task.created_at_unix
            or row["updated_at"] != task.updated_at_unix
            or not task.goal.strip()
            or len(task.goal) > 20_000
            or len(task.steps) == 0
            or len(task.steps) > 100
        ):
            raise TaskStoreIntegrityError("Stored task snapshot failed integrity checks.")
        try:
            TaskStateStore._validate_identifier(task.task_id, "task_id")
            TaskStateStore._validate_identifier(task.request_id, "request_id")
            step_ids = [step.plan.step_id for step in task.steps]
            if len(step_ids) != len(set(step_ids)):
                raise ValueError("Duplicate step identifiers.")
            known_step_ids = set(step_ids)
            if any(set(step.plan.depends_on) - known_step_ids for step in task.steps):
                raise ValueError("Unknown step dependency.")
            TaskStateStore._validate_acyclic(tuple(step.plan for step in task.steps))
        except (ValueError, KeyError) as exc:
            raise TaskStoreIntegrityError("Stored task dependency graph is invalid.") from exc
        return task

    def _load_task(self, task_id: str) -> TaskSnapshot:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        if row is None:
            raise TaskNotFoundError(task_id)
        events = self._connection.execute(
            "SELECT sequence, event_json FROM task_events "
            "WHERE task_id = ? ORDER BY sequence",
            (task_id,),
        ).fetchall()
        return self._snapshot_from_row(row, events)

    def _insert_task(self, task: TaskSnapshot) -> None:
        self._connection.execute(
            "INSERT INTO tasks(task_id, revision, snapshot_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                task.task_id,
                task.revision,
                self._snapshot_to_json(task),
                task.created_at_unix,
                task.updated_at_unix,
            ),
        )
        self._append_events(task, 0)

    def _append_events(self, task: TaskSnapshot, previous_event_count: int) -> None:
        new_events = task.events[previous_event_count:]
        if not new_events:
            return
        self._connection.executemany(
            "INSERT INTO task_events(task_id, sequence, event_json) VALUES (?, ?, ?)",
            [
                (
                    task.task_id,
                    event.sequence,
                    json.dumps(
                        self._event_to_dict(event),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    ),
                )
                for event in new_events
            ],
        )

    def _write_task(self, previous: TaskSnapshot, updated: TaskSnapshot) -> None:
        cursor = self._connection.execute(
            """
            UPDATE tasks
            SET revision = ?, snapshot_json = ?, updated_at = ?
            WHERE task_id = ? AND revision = ?
            """,
            (
                updated.revision,
                self._snapshot_to_json(updated),
                updated.updated_at_unix,
                previous.task_id,
                previous.revision,
            ),
        )
        if cursor.rowcount != 1:
            raise StaleTaskRevisionError("Task changed concurrently; reload before retrying.")
        self._append_events(updated, len(previous.events))

    def create_task(
        self,
        *,
        task_id: str,
        request_id: str,
        goal: str,
        steps: Iterable[PlanStep],
    ) -> TaskSnapshot:
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                memory_store = TaskStateStore(clock=self._clock)
                task = memory_store.create_task(
                    task_id=task_id, request_id=request_id, goal=goal, steps=steps
                )
                self._insert_task(task)
                self._connection.commit()
                return task
            except sqlite3.IntegrityError as exc:
                self._connection.rollback()
                raise TaskStateError(f"Task already exists: {task_id}") from exc
            except Exception:
                self._connection.rollback()
                raise

    def get_task(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            self._ensure_open()
            return self._load_task(task_id)

    def list_tasks(self) -> tuple[TaskSnapshot, ...]:
        with self._lock:
            self._ensure_open()
            rows = self._connection.execute(
                "SELECT task_id FROM tasks ORDER BY task_id"
            ).fetchall()
            return tuple(self._load_task(row["task_id"]) for row in rows)

    def _mutate(self, task_id: str, method: str, *args, **kwargs) -> TaskSnapshot:
        with self._lock:
            self._ensure_open()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                previous = self._load_task(task_id)
                memory_store = TaskStateStore(clock=self._clock)
                memory_store._tasks[task_id] = previous
                updated = getattr(memory_store, method)(task_id, *args, **kwargs)
                if updated.revision != previous.revision:
                    self._write_task(previous, updated)
                self._connection.commit()
                return updated
            except Exception:
                self._connection.rollback()
                raise

    def start_step(self, task_id: str, step_id: str, *, expected_revision: int | None = None) -> TaskSnapshot:
        return self._mutate(task_id, "start_step", step_id, expected_revision=expected_revision)

    def require_approval(self, task_id: str, step_id: str, *, expected_revision: int | None = None) -> TaskSnapshot:
        return self._mutate(task_id, "require_approval", step_id, expected_revision=expected_revision)

    def approve_step(self, task_id: str, step_id: str, *, expected_revision: int | None = None) -> TaskSnapshot:
        return self._mutate(task_id, "approve_step", step_id, expected_revision=expected_revision)

    def deny_step(self, task_id: str, step_id: str, *, expected_revision: int | None = None) -> TaskSnapshot:
        return self._mutate(task_id, "deny_step", step_id, expected_revision=expected_revision)

    def complete_step(
        self,
        task_id: str,
        step_id: str,
        *,
        result_ref: str | None = None,
        expected_revision: int | None = None,
    ) -> TaskSnapshot:
        return self._mutate(
            task_id, "complete_step", step_id,
            result_ref=result_ref, expected_revision=expected_revision,
        )

    def fail_step(
        self,
        task_id: str,
        step_id: str,
        *,
        error_code: str,
        side_effects_may_have_occurred: bool,
        expected_revision: int | None = None,
    ) -> TaskSnapshot:
        return self._mutate(
            task_id, "fail_step", step_id,
            error_code=error_code,
            side_effects_may_have_occurred=side_effects_may_have_occurred,
            expected_revision=expected_revision,
        )

    def cancel_task(self, task_id: str, *, expected_revision: int | None = None) -> TaskSnapshot:
        return self._mutate(task_id, "cancel_task", expected_revision=expected_revision)
