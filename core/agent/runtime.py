"""Durable orchestration layer joining task state, approvals and tool dispatch.

This is a library component only. It does not expose HTTP endpoints, choose tools
from model output, or grant permissions. Trusted host code registers adapters,
supplies permissions and confirms risky operations.
"""
from __future__ import annotations

import math
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .contracts import (
    ApprovalGrant,
    ResultStatus,
    ToolAdapter,
    ToolCall,
    ToolDispatcher,
    ToolRegistry,
    ToolResult,
    arguments_digest,
    validate_json_value,
)
from .sqlite_approval_store import SQLiteApprovalStore
from .sqlite_task_state import SQLiteTaskStateStore
from .task_state import (
    PlanStep,
    StaleTaskRevisionError,
    StepStatus,
    TaskSnapshot,
    TaskStateError,
)


_MAX_APPROVAL_LIFETIME_SECONDS = 15 * 60
_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,95}")


class RuntimeClosedError(RuntimeError):
    """The automation runtime has already been closed."""


class RuntimeBusyError(RuntimeError):
    """The runtime still has active adapter calls and cannot close safely."""


class RuntimeRecoveryError(RuntimeError):
    """Recovery was requested without an exclusive startup guarantee."""


@dataclass(frozen=True)
class RuntimeStepResult:
    """The persisted task view and the in-memory result of one dispatch attempt.

    Tool output is returned to the caller but is never copied into the task
    ledger. Store a separately reviewed, redacted result reference when needed.
    """

    task: TaskSnapshot
    tool_result: ToolResult | None
    awaiting_approval: bool = False
    approval_id: str | None = None


class AutomationRuntime:
    """Single-process orchestration facade over the durable task and approval stores.

    The task ledger and approval ledger share one SQLite database file. Only
    registered adapters can execute, and every invocation passes through
    ToolDispatcher. Permission grants are supplied by trusted host code for
    each call; they are not inferred from plans or persisted by this class.

    A caller that starts a new process must explicitly call
    recover_interrupted_tasks(exclusive_startup=True) only after ensuring that
    no other runtime process is still executing against this database.
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
        self._clock = clock
        self._lock = threading.RLock()
        self._active: dict[tuple[str, str], threading.Event] = {}
        self._closed = False
        self._tasks = SQLiteTaskStateStore(
            database_path, clock=clock, timeout_seconds=timeout_seconds
        )
        try:
            self._approvals = SQLiteApprovalStore(
                database_path, clock=clock, timeout_seconds=timeout_seconds
            )
        except Exception:
            self._tasks.close()
            raise
        self._registry = ToolRegistry(approval_consumer=self._approvals.consume)
        self._dispatcher = ToolDispatcher(self._registry)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeClosedError("Automation runtime is closed.")

    def register_adapter(self, adapter: ToolAdapter) -> None:
        """Register an adapter from trusted host code, never from model output."""
        with self._lock:
            self._ensure_open()
            self._registry.register(adapter)

    def list_tools(self):
        """Return immutable tool manifests registered by the host."""
        with self._lock:
            self._ensure_open()
            return self._registry.manifests()

    def create_task(
        self,
        *,
        task_id: str,
        request_id: str,
        goal: str,
        steps: Iterable[PlanStep],
    ) -> TaskSnapshot:
        plan = tuple(steps)
        if any(
            not isinstance(step, PlanStep)
            or step.tool_id is None
            or step.operation is None
            for step in plan
        ):
            raise ValueError(
                "Every AutomationRuntime step must bind a tool_id and operation."
            )
        with self._lock:
            self._ensure_open()
            return self._tasks.create_task(
                task_id=task_id,
                request_id=request_id,
                goal=goal,
                steps=plan,
            )

    def get_task(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            self._ensure_open()
            return self._tasks.get_task(task_id)

    def _resolve_call(
        self, call: ToolCall
    ) -> tuple[TaskSnapshot, object | None, str | None]:
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall.")
        task = self._tasks.get_task(call.task_id)
        if task.request_id != call.request_id:
            return task, None, "request_id_mismatch"
        step = task.get_step(call.step_id)
        if step is None:
            return task, None, "unknown_step"
        if (
            step.plan.tool_id != call.tool_id
            or step.plan.operation != call.operation
        ):
            return task, None, "step_tool_binding_mismatch"
        manifest = self._registry.get_manifest(call.tool_id)
        if manifest is None or self._registry.get(call.tool_id) is None:
            return task, None, "unknown_tool"
        if call.manifest_version != manifest.manifest_version:
            return task, None, "unsupported_manifest_version"
        if call.tool_version != manifest.version:
            return task, None, "tool_version_mismatch"
        operation = manifest.get_operation(call.operation)
        if operation is None:
            return task, None, "unknown_operation"
        return task, operation, None

    @staticmethod
    def _grant_matches_call(grant: ApprovalGrant, call: ToolCall) -> bool:
        return (
            grant.request_id == call.request_id
            and grant.task_id == call.task_id
            and grant.step_id == call.step_id
            and grant.tool_id == call.tool_id
            and grant.tool_version == call.tool_version
            and grant.operation == call.operation
            and grant.idempotency_key == call.idempotency_key
            and grant.arguments_sha256 == arguments_digest(call.arguments)
            and grant.user_confirmed is True
        )

    @staticmethod
    def _denied(task: TaskSnapshot, code: str) -> RuntimeStepResult:
        return RuntimeStepResult(
            task=task,
            tool_result=ToolResult(status=ResultStatus.DENIED, error_code=code),
        )

    def _await_approval(
        self, task: TaskSnapshot, step_id: str, error_code: str | None = None
    ) -> RuntimeStepResult:
        step = task.get_step(step_id)
        if step is None:
            return self._denied(task, "unknown_step")
        if step.status not in {StepStatus.PENDING, StepStatus.WAITING_APPROVAL}:
            return self._denied(task, "invalid_step_state")
        approval_id = step.approval_id or secrets.token_urlsafe(32)
        task = self._tasks.require_approval(
            task.task_id,
            step_id,
            approval_id=approval_id,
            expected_revision=task.revision,
        )
        result = (
            None
            if error_code is None
            else ToolResult(status=ResultStatus.DENIED, error_code=error_code)
        )
        return RuntimeStepResult(
            task=task,
            tool_result=result,
            awaiting_approval=True,
            approval_id=approval_id,
        )

    def issue_approval(
        self,
        call: ToolCall,
        *,
        user_confirmed: bool,
        lifetime_seconds: float = 300.0,
    ) -> ApprovalGrant:
        """Issue a cryptographically random grant after a trusted host confirms.

        This method is not an approval UI. The host must invoke it only after an
        explicit user action that identifies the exact operation and arguments.
        """
        if user_confirmed is not True:
            raise ValueError("Explicit user confirmation is required.")
        if isinstance(lifetime_seconds, bool) or not isinstance(lifetime_seconds, (int, float)):
            raise ValueError("lifetime_seconds must be a finite positive number.")
        if (
            not math.isfinite(lifetime_seconds)
            or lifetime_seconds <= 0
            or lifetime_seconds > _MAX_APPROVAL_LIFETIME_SECONDS
        ):
            raise ValueError(
                f"lifetime_seconds must be greater than zero and at most "
                f"{_MAX_APPROVAL_LIFETIME_SECONDS}."
            )
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall.")

        with self._lock:
            self._ensure_open()
            task, operation, error = self._resolve_call(call)
            if error is not None:
                raise ValueError(error)
            if not operation.requires_approval:
                raise ValueError("This operation does not require an approval grant.")
            step = task.get_step(call.step_id)
            if step is None or step.status != StepStatus.WAITING_APPROVAL:
                raise TaskStateError("The task step must be waiting for approval.")
            validate_json_value(call.arguments, operation.input_schema)
            now = self._clock()
            if type(now) not in (int, float) or not math.isfinite(now):
                raise RuntimeError("Runtime clock must return a finite Unix timestamp.")

            # Reuse an active grant after a crash between durable grant issuance
            # and task-state approval. Otherwise rotate the opaque persisted ID.
            existing = (
                self._approvals.get(step.approval_id)
                if step.approval_id is not None
                else None
            )
            if existing is not None:
                if not self._grant_matches_call(existing, call):
                    raise ValueError("The persisted approval reference has a different scope.")
                grant = existing
            else:
                approval_id = secrets.token_urlsafe(32)
                task = self._tasks.require_approval(
                    task.task_id,
                    call.step_id,
                    approval_id=approval_id,
                    expected_revision=task.revision,
                )
                grant = ApprovalGrant(
                    approval_id=approval_id,
                    request_id=call.request_id,
                    task_id=call.task_id,
                    step_id=call.step_id,
                    tool_id=call.tool_id,
                    tool_version=call.tool_version,
                    operation=call.operation,
                    arguments_sha256=arguments_digest(call.arguments),
                    idempotency_key=call.idempotency_key,
                    expires_at_unix=float(now) + float(lifetime_seconds),
                    user_confirmed=True,
                )
                self._approvals.issue(grant)
            try:
                current = self._tasks.get_task(task.task_id)
                current_step = current.get_step(call.step_id)
                if current_step is None or current_step.status != StepStatus.WAITING_APPROVAL:
                    raise TaskStateError("The task step changed before approval was recorded.")
                self._tasks.approve_step(
                    task.task_id, call.step_id, expected_revision=current.revision
                )
            except Exception:
                # The approval must not remain usable if task-state advancement failed.
                self._approvals.revoke(grant.approval_id)
                raise
            return grant

    def execute_step(
        self,
        call: ToolCall,
        *,
        granted_permissions: frozenset[str] | set[str],
        approval_id: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> RuntimeStepResult:
        """Run one planned step through the dispatcher and persist its outcome."""
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall.")
        if cancel_event is not None and not isinstance(cancel_event, threading.Event):
            raise TypeError("cancel_event must be a threading.Event or None.")
        event = cancel_event if cancel_event is not None else threading.Event()
        active_key = (call.task_id, call.step_id)

        with self._lock:
            self._ensure_open()
            task, operation, error = self._resolve_call(call)
            if error is not None:
                return self._denied(task, error)
            if task.cancel_requested:
                return RuntimeStepResult(
                    task=task,
                    tool_result=ToolResult(
                        status=ResultStatus.CANCELLED,
                        error_code="task_cancel_requested",
                    ),
                )
            if event.is_set():
                return RuntimeStepResult(
                    task=task,
                    tool_result=ToolResult(
                        status=ResultStatus.CANCELLED,
                        error_code="cancelled_before_dispatch",
                    ),
                )
            if not isinstance(granted_permissions, (set, frozenset)) or any(
                not isinstance(value, str) or not value.strip()
                for value in granted_permissions
            ):
                return self._denied(task, "invalid_permission_context")
            if not operation.required_permissions.issubset(granted_permissions):
                return self._denied(task, "missing_permission")
            try:
                validate_json_value(call.arguments, operation.input_schema)
            except (KeyError, TypeError, ValueError):
                return self._denied(task, "invalid_arguments")

            step = task.get_step(call.step_id)
            if step is None:
                return self._denied(task, "unknown_step")
            approval: ApprovalGrant | None = None
            if operation.requires_approval:
                # The task snapshot is the source of truth for the reference;
                # callers may omit approval_id after a process restart.
                if approval_id is None:
                    approval_id = step.approval_id
                if approval_id is None:
                    return self._await_approval(task, call.step_id)
                if (
                    not isinstance(approval_id, str)
                    or not approval_id.strip()
                    or len(approval_id) > 128
                ):
                    return self._await_approval(
                        task, call.step_id, "approval_unavailable"
                    )
                if step.approval_id is not None and step.approval_id != approval_id:
                    return self._denied(task, "approval_reference_mismatch")
                approval = self._approvals.get(approval_id)
                if approval is None:
                    return self._await_approval(
                        task, call.step_id, "approval_required"
                    )
                if not self._grant_matches_call(approval, call):
                    return self._await_approval(
                        task, call.step_id, "approval_scope_mismatch"
                    )
                if step.status == StepStatus.WAITING_APPROVAL:
                    task = self._tasks.approve_step(
                        task.task_id, call.step_id, expected_revision=task.revision
                    )
                    step = task.get_step(call.step_id)
            elif approval_id is not None:
                return self._denied(task, "unexpected_approval")

            if step is None or step.status != StepStatus.PENDING:
                return self._denied(task, "invalid_step_state")
            if active_key in self._active:
                return self._denied(task, "step_already_active")

            self._active[active_key] = event
            try:
                task = self._tasks.start_step(
                    task.task_id, call.step_id, expected_revision=task.revision
                )
            except Exception:
                self._active.pop(active_key, None)
                raise

        try:
            try:
                result = self._dispatcher.dispatch(
                    call,
                    granted_permissions=granted_permissions,
                    approval=approval,
                    cancel_event=event,
                )
            except Exception:
                # An unexpected dispatcher exception is conservatively uncertain:
                # do not infer that an adapter had no external side effects.
                result = ToolResult(
                    status=ResultStatus.FAILED,
                    error_code="runtime_dispatch_error",
                    side_effects_may_have_occurred=True,
                )

            latest = self._tasks.get_task(call.task_id)
            if result.status == ResultStatus.SUCCESS:
                latest = self._tasks.complete_step(
                    call.task_id,
                    call.step_id,
                    expected_revision=latest.revision,
                )
            else:
                if result.status == ResultStatus.CANCELLED and not latest.cancel_requested:
                    latest = self._tasks.cancel_task(
                        call.task_id, expected_revision=latest.revision
                    )
                error_code = result.error_code
                if (
                    not isinstance(error_code, str)
                    or _SAFE_ERROR_CODE.fullmatch(error_code) is None
                ):
                    error_code = {
                        ResultStatus.PARTIAL: "partial_result_requires_review",
                        ResultStatus.FAILED: "tool_execution_failed",
                        ResultStatus.DENIED: "tool_dispatch_denied",
                        ResultStatus.TIMEOUT: "tool_timeout",
                        ResultStatus.CANCELLED: "tool_cancelled",
                    }.get(result.status, "tool_execution_failed")
                latest = self._tasks.fail_step(
                    call.task_id,
                    call.step_id,
                    error_code=error_code,
                    side_effects_may_have_occurred=result.side_effects_may_have_occurred,
                    expected_revision=latest.revision,
                )
            return RuntimeStepResult(task=latest, tool_result=result)
        finally:
            with self._lock:
                self._active.pop(active_key, None)

    def cancel_task(self, task_id: str) -> TaskSnapshot:
        """Persist a cancellation request and signal all active steps for the task."""
        with self._lock:
            self._ensure_open()
            task = self._tasks.get_task(task_id)
            updated = self._tasks.cancel_task(
                task_id, expected_revision=task.revision
            )
            for (active_task_id, _), event in tuple(self._active.items()):
                if active_task_id == task_id:
                    event.set()
            return updated

    def recover_interrupted_tasks(
        self, *, exclusive_startup: bool
    ) -> tuple[TaskSnapshot, ...]:
        """Mark persisted RUNNING steps uncertain after a confirmed exclusive restart.

        This never retries an operation. The caller must guarantee that no other
        process/runtime is currently executing against the same database.
        """
        if exclusive_startup is not True:
            raise RuntimeRecoveryError(
                "Recovery requires an explicit exclusive_startup=True guarantee."
            )
        with self._lock:
            self._ensure_open()
            if self._active:
                raise RuntimeBusyError("Cannot recover while this runtime has active steps.")
            recovered: list[TaskSnapshot] = []
            for task in self._tasks.list_tasks():
                current = self._tasks.get_task(task.task_id)
                for step in current.steps:
                    if step.status == StepStatus.RUNNING:
                        current = self._tasks.fail_step(
                            current.task_id,
                            step.plan.step_id,
                            error_code="runtime_restarted",
                            side_effects_may_have_occurred=True,
                            expected_revision=current.revision,
                        )
                if current.revision != task.revision:
                    recovered.append(current)
            return tuple(recovered)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active:
                for event in self._active.values():
                    event.set()
                raise RuntimeBusyError(
                    "Active steps were signalled to stop; wait for them to finish before closing."
                )
            self._approvals.close()
            self._tasks.close()
            self._closed = True

    def __enter__(self) -> "AutomationRuntime":
        with self._lock:
            self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
