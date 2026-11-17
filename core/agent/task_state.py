"""Process-local task and plan state engine for automation orchestration.

This module deliberately does not execute tools, grant permissions, or persist
sensitive payloads. It tracks workflow control state; the host remains
responsible for durable storage, cancellation signalling, approvals and audit
retention.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, Iterable


_MAX_STEPS = 100
_MAX_EVENTS_PER_TASK = 10_000
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_APPROVAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9_.:-]{0,127}$")


class TaskStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    WAITING_APPROVAL = "waiting_approval"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


_TERMINAL_STEP_STATUSES = {
    StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.DENIED,
    StepStatus.CANCELLED, StepStatus.BLOCKED,
}


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    description: str
    depends_on: tuple[str, ...] = ()
    tool_id: str | None = None
    operation: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.step_id, str) or not _ID_PATTERN.fullmatch(self.step_id):
            raise ValueError("step_id must be a stable identifier (1-128 characters).")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Step description is required.")
        dependencies = tuple(self.depends_on)
        if any(not isinstance(item, str) or not _ID_PATTERN.fullmatch(item)
               for item in dependencies):
            raise ValueError("Step dependencies must be valid step identifiers.")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("Step dependencies must be unique.")
        if self.step_id in dependencies:
            raise ValueError("A step cannot depend on itself.")
        for label, value in (("tool_id", self.tool_id), ("operation", self.operation)):
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > 128
            ):
                raise ValueError(f"{label} must be a non-empty string of at most 128 characters.")
        if (self.tool_id is None) != (self.operation is None):
            raise ValueError("tool_id and operation must either both be set or both be omitted.")
        object.__setattr__(self, "depends_on", dependencies)


@dataclass(frozen=True)
class StepSnapshot:
    plan: PlanStep
    status: StepStatus = StepStatus.PENDING
    attempt: int = 0
    result_ref: str | None = None
    error_code: str | None = None
    side_effects_may_have_occurred: bool = False
    # Opaque lookup reference only; the dispatcher must still validate the grant.
    approval_id: str | None = None

    def __post_init__(self) -> None:
        if self.approval_id is not None and (
            not isinstance(self.approval_id, str)
            or not _APPROVAL_ID_PATTERN.fullmatch(self.approval_id)
        ):
            raise ValueError("approval_id must be a stable identifier (1-128 characters).")


@dataclass(frozen=True)
class TaskEvent:
    sequence: int
    event_type: str
    step_id: str | None
    timestamp_unix: float
    revision: int
    error_code: str | None = None


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: str
    request_id: str
    goal: str
    status: TaskStatus
    revision: int
    created_at_unix: float
    updated_at_unix: float
    cancel_requested: bool
    steps: tuple[StepSnapshot, ...]
    events: tuple[TaskEvent, ...]

    def get_step(self, step_id: str) -> StepSnapshot | None:
        return next((step for step in self.steps if step.plan.step_id == step_id), None)


class TaskStateError(ValueError):
    """A requested task transition violates the workflow state machine."""


class TaskNotFoundError(KeyError):
    """The requested task does not exist in this store."""


class StaleTaskRevisionError(TaskStateError):
    """A caller attempted to update a task snapshot that is no longer current."""


class TaskStateStore:
    """Thread-safe, process-local task ledger with optimistic revision checks.

    This is intentionally an in-memory foundation, not durable storage. Do not
    treat it as an audit log or rely on it across process restarts.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._tasks: dict[str, TaskSnapshot] = {}

    def create_task(
        self,
        *,
        task_id: str,
        request_id: str,
        goal: str,
        steps: Iterable[PlanStep],
    ) -> TaskSnapshot:
        self._validate_identifier(task_id, "task_id")
        self._validate_identifier(request_id, "request_id")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 20_000:
            raise ValueError("goal must be non-empty and at most 20,000 characters.")
        plan = tuple(steps)
        if not plan or len(plan) > _MAX_STEPS:
            raise ValueError(f"A task must contain between 1 and {_MAX_STEPS} steps.")
        if any(not isinstance(step, PlanStep) for step in plan):
            raise ValueError("steps must contain PlanStep values.")
        ids = [step.step_id for step in plan]
        if len(ids) != len(set(ids)):
            raise ValueError("Step identifiers must be unique.")
        known = set(ids)
        for step in plan:
            missing = set(step.depends_on) - known
            if missing:
                raise ValueError(f"Step {step.step_id} has unknown dependencies: {sorted(missing)}.")
        self._validate_acyclic(plan)

        with self._lock:
            if task_id in self._tasks:
                raise TaskStateError(f"Task already exists: {task_id}")
            now = self._clock()
            event = TaskEvent(1, "task_created", None, now, 1)
            snapshot = TaskSnapshot(
                task_id=task_id,
                request_id=request_id,
                goal=goal.strip(),
                status=TaskStatus.READY,
                revision=1,
                created_at_unix=now,
                updated_at_unix=now,
                cancel_requested=False,
                steps=tuple(StepSnapshot(step) for step in plan),
                events=(event,),
            )
            self._tasks[task_id] = snapshot
            return snapshot

    def get_task(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            try:
                return self._tasks[task_id]
            except KeyError as exc:
                raise TaskNotFoundError(task_id) from exc

    def list_tasks(self) -> tuple[TaskSnapshot, ...]:
        with self._lock:
            return tuple(self._tasks[key] for key in sorted(self._tasks))

    def start_step(
        self, task_id: str, step_id: str, *, expected_revision: int | None = None
    ) -> TaskSnapshot:
        with self._lock:
            task = self._current(task_id, expected_revision)
            self._ensure_event_capacity(task, 2)
            if task.cancel_requested:
                raise TaskStateError("A cancellation-requested task cannot start new steps.")
            step = self._step(task, step_id)
            if step.status != StepStatus.PENDING:
                raise TaskStateError("Only pending steps can start.")
            by_id = {item.plan.step_id: item for item in task.steps}
            if any(by_id[dep].status != StepStatus.SUCCEEDED for dep in step.plan.depends_on):
                raise TaskStateError("All dependencies must succeed before a step can start.")
            return self._replace_step(
                task, replace(step, status=StepStatus.RUNNING, attempt=step.attempt + 1),
                "step_started", step_id,
            )

    def require_approval(
        self,
        task_id: str,
        step_id: str,
        *,
        approval_id: str | None = None,
        expected_revision: int | None = None,
    ) -> TaskSnapshot:
        if approval_id is not None:
            self._validate_approval_identifier(approval_id)
        with self._lock:
            task = self._current(task_id, expected_revision)
            self._ensure_event_capacity(task, 2)
            if task.cancel_requested:
                raise TaskStateError("A cancellation-requested task cannot request approval.")
            step = self._step(task, step_id)
            if step.status == StepStatus.WAITING_APPROVAL:
                if approval_id is None or step.approval_id == approval_id:
                    return task
                event_type = (
                    "approval_reference_attached"
                    if step.approval_id is None
                    else "approval_reference_replaced"
                )
                return self._replace_step(
                    task,
                    replace(step, approval_id=approval_id),
                    event_type, step_id,
                )
            if step.status != StepStatus.PENDING:
                raise TaskStateError("Approval can only be requested for a pending or waiting step.")
            return self._replace_step(
                task,
                replace(
                    step,
                    status=StepStatus.WAITING_APPROVAL,
                    approval_id=approval_id,
                ),
                "approval_requested", step_id,
            )

    def approve_step(
        self, task_id: str, step_id: str, *, expected_revision: int | None = None
    ) -> TaskSnapshot:
        """Record that approval was obtained; dispatcher must still validate its grant."""
        with self._lock:
            task = self._current(task_id, expected_revision)
            self._ensure_event_capacity(task, 2)
            if task.cancel_requested:
                raise TaskStateError("A cancellation-requested task cannot be approved.")
            step = self._step(task, step_id)
            if step.status != StepStatus.WAITING_APPROVAL:
                raise TaskStateError("Only steps waiting for approval can be approved.")
            return self._replace_step(
                task, replace(step, status=StepStatus.PENDING),
                "approval_recorded", step_id,
            )

    def deny_step(
        self, task_id: str, step_id: str, *, expected_revision: int | None = None
    ) -> TaskSnapshot:
        with self._lock:
            task = self._current(task_id, expected_revision)
            self._ensure_event_capacity(task, 4)
            step = self._step(task, step_id)
            if step.status != StepStatus.WAITING_APPROVAL:
                raise TaskStateError("Only steps waiting for approval can be denied.")
            updated = self._replace_step(
                task, replace(step, status=StepStatus.DENIED, error_code="user_denied"),
                "approval_denied", step_id, error_code="user_denied",
            )
            return self._block_dependents(updated, step_id)

    def complete_step(
        self,
        task_id: str,
        step_id: str,
        *,
        result_ref: str | None = None,
        expected_revision: int | None = None,
    ) -> TaskSnapshot:
        if result_ref is not None and (
            not isinstance(result_ref, str) or not result_ref.strip() or len(result_ref) > 512
        ):
            raise ValueError("result_ref must be a non-empty reference of at most 512 characters.")
        with self._lock:
            task = self._current(task_id, expected_revision)
            self._ensure_event_capacity(task, 2)
            step = self._step(task, step_id)
            if step.status != StepStatus.RUNNING:
                raise TaskStateError("Only running steps can complete.")
            updated = self._replace_step(
                task, replace(step, status=StepStatus.SUCCEEDED, result_ref=result_ref),
                "step_succeeded", step_id,
            )
            return self._refresh_status(updated)

    def fail_step(
        self,
        task_id: str,
        step_id: str,
        *,
        error_code: str,
        side_effects_may_have_occurred: bool,
        expected_revision: int | None = None,
    ) -> TaskSnapshot:
        self._validate_error(error_code, side_effects_may_have_occurred)
        with self._lock:
            task = self._current(task_id, expected_revision)
            self._ensure_event_capacity(task, 4)
            step = self._step(task, step_id)
            if step.status != StepStatus.RUNNING:
                raise TaskStateError("Only running steps can fail.")
            updated = self._replace_step(
                task,
                replace(step, status=StepStatus.FAILED, error_code=error_code,
                        side_effects_may_have_occurred=side_effects_may_have_occurred),
                "step_failed", step_id, error_code=error_code,
            )
            return self._block_dependents(updated, step_id)

    def cancel_task(
        self, task_id: str, *, expected_revision: int | None = None
    ) -> TaskSnapshot:
        with self._lock:
            task = self._current(task_id, expected_revision)
            self._ensure_event_capacity(task, 2)
            if task.status in {TaskStatus.SUCCEEDED, TaskStatus.PARTIAL,
                               TaskStatus.FAILED, TaskStatus.CANCELLED}:
                raise TaskStateError("A terminal task cannot be cancelled.")
            steps = tuple(
                replace(step, status=StepStatus.CANCELLED)
                if step.status in {StepStatus.PENDING, StepStatus.WAITING_APPROVAL}
                else step
                for step in task.steps
            )
            updated = self._append_event(
                replace(task, steps=steps, cancel_requested=True),
                "task_cancel_requested", None,
            )
            return self._refresh_status(updated)

    @staticmethod
    def _validate_identifier(value: str, label: str) -> None:
        if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
            raise ValueError(f"{label} must be a stable identifier (1-128 characters).")

    @staticmethod
    def _validate_approval_identifier(value: str) -> None:
        if not isinstance(value, str) or not _APPROVAL_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "approval_id must be a URL-safe stable identifier (1-128 characters)."
            )

    @staticmethod
    def _validate_acyclic(steps: tuple[PlanStep, ...]) -> None:
        dependencies = {step.step_id: step.depends_on for step in steps}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in visiting:
                raise ValueError("Plan dependencies must be acyclic.")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in dependencies[step_id]:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in dependencies:
            visit(step_id)

    @staticmethod
    def _ensure_event_capacity(task: TaskSnapshot, required: int) -> None:
        if len(task.events) + required > _MAX_EVENTS_PER_TASK:
            raise TaskStateError(
                "Task event limit would be exceeded; archive the task before continuing."
            )

    def _current(self, task_id: str, expected_revision: int | None) -> TaskSnapshot:
        task = self.get_task(task_id)
        if expected_revision is not None:
            if type(expected_revision) is not int or expected_revision < 1:
                raise ValueError("expected_revision must be a positive integer.")
            if task.revision != expected_revision:
                raise StaleTaskRevisionError(
                    f"Expected task revision {expected_revision}, found {task.revision}."
                )
        return task

    @staticmethod
    def _step(task: TaskSnapshot, step_id: str) -> StepSnapshot:
        for step in task.steps:
            if step.plan.step_id == step_id:
                return step
        raise TaskStateError(f"Unknown step: {step_id}")

    def _replace_step(
        self,
        task: TaskSnapshot,
        replacement: StepSnapshot,
        event_type: str,
        step_id: str,
        *,
        error_code: str | None = None,
    ) -> TaskSnapshot:
        steps = tuple(
            replacement if step.plan.step_id == step_id else step
            for step in task.steps
        )
        updated = self._append_event(
            replace(task, steps=steps), event_type, step_id, error_code=error_code
        )
        return self._refresh_status(updated)

    def _append_event(
        self,
        task: TaskSnapshot,
        event_type: str,
        step_id: str | None,
        *,
        error_code: str | None = None,
    ) -> TaskSnapshot:
        if len(task.events) >= _MAX_EVENTS_PER_TASK:
            raise TaskStateError("Task event limit reached; archive the task before continuing.")
        now = self._clock()
        revision = task.revision + 1
        event = TaskEvent(
            sequence=len(task.events) + 1,
            event_type=event_type,
            step_id=step_id,
            timestamp_unix=now,
            revision=revision,
            error_code=error_code,
        )
        updated = replace(
            task, revision=revision, updated_at_unix=now,
            events=task.events + (event,),
        )
        self._tasks[task.task_id] = updated
        return updated

    def _refresh_status(self, task: TaskSnapshot) -> TaskSnapshot:
        statuses = [step.status for step in task.steps]
        if any(status == StepStatus.RUNNING for status in statuses):
            status = TaskStatus.RUNNING
        elif any(status == StepStatus.WAITING_APPROVAL for status in statuses):
            status = TaskStatus.WAITING_APPROVAL
        elif any(status == StepStatus.PENDING for status in statuses):
            status = TaskStatus.READY
        elif all(status == StepStatus.SUCCEEDED for status in statuses):
            status = TaskStatus.SUCCEEDED
        elif any(status == StepStatus.SUCCEEDED for status in statuses):
            status = TaskStatus.PARTIAL
        elif task.cancel_requested or all(
            status in {StepStatus.CANCELLED, StepStatus.DENIED, StepStatus.BLOCKED}
            for status in statuses
        ):
            status = TaskStatus.CANCELLED
        else:
            status = TaskStatus.FAILED
        if status == task.status:
            return task
        return self._append_event(replace(task, status=status), "task_status_changed", None)

    def _block_dependents(self, task: TaskSnapshot, failed_step_id: str) -> TaskSnapshot:
        failed = {failed_step_id}
        changed = True
        while changed:
            changed = False
            for step in task.steps:
                if (
                    step.plan.step_id not in failed
                    and step.status in {StepStatus.PENDING, StepStatus.WAITING_APPROVAL}
                    and any(dependency in failed for dependency in step.plan.depends_on)
                ):
                    failed.add(step.plan.step_id)
                    changed = True
        if len(failed) == 1:
            return self._refresh_status(task)
        steps = tuple(
            replace(step, status=StepStatus.BLOCKED, error_code="dependency_not_satisfied")
            if step.plan.step_id in failed and step.plan.step_id != failed_step_id
            else step
            for step in task.steps
        )
        updated = self._append_event(
            replace(task, steps=steps), "dependent_steps_blocked", failed_step_id,
            error_code="dependency_not_satisfied",
        )
        return self._refresh_status(updated)

    @staticmethod
    def _validate_error(error_code: str, side_effects_may_have_occurred: bool) -> None:
        if not isinstance(error_code, str) or not re.fullmatch(
            r"[a-z][a-z0-9_.-]{0,95}", error_code
        ):
            raise ValueError("error_code must be a stable lowercase error identifier.")
        if type(side_effects_may_have_occurred) is not bool:
            raise ValueError("side_effects_may_have_occurred must be a boolean.")
