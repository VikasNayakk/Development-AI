import threading
import unittest
from unittest.mock import patch

from core.agent.task_state import (
    PlanStep,
    StepStatus,
    StaleTaskRevisionError,
    TaskStateError,
    TaskStateStore,
    TaskStatus,
)


def make_store():
    return TaskStateStore(clock=lambda: 100.0)


def linear_plan():
    return (
        PlanStep("inspect", "Inspect the selected page."),
        PlanStep("prepare", "Prepare the requested change.", ("inspect",)),
        PlanStep("verify", "Verify the resulting page.", ("prepare",)),
    )


class TaskPlanValidationTests(unittest.TestCase):
    def test_rejects_empty_or_oversized_plans(self):
        with self.assertRaises(ValueError):
            make_store().create_task(task_id="t1", request_id="r1", goal="goal", steps=())
        with self.assertRaises(ValueError):
            make_store().create_task(
                task_id="t1", request_id="r1", goal="goal",
                steps=tuple(PlanStep(f"s{i}", "step") for i in range(101)),
            )

    def test_rejects_duplicate_ids_unknown_dependencies_and_cycles(self):
        cases = (
            (PlanStep("a", "A"), PlanStep("a", "duplicate")),
            (PlanStep("a", "A", ("missing",)),),
            (PlanStep("a", "A", ("b",)), PlanStep("b", "B", ("a",))),
        )
        for steps in cases:
            with self.subTest(steps=steps):
                with self.assertRaises(ValueError):
                    make_store().create_task(
                        task_id="t1", request_id="r1", goal="goal", steps=steps
                    )

    def test_rejects_invalid_tool_binding_and_self_dependency(self):
        with self.assertRaises(ValueError):
            PlanStep("a", "A", ("a",))
        with self.assertRaises(ValueError):
            PlanStep("a", "A", tool_id="browser.read")


class TaskLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.store = make_store()
        self.task = self.store.create_task(
            task_id="task-1", request_id="request-1",
            goal="Inspect and verify a page.", steps=linear_plan(),
        )

    def test_dependencies_must_succeed_before_step_can_start(self):
        with self.assertRaises(TaskStateError):
            self.store.start_step("task-1", "prepare")
        started = self.store.start_step(
            "task-1", "inspect", expected_revision=self.task.revision
        )
        self.assertEqual(started.status, TaskStatus.RUNNING)
        self.assertEqual(started.get_step("inspect").attempt, 1)

    def test_linear_workflow_reaches_success_only_after_every_step_succeeds(self):
        snapshot = self.store.start_step("task-1", "inspect")
        snapshot = self.store.complete_step("task-1", "inspect")
        self.assertEqual(snapshot.status, TaskStatus.READY)
        snapshot = self.store.start_step("task-1", "prepare")
        snapshot = self.store.complete_step("task-1", "prepare", result_ref="result://prepare")
        snapshot = self.store.start_step("task-1", "verify")
        snapshot = self.store.complete_step("task-1", "verify")
        self.assertEqual(snapshot.status, TaskStatus.SUCCEEDED)
        self.assertEqual(snapshot.get_step("prepare").result_ref, "result://prepare")

    def test_failure_blocks_transitive_dependents_but_not_independent_steps(self):
        plan = (
            PlanStep("root", "Root step."),
            PlanStep("child", "Child step.", ("root",)),
            PlanStep("grandchild", "Grandchild step.", ("child",)),
            PlanStep("independent", "Independent step."),
        )
        self.store.create_task(
            task_id="task-2", request_id="request-2", goal="Run a DAG.", steps=plan
        )
        snapshot = self.store.start_step("task-2", "root")
        snapshot = self.store.fail_step(
            "task-2", "root", error_code="adapter_error",
            side_effects_may_have_occurred=True,
        )
        self.assertEqual(snapshot.get_step("child").status, StepStatus.BLOCKED)
        self.assertEqual(snapshot.get_step("grandchild").status, StepStatus.BLOCKED)
        self.assertEqual(snapshot.get_step("independent").status, StepStatus.PENDING)
        snapshot = self.store.start_step("task-2", "independent")
        snapshot = self.store.complete_step("task-2", "independent")
        self.assertEqual(snapshot.status, TaskStatus.PARTIAL)

    def test_approval_reference_is_retained_but_is_not_execution_permission(self):
        snapshot = self.store.require_approval(
            "task-1", "inspect", approval_id="_approval-opaque-123"
        )
        self.assertEqual(snapshot.status, TaskStatus.WAITING_APPROVAL)
        self.assertEqual(
            snapshot.get_step("inspect").approval_id, "_approval-opaque-123"
        )
        snapshot = self.store.approve_step("task-1", "inspect")
        self.assertEqual(snapshot.get_step("inspect").status, StepStatus.PENDING)
        self.assertEqual(
            snapshot.get_step("inspect").approval_id, "approval-opaque-123"
        )
        self.assertEqual(snapshot.status, TaskStatus.READY)
        self.assertIn("approval_recorded", [event.event_type for event in snapshot.events])

    def test_invalid_approval_reference_is_rejected_without_state_change(self):
        with self.assertRaises(ValueError):
            self.store.require_approval(
                "task-1", "inspect", approval_id="not a valid id"
            )
        snapshot = self.store.get_task("task-1")
        self.assertEqual(snapshot.revision, self.task.revision)
        self.assertEqual(snapshot.get_step("inspect").status, StepStatus.PENDING)

    def test_denial_blocks_dependents(self):
        snapshot = self.store.require_approval("task-1", "inspect")
        snapshot = self.store.deny_step("task-1", "inspect")
        self.assertEqual(snapshot.get_step("inspect").status, StepStatus.DENIED)
        self.assertEqual(snapshot.get_step("prepare").status, StepStatus.BLOCKED)
        self.assertEqual(snapshot.status, TaskStatus.CANCELLED)

    def test_cancellation_does_not_claim_running_adapter_was_stopped(self):
        snapshot = self.store.start_step("task-1", "inspect")
        snapshot = self.store.cancel_task("task-1")
        self.assertTrue(snapshot.cancel_requested)
        self.assertEqual(snapshot.get_step("inspect").status, StepStatus.RUNNING)
        self.assertEqual(snapshot.status, TaskStatus.RUNNING)
        with self.assertRaises(TaskStateError):
            self.store.start_step("task-1", "prepare")
        snapshot = self.store.complete_step("task-1", "inspect")
        self.assertEqual(snapshot.status, TaskStatus.PARTIAL)

    def test_stale_revision_is_rejected(self):
        snapshot = self.store.start_step("task-1", "inspect")
        with self.assertRaises(StaleTaskRevisionError):
            self.store.complete_step(
                "task-1", "inspect", expected_revision=self.task.revision
            )
        self.assertEqual(snapshot.revision, self.store.get_task("task-1").revision)

    def test_only_running_steps_can_complete_or_fail(self):
        with self.assertRaises(TaskStateError):
            self.store.complete_step("task-1", "inspect")
        with self.assertRaises(TaskStateError):
            self.store.fail_step(
                "task-1", "inspect", error_code="adapter_error",
                side_effects_may_have_occurred=False,
            )

    def test_parallel_independent_steps_are_supported_and_serialized(self):
        plan = (PlanStep("a", "A"), PlanStep("b", "B"))
        self.store.create_task(
            task_id="task-2", request_id="request-2", goal="Parallel steps.", steps=plan
        )
        barrier = threading.Barrier(3)
        errors = []

        def start(step_id):
            try:
                barrier.wait()
                self.store.start_step("task-2", step_id)
            except Exception as exc:  # captured for assertion in the main thread
                errors.append(exc)

        threads = [threading.Thread(target=start, args=(step,)) for step in ("a", "b")]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        snapshot = self.store.get_task("task-2")
        self.assertEqual(
            {step.status for step in snapshot.steps}, {StepStatus.RUNNING}
        )

    def test_event_limit_rejects_transition_without_partial_state_mutation(self):
        with patch("core.agent.task_state._MAX_EVENTS_PER_TASK", 1):
            with self.assertRaises(TaskStateError):
                self.store.start_step("task-1", "inspect")
        snapshot = self.store.get_task("task-1")
        self.assertEqual(snapshot.revision, self.task.revision)
        self.assertEqual(snapshot.get_step("inspect").status, StepStatus.PENDING)

    def test_event_sequence_and_revisions_are_monotonic(self):
        snapshot = self.store.start_step("task-1", "inspect")
        snapshot = self.store.complete_step("task-1", "inspect")
        self.assertEqual(
            [event.sequence for event in snapshot.events],
            list(range(1, len(snapshot.events) + 1)),
        )
        self.assertEqual(
            [event.revision for event in snapshot.events],
            sorted(event.revision for event in snapshot.events),
        )
        self.assertEqual(snapshot.revision, snapshot.events[-1].revision)


if __name__ == "__main__":
    unittest.main()
