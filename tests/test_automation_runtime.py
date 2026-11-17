import tempfile
import threading
import time
import unittest
from pathlib import Path

from core.agent import (
    AutomationRuntime,
    RuntimeIdentity,
    PlanStep,
    ResultStatus,
    RiskLevel,
    SideEffect,
    StepStatus,
    TaskStatus,
    ToolCall,
    ToolManifest,
    ToolOperation,
    ToolResult,
)
from core.agent.runtime import RuntimeBusyError, RuntimeRecoveryError


INPUT_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string", "minLength": 1}},
    "required": ["url"],
    "additionalProperties": False,
}
IDENTITY = RuntimeIdentity(principal_id="owner-1", session_id="session-1")

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}


def make_operation(*, name="read_page", approval=False, side_effect=SideEffect.READ_ONLY):
    return ToolOperation(
        name=name,
        description="Run a deterministic runtime test operation.",
        input_schema=INPUT_SCHEMA,
        output_schema=OUTPUT_SCHEMA,
        required_permissions=frozenset({"browser.write" if approval else "browser.read"}),
        risk_level=RiskLevel.HIGH if approval else RiskLevel.LOW,
        side_effect=side_effect,
        requires_approval=approval,
        idempotent=not approval,
        timeout_seconds=3.0,
    )


def make_call(
    *,
    request_id="request-1",
    task_id="task-1",
    step_id="step-1",
    operation="read_page",
    arguments=None,
):
    return ToolCall(
        request_id=request_id,
        task_id=task_id,
        step_id=step_id,
        tool_id="browser.test",
        tool_version="1.0.0",
        operation=operation,
        arguments=arguments or {"url": "https://example.test"},
        idempotency_key="idem-1",
    )


class FakeAdapter:
    def __init__(self, operation=None, *, result=None, error=None, on_execute=None):
        self.manifest = ToolManifest(
            tool_id="browser.test",
            display_name="Runtime Test Browser",
            description="Fake adapter used only by AutomationRuntime tests.",
            version="1.0.0",
            operations=(operation or make_operation(),),
        )
        self.result = result or ToolResult(
            status=ResultStatus.SUCCESS, output={"title": "Example"}
        )
        self.error = error
        self.on_execute = on_execute
        self.calls = []

    def execute(self, call, operation, context):
        self.calls.append((call, operation, context))
        if self.on_execute is not None:
            return self.on_execute(call, operation, context)
        if self.error is not None:
            raise self.error
        return self.result


class AutomationRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.database = Path(self.temp_dir.name) / "runtime.sqlite3"
        self.clock_value = [100.0]
        self.runtime = self.open_runtime()
        self.runtime.register_session(IDENTITY, expires_at_unix=1000.0)
        self.addCleanup(self.close_runtime)

    def open_runtime(self):
        return AutomationRuntime(self.database, clock=lambda: self.clock_value[0])

    def close_runtime(self):
        if self.runtime is not None:
            try:
                self.runtime.close()
            except RuntimeBusyError:
                pass
            self.runtime = None

    def register(self, operation=None, **adapter_kwargs):
        adapter = FakeAdapter(operation, **adapter_kwargs)
        self.runtime.register_adapter(adapter)
        return adapter

    def create_task(self, *steps):
        return self.runtime.create_task(
            task_id="task-1",
            request_id="request-1",
            goal="Complete the requested browser automation safely.",
            identity=IDENTITY,
            steps=steps or (
                PlanStep(
                    "step-1", "Read the selected page.",
                    tool_id="browser.test", operation="read_page",
                ),
            ),
        )

    def test_low_risk_step_flows_through_dispatcher_and_persists_success(self):
        adapter = self.register()
        created = self.create_task()
        result = self.runtime.execute_step(
            make_call(), granted_permissions={"browser.read"}, identity=IDENTITY
        )

        self.assertEqual(result.tool_result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.task.status, TaskStatus.SUCCEEDED)
        self.assertEqual(result.task.get_step("step-1").status, StepStatus.SUCCEEDED)
        self.assertEqual(len(adapter.calls), 1)
        self.assertEqual(created.revision, 1)
        self.assertIn(
            "step_succeeded",
            [event.event_type for event in result.task.events],
        )

    def test_high_risk_step_waits_for_approval_without_running_adapter(self):
        adapter = self.register(make_operation(approval=True, side_effect=SideEffect.IRREVERSIBLE))
        self.create_task(PlanStep(
            "step-1", "Submit the selected form.",
            tool_id="browser.test", operation="read_page",
        ))
        call = make_call()

        result = self.runtime.execute_step(call, granted_permissions={"browser.write"}, identity=IDENTITY)
        self.assertTrue(result.awaiting_approval)
        self.assertIsNone(result.tool_result)
        self.assertTrue(result.approval_id)
        self.assertEqual(
            result.task.get_step("step-1").status, StepStatus.WAITING_APPROVAL
        )
        self.assertEqual(
            result.task.get_step("step-1").approval_id, result.approval_id
        )
        self.assertEqual(len(adapter.calls), 0)

    def test_explicit_approval_survives_runtime_restart_and_executes_once(self):
        operation = make_operation(approval=True, side_effect=SideEffect.IRREVERSIBLE)
        first_adapter = FakeAdapter(operation)
        self.runtime.register_adapter(first_adapter)
        self.create_task(PlanStep(
            "step-1", "Submit the selected form.",
            tool_id="browser.test", operation="read_page",
        ))
        call = make_call()
        waiting = self.runtime.execute_step(call, granted_permissions={"browser.write"})
        self.assertTrue(waiting.awaiting_approval)
        self.assertTrue(waiting.approval_id)
        grant = self.runtime.issue_approval(call, user_confirmed=True, identity=IDENTITY)
        approval_id = grant.approval_id
        self.assertEqual(approval_id, waiting.approval_id)

        self.runtime.close()
        self.runtime = self.open_runtime()
        second_adapter = FakeAdapter(operation)
        self.runtime.register_adapter(second_adapter)

        restored = self.runtime.get_task("task-1", identity=IDENTITY)
        self.assertEqual(restored.get_step("step-1").approval_id, approval_id)
        result = self.runtime.execute_step(
            call,
            granted_permissions={"browser.write"},
        )
        self.assertEqual(result.tool_result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.task.get_step("step-1").status, StepStatus.SUCCEEDED)
        self.assertEqual(len(second_adapter.calls), 1)

        repeated = self.runtime.execute_step(
            call,
            granted_permissions={"browser.write"},
            approval_id=approval_id,
        )
        self.assertEqual(repeated.tool_result.status, ResultStatus.DENIED)
        self.assertEqual(len(second_adapter.calls), 1)

    def test_approval_is_bound_to_exact_arguments(self):
        operation = make_operation(approval=True, side_effect=SideEffect.IRREVERSIBLE)
        adapter = self.register(operation)
        self.create_task(PlanStep(
            "step-1", "Submit the selected form.",
            tool_id="browser.test", operation="read_page",
        ))
        original = make_call()
        self.runtime.execute_step(original, granted_permissions={"browser.write"})
        grant = self.runtime.issue_approval(original, user_confirmed=True)

        altered = make_call(arguments={"url": "https://other.example"})
        mismatch = self.runtime.execute_step(
            altered,
            granted_permissions={"browser.write"},
            approval_id=grant.approval_id,
        )
        self.assertTrue(mismatch.awaiting_approval)
        self.assertEqual(mismatch.tool_result.error_code, "approval_scope_mismatch")
        self.assertEqual(len(adapter.calls), 0)

        accepted = self.runtime.execute_step(
            original,
            granted_permissions={"browser.write"},
            approval_id=grant.approval_id,
        )
        self.assertEqual(accepted.tool_result.status, ResultStatus.SUCCESS)
        self.assertEqual(len(adapter.calls), 1)

    def test_expired_approval_returns_step_to_waiting_state(self):
        operation = make_operation(approval=True, side_effect=SideEffect.IRREVERSIBLE)
        adapter = self.register(operation)
        self.create_task(PlanStep(
            "step-1", "Submit the selected form.",
            tool_id="browser.test", operation="read_page",
        ))
        call = make_call()
        self.runtime.execute_step(call, granted_permissions={"browser.write"})
        grant = self.runtime.issue_approval(
            call, user_confirmed=True, lifetime_seconds=10
        )
        self.clock_value[0] = 200.0

        result = self.runtime.execute_step(
            call,
            granted_permissions={"browser.write"},
            approval_id=grant.approval_id,
        )
        self.assertTrue(result.awaiting_approval)
        self.assertEqual(result.task.get_step("step-1").status, StepStatus.WAITING_APPROVAL)
        self.assertEqual(len(adapter.calls), 0)

        replacement = self.runtime.issue_approval(call, user_confirmed=True)
        self.assertNotEqual(replacement.approval_id, grant.approval_id)
        self.assertEqual(
            self.runtime.get_task("task-1").get_step("step-1").approval_id,
            replacement.approval_id,
        )

    def test_pre_dispatch_cancellation_does_not_start_step_or_consume_grant(self):
        operation = make_operation(approval=True, side_effect=SideEffect.IRREVERSIBLE)
        adapter = self.register(operation)
        self.create_task(PlanStep(
            "step-1", "Submit the selected form.",
            tool_id="browser.test", operation="read_page",
        ))
        call = make_call()
        self.runtime.execute_step(call, granted_permissions={"browser.write"})
        grant = self.runtime.issue_approval(call, user_confirmed=True)
        cancellation = threading.Event()
        cancellation.set()

        stopped = self.runtime.execute_step(
            call,
            granted_permissions={"browser.write"},
            approval_id=grant.approval_id,
            cancel_event=cancellation,
        )
        self.assertEqual(stopped.tool_result.status, ResultStatus.CANCELLED)
        self.assertEqual(stopped.task.get_step("step-1").status, StepStatus.PENDING)
        self.assertEqual(len(adapter.calls), 0)

        accepted = self.runtime.execute_step(
            call,
            granted_permissions={"browser.write"},
            approval_id=grant.approval_id,
        )
        self.assertEqual(accepted.tool_result.status, ResultStatus.SUCCESS)
        self.assertEqual(len(adapter.calls), 1)

    def test_missing_permissions_do_not_start_or_consume_step(self):
        adapter = self.register()
        self.create_task()
        result = self.runtime.execute_step(make_call(), granted_permissions=set(), identity=IDENTITY)
        self.assertEqual(result.tool_result.error_code, "missing_permission")
        self.assertEqual(result.task.get_step("step-1").status, StepStatus.PENDING)
        self.assertEqual(len(adapter.calls), 0)

    def test_failure_blocks_dependents_and_marks_uncertain_side_effects(self):
        failure = ToolResult(
            status=ResultStatus.FAILED,
            error_code="browser_submit_failed",
            side_effects_may_have_occurred=True,
        )
        adapter = self.register(
            make_operation(approval=False, side_effect=SideEffect.READ_ONLY),
            result=failure,
        )
        self.create_task(
            PlanStep("step-1", "Run first step.", tool_id="browser.test", operation="read_page"),
            PlanStep(
                "step-2", "Run dependent step.", depends_on=("step-1",),
                tool_id="browser.test", operation="read_page",
            ),
            PlanStep("step-3", "Run independent step.", tool_id="browser.test", operation="read_page"),
        )
        result = self.runtime.execute_step(make_call(), granted_permissions={"browser.read"})
        self.assertEqual(result.task.get_step("step-1").status, StepStatus.FAILED)
        self.assertTrue(result.task.get_step("step-1").side_effects_may_have_occurred)
        self.assertEqual(result.task.get_step("step-2").status, StepStatus.BLOCKED)
        self.assertEqual(result.task.get_step("step-3").status, StepStatus.PENDING)
        self.assertEqual(len(adapter.calls), 1)

    def test_cancellation_during_adapter_execution_is_persisted(self):
        started = threading.Event()

        def wait_for_cancellation(call, operation, context):
            started.set()
            context.cancel_event.wait(timeout=3)
            return ToolResult(
                status=ResultStatus.CANCELLED,
                error_code="adapter_cancelled",
                side_effects_may_have_occurred=True,
            )

        adapter = self.register(on_execute=wait_for_cancellation)
        self.create_task()
        outcomes = []
        errors = []

        def run_step():
            try:
                outcomes.append(self.runtime.execute_step(
                    make_call(), granted_permissions={"browser.read"}
                ))
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=run_step)
        worker.start()
        self.assertTrue(started.wait(timeout=2))
        self.runtime.cancel_task("task-1", identity=IDENTITY)
        worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        outcome = outcomes[0]
        self.assertEqual(outcome.task.status, TaskStatus.CANCELLED)
        step = outcome.task.get_step("step-1")
        self.assertEqual(step.status, StepStatus.FAILED)
        self.assertTrue(step.side_effects_may_have_occurred)
        self.assertEqual(len(adapter.calls), 1)

    def test_interrupted_running_step_is_never_automatically_retried(self):
        self.register()
        self.create_task()
        snapshot = self.runtime._tasks.start_step("task-1", "step-1", expected_revision=1)
        self.assertEqual(snapshot.get_step("step-1").status, StepStatus.RUNNING)
        self.runtime.close()

        self.runtime = self.open_runtime()
        with self.assertRaises(RuntimeRecoveryError):
            self.runtime.recover_interrupted_tasks(exclusive_startup=False)
        recovered = self.runtime.recover_interrupted_tasks(exclusive_startup=True)

        self.assertEqual(len(recovered), 1)
        step = recovered[0].get_step("step-1")
        self.assertEqual(step.status, StepStatus.FAILED)
        self.assertEqual(step.error_code, "runtime_restarted")
        self.assertTrue(step.side_effects_may_have_occurred)

    def test_task_is_bound_to_exact_principal_and_extension_session(self):
        self.register()
        self.create_task()
        other_session = RuntimeIdentity("owner-1", "session-2")
        self.runtime.register_session(other_session, expires_at_unix=1000.0)
        with self.assertRaises(PermissionError):
            self.runtime.get_task("task-1", identity=other_session)
        with self.assertRaises(PermissionError):
            self.runtime.execute_step(
                make_call(),
                granted_permissions={"browser.read"},
                identity=other_session,
            )

    def test_revoked_session_cannot_read_or_execute_existing_task(self):
        adapter = self.register()
        self.create_task()
        self.assertTrue(self.runtime.revoke_session(IDENTITY))
        self.assertFalse(self.runtime.revoke_session(IDENTITY))
        with self.assertRaises(PermissionError):
            self.runtime.get_task("task-1", identity=IDENTITY)
        with self.assertRaises(PermissionError):
            self.runtime.execute_step(
                make_call(),
                granted_permissions={"browser.read"},
                identity=IDENTITY,
            )
        self.assertEqual(len(adapter.calls), 0)

    def test_expired_session_cannot_create_or_execute_tasks(self):
        self.clock_value[0] = 1001.0
        with self.assertRaises(PermissionError):
            self.create_task()

    def test_close_refuses_to_close_while_adapter_is_running(self):
        started = threading.Event()
        release = threading.Event()

        def hold(call, operation, context):
            started.set()
            release.wait(timeout=3)
            return ToolResult(status=ResultStatus.SUCCESS, output={"title": "Done"})

        self.register(on_execute=hold)
        self.create_task()
        worker = threading.Thread(target=lambda: self.runtime.execute_step(
            make_call(), granted_permissions={"browser.read"}
        ))
        worker.start()
        self.assertTrue(started.wait(timeout=2))
        with self.assertRaises(RuntimeBusyError):
            self.runtime.close()
        release.set()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.runtime.close()
        self.assertTrue(self.runtime._closed)


if __name__ == "__main__":
    unittest.main()
