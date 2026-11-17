import threading
import unittest

from core.agent.contracts import (
    ApprovalGrant,
    ResultStatus,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolDispatcher,
    ToolManifest,
    ToolOperation,
    ToolRegistry,
    ToolResult,
    arguments_digest,
    validate_json_value,
)


INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "minLength": 1},
    },
    "required": ["url"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string"}},
    "required": ["title"],
    "additionalProperties": False,
}


def make_operation(**overrides):
    values = {
        "name": "read_page",
        "description": "Read the explicitly selected page title.",
        "input_schema": INPUT_SCHEMA,
        "output_schema": OUTPUT_SCHEMA,
        "required_permissions": frozenset({"browser.read"}),
        "risk_level": RiskLevel.LOW,
        "side_effect": SideEffect.READ_ONLY,
        "requires_approval": False,
        "idempotent": True,
        "timeout_seconds": 5.0,
    }
    values.update(overrides)
    return ToolOperation(**values)


def make_call(**overrides):
    values = {
        "request_id": "req-1",
        "task_id": "task-1",
        "step_id": "step-1",
        "tool_id": "browser.test",
        "tool_version": "1.0.0",
        "operation": "read_page",
        "arguments": {"url": "https://example.test"},
    }
    values.update(overrides)
    return ToolCall(**values)


class FakeAdapter:
    def __init__(self, operation=None, result=None, error=None):
        self.manifest = ToolManifest(
            tool_id="browser.test",
            display_name="Test Browser",
            description="A deterministic test adapter.",
            version="1.0.0",
            operations=(operation or make_operation(),),
        )
        self.result = result or ToolResult(
            status=ResultStatus.SUCCESS, output={"title": "Example"}
        )
        self.error = error
        self.calls = []

    def execute(self, call, operation, context):
        self.calls.append((call, operation, context))
        if self.error:
            raise self.error
        return self.result


class ToolSchemaTests(unittest.TestCase):
    def test_validates_nested_objects_and_arrays(self):
        schema = {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "count": {"type": "integer", "minimum": 1},
                        },
                        "required": ["name", "count"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["steps"],
            "additionalProperties": False,
        }
        validate_json_value(
            {"steps": [{"name": "inspect", "count": 2}]}, schema
        )

    def test_rejects_wrong_types_extra_keys_missing_fields_and_non_finite_numbers(self):
        with self.assertRaises(ValueError):
            validate_json_value({"url": "x", "extra": True}, INPUT_SCHEMA)
        with self.assertRaises(ValueError):
            validate_json_value({}, INPUT_SCHEMA)
        with self.assertRaises(ValueError):
            validate_json_value({"url": 3}, INPUT_SCHEMA)
        with self.assertRaises(ValueError):
            validate_json_value({"value": float("nan")}, {
                "type": "object",
                "properties": {"value": {"type": "number"}},
                "required": ["value"],
                "additionalProperties": False,
            })

    def test_rejects_unsupported_or_unsafe_schema_definitions(self):
        invalid = (
            {"type": "unknown"},
            {"type": ["string", "null"]},
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "additionalProperties": True,
            },
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["missing"],
            },
            {"type": "array"},
        )
        for schema in invalid:
            with self.subTest(schema=schema):
                with self.assertRaises(ValueError):
                    make_operation(input_schema=schema)

    def test_rejects_high_risk_and_irreversible_operations_without_approval(self):
        with self.assertRaises(ValueError):
            make_operation(risk_level=RiskLevel.HIGH)
        with self.assertRaises(ValueError):
            make_operation(side_effect=SideEffect.IRREVERSIBLE)


class ToolRegistryAndDispatcherTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.registry = ToolRegistry()
        self.registry.register(self.adapter)
        self.dispatcher = ToolDispatcher(self.registry)
        self.call = make_call()

    def make_approval(self, call=None, *, expires=1000, confirmed=True):
        call = call or self.call
        return ApprovalGrant(
            approval_id="approval-1",
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
        )

    def test_registers_and_lists_manifest_without_exposing_execution(self):
        manifests = self.registry.manifests()
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0].tool_id, "browser.test")
        self.assertIs(self.registry.get("browser.test"), self.adapter)

    def test_rejects_duplicate_tool_registration(self):
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.registry.register(FakeAdapter())

    def test_dispatcher_passes_a_defensive_call_snapshot_to_adapter(self):
        result = self.dispatcher.dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertIsNot(self.adapter.calls[0][0], self.call)
        self.assertEqual(self.adapter.calls[0][0].arguments, self.call.arguments)

    def test_dispatches_only_after_schema_and_permission_validation(self):
        result = self.dispatcher.dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)
        self.assertEqual(result.output["title"], "Example")
        self.assertEqual(len(self.adapter.calls), 1)

    def test_tool_version_must_match_registered_manifest(self):
        result = self.dispatcher.dispatch(
            make_call(tool_version="2.0.0"),
            granted_permissions={"browser.read"},
        )
        self.assertEqual(result.error_code, "tool_version_mismatch")
        self.assertEqual(len(self.adapter.calls), 0)

    def test_approval_is_bound_to_idempotency_key(self):
        operation = make_operation(
            name="submit_form",
            risk_level=RiskLevel.HIGH,
            side_effect=SideEffect.IRREVERSIBLE,
            requires_approval=True,
        )
        adapter = FakeAdapter(operation=operation)
        registry = ToolRegistry()
        registry.register(adapter)
        dispatcher = ToolDispatcher(registry)
        approved_call = make_call(
            operation="submit_form", idempotency_key="idem-001"
        )
        approval = self.make_approval(approved_call)
        changed_call = make_call(
            operation="submit_form", idempotency_key="idem-002"
        )
        result = dispatcher.dispatch(
            changed_call,
            granted_permissions={"browser.read"},
            approval=approval,
        )
        self.assertEqual(result.error_code, "approval_scope_mismatch")
        self.assertEqual(len(adapter.calls), 0)

    def test_unknown_tool_and_operation_are_denied(self):
        unknown_tool = self.dispatcher.dispatch(
            make_call(tool_id="unknown.tool"),
            granted_permissions={"browser.read"},
        )
        unknown_operation = self.dispatcher.dispatch(
            make_call(operation="delete_everything"),
            granted_permissions={"browser.read"},
        )
        self.assertEqual(unknown_tool.error_code, "unknown_tool")
        self.assertEqual(unknown_operation.error_code, "unknown_operation")
        self.assertEqual(len(self.adapter.calls), 0)

    def test_invalid_arguments_or_permission_context_are_denied(self):
        invalid_arguments = self.dispatcher.dispatch(
            make_call(arguments={"url": "x", "unexpected": "value"}),
            granted_permissions={"browser.read"},
        )
        invalid_permissions = self.dispatcher.dispatch(
            self.call, granted_permissions=None,
        )
        missing_permission = self.dispatcher.dispatch(
            self.call, granted_permissions=set(),
        )
        self.assertEqual(invalid_arguments.error_code, "invalid_arguments")
        self.assertEqual(invalid_permissions.error_code, "invalid_permission_context")
        self.assertEqual(missing_permission.error_code, "missing_permission")
        self.assertEqual(len(self.adapter.calls), 0)

    def test_does_not_dispatch_unregistered_or_model_selected_permission(self):
        # A model-provided permission field is not part of ToolCall and cannot
        # satisfy the host-supplied capability context.
        result = self.dispatcher.dispatch(
            self.call, granted_permissions={"browser.write"}
        )
        self.assertEqual(result.error_code, "missing_permission")
        self.assertEqual(len(self.adapter.calls), 0)

    def test_high_impact_operation_requires_exact_fresh_approval(self):
        operation = make_operation(
            name="submit_form",
            risk_level=RiskLevel.HIGH,
            side_effect=SideEffect.IRREVERSIBLE,
            requires_approval=True,
        )
        adapter = FakeAdapter(operation=operation)
        registry = ToolRegistry()
        registry.register(adapter)
        dispatcher = ToolDispatcher(registry)
        call = make_call(operation="submit_form")

        no_approval = dispatcher.dispatch(
            call, granted_permissions={"browser.read"}
        )
        mismatch = dispatcher.dispatch(
            call,
            granted_permissions={"browser.read"},
            approval=self.make_approval(make_call(arguments={"url": "https://other.test"})),
        )
        step_mismatch = dispatcher.dispatch(
            make_call(operation="submit_form", step_id="step-2"),
            granted_permissions={"browser.read"},
            approval=self.make_approval(call),
        )
        expired = dispatcher.dispatch(
            call,
            granted_permissions={"browser.read"},
            approval=self.make_approval(call, expires=99),
            wall_clock=lambda: 100,
        )
        approved = dispatcher.dispatch(
            call,
            granted_permissions={"browser.read"},
            approval=self.make_approval(call),
            wall_clock=lambda: 100,
        )
        self.assertEqual(no_approval.error_code, "approval_required")
        self.assertEqual(mismatch.error_code, "approval_scope_mismatch")
        self.assertEqual(step_mismatch.error_code, "approval_scope_mismatch")
        self.assertEqual(expired.error_code, "approval_expired")
        self.assertEqual(approved.status, ResultStatus.SUCCESS)
        self.assertEqual(len(adapter.calls), 1)

    def test_approval_cannot_be_reused_for_different_arguments(self):
        operation = make_operation(
            name="submit_form",
            risk_level=RiskLevel.HIGH,
            side_effect=SideEffect.IRREVERSIBLE,
            requires_approval=True,
        )
        adapter = FakeAdapter(operation=operation)
        registry = ToolRegistry()
        registry.register(adapter)
        dispatcher = ToolDispatcher(registry)
        first = make_call(operation="submit_form")
        approval = self.make_approval(first)
        changed = make_call(
            operation="submit_form", arguments={"url": "https://changed.test"}
        )
        result = dispatcher.dispatch(
            changed,
            granted_permissions={"browser.read"},
            approval=approval,
        )
        self.assertEqual(result.error_code, "approval_scope_mismatch")
        self.assertEqual(len(adapter.calls), 0)

    def test_approval_grant_is_single_use(self):
        operation = make_operation(
            name="submit_form",
            risk_level=RiskLevel.HIGH,
            side_effect=SideEffect.IRREVERSIBLE,
            requires_approval=True,
        )
        adapter = FakeAdapter(operation=operation)
        registry = ToolRegistry()
        registry.register(adapter)
        dispatcher = ToolDispatcher(registry)
        call = make_call(operation="submit_form")
        approval = self.make_approval(call)
        first = dispatcher.dispatch(
            call,
            granted_permissions={"browser.read"},
            approval=approval,
            wall_clock=lambda: 100,
        )
        second = ToolDispatcher(registry).dispatch(
            call,
            granted_permissions={"browser.read"},
            approval=approval,
            wall_clock=lambda: 100,
        )
        self.assertEqual(first.status, ResultStatus.SUCCESS)
        self.assertEqual(second.error_code, "approval_reused")
        self.assertEqual(len(adapter.calls), 1)

    def test_non_idempotent_operations_cannot_advertise_retryable_result(self):
        operation = make_operation(
            side_effect=SideEffect.REVERSIBLE,
            idempotent=False,
        )
        adapter = FakeAdapter(operation=operation, result=ToolResult(
            status=ResultStatus.FAILED,
            error_code="uncertain_completion",
            retryable=True,
            side_effects_may_have_occurred=True,
        ))
        registry = ToolRegistry()
        registry.register(adapter)
        result = ToolDispatcher(registry).dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertFalse(result.retryable)

    def test_tool_arguments_have_a_hard_size_limit(self):
        with self.assertRaises(ValueError):
            make_call(arguments={"url": "x" * (256 * 1024)})

    def test_explicit_pre_dispatch_cancellation_skips_adapter(self):
        cancellation = threading.Event()
        cancellation.set()
        result = self.dispatcher.dispatch(
            self.call,
            granted_permissions={"browser.read"},
            cancel_event=cancellation,
        )
        self.assertEqual(result.status, ResultStatus.CANCELLED)
        self.assertEqual(len(self.adapter.calls), 0)

    def test_adapter_timeout_and_exception_are_normalized_without_tracebacks(self):
        timeout_adapter = FakeAdapter(error=TimeoutError("private details"))
        registry = ToolRegistry()
        registry.register(timeout_adapter)
        result = ToolDispatcher(registry).dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(result.status, ResultStatus.TIMEOUT)
        self.assertEqual(result.error_code, "adapter_timeout")
        self.assertTrue(result.side_effects_may_have_occurred)
        self.assertNotIn("private details", str(result))

        failed_adapter = FakeAdapter(error=RuntimeError("secret traceback"))
        registry = ToolRegistry()
        registry.register(failed_adapter)
        failed = ToolDispatcher(registry).dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(failed.status, ResultStatus.FAILED)
        self.assertEqual(failed.error_code, "adapter_error")
        self.assertNotIn("secret traceback", str(failed))

    def test_adapter_reported_timeout_is_never_safe_to_retry_automatically(self):
        adapter = FakeAdapter(result=ToolResult(
            status=ResultStatus.TIMEOUT,
            error_code="network_timeout",
            retryable=True,
            side_effects_may_have_occurred=False,
        ))
        registry = ToolRegistry()
        registry.register(adapter)
        result = ToolDispatcher(registry).dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(result.status, ResultStatus.TIMEOUT)
        self.assertFalse(result.retryable)
        self.assertTrue(result.side_effects_may_have_occurred)

    def test_late_adapter_return_is_timed_out_and_not_retryable_by_default(self):
        clock_values = iter((10.0, 16.0))
        result = self.dispatcher.dispatch(
            self.call,
            granted_permissions={"browser.read"},
            monotonic_clock=lambda: next(clock_values),
        )
        self.assertEqual(result.status, ResultStatus.TIMEOUT)
        self.assertEqual(result.error_code, "adapter_deadline_exceeded")
        self.assertTrue(result.side_effects_may_have_occurred)
        self.assertFalse(result.retryable)

    def test_invalid_adapter_output_is_rejected(self):
        adapter = FakeAdapter(result=ToolResult(
            status=ResultStatus.SUCCESS, output={"unexpected": "field"}
        ))
        registry = ToolRegistry()
        registry.register(adapter)
        result = ToolDispatcher(registry).dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(result.status, ResultStatus.FAILED)
        self.assertEqual(result.error_code, "invalid_adapter_output")
        self.assertTrue(result.side_effects_may_have_occurred)

    def test_tool_manifest_snapshot_does_not_follow_adapter_reassignment(self):
        original = self.adapter.manifest
        self.adapter.manifest = ToolManifest(
            tool_id="browser.test",
            display_name="Changed Tool",
            description="Changed after registration.",
            version="2.0.0",
            operations=(make_operation(name="replacement"),),
        )
        self.assertIs(self.registry.get_manifest("browser.test"), original)
        result = self.dispatcher.dispatch(
            self.call, granted_permissions={"browser.read"}
        )
        self.assertEqual(result.status, ResultStatus.SUCCESS)

    def test_dispatch_rejects_non_tool_call(self):
        result = self.dispatcher.dispatch(
            {"tool_id": "browser.test"},  # type: ignore[arg-type]
            granted_permissions={"browser.read"},
        )
        self.assertEqual(result.error_code, "invalid_call")


if __name__ == "__main__":
    unittest.main()
