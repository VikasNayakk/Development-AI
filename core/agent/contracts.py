"""Versioned, model-independent contracts for connected automation tools.

This module describes and validates tool operations. It does not execute model
output as code, grant permissions, or store credentials.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SideEffect(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


class ResultStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    DENIED = "denied"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


_JSON_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_SCHEMA_KEYS = {
    "object": {"type", "properties", "required", "additionalProperties", "enum"},
    "array": {"type", "items", "minItems", "maxItems", "enum"},
    "string": {"type", "minLength", "maxLength", "enum"},
    "integer": {"type", "minimum", "maximum", "enum"},
    "number": {"type", "minimum", "maximum", "enum"},
    "boolean": {"type", "enum"},
    "null": {"type", "enum"},
}


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain JSON-compatible values.") from exc


def _validate_schema_definition(schema: Mapping[str, Any], path: str = "$") -> None:
    if not isinstance(schema, Mapping):
        raise ValueError(f"{path}: schema must be an object.")
    schema_type = schema.get("type")
    if not isinstance(schema_type, str) or schema_type not in _JSON_TYPES:
        raise ValueError(f"{path}: unsupported or missing schema type.")
    unknown = set(schema) - _SCHEMA_KEYS[schema_type]
    if unknown:
        raise ValueError(f"{path}: unsupported schema keywords: {sorted(unknown)}")
    if "enum" in schema and (
        not isinstance(schema["enum"], list) or not schema["enum"]
    ):
        raise ValueError(f"{path}: enum must be a non-empty list.")
    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required", [])
        if not isinstance(properties, Mapping):
            raise ValueError(f"{path}: object properties must be an object.")
        if not isinstance(required, list) or any(
            not isinstance(name, str) for name in required
        ):
            raise ValueError(f"{path}: required must be a list of property names.")
        if len(set(required)) != len(required) or not set(required).issubset(properties):
            raise ValueError(f"{path}: required properties must be unique and declared.")
        if schema.get("additionalProperties", False) is not False:
            raise ValueError(f"{path}: additionalProperties must be false.")
        for name, child in properties.items():
            if not isinstance(name, str):
                raise ValueError(f"{path}: property names must be strings.")
            _validate_schema_definition(child, f"{path}.{name}")
    elif schema_type == "array":
        if "items" not in schema:
            raise ValueError(f"{path}: array schemas must declare items.")
        _validate_schema_definition(schema["items"], f"{path}[]")
        for key in ("minItems", "maxItems"):
            if key in schema and (
                type(schema[key]) is not int or schema[key] < 0
            ):
                raise ValueError(f"{path}: {key} must be a non-negative integer.")
        if schema.get("minItems", 0) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: minItems cannot exceed maxItems.")
    elif schema_type == "string":
        for key in ("minLength", "maxLength"):
            if key in schema and (
                type(schema[key]) is not int or schema[key] < 0
            ):
                raise ValueError(f"{path}: {key} must be a non-negative integer.")
        if schema.get("minLength", 0) > schema.get("maxLength", math.inf):
            raise ValueError(f"{path}: minLength cannot exceed maxLength.")
    elif schema_type in {"integer", "number"}:
        for key in ("minimum", "maximum"):
            if key in schema and (
                isinstance(schema[key], bool)
                or not isinstance(schema[key], (int, float))
                or not math.isfinite(schema[key])
            ):
                raise ValueError(f"{path}: {key} must be a finite number.")
        if schema.get("minimum", -math.inf) > schema.get("maximum", math.inf):
            raise ValueError(f"{path}: minimum cannot exceed maximum.")


def validate_json_value(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate the deliberately small, strict JSON Schema subset used by tools."""
    schema_type = schema["type"]
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
        ),
        "boolean": type(value) is bool,
        "null": value is None,
    }[schema_type]
    if not valid:
        raise ValueError(f"{path}: expected {schema_type}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in the allowed enum.")
    if schema_type == "object":
        properties = schema["properties"]
        required = schema.get("required", [])
        missing = set(required) - set(value)
        extra = set(value) - set(properties)
        if missing:
            raise ValueError(f"{path}: missing required properties {sorted(missing)}.")
        if extra:
            raise ValueError(f"{path}: undeclared properties {sorted(extra)}.")
        for name, child in properties.items():
            if name in value:
                validate_json_value(value[name], child, f"{path}.{name}")
    elif schema_type == "array":
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: array has too few items.")
        if len(value) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: array has too many items.")
        for index, item in enumerate(value):
            validate_json_value(item, schema["items"], f"{path}[{index}]")
    elif schema_type == "string":
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path}: string is too short.")
        if len(value) > schema.get("maxLength", math.inf):
            raise ValueError(f"{path}: string is too long.")
    elif schema_type in {"integer", "number"}:
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: number is below minimum.")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: number exceeds maximum.")


def arguments_digest(arguments: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        arguments, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class ToolOperation:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    output_schema: Mapping[str, Any]
    required_permissions: frozenset[str] = field(default_factory=frozenset)
    risk_level: RiskLevel = RiskLevel.LOW
    side_effect: SideEffect = SideEffect.READ_ONLY
    requires_approval: bool = False
    idempotent: bool = True
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", self.name):
            raise ValueError("Operation name must be a stable lowercase identifier.")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("Operation description is required.")
        _validate_schema_definition(self.input_schema, "$input")
        _validate_schema_definition(self.output_schema, "$output")
        if any(not isinstance(item, str) or not item.strip()
               for item in self.required_permissions):
            raise ValueError("Required permissions must be non-empty strings.")
        if not isinstance(self.risk_level, RiskLevel):
            raise ValueError("risk_level must be a RiskLevel.")
        if not isinstance(self.side_effect, SideEffect):
            raise ValueError("side_effect must be a SideEffect.")
        if self.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL} and not self.requires_approval:
            raise ValueError("High and critical risk operations must require approval.")
        if self.side_effect == SideEffect.IRREVERSIBLE and not self.requires_approval:
            raise ValueError("Irreversible operations must require approval.")
        if type(self.requires_approval) is not bool or type(self.idempotent) is not bool:
            raise ValueError("Approval and idempotency settings must be booleans.")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a finite positive number.")
        object.__setattr__(self, "required_permissions", frozenset(self.required_permissions))
        object.__setattr__(self, "input_schema", _json_copy(self.input_schema, label="input_schema"))
        object.__setattr__(self, "output_schema", _json_copy(self.output_schema, label="output_schema"))


@dataclass(frozen=True)
class ToolManifest:
    tool_id: str
    display_name: str
    description: str
    version: str
    operations: tuple[ToolOperation, ...]
    manifest_version: int = 1

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.-]{1,95}", self.tool_id):
            raise ValueError("tool_id must be a stable lowercase identifier.")
        for label, value in (
            ("display_name", self.display_name),
            ("description", self.description),
            ("version", self.version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} is required.")
        if type(self.manifest_version) is not int or self.manifest_version != 1:
            raise ValueError("Only manifest_version=1 is currently supported.")
        operations = tuple(self.operations)
        if not operations or any(not isinstance(op, ToolOperation) for op in operations):
            raise ValueError("A manifest must contain ToolOperation entries.")
        names = [op.name for op in operations]
        if len(names) != len(set(names)):
            raise ValueError("Operation names must be unique within a tool.")
        object.__setattr__(self, "operations", operations)

    def get_operation(self, name: str) -> ToolOperation | None:
        return next((op for op in self.operations if op.name == name), None)


@dataclass(frozen=True)
class ToolCall:
    request_id: str
    task_id: str
    step_id: str
    tool_id: str
    operation: str
    arguments: Mapping[str, Any]
    manifest_version: int = 1

    def __post_init__(self) -> None:
        for label in ("request_id", "task_id", "step_id", "tool_id", "operation"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} is required.")
        if type(self.manifest_version) is not int or self.manifest_version != 1:
            raise ValueError("Unsupported tool manifest version.")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("Tool arguments must be a JSON object.")
        copied = _json_copy(dict(self.arguments), label="arguments")
        if not isinstance(copied, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        object.__setattr__(self, "arguments", copied)


@dataclass(frozen=True)
class ApprovalGrant:
    """Host-created record of a user's approval for one exact operation payload."""

    approval_id: str
    request_id: str
    tool_id: str
    operation: str
    arguments_sha256: str
    expires_at_unix: float
    user_confirmed: bool

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip() for value in (
            self.approval_id, self.request_id, self.tool_id, self.operation,
            self.arguments_sha256,
        )):
            raise ValueError("Approval grant identifiers and digest are required.")
        if (
            isinstance(self.expires_at_unix, bool)
            or not isinstance(self.expires_at_unix, (int, float))
            or not math.isfinite(self.expires_at_unix)
        ):
            raise ValueError("Approval expiry must be a finite Unix timestamp.")
        if type(self.user_confirmed) is not bool:
            raise ValueError("user_confirmed must be a boolean.")


@dataclass(frozen=True)
class ExecutionContext:
    request_id: str
    task_id: str
    step_id: str
    deadline_monotonic: float
    cancel_event: threading.Event


@dataclass(frozen=True)
class ToolResult:
    status: ResultStatus
    output: Mapping[str, Any] = field(default_factory=dict)
    changed_resources: tuple[str, ...] = ()
    error_code: str | None = None
    retryable: bool = False
    side_effects_may_have_occurred: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.status, ResultStatus):
            raise ValueError("Tool result status must be a ResultStatus.")
        copied = _json_copy(dict(self.output), label="tool output")
        if not isinstance(copied, dict):
            raise ValueError("Tool output must be a JSON object.")
        if any(not isinstance(item, str) or not item.strip()
               for item in self.changed_resources):
            raise ValueError("Changed resource references must be non-empty strings.")
        if type(self.retryable) is not bool or type(self.side_effects_may_have_occurred) is not bool:
            raise ValueError("Result flags must be booleans.")
        object.__setattr__(self, "output", copied)
        object.__setattr__(self, "changed_resources", tuple(self.changed_resources))


class ToolAdapter(Protocol):
    manifest: ToolManifest

    def execute(
        self, call: ToolCall, operation: ToolOperation, context: ExecutionContext
    ) -> ToolResult:
        """Execute one validated operation within the adapter's own permissions."""
        ...


class ToolRegistry:
    """Explicit registry; tools are added by trusted host code, never model output."""

    def __init__(self) -> None:
        self._adapters: dict[str, ToolAdapter] = {}
        self._manifests: dict[str, ToolManifest] = {}

    def register(self, adapter: ToolAdapter) -> None:
        manifest = getattr(adapter, "manifest", None)
        execute = getattr(adapter, "execute", None)
        if not isinstance(manifest, ToolManifest) or not callable(execute):
            raise ValueError("Adapter must expose a ToolManifest and callable execute().")
        if manifest.tool_id in self._adapters:
            raise ValueError(f"Tool already registered: {manifest.tool_id}")
        self._adapters[manifest.tool_id] = adapter
        self._manifests[manifest.tool_id] = manifest

    def get(self, tool_id: str) -> ToolAdapter | None:
        return self._adapters.get(tool_id)

    def get_manifest(self, tool_id: str) -> ToolManifest | None:
        return self._manifests.get(tool_id)

    def manifests(self) -> tuple[ToolManifest, ...]:
        return tuple(self._manifests[key] for key in sorted(self._manifests))


class ToolDispatcher:
    """Validate, authorize and invoke a registered adapter.

    Adapters must enforce the supplied deadline and cancellation event while
    performing I/O. A synchronous adapter cannot be forcibly interrupted safely
    by Python; late returns are detected and reported as timed out, with side
    effects marked as potentially having occurred.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @staticmethod
    def _denied(code: str) -> ToolResult:
        return ToolResult(status=ResultStatus.DENIED, error_code=code)

    def dispatch(
        self,
        call: ToolCall,
        *,
        granted_permissions: frozenset[str] | set[str] = frozenset(),
        approval: ApprovalGrant | None = None,
        cancel_event: threading.Event | None = None,
        wall_clock=time.time,
        monotonic_clock=time.monotonic,
    ) -> ToolResult:
        if not isinstance(call, ToolCall):
            return self._denied("invalid_call")
        if not isinstance(granted_permissions, (set, frozenset)) or any(
            not isinstance(item, str) or not item.strip()
            for item in granted_permissions
        ):
            return self._denied("invalid_permission_context")
        adapter = self._registry.get(call.tool_id)
        manifest = self._registry.get_manifest(call.tool_id)
        if adapter is None or manifest is None:
            return self._denied("unknown_tool")
        if call.manifest_version != manifest.manifest_version:
            return self._denied("unsupported_manifest_version")
        operation = manifest.get_operation(call.operation)
        if operation is None:
            return self._denied("unknown_operation")
        try:
            validate_json_value(call.arguments, operation.input_schema)
        except (KeyError, TypeError, ValueError):
            return self._denied("invalid_arguments")
        if not operation.required_permissions.issubset(granted_permissions):
            return self._denied("missing_permission")
        if operation.requires_approval:
            if (
                not isinstance(approval, ApprovalGrant)
                or not approval.user_confirmed
            ):
                return self._denied("approval_required")
            if (
                approval.request_id != call.request_id
                or approval.tool_id != call.tool_id
                or approval.operation != call.operation
                or approval.arguments_sha256 != arguments_digest(call.arguments)
            ):
                return self._denied("approval_scope_mismatch")
            if approval.expires_at_unix <= wall_clock():
                return self._denied("approval_expired")

        cancel_event = cancel_event or threading.Event()
        if cancel_event.is_set():
            return ToolResult(status=ResultStatus.CANCELLED, error_code="cancelled_before_dispatch")
        context = ExecutionContext(
            request_id=call.request_id,
            task_id=call.task_id,
            step_id=call.step_id,
            deadline_monotonic=monotonic_clock() + operation.timeout_seconds,
            cancel_event=cancel_event,
        )
        try:
            result = adapter.execute(call, operation, context)
        except TimeoutError:
            return ToolResult(
                status=ResultStatus.TIMEOUT, error_code="adapter_timeout",
                side_effects_may_have_occurred=True,
            )
        except InterruptedError:
            return ToolResult(
                status=ResultStatus.CANCELLED, error_code="adapter_cancelled",
                side_effects_may_have_occurred=True,
            )
        except Exception:
            return ToolResult(
                status=ResultStatus.FAILED, error_code="adapter_error",
                side_effects_may_have_occurred=True,
            )
        if monotonic_clock() > context.deadline_monotonic:
            return ToolResult(
                status=ResultStatus.TIMEOUT, error_code="adapter_deadline_exceeded",
                side_effects_may_have_occurred=True,
            )
        if cancel_event.is_set():
            return ToolResult(
                status=ResultStatus.CANCELLED, error_code="cancelled_during_dispatch",
                side_effects_may_have_occurred=True,
            )
        if not isinstance(result, ToolResult):
            return ToolResult(
                status=ResultStatus.FAILED, error_code="invalid_adapter_result",
                side_effects_may_have_occurred=True,
            )
        if result.status in {ResultStatus.SUCCESS, ResultStatus.PARTIAL}:
            try:
                validate_json_value(result.output, operation.output_schema, "$output")
            except (KeyError, TypeError, ValueError):
                return ToolResult(
                    status=ResultStatus.FAILED, error_code="invalid_adapter_output",
                    side_effects_may_have_occurred=True,
                )
        return result
